"""Mira-owned inference server over mlx-lm's serving primitives.

Stock `mlx_lm.server` (Phase 1) works, but as a pip dependency it can't expose
things Mira actually wants to tune for Mistral-family models: KV-cache
quantization, `fix_mistral_regex`, or the exact tool-call response shape. This
module drives mlx-lm's `BatchGenerator` (continuous batching), `LRUPromptCache`
(cross-turn prefix reuse), and `ToolCallFormatter` (tool-call parsing) directly,
and exposes only the two endpoints Mira's orchestrator needs:
`/v1/chat/completions` and `/v1/models`.

MLX streams are thread-local — a stream created in one thread cannot be driven
from another (this is what silently hung vllm-mlx). So exactly one thread ("the
engine thread") ever touches the model, the BatchGenerator, or the prompt
cache, for the lifetime of the process. HTTP handlers only tokenize (plain
Python) and hand off work through a thread-safe queue.Queue; they never call
into MLX directly.
"""

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from mlx_lm.generate import BatchGenerator, SequenceStateMachine
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.server import ToolCallFormatter
from mlx_lm.utils import load

from core.inference.disk_prompt_cache import DiskBackedPromptCache, DiskPromptCacheStore

logger = logging.getLogger("mira_mlx_server")

DONE = object()  # sentinel pushed onto a job's out_queue when generation finishes


def _prepare_messages(messages: list) -> list:
    """Normalize OpenAI-shape messages for `apply_chat_template`.

    Mirrors mlx_lm.server's `process_message_content`: chat templates render
    `content` as a string (None crashes Jinja's `len()` calls) and expect
    `tool_calls[].function.arguments` as a parsed object, not the JSON string
    the OpenAI wire format uses.
    """
    prepared = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if content is None:
            message["content"] = ""
        elif isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in ("image_url", "image")
            for part in content
        ):
            raise ValueError(
                "mira-mlx does not support image inputs yet; switch to the omlx "
                "backend for vision requests"
            )
        if tool_calls := message.get("tool_calls"):
            message["tool_calls"] = [dict(tc) for tc in tool_calls]
            for tc in message["tool_calls"]:
                func = tc.get("function")
                if func and isinstance(func.get("arguments"), str):
                    func = dict(func)
                    func["arguments"] = json.loads(func["arguments"])
                    tc["function"] = func
        prepared.append(message)
    return prepared


def _build_state_machine(tokenizer, stop_words=()):
    """Minimal re-derivation of mlx_lm.server's ModelProvider._make_state_machine,
    trimmed to a single always-loaded model (no per-model-key cache needed)."""
    transitions = {}
    common_stops = []
    for t in tokenizer.eos_token_ids:
        common_stops.append(((t,), None))
    for w in stop_words:
        common_stops.append((tuple(tokenizer.encode(w, add_special_tokens=False)), None))

    transitions["normal"] = list(common_stops)

    if tokenizer.has_thinking:
        ts, te = tokenizer.think_start_tokens, tokenizer.think_end_tokens
        transitions["normal"].append((ts, "reasoning"))
        transitions["reasoning"] = [(te, "normal")] + common_stops

    if tokenizer.has_tool_calling:
        ts, te = tokenizer.tool_call_start_tokens, tokenizer.tool_call_end_tokens
        transitions["normal"].append((ts, "tool"))
        # No end marker (Mistral's [TOOL_CALLS] is one-sided) -> only EOS exits "tool".
        transitions["tool"] = ([(te, "normal")] if te else []) + common_stops

    return SequenceStateMachine(transitions, initial="normal")


@dataclass
class ChatJob:
    messages: list
    tools: Optional[list]
    stream: bool
    max_tokens: int
    temperature: float
    top_p: float
    chat_template_kwargs: Optional[dict] = None
    out_queue: "queue.Queue" = field(default_factory=queue.Queue)
    request_id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")


class GenerationEngine:
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 4096,
        prefill_step_size: int = 1024,
        completion_batch_size: int = 1,
        prompt_cache_max_bytes: int = 12 * 1024**3,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        fix_mistral_regex: bool = False,
        # Off by default: loading a repo with this on executes that repo's own
        # Python (tokenizer auto_map) inside this process. Opt in per-model via
        # mira.yaml when a model genuinely needs a custom tokenizer class.
        trust_remote_code: bool = False,
        disk_cache_dir: Optional[str] = None,
        disk_cache_max_bytes: int = 0,
        profile_experts: bool = False,
        expert_profile_path: Optional[str] = None,
        resident_expert_fraction: Optional[float] = None,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.prefill_step_size = prefill_step_size
        self.completion_batch_size = completion_batch_size
        self.prompt_cache_max_bytes = prompt_cache_max_bytes
        self.max_kv_size = max_kv_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.fix_mistral_regex = fix_mistral_regex
        self.trust_remote_code = trust_remote_code
        self.disk_cache_dir = disk_cache_dir
        self.disk_cache_max_bytes = disk_cache_max_bytes
        self.profile_experts = profile_experts
        self.expert_profile_path = expert_profile_path
        self.resident_expert_fraction = resident_expert_fraction

        self._inbox: "queue.Queue[ChatJob]" = queue.Queue()
        self._pending: dict[int, dict] = {}  # uid -> job bookkeeping
        self._ready = threading.Event()
        self._shutdown = False
        self._error: Optional[BaseException] = None

        self.model = None
        self.tokenizer = None
        self.batch_generator: Optional[BatchGenerator] = None
        self.prompt_cache: Optional[DiskBackedPromptCache] = None
        self.state_machine: Optional[SequenceStateMachine] = None
        self.expert_profiler = None  # ExpertProfiler, only set when profile_experts=True
        self.expert_cache_store = None  # DiskExpertCacheStore, only set when resident_expert_fraction is not None

        # Diagnostics only (GET /v1/stats) — mutated on the engine thread, read
        # cross-thread from the async HTTP layer, so a lock guards both sides.
        self._stats_lock = threading.Lock()
        self._start_time = time.time()
        self._cache_hits = 0
        self._cache_misses = 0
        self._memory_pressure_trim_events = 0
        self._total_prompt_tokens = 0
        self._total_generated_tokens = 0
        self._recent_latencies_ms: "deque[float]" = deque(maxlen=200)
        self._active_memory_bytes = 0
        self._cache_memory_bytes = 0
        self._peak_memory_bytes = 0
        self._wired_limit_bytes = 0

    def start(self) -> None:
        threading.Thread(target=self._run, name="mira-mlx-engine", daemon=True).start()
        self._ready.wait(timeout=180)
        if self._error is not None:
            raise self._error
        if not self._ready.is_set():
            raise TimeoutError("mira-mlx engine did not finish loading in time")

    def submit(self, job: ChatJob) -> None:
        self._inbox.put(job)

    # --- everything below this line runs ONLY on the engine thread ---

    def _run(self) -> None:
        try:
            tokenizer_config = {"trust_remote_code": self.trust_remote_code}
            if self.fix_mistral_regex:
                tokenizer_config["fix_mistral_regex"] = True
            # When expert offload is enabled, load lazily. lazy=False makes
            # mlx_lm.load call mx.eval(model.parameters()), which eagerly
            # materializes the full stacked (num_experts, ...) expert table into
            # unified memory at load — 18.17GB on Qwen3.6-35B-A3B, the single
            # largest memory event in the whole lifecycle and the wall that stops
            # a model whose expert table exceeds DRAM from ever opening. With
            # lazy=True the table stays an unevaluated graph node (0 bytes wired);
            # install_expert_offload() below then seeds the resident set straight
            # from disk and its stand-in swap drops those nodes before anything
            # forces their eval, so the full table never becomes a live buffer at
            # any point (measured whole-run peak 18.21GB -> 7.48GB at fraction
            # 0.3; peak now scales with the resident fraction, not table size).
            # Without offload there is no stand-in swap to drop a deferred table,
            # so keep the eager default (lazy=False) — unchanged behavior.
            lazy_load = self.resident_expert_fraction is not None
            self.model, self.tokenizer = load(
                self.model_path, tokenizer_config=tokenizer_config, lazy=lazy_load
            )
            if self.profile_experts:
                from core.inference.expert_profiler import ExpertProfiler
                from core.inference.expert_profiler import install as install_expert_profiler

                if self.expert_profile_path:
                    profile_path = Path(self.expert_profile_path)
                else:
                    from core.config import DB_PATH
                    profile_path = DB_PATH.parent / "expert_profile" / f"{int(time.time())}.jsonl"
                self.expert_profiler = ExpertProfiler(
                    profile_path,
                    active_job_ids_fn=lambda: [str(uid) for uid in self._pending.keys()],
                )
                install_expert_profiler(self.model, self.expert_profiler)
            if self.resident_expert_fraction is not None:
                from core.inference.expert_offload import install as install_expert_offload

                self.expert_cache_store = install_expert_offload(
                    self.model, self.model_path, self.resident_expert_fraction
                )
            self.state_machine = _build_state_machine(self.tokenizer)
            self._eos_ids = set(self.tokenizer.eos_token_ids)
            # Capture this thread's stream so every later insert/next call is
            # pinned to it — this is the fix for the MLX cross-thread hang class.
            generation_stream = mx.default_stream(mx.default_device())
            self.batch_generator = BatchGenerator(
                self.model,
                max_tokens=self.max_tokens,
                completion_batch_size=self.completion_batch_size,
                prefill_step_size=self.prefill_step_size,
                max_kv_size=self.max_kv_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
                quantized_kv_start=self.quantized_kv_start,
                stream=generation_stream,
            )
            disk_store = None
            if self.disk_cache_dir and self.disk_cache_max_bytes > 0:
                disk_store = DiskPromptCacheStore(
                    Path(self.disk_cache_dir),
                    self.disk_cache_max_bytes,
                    kv_bits=self.kv_bits,
                    kv_group_size=self.kv_group_size,
                )
            self.prompt_cache = DiskBackedPromptCache(
                max_bytes=self.prompt_cache_max_bytes, disk_store=disk_store
            )
        except BaseException as exc:  # noqa: BLE001 - surface to start()
            self._error = exc
            self._ready.set()
            return

        from core import hardware
        self._memory_ceiling_bytes = hardware.get_total_ram_bytes() - hardware.SAFETY_MARGIN_BYTES

        # BatchGenerator's own __init__ already calls mx.set_wired_limit() to the
        # GPU driver's recommended working set (confirmed in mlx_lm.generate) —
        # this is that same value, recorded for /v1/stats rather than re-set here.
        # A NAX-capable Metal 4 GPU (M5+) reports this via device_info(); logging
        # the architecture is a cheap regression guard if a future mlx build ever
        # silently loses that acceleration path.
        device_info = mx.device_info() if mx.metal.is_available() else {}
        with self._stats_lock:
            self._wired_limit_bytes = device_info.get("max_recommended_working_set_size", 0)

        cache_limit_bytes = hardware.derive_cache_limit_bytes()
        mx.set_cache_limit(cache_limit_bytes)

        logger.info(
            "mira-mlx engine ready: %s (gpu=%s wired_limit=%.2fGB cache_limit=%.2fGB)",
            self.model_path,
            device_info.get("architecture", "unknown"),
            self._wired_limit_bytes / (1024**3),
            cache_limit_bytes / (1024**3),
        )
        self._ready.set()

        while not self._shutdown:
            drained = self._drain_inbox()
            if self._pending:
                _tn = time.time()
                responses = self.batch_generator.next_generated()
                _dt = time.time() - _tn
                if _dt > 1.0:
                    logger.info("next_generated() took %.2fs, returned %d responses", _dt, len(responses))
                for r in responses:
                    self._handle_response(r)
                self._check_memory_pressure()
            elif not drained:
                time.sleep(0.02)

    def _check_memory_pressure(self) -> None:
        """Proactively shrink the prompt-cache pool if real MLX memory use is
        approaching the machine's ceiling — catches pressure from the active
        generation's own live KV cache, which sits outside prompt_cache's own
        byte budget and isn't covered by its self-contained eviction alone."""
        active = mx.get_active_memory()
        cache = mx.get_cache_memory()
        with self._stats_lock:
            self._active_memory_bytes = active
            self._cache_memory_bytes = cache
            self._peak_memory_bytes = mx.get_peak_memory()

        used = active + cache
        if used <= self._memory_ceiling_bytes:
            return
        target = max(self.prompt_cache.nbytes // 2, 0)
        logger.warning(
            "memory pressure: %.2fGB used vs %.2fGB ceiling — trimming prompt cache pool to %.2fGB",
            used / (1024**3), self._memory_ceiling_bytes / (1024**3), target / (1024**3),
        )
        self.prompt_cache.trim_to(n_bytes=target)
        mx.clear_cache()
        with self._stats_lock:
            self._memory_pressure_trim_events += 1

    def stats_snapshot(self) -> dict:
        """Cross-thread read of the diagnostics counters (see GET /v1/stats)."""
        with self._stats_lock:
            hits, misses = self._cache_hits, self._cache_misses
            trims = self._memory_pressure_trim_events
            prompt_tokens = self._total_prompt_tokens
            generated_tokens = self._total_generated_tokens
            latencies = sorted(self._recent_latencies_ms)
            active_memory_bytes = self._active_memory_bytes
            cache_memory_bytes = self._cache_memory_bytes
            peak_memory_bytes = self._peak_memory_bytes
            wired_limit_bytes = self._wired_limit_bytes

        total_requests = hits + misses

        def _percentile(sorted_values, pct):
            if not sorted_values:
                return None
            idx = min(int(len(sorted_values) * pct), len(sorted_values) - 1)
            return round(sorted_values[idx], 1)

        disk_store = getattr(self.prompt_cache, "disk_store", None)
        expert_cache_stats = None
        if self.expert_cache_store is not None:
            from core.inference.expert_offload import stats as expert_offload_stats

            ec = expert_offload_stats(self.model)
            ec_total = ec["hits"] + ec["misses"]
            ec_total_decode = ec["hits_decode"] + ec["misses_decode"]
            expert_cache_stats = {
                "hits": ec["hits"],
                "misses": ec["misses"],
                # Blended over the process lifetime and summed across all modules,
                # so a cold prefill drags it down; decode_hit_rate is the steady-
                # state number a residency/policy change actually moves.
                "hit_rate": round(ec["hits"] / ec_total, 3) if ec_total else None,
                "decode_hit_rate": round(ec["hits_decode"] / ec_total_decode, 3) if ec_total_decode else None,
            }
        return {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": round(hits / total_requests, 3) if total_requests else None,
            "disk_cache_hits": disk_store.hits if disk_store is not None else 0,
            "expert_cache": expert_cache_stats,
            "memory_pressure_trim_events": trims,
            "total_prompt_tokens": prompt_tokens,
            "total_generated_tokens": generated_tokens,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "latency_sample_size": len(latencies),
            "active_memory_bytes": active_memory_bytes,
            "cache_memory_bytes": cache_memory_bytes,
            "peak_memory_bytes": peak_memory_bytes,
            "wired_limit_bytes": wired_limit_bytes,
        }

    def _drain_inbox(self) -> bool:
        drained_any = False
        while True:
            try:
                job = self._inbox.get_nowait()
            except queue.Empty:
                break
            drained_any = True
            self._start_job(job)
        return drained_any

    def _start_job(self, job: ChatJob) -> None:
        try:
            ckwargs = job.chat_template_kwargs or {}
            prompt_tokens = self.tokenizer.apply_chat_template(
                _prepare_messages(job.messages), tools=job.tools, add_generation_prompt=True,
                tokenize=True, **ckwargs
            )
        except BaseException as exc:  # noqa: BLE001
            job.out_queue.put(exc)
            job.out_queue.put(DONE)
            return

        # A single prompt at or beyond max_kv_size leaves no room for even one
        # generated token, and RotatingKVCache's behavior in that case is
        # undefined — reject clearly instead of letting it degrade silently.
        if self.max_kv_size is not None and len(prompt_tokens) >= self.max_kv_size:
            exc = ValueError(
                f"prompt is {len(prompt_tokens)} tokens, this machine's context "
                f"ceiling is {self.max_kv_size} tokens"
            )
            job.out_queue.put(exc)
            job.out_queue.put(DONE)
            return

        _t0 = time.time()
        cache, rest = self.prompt_cache.fetch_nearest_cache(self.model_path, prompt_tokens)
        _t1 = time.time()
        prompt_cache_count = len(prompt_tokens) - len(rest)
        logger.info(
            "cache %s: %d/%d prompt tokens reused (fetch_nearest_cache took %.2fs)",
            "HIT" if prompt_cache_count > 0 else "MISS",
            prompt_cache_count,
            len(prompt_tokens),
            _t1 - _t0,
        )
        with self._stats_lock:
            if prompt_cache_count > 0:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            self._total_prompt_tokens += len(prompt_tokens)

        sampler = make_sampler(temp=job.temperature, top_p=job.top_p)
        logits_processors = make_logits_processors()

        _t2 = time.time()
        (uid,) = self.batch_generator.insert_segments(
            segments=[[rest]],
            max_tokens=[min(job.max_tokens, self.max_tokens)],
            caches=[cache],
            all_tokens=[prompt_tokens[:prompt_cache_count]],
            samplers=[sampler],
            logits_processors=[logits_processors],
            state_machines=[self.state_machine],
        )
        logger.info("insert_segments took %.2fs (rest=%d tokens)", time.time() - _t2, len(rest))
        self._pending[uid] = {
            "job": job,
            "detokenizer": self.tokenizer.detokenizer,
            "tool_formatter": ToolCallFormatter(self.tokenizer.tool_parser, job.tools, streaming=job.stream),
            "tool_text": "",
            "tool_calls_raw": [],
            "made_tool_call": False,
            "in_tool_state": False,
            "prev_state": "normal",
            "created": int(time.time()),
            "start_time": time.time(),
        }

    def _handle_response(self, r) -> None:
        state = self._pending.get(r.uid)
        if state is None:
            return
        job = state["job"]
        state["detokenizer"].add_token(r.token)
        with self._stats_lock:
            self._total_generated_tokens += 1

        # Buffer raw text while inside a "tool" state segment.
        in_tool_text = r.current_state == "tool" or state["prev_state"] == "tool"
        if in_tool_text:
            state["tool_text"] += state["detokenizer"].last_segment
        # EOS/stop tokens decode to literal special-token text (e.g. "</s>") —
        # never part of the visible message, regardless of state.
        if in_tool_text or r.token in self._eos_ids:
            text_delta = ""
        else:
            text_delta = state["detokenizer"].last_segment
        state["prev_state"] = r.current_state or state["prev_state"]

        finished = r.finish_reason is not None
        if finished:
            state["detokenizer"].finalize()
            # finalize() can flush trailing bytes (e.g. a multi-byte char cut
            # off mid-token by a "length" stop) that last_segment hasn't
            # returned yet — same in_tool_text/EOS routing as above.
            trailing = state["detokenizer"].last_segment
            if trailing:
                if in_tool_text:
                    state["tool_text"] += trailing
                elif r.token not in self._eos_ids:
                    text_delta += trailing
            # Flush any buffered tool call unconditionally — this is the
            # ml-explore/mlx-lm#1373 fix: with no tool-call end marker (Mistral),
            # EOS is the only way out of "tool" state and it clears prev_state,
            # so gating the flush on prev_state == "tool" silently drops it.
            if state["tool_text"]:
                # For two-sided markers (e.g. Qwen's <tool_call>...</tool_call>),
                # the trailing prev_state=="tool" fallback above (needed for
                # Mistral's one-sided marker) also captures the closing marker
                # token itself, leaving a literal "</tool_call>" tail that
                # breaks tool_parsers expecting the call to end at "</function>".
                # Strip it — a no-op for Mistral, which has no tool_call_end.
                tool_text = state["tool_text"]
                end_marker = self.tokenizer.tool_call_end
                if end_marker and tool_text.endswith(end_marker):
                    tool_text = tool_text[: -len(end_marker)]
                state["tool_calls_raw"].append(tool_text)
                state["made_tool_call"] = True

            finish_reason = r.finish_reason
            if finish_reason == "stop" and state["made_tool_call"]:
                finish_reason = "tool_calls"
            tool_calls = state["tool_formatter"](state["tool_calls_raw"])

            if job.stream:
                if text_delta:
                    job.out_queue.put(self._chunk(job, delta={"content": text_delta}, finish_reason=None))
                if tool_calls:
                    job.out_queue.put(self._chunk(job, delta={"tool_calls": tool_calls}, finish_reason=None))
                job.out_queue.put(self._chunk(job, delta={}, finish_reason=finish_reason))
            else:
                message = {"role": "assistant", "content": state.get("full_text", "") + text_delta}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                    message["content"] = message["content"] or None
                job.out_queue.put(self._response(job, message=message, finish_reason=finish_reason))
            job.out_queue.put(DONE)

            if r.prompt_cache is not None and r.all_tokens is not None:
                self.prompt_cache.insert_cache(self.model_path, r.all_tokens, r.prompt_cache)
                logger.info("insert_cache: registered %d tokens (finish_reason=%s)", len(r.all_tokens), r.finish_reason)
            else:
                logger.info(
                    "insert_cache SKIPPED: prompt_cache=%s all_tokens=%s (finish_reason=%s)",
                    r.prompt_cache is not None, r.all_tokens is not None, r.finish_reason,
                )
            with self._stats_lock:
                self._recent_latencies_ms.append((time.time() - state["start_time"]) * 1000)
            del self._pending[r.uid]
        elif job.stream and text_delta:
            job.out_queue.put(self._chunk(job, delta={"content": text_delta}, finish_reason=None))
        elif not job.stream and text_delta:
            state["full_text"] = state.get("full_text", "") + text_delta

    def _chunk(self, job: ChatJob, delta: dict, finish_reason: Optional[str]) -> dict:
        return {
            "id": job.request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def _response(self, job: ChatJob, message: dict, finish_reason: Optional[str]) -> dict:
        return {
            "id": job.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        }


def create_app(engine: GenerationEngine) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": engine.model_path, "object": "model"}]}

    @app.get("/v1/stats")
    async def stats():
        return engine.stats_snapshot()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        job = ChatJob(
            messages=body["messages"],
            tools=body.get("tools"),
            stream=bool(body.get("stream", False)),
            max_tokens=body.get("max_tokens") or engine.max_tokens,
            temperature=body.get("temperature", 0.0) or 0.0,
            top_p=body.get("top_p", 0.0) or 0.0,
            chat_template_kwargs=body.get("chat_template_kwargs"),
        )
        loop = asyncio.get_event_loop()
        engine.submit(job)

        if job.stream:
            async def event_gen():
                while True:
                    item = await loop.run_in_executor(None, job.out_queue.get)
                    if item is DONE:
                        yield {"data": "[DONE]"}
                        break
                    if isinstance(item, BaseException):
                        # OpenAI-compatible clients (the openai SDK, orchestrator's
                        # bc.normalize_oai_stream) expect error.message, not a bare
                        # string — a bare string here silently becomes the SDK's
                        # own generic "An error occurred during streaming",
                        # discarding whatever the real exception said.
                        yield {"data": json.dumps({"error": {"message": str(item), "type": type(item).__name__}})}
                        break
                    yield {"data": json.dumps(item)}

            return EventSourceResponse(event_gen())

        # Non-streaming: drain until DONE, keep the last real payload.
        result = None
        while True:
            item = await loop.run_in_executor(None, job.out_queue.get)
            if item is DONE:
                break
            if isinstance(item, BaseException):
                # A bare `raise item` here hits FastAPI's default 500 handler,
                # which discards the exception message entirely (client just
                # sees "Internal Server Error") — surface it explicitly instead.
                status = 400 if isinstance(item, ValueError) else 500
                raise HTTPException(status_code=status, detail=str(item))
            result = item
        return result

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Mira-owned mlx-lm-based inference server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--prefill-step-size", type=int, default=1024)
    parser.add_argument("--completion-batch-size", type=int, default=1)
    # A single ~21K-token KV cache entry measured ~3.3GB for this model; too low a
    # cap here silently evicts an entry right after inserting it (cache never hits).
    parser.add_argument("--prompt-cache-max-bytes", type=int, default=12 * 1024**3)
    # Bounds a single conversation's own KV cache (rotating/sliding-window) so
    # one very long conversation can't alone exhaust memory. None = unbounded
    # (fine on high-RAM machines; backend_manager.py derives a real value
    # per-machine for normal use).
    parser.add_argument("--max-kv-size", type=int, default=None)
    # KV-cache quantization (opt-in; None = today's unquantized fp16 behavior).
    # 8-bit is the only numerically-validated setting (fork's own test suite,
    # rtol=4e-2); 4-bit is unproven anywhere in this codebase.
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    # Delayed quantization (quantize only after this many tokens) is unsupported
    # on the batching path — BatchGenerator raises if this is nonzero alongside
    # kv_bits. Left as a CLI arg for parity with mlx-lm's single-sequence API.
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--fix-mistral-regex", action="store_true")
    # Default off. --no-trust-remote-code kept as an accepted no-op so any
    # existing caller passing it keeps working.
    parser.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code",
                        default=False)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    # Disk overflow for evicted prompt-cache entries. Unset dir or a zero byte
    # budget disables persistence entirely (in-memory-only, today's behavior).
    parser.add_argument("--disk-cache-dir", default=None)
    parser.add_argument("--disk-cache-max-bytes", type=int, default=0)
    # Opt-in MoE expert-activation logging for the offloading go/no-go decision
    # (specs/moe-expert-offload-01-profiling.md). No-op on dense models; zero
    # overhead unless explicitly set. See core/inference/expert_profiler.py.
    parser.add_argument("--profile-experts", action="store_true")
    parser.add_argument("--expert-profile-path", default=None)
    # Opt-in MoE expert disk offloading (specs/moe-expert-offload-02-runtime-cache.md).
    # None (default) = every expert resident, today's behavior. No-op on dense
    # models. See core/inference/expert_offload.py.
    parser.add_argument("--resident-expert-fraction", type=float, default=None)
    args = parser.parse_args()

    engine = GenerationEngine(
        model_path=args.model,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
        completion_batch_size=args.completion_batch_size,
        prompt_cache_max_bytes=args.prompt_cache_max_bytes,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        fix_mistral_regex=args.fix_mistral_regex,
        trust_remote_code=args.trust_remote_code,
        disk_cache_dir=args.disk_cache_dir,
        disk_cache_max_bytes=args.disk_cache_max_bytes,
        profile_experts=args.profile_experts,
        expert_profile_path=args.expert_profile_path,
        resident_expert_fraction=args.resident_expert_fraction,
    )
    engine.start()

    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
