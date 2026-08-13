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
import gc
import json
import logging
import queue
import secrets
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

from mlx_lm.generate import BatchGenerator, SequenceStateMachine, _embed_tokens
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.server import ToolCallFormatter
from mlx_lm.utils import load

from core.inference.disk_prompt_cache import DiskBackedPromptCache, DiskPromptCacheStore

logger = logging.getLogger("mira_mlx_server")

# How often the idle loop re-derives the memory ceiling from real system state.
# The loop itself spins at 50Hz; the probe is ~2.5ms of subprocess, so it needs
# its own timer rather than running per-iteration. 30s is well under the time it
# takes someone to open an IDE and start a build, and costs ~0.008% of a core.
SYSTEM_MEMORY_PROBE_INTERVAL_S = 30.0

# Availability required before Mira will decompress itself back into RAM.
# Deliberately NOT "as much as is currently compressed": the measured net cost of
# a decompression is about 1GB, not the full expansion, because emptying the
# compressor frees most of what the expansion needs. The real 18.44GB event ran
# fine with 6.49GB available and left 5.12GB, and a precondition sized against
# the compressed total would have blocked precisely the case worth acting on.
# 2GB covers both measured events (1.37GB and 1.15GB net) with room to spare.
DECOMPRESS_MIN_AVAILABLE_BYTES = 2 * 1024**3

# Prompt length for the decompression touch. Derived, not chosen: covering an
# E-expert table routed top-k is a coupon-collector problem needing about
# (E/k)*ln(E) tokens, which for 256 experts at top-8 is ~177. A single-token
# pass was measured first and reclaimed nothing at all. 512 leaves margin for
# models with wider tables and still prefills in well under a second warm.
TOUCH_TOKENS = 512
# Prime, so ids do not land in a repeating pattern that routes to the same
# experts and quietly defeats the point of the length above.
TOUCH_ID_STRIDE = 7919

# Below this, a boundary snapshot is not worth the LRU slot it occupies. The
# saving scales with how much of the next prompt is history: measured at 95% of
# a long conversation's prompt but only ~44% of a short exchange, and the LRU
# holds few entries, so a one-shot question would evict a useful one to store
# something that saves a second.
SNAPSHOT_MIN_BOUNDARY_TOKENS = 1024

# Entry-count ceiling for the in-memory prompt cache. Up to three entries per
# turn now (the completed turn, its history snapshot, and the shared system
# snapshot), so this is triple mlx-lm's default of 10 to keep the same number of
# conversations warm. Bytes are capped separately by prompt_cache_max_bytes,
# which is what actually bounds memory.
PROMPT_CACHE_MAX_ENTRIES = 30

# Which eviction class each boundary snapshot goes into.
#
# LRUPromptCache.CacheOrder evicts over ["assistant", "user", "system"] and
# drains the earlier classes before touching the later ones, so the tag is a
# priority: an entry tagged "system" is the last thing discarded. That matters
# because the entries are not equally valuable and the LRU cannot tell — the
# system snapshot is one ~88 MB object serving *every* conversation on the
# machine, while a history snapshot serves exactly one and a completed turn is
# the most replaceable thing in the pool. Losing the shared entry costs every
# subsequent conversation a full system-prompt prefill; losing a completed turn
# costs one request the tail of its own prompt.
#
# It is priority, not pinning: trim_to() still reaches a "system" entry once the
# earlier classes are empty, which is what keeps this from becoming a leak.
SNAPSHOT_CACHE_TYPE = {"system": "system", "history": "user"}

DONE = object()  # sentinel pushed onto a job's out_queue when generation finishes


class EngineDead(RuntimeError):
    """The engine thread is gone and will not come back without a restart.

    Raised by submit() rather than letting the job onto an inbox nothing
    drains. A queued job on a dead engine is indistinguishable, from the
    client's side, from a slow one: it waits on out_queue forever.
    """


def plan_prefill_segments(rest, prompt_cache_count, boundaries):
    """Decide where prefill should stop to leave reusable snapshots behind.

    Returns (segments, boundaries_to_cache). `boundaries_to_cache` lines up with
    the segment ends that precede the final segment: when segment i finishes,
    the live state corresponds to `boundaries_to_cache[i]` tokens of the whole
    prompt. It is empty when the prompt should be prefilled in one piece exactly
    as before.

    Each boundary indexes the whole prompt, while `rest` is only what the cache
    did not already cover, so each split point is the difference. A boundary at
    or behind what was already reused needs no segment: the entry that supplied
    the reuse already covers it. A boundary at or past the end of `rest` would
    make a trailing segment empty, and an empty trailing segment has nothing to
    generate from.

    Boundaries are sorted and de-duplicated because the callers find them
    independently: a conversation whose history is nothing but the system
    prompt yields the same index twice, and two segments split at the same
    point would put an empty segment in the middle.
    """
    cuts = []
    for n in sorted({b for b in (boundaries or ()) if b is not None}):
        split_at = n - prompt_cache_count
        if 0 < split_at < len(rest):
            cuts.append((split_at, n))
    if not cuts:
        return [rest], []
    segments, to_cache, prev = [], [], 0
    for split_at, n in cuts:
        segments.append(rest[prev:split_at])
        to_cache.append(n)
        prev = split_at
    segments.append(rest[prev:])
    return segments, to_cache


def _decode_image_part(part: dict):
    """Turn one OpenAI `image_url` content part into a PIL image.

    Only data URLs are accepted. Fetching a remote URL here would give the model
    a way to make the server issue arbitrary outbound requests, which is exactly
    the SSRF shape the rest of the codebase guards against.
    """
    import base64
    import io

    from PIL import Image

    url = part.get("image_url", {})
    if isinstance(url, dict):
        url = url.get("url", "")
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError(
            "image parts must be data URLs; mira-mlx does not fetch remote images"
        )
    _, _, payload = url.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def _prepare_messages(messages: list, vision: bool = False) -> tuple:
    """Normalize OpenAI-shape messages for `apply_chat_template`.

    Mirrors mlx_lm.server's `process_message_content`: chat templates render
    `content` as a string (None crashes Jinja's `len()` calls) and expect
    `tool_calls[].function.arguments` as a parsed object, not the JSON string
    the OpenAI wire format uses.

    With `vision` on, image parts are decoded and pulled out, and the part is
    rewritten to the bare `{"type": "image"}` the chat template understands. It
    renders as a single `<|image_pad|>`, which the caller expands to one token
    per image patch, since we drive the tokenizer directly and so never get the
    HF processor that would normally do that expansion.

    Returns (prepared_messages, images) where images is in prompt order.
    """
    prepared = []
    images = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if content is None:
            message["content"] = ""
        elif isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in ("image_url", "image")
            for part in content
        ):
            if not vision:
                raise ValueError(
                    "mira-mlx does not support image inputs yet; switch to the omlx "
                    "backend for vision requests"
                )
            rewritten = []
            for part in content:
                if not isinstance(part, dict):
                    rewritten.append(part)
                elif part.get("type") == "image_url":
                    images.append(_decode_image_part(part))
                    rewritten.append({"type": "image"})
                elif part.get("type") == "image":
                    raise ValueError(
                        "bare 'image' parts carry no data; send an image_url data URL"
                    )
                else:
                    rewritten.append(part)
            message["content"] = rewritten
        if tool_calls := message.get("tool_calls"):
            message["tool_calls"] = [dict(tc) for tc in tool_calls]
            for tc in message["tool_calls"]:
                func = tc.get("function")
                if func and isinstance(func.get("arguments"), str):
                    func = dict(func)
                    func["arguments"] = json.loads(func["arguments"])
                    tc["function"] = func
        prepared.append(message)
    return prepared, images


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


def _rfind_subseq(tokens, needle) -> int:
    """Index of the last occurrence of ``needle`` in ``tokens``, or -1."""
    n = len(needle)
    if not n or len(tokens) < n:
        return -1
    for i in range(len(tokens) - n, -1, -1):
        if all(tokens[i + j] == needle[j] for j in range(n)):
            return i
    return -1


def _think_preopened(prompt_tokens, think_start_tokens, think_end_tokens=()) -> bool:
    """True when the chat template left a thinking block open at the end of the prompt.

    Qwen3-style templates put the thinking marker in the prompt, so the model never
    emits think_start and the state machine stays in "normal" for the whole reasoning
    stretch. Without this, every reasoning token on the default model is counted as
    answer text.

    Testing the prompt's last tokens is not enough: Qwen3.6 ends the prompt with
    ``<think>\\n``, a newline token *after* the marker, and with thinking disabled it
    emits a pre-*closed* ``<think>\\n\\n</think>\\n\\n``. What decides it is whether the
    last opener comes after the last closer.
    """
    start = _rfind_subseq(prompt_tokens, tuple(think_start_tokens))
    if start < 0:
        return False
    return _rfind_subseq(prompt_tokens, tuple(think_end_tokens)) < start


def _passthrough_processor(tokens, logits):
    """A logits processor that does nothing, to keep a sequence's list non-empty.

    See the note at its call site: mlx-lm turns a falsy per-sequence processor
    list into None when batching mixes sequences with and without processors.
    """
    return logits


class ThinkingBudget:
    """Closes the reasoning block once it has run past its budget.

    The budget was nominally configurable since MAX_THINKING_TOKENS was added,
    but it travelled as a chat-template kwarg and **Qwen3.6's template does not
    reference `thinking_budget` at all** — Jinja discards unknown kwargs in
    silence, so nothing ever enforced it. This does.

    It forces `</think>` rather than stopping generation, which is the whole
    point: a hard stop at the budget yields a reply that is all reasoning and no
    answer, which is the truncated-thinking failure this project already fixed
    once. Forcing the closer lets the model spend its remaining tokens saying
    something.

    Attached per sequence, so the state here is one request's.
    """

    def __init__(self, budget: int, think_start, think_end, preopened: bool):
        self.budget = budget
        self.think_start = tuple(think_start or ())
        self.think_end = tuple(think_end or ())
        # Qwen3-style templates open the block in the prompt, so the model never
        # emits think_start and reasoning is already running at token 0.
        self.started = preopened
        self.closed = False
        self.forced = 0
        self.count = 0

    def __call__(self, tokens, logits):
        if self.closed or not self.think_end:
            return logits

        # `tokens` is this sequence's whole history and it grows by one per step,
        # so only ever look at the tail: reading all of it would mean a Python
        # list of thousands of ints on every decode step. Shapes differ between
        # mlx-lm's single and batched paths, hence the flatten.
        if not self.started or self.forced == 0:
            window = max(len(self.think_start), len(self.think_end))
            flat = tokens.reshape(-1) if hasattr(tokens, "reshape") else tokens
            tail = flat[-window:]
            tail = [int(t) for t in (tail.tolist() if hasattr(tail, "tolist") else tail)]
            if not self.started and self.think_start and \
                    _rfind_subseq(tail, self.think_start) >= 0:
                self.started = True
            if _rfind_subseq(tail, self.think_end) >= 0:
                # The model closed it on its own, inside budget. Nothing to do,
                # ever again for this request.
                self.closed = True
                return logits

        if not self.started:
            return logits

        self.count += 1
        if self.count < self.budget:
            return logits

        # Over budget: emit think_end one token at a time by flooring every
        # other logit. -1e4 rather than -inf so this is representable in float16
        # and cannot produce a NaN in softmax; it is far below any real logit.
        target = self.think_end[self.forced]
        self.forced += 1
        if self.forced >= len(self.think_end):
            self.closed = True
        logger.info("thinking budget HIT at %d tokens: forcing token %d (%d/%d)",
                    self.count, target, self.forced, len(self.think_end))
        idx = mx.arange(logits.shape[-1])
        return mx.where(idx == target, logits, mx.array(-1e4, logits.dtype))


def _build_logits_processors(thinking_budget, think_start, think_end,
                             prompt_tokens, enable_thinking, penalties=None):
    """The per-sequence logits processors for one job. Never returns an empty list.

    The budget is attached only when thinking is actually on for this turn: with
    it off the model never opens a block, and a processor watching for a closer
    that will never come could force one into ordinary answer text.

    `penalties` carries the repetition/presence/frequency knobs. mlx-lm builds a
    processor only for a penalty that is neither None nor 0, so an unset install
    gets the same empty list it always got. They are per sequence, which matters
    under continuous batching: each job owns its own penalty state and nothing
    leaks between two requests sharing a batch.

    The list is never empty because mlx-lm's PromptProcessingBatch.extend
    replaces a sequence's processors with None whenever `any()` of the batch's
    lists is falsy. Batch one sequence that has a processor with one that has
    none and the second ends up holding None, and _step() then runs
    `for processor in None` -- which raises inside the engine thread and takes
    the whole engine down. A no-op keeps every sequence's list truthy.
    """
    processors = list(make_logits_processors(**(penalties or {})))
    if thinking_budget and think_end and enable_thinking is not False:
        preopened = _think_preopened(prompt_tokens, think_start, think_end)
        processors.append(ThinkingBudget(
            budget=thinking_budget,
            think_start=think_start,
            think_end=think_end,
            preopened=preopened,
        ))
        logger.info("thinking budget active: %d tokens (preopened=%s, end_ids=%s)",
                    thinking_budget, preopened, think_end)
    return processors or [_passthrough_processor]


@dataclass
class ChatJob:
    messages: list
    tools: Optional[list]
    stream: bool
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int = 0
    # None means "draw a fresh one at admission", which is what makes a
    # regenerate resample instead of returning the previous reply verbatim.
    seed: Optional[int] = None
    # Keyword arguments for make_logits_processors: repetition/presence/
    # frequency penalties and their context sizes. Empty means every penalty
    # stays off, which is the shipped default.
    penalties: Optional[dict] = None
    chat_template_kwargs: Optional[dict] = None
    # OpenAI's `stream_options.include_usage`. Streaming responses carry no
    # usage unless the client asks, and then it arrives as one extra chunk with
    # an empty `choices` list after the finish_reason chunk.
    include_usage: bool = False
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
        vision: bool = False,
        vision_max_pixels: Optional[int] = None,
        vision_tower_idle_timeout: float = 300.0,
        proactive_decompress: bool = False,
        boundary_snapshot: bool = False,
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
        self.vision = vision
        self.vision_max_pixels = vision_max_pixels
        self.vision_tower_idle_timeout = vision_tower_idle_timeout
        self.proactive_decompress = proactive_decompress
        self.boundary_snapshot = boundary_snapshot
        self._snapshots_taken = 0
        # Split by boundary kind, because the two answer different questions:
        # "system" is what makes conversation openers cheap, "history" is what
        # makes continuations cheap, and a change that helps one can quietly
        # stop the other from firing at all.
        self._snapshots_by_kind = {"system": 0, "history": 0}
        # Memo for the two-render system probe. Rendering and encoding ~3,600
        # tokens twice per request is not free, and the system prompt changes
        # only when the project, memories or date do.
        self._system_probe_cache = {}
        self._snapshots_skipped_short = 0
        self._snapshot_failures = 0
        self._snapshot_last_seconds = 0.0
        self._snapshot_last_bytes = 0
        # Loaded lazily on the first image turn and released again after
        # vision_tower_idle_timeout, so a text-only session never pays the
        # 0.89GB and an occasional image costs it only while in use.
        self.vision_tower = None
        self._tower_last_used = 0.0
        self._tower_loads = 0
        self._tower_unloads = 0
        self._tower_last_reclaimed_bytes = 0
        self._vision_error: Optional[str] = None
        self._image_count = 0
        self._image_tokens = 0

        self._inbox: "queue.Queue[ChatJob]" = queue.Queue()
        self._pending: dict[int, dict] = {}  # uid -> job bookkeeping
        self._ready = threading.Event()
        self._shutdown = False
        self._error: Optional[BaseException] = None
        # Held for the check-then-put in submit() and for the set-then-drain in
        # _die(). Without it a job can pass the liveness check on an HTTP thread
        # and land in the inbox just after the engine thread finished draining
        # it — the one job that would still hang on an engine that is already
        # reporting itself dead to everyone else.
        self._admit_lock = threading.Lock()

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
        # Times the ceiling was breached but releasing MLX's reuse cache was
        # enough, so no prefill was thrown away. Counted separately from
        # _memory_pressure_trim_events: they are the same trigger with very
        # different costs, and collapsing them hides which one is happening.
        self._buffer_cache_sufficed = 0
        self._last_trim: dict | None = None
        self._total_prompt_tokens = 0
        self._total_generated_tokens = 0
        self._recent_latencies_ms: "deque[float]" = deque(maxlen=200)
        # The whole-request latency above answers "was that slow" and nothing
        # else. These four split it, because every throughput argument this
        # project has had came down to not knowing which half moved: a request
        # is time-to-first-token (queue + cache fetch + prefill) followed by
        # decode, and only the second half is the tok/s figure people quote.
        self._recent_ttft_ms: "deque[float]" = deque(maxlen=200)
        self._recent_decode_ms: "deque[float]" = deque(maxlen=200)
        self._recent_decode_tps: "deque[float]" = deque(maxlen=200)
        self._recent_prefill_tps: "deque[float]" = deque(maxlen=200)
        self._active_memory_bytes = 0
        self._cache_memory_bytes = 0
        self._peak_memory_bytes = 0
        self._wired_limit_bytes = 0
        # System-memory advisory. The ceiling used to be a hardware constant
        # (hw.memsize - margin) computed once at start, which cannot notice that
        # this is a Mac someone also uses. These are refreshed on the idle branch.
        self._system_state: dict = {"advisory": "unknown", "source": "not yet probed"}
        self._system_state_checked_at = 0.0
        # Set for real on the model thread once the engine starts, and re-derived
        # from live system state on the idle branch after that. Declared here so
        # stats_snapshot() works on an engine that has not been started.
        self._memory_ceiling_bytes = 0
        # Proactive decompression. `_decompress_done_for_event` is what makes this
        # once-per-eviction rather than a loop: it latches on acting and only
        # clears when the advisory leaves `evicted`, so a model the OS keeps
        # re-compressing is left alone instead of being fought over.
        self._decompress_events = 0
        self._decompress_last_seconds = 0.0
        self._decompress_last_reclaimed_bytes = 0
        self._decompress_skipped_no_headroom = 0
        self._decompress_failures = 0
        self._decompress_done_for_event = False

    def start(self) -> None:
        threading.Thread(target=self._run, name="mira-mlx-engine", daemon=True).start()
        self._ready.wait(timeout=180)
        if self._error is not None:
            raise self._error
        if not self._ready.is_set():
            raise TimeoutError("mira-mlx engine did not finish loading in time")

    def submit(self, job: ChatJob) -> None:
        with self._admit_lock:
            if self._error is not None:
                raise EngineDead(
                    f"mira-mlx engine is not running: "
                    f"{type(self._error).__name__}: {self._error}"
                )
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
            # max_size was previously left at mlx-lm's default of 10. With the
            # boundary snapshot a turn stores two entries instead of one, which
            # would have quietly halved the number of conversations that stay
            # warm. 20 restores it. Bytes remain the real ceiling and are
            # enforced separately (insert_cache evicts on both), so this cannot
            # grow memory beyond prompt_cache_max_bytes.
            self.prompt_cache = DiskBackedPromptCache(
                max_size=PROMPT_CACHE_MAX_ENTRIES,
                max_bytes=self.prompt_cache_max_bytes,
                disk_store=disk_store,
            )
            # The tower is NOT built here. Startup stays text-only and a session
            # that never sends an image never pays the 0.89GB. It is loaded on
            # the first image turn instead - measured at 0.14s page-cached, 1.94s
            # cold - and released again after an idle period. See _ensure_tower().
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
        self._serve_forever()

    def _serve_forever(self) -> None:
        """The generation loop. Split out of _run so it can be exercised without
        a model load — the failure path below is the one that used to be
        untestable and therefore untested."""
        while not self._shutdown:
            try:
                drained = self._drain_inbox()
                if self._pending:
                    _tn = time.time()
                    # next() rather than next_generated(): the end_of_segment signal
                    # the boundary snapshot needs rides on the prompt responses that
                    # next_generated() discards. A step with no generated tokens
                    # simply loops again, which is what next_generated() did
                    # internally anyway.
                    prompt_responses, responses = self.batch_generator.next()
                    _dt = time.time() - _tn
                    if _dt > 1.0:
                        logger.info("next() took %.2fs, returned %d responses", _dt, len(responses))
                    for pr in prompt_responses:
                        self._maybe_snapshot_boundary(pr)
                    for r in responses:
                        self._handle_response(r)
                    self._check_memory_pressure()
                elif not drained:
                    self._release_idle_tower()
                    self._refresh_system_memory_state()
                    self._maybe_decompress_model()
                    time.sleep(0.02)
            except BaseException as exc:  # noqa: BLE001 - see _die()
                self._die(exc)
                return

    def _die(self, exc: BaseException) -> None:
        """Fail every job the engine still owes an answer to, then stop.

        Reached only when the generation loop itself raises. Admission failures
        are handled per job in _start_job and never get here; this is the other
        kind — the batch generator, the prompt cache or the memory watchdog
        blew up mid-step, which leaves the batch's KV state and MLX stream of
        unknown validity. Decoding the next token on top of that would produce
        wrong output rather than an error, so the loop does not continue.

        What it must not do is continue *silently*. Before this existed the
        thread simply unwound: every in-flight request stayed blocked on its
        out_queue, every later request queued onto an inbox with no reader, and
        the server went on answering /v1/stats as if it were healthy. One
        exception became a permanently hung process with nothing in the log to
        say why. Now the exception reaches the clients that were waiting for
        it, submit() refuses new work with the same cause attached, and the
        traceback is on disk.
        """
        logger.exception(
            "mira-mlx engine loop died; failing %d in-flight job(s)", len(self._pending)
        )
        # Set the flag before draining the inbox: submit() takes the same lock,
        # so from here on a caller either sees the flag and raises, or already
        # put its job where the drain below will find it.
        with self._admit_lock:
            self._error = exc
            orphans = []
            while True:
                try:
                    orphans.append(self._inbox.get_nowait())
                except queue.Empty:
                    break
        for state in list(self._pending.values()):
            orphans.append(state["job"])
        self._pending.clear()
        for job in orphans:
            # Best effort per job: a client that has already disconnected must
            # not stop the next one from being told.
            try:
                job.out_queue.put(exc)
                job.out_queue.put(DONE)
            except Exception:  # noqa: BLE001
                logger.exception("could not deliver engine failure to job %s", job.request_id)

    def _ensure_tower(self):
        """Load the vision tower on first use, on the model thread.

        Called only from _admit_job, which runs on this thread, because the
        tower's arrays have to live on the same MLX stream as everything else.

        A tower that fails to load is not fatal and, importantly, is not retried:
        the first failure turns vision off for the process, so a broken
        checkpoint costs one load attempt rather than one per image. The backend
        keeps serving text and the orchestrator's OCR fallback takes over.
        """
        if self.vision_tower is not None:
            self._tower_last_used = time.time()
            return self.vision_tower
        if not self.vision:
            return None
        try:
            from core.inference.vision_tower import VisionTower

            _t = time.time()
            tower = VisionTower(self.model_path, max_pixels=self.vision_max_pixels)
            tower.load()
            self.vision_tower = tower
            self._tower_last_used = time.time()
            with self._stats_lock:
                self._tower_loads += 1
            logger.info(
                "vision: tower loaded in %.2fs (%.2f GB, image_token_id=%d, "
                "max_pixels=%d)",
                time.time() - _t,
                tower.weight_bytes / 1e9,
                tower.image_token_id,
                tower.max_pixels,
            )
            return tower
        except BaseException as exc:  # noqa: BLE001
            self.vision = False
            self.vision_tower = None
            # Recorded, not just logged: this subprocess's stdout goes to
            # DEVNULL, so a warning here would be invisible and vision would
            # look like it was never asked for. /v1/stats is the only channel
            # that actually reaches anyone.
            self._vision_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "vision requested but the tower failed to load (%s); "
                "continuing text-only, images will fall back to OCR",
                exc,
            )
            return None

    def _release_idle_tower(self) -> None:
        """Free the tower's 0.89GB after an idle stretch.

        Only ever called from the idle branch of the model loop, so it can never
        run while a job holds a reference to the tower. Metal kernels survive the
        round trip - measured, a reload plus first embed runs at normal warm
        speed - so the cost of getting it wrong is ~2s on the next image, not a
        recompile.
        """
        if self.vision_tower is None or self.vision_tower_idle_timeout <= 0:
            return
        if time.time() - self._tower_last_used < self.vision_tower_idle_timeout:
            return
        claimed = self.vision_tower.weight_bytes
        before = mx.get_active_memory()
        self.vision_tower = None
        gc.collect()
        mx.clear_cache()
        after = mx.get_active_memory()

        # Sample memory here rather than trusting the tower's own weight_bytes.
        # _check_memory_pressure only runs while jobs are in flight, so during an
        # idle stretch the stats counters are frozen at their last active value
        # and would report the release as having reclaimed nothing. Recording the
        # measured delta is also the only way to notice if something still holds
        # a reference and the memory does not actually come back.
        with self._stats_lock:
            self._tower_unloads += 1
            self._active_memory_bytes = after
            self._cache_memory_bytes = mx.get_cache_memory()
            self._tower_last_reclaimed_bytes = before - after
        logger.info(
            "vision: tower released after %.0fs idle, %.2f GB reclaimed "
            "(tower weights are %.2f GB)",
            self.vision_tower_idle_timeout,
            (before - after) / 1e9,
            claimed / 1e9,
        )

    def _touch_model_weights(self) -> tuple[float, int]:
        """Fault the model's weights back into RAM with one real forward pass.

        **The prompt has to be long, and this is the whole subtlety.** A
        single-token pass was tried first and measured: it ran in 1.19s and
        reclaimed exactly zero, leaving the model 13.39GB compressed, because
        top-8-of-256 routing reaches about 3% of the expert table per token. The
        earlier observation that a real request faults in ~19.8GB was not
        evidence against that; a chat-templated request is several dozen tokens,
        which is a different thing entirely.

        TOUCH_TOKENS follows from the routing arithmetic rather than taste. With
        top-k of E experts, covering the table is a coupon-collector problem
        needing about (E/k)*ln(E) tokens: for Qwen3.6's 256 experts at top-8 that
        is ~177, so 512 leaves a wide margin and still prefills in well under a
        second warm. Ids are spread with a prime stride because consecutive low
        ids are mostly special tokens and would route through the same handful
        of experts.

        Returns (seconds, reclaimed_bytes), where reclaimed is the MEASURED drop
        in this process's own compressed bytes. It is not inferred from what the
        pass should have touched, which is exactly how the single-token version
        was caught: it reported 0 instead of looking like a success.
        """
        from core import hardware

        vocab = (
            getattr(self.tokenizer, "vocab_size", None)
            or len(getattr(self.tokenizer, "vocab", ()) or ())
            or 32_000
        )
        # Keep clear of the special-token block at the bottom of the vocab.
        ids = [(i * TOUCH_ID_STRIDE) % max(vocab - 1024, 1024) + 1024
               for i in range(TOUCH_TOKENS)]

        before = hardware.read_self_memory_state()
        started = time.time()
        out = self.model(mx.array([ids]))
        mx.eval(out)
        del out
        mx.clear_cache()
        elapsed = time.time() - started

        after = hardware.read_self_memory_state()
        reclaimed = 0
        if before is not None and after is not None:
            reclaimed = max(before["compressed_bytes"] - after["compressed_bytes"], 0)
        return elapsed, reclaimed

    def _maybe_decompress_model(self) -> None:
        """Pay the decompression on the idle branch instead of on the next reply.

        Only ever called from the idle branch, so the inbox drained empty and no
        job is in flight - the same safety argument as _release_idle_tower().

        On the one request that could arrive DURING a touch: it waits behind it,
        because the model thread is the only thread that can serve it. That is
        acceptable rather than merely tolerated. Without the touch that request
        pays the full fault-in itself (17.60s measured), so waiting behind a
        touch already in progress is never worse than doing nothing, and every
        request that arrives after it is served warm.
        """
        if not self.proactive_decompress:
            return
        from core import hardware

        with self._stats_lock:
            advisory = self._system_state.get("advisory")
            compressed = self._system_state.get("self_compressed_bytes")
            available = self._system_state.get("available_bytes")
            pressure = self._system_state.get("pressure_level")

        if advisory != "evicted":
            # The event is over. Re-arm here rather than on a timer so that
            # "once per eviction" means once per eviction, not once per interval.
            self._decompress_done_for_event = False
            return
        if self._decompress_done_for_event:
            return
        if compressed is None:
            # Only act on the per-process reading. The system-wide fallback
            # cannot tell Mira's compressed pages from another app's, and acting
            # on it would mean doing 17s of memory traffic because Xcode is busy.
            return
        if pressure is not None and pressure >= hardware.PRESSURE_CRITICAL:
            return
        if available is not None and available < DECOMPRESS_MIN_AVAILABLE_BYTES:
            with self._stats_lock:
                self._decompress_skipped_no_headroom += 1
            logger.info(
                "memory: model evicted (%.2f GB compressed) but only %.2f GB available; "
                "leaving it alone rather than pushing the rest of the Mac out",
                compressed / 1e9, available / 1e9,
            )
            self._decompress_done_for_event = True
            return
        if hardware.on_battery():
            logger.info("memory: model evicted but running on battery; not spending the traffic")
            self._decompress_done_for_event = True
            return

        self._decompress_done_for_event = True
        try:
            elapsed, reclaimed = self._touch_model_weights()
        except Exception as exc:  # noqa: BLE001
            # This runs on the model thread. An unhandled exception here would
            # take the engine loop down and stop the server serving, which is a
            # catastrophic price for an optimization nobody asked for. Turn the
            # feature off for the process instead: the failure is a property of
            # this model's forward signature, so retrying it every 30s would
            # only log the same traceback forever.
            self.proactive_decompress = False
            with self._stats_lock:
                self._decompress_failures += 1
            logger.warning(
                "memory: proactive decompression failed (%s); disabled for this "
                "process. Replies are unaffected, they just pay the fault-in.", exc,
            )
            return
        with self._stats_lock:
            self._decompress_events += 1
            self._decompress_last_seconds = round(elapsed, 3)
            self._decompress_last_reclaimed_bytes = reclaimed
            self._active_memory_bytes = mx.get_active_memory()
            self._cache_memory_bytes = mx.get_cache_memory()
        logger.info(
            "memory: decompressed the model on idle in %.2fs, %.2f GB reclaimed "
            "(was %.2f GB compressed) - the next reply does not pay for this",
            elapsed, reclaimed / 1e9, compressed / 1e9,
        )
        # Re-probe on the next pass instead of waiting out the 30s gate, so the
        # advisory and the notification stop reporting a state that is now fixed.
        self._system_state_checked_at = 0.0

    def _refresh_system_memory_state(self) -> None:
        """Re-derive the memory ceiling from what the whole Mac is doing.

        Called only from the idle branch of the model loop, which spins at 50Hz,
        so this is time-gated: the probe costs about 2.5ms of subprocess and at
        50Hz that would be roughly 12% of a core spent asking the same question.

        The ceiling only ever shrinks Mira's own appetite. It is clamped to the
        original static ceiling inside derive_dynamic_ceiling_bytes, so a
        transient "lots free" reading cannot let the model claim memory it will
        not be able to keep.
        """
        now = time.time()
        if now - self._system_state_checked_at < SYSTEM_MEMORY_PROBE_INTERVAL_S:
            return
        self._system_state_checked_at = now
        try:
            from core import hardware
            used = mx.get_active_memory() + mx.get_cache_memory()
            # This runs in the process that holds the weights, so its own
            # compressed bytes answer "is Mira evicted" as a fact, rather than
            # inferring it from a system-wide compressor that belongs to
            # everybody. ~1.4us; None off-darwin, which falls back cleanly.
            self_state = hardware.read_self_memory_state()
            ceiling, diag = hardware.derive_dynamic_ceiling_bytes(
                mira_used_bytes=used,
                self_compressed_bytes=(
                    None if self_state is None else self_state["compressed_bytes"]
                ),
            )
            if self_state is not None:
                diag["self_memory"] = self_state
        except Exception as exc:  # noqa: BLE001 - advisory only, never fatal
            logger.debug("system memory probe failed (%s)", exc)
            with self._stats_lock:
                self._system_state = {"advisory": "unknown", "source": f"probe failed: {exc}"}
            return

        previous = self._system_state.get("advisory")
        with self._stats_lock:
            self._system_state = diag
            self._memory_ceiling_bytes = ceiling
            # Refresh the memory counters here too. _check_memory_pressure is
            # the only other writer and it runs solely while jobs are in flight,
            # so an idle server reported active_memory_bytes: 0 while actually
            # holding the whole model — which reads as "nothing loaded" rather
            # than "nobody has asked yet". We already have the live values.
            self._active_memory_bytes = mx.get_active_memory()
            self._cache_memory_bytes = mx.get_cache_memory()
            self._peak_memory_bytes = mx.get_peak_memory()
        if diag["advisory"] != previous and diag["advisory"] != "ok":
            logger.warning(
                "system memory advisory %s -> %s: ceiling %.2fGB (static %.2fGB), "
                "available %.2fGB, compressor %.2fGB",
                previous, diag["advisory"], ceiling / (1024**3),
                diag["static_ceiling_bytes"] / (1024**3),
                (diag.get("available_bytes") or 0) / (1024**3),
                (diag.get("compressor_bytes") or 0) / (1024**3),
            )

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

        ceiling = self._memory_ceiling_bytes
        used = active + cache
        if used <= ceiling:
            return
        overshoot = used - ceiling

        # Give back MLX's own reuse cache FIRST, then re-measure. `used` counts
        # mx.get_cache_memory(), which is a reuse pool capped at ceiling//8
        # (~3.6GB here) and is free to discard: it costs a later reallocation,
        # never a recomputation. The prompt cache is the opposite -- every entry
        # discarded is a prefill somebody pays again, and a system checkpoint is
        # ~88MB that takes a full system-prompt prefill to rebuild.
        #
        # Ordering used to be the other way round, and it was expensive. Across
        # the five trims in the 2026-08-08..11 log the overshoot was 0.00-0.11GB
        # while the response freed 0.53-1.14GB of prompt cache, 7x to 53x more
        # than needed, with clear_cache() running immediately afterwards anyway.
        mx.clear_cache()
        used = mx.get_active_memory() + mx.get_cache_memory()
        with self._stats_lock:
            self._cache_memory_bytes = mx.get_cache_memory()

        if used <= ceiling:
            # The reuse cache alone covered it. Nothing recomputable was lost.
            with self._stats_lock:
                self._buffer_cache_sufficed += 1
                self._last_trim = {
                    "overshoot_bytes": overshoot,
                    "mlx_cache_bytes": cache,
                    "freed_by_buffer_cache_alone": True,
                    "pool_before_bytes": self.prompt_cache.nbytes,
                    "pool_after_bytes": self.prompt_cache.nbytes,
                }
            logger.info(
                "memory pressure: over by %.3fGB; releasing the %.3fGB mlx reuse "
                "cache cleared it, prompt cache untouched",
                overshoot / (1024**3), cache / (1024**3),
            )
            return

        # Still over. Free what is actually needed and no more. The margin is the
        # remaining overshoot again: trim_to() pops whole entries, so the real
        # granularity is one ~88MB entry regardless, and doubling only guarantees
        # the check does not retrigger on the very next pass.
        remaining = used - ceiling
        pool_before = self.prompt_cache.nbytes
        target = max(pool_before - remaining * 2, 0)
        logger.warning(
            "memory pressure: %.2fGB used vs %.2fGB ceiling, still over by %.3fGB "
            "after releasing the reuse cache — trimming prompt cache %.2fGB -> %.2fGB",
            used / (1024**3), ceiling / (1024**3), remaining / (1024**3),
            pool_before / (1024**3), target / (1024**3),
        )
        self.prompt_cache.trim_to(n_bytes=target)
        with self._stats_lock:
            self._memory_pressure_trim_events += 1
            self._last_trim = {
                "overshoot_bytes": overshoot,
                "mlx_cache_bytes": cache,
                "freed_by_buffer_cache_alone": False,
                "remaining_after_clear_bytes": remaining,
                "pool_before_bytes": pool_before,
                "pool_after_bytes": self.prompt_cache.nbytes,
            }

    def stats_snapshot(self) -> dict:
        """Cross-thread read of the diagnostics counters (see GET /v1/stats)."""
        with self._stats_lock:
            hits, misses = self._cache_hits, self._cache_misses
            trims = self._memory_pressure_trim_events
            buffer_sufficed = self._buffer_cache_sufficed
            last_trim = self._last_trim
            prompt_tokens = self._total_prompt_tokens
            generated_tokens = self._total_generated_tokens
            latencies = sorted(self._recent_latencies_ms)
            ttfts = sorted(self._recent_ttft_ms)
            decode_mss = sorted(self._recent_decode_ms)
            decode_tpss = sorted(self._recent_decode_tps)
            prefill_tpss = sorted(self._recent_prefill_tps)
            active_memory_bytes = self._active_memory_bytes
            cache_memory_bytes = self._cache_memory_bytes
            peak_memory_bytes = self._peak_memory_bytes
            wired_limit_bytes = self._wired_limit_bytes
            image_count = self._image_count
            image_tokens = self._image_tokens
            tower_loads = self._tower_loads
            tower_unloads = self._tower_unloads
            tower_reclaimed = self._tower_last_reclaimed_bytes
            system_state = dict(self._system_state)
            memory_ceiling_bytes = self._memory_ceiling_bytes
            decompress_events = self._decompress_events
            decompress_last_seconds = self._decompress_last_seconds
            decompress_last_bytes = self._decompress_last_reclaimed_bytes
            decompress_skipped = self._decompress_skipped_no_headroom
            decompress_failures = self._decompress_failures
            snapshots_taken = self._snapshots_taken
            snapshots_by_kind = dict(self._snapshots_by_kind)
            snapshots_skipped_short = self._snapshots_skipped_short
            snapshot_failures = self._snapshot_failures
            snapshot_last_seconds = self._snapshot_last_seconds
            snapshot_last_bytes = self._snapshot_last_bytes

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

        try:
            prompt_cache_by_type = self.prompt_cache.stats_by_type()
        except Exception:  # noqa: BLE001 - diagnostics must never break /v1/stats
            prompt_cache_by_type = None

        return {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": round(hits / total_requests, 3) if total_requests else None,
            "disk_cache_hits": disk_store.hits if disk_store is not None else 0,
            "expert_cache": expert_cache_stats,
            # Both counters share a trigger and differ in cost: a trim discards
            # prefills somebody pays for again, the other discards a reuse pool
            # that only costs a reallocation. Read them as a ratio.
            "memory_pressure_trim_events": trims,
            "memory_pressure_buffer_cache_sufficed": buffer_sufficed,
            "last_trim": last_trim,
            # Sequences and bytes per eviction class (SNAPSHOT_CACHE_TYPE). The
            # counters above cannot show whether a miss was a cold start or the
            # shared system entry being discarded, and those call for opposite
            # responses. Best-effort: it reads LRUPromptCache internals, so a
            # future upstream rename must degrade to None rather than 500 /v1/stats.
            "prompt_cache_by_type": prompt_cache_by_type,
            "total_prompt_tokens": prompt_tokens,
            "total_generated_tokens": generated_tokens,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "latency_sample_size": len(latencies),
            # The split. latency_* above is ttft + decode for the same request,
            # so these three read together: how long before it started talking,
            # how long it talked, and how fast while talking. decode_tps is the
            # only one of them that is a property of the hardware; the other two
            # move with prompt length and cache hit rate.
            "ttft_p50_ms": _percentile(ttfts, 0.50),
            "ttft_p95_ms": _percentile(ttfts, 0.95),
            "decode_p50_ms": _percentile(decode_mss, 0.50),
            "decode_tps_p50": _percentile(decode_tpss, 0.50),
            "decode_tps_p95": _percentile(decode_tpss, 0.95),
            "decode_tps_sample_size": len(decode_tpss),
            # Contaminated by queueing under batching — a floor, not a rate.
            "prefill_tps_p50": _percentile(prefill_tpss, 0.50),
            "active_memory_bytes": active_memory_bytes,
            "cache_memory_bytes": cache_memory_bytes,
            "peak_memory_bytes": peak_memory_bytes,
            "wired_limit_bytes": wired_limit_bytes,
            # What the rest of the Mac is doing to us. `advisory` is the field a
            # client should show: "evicted" means the model has been compressed
            # out and the next reply pays to bring it back (measured 15.37s
            # against a warm 0.47s). The ceiling is re-derived on the idle branch
            # rather than being the boot-time hardware constant it used to be.
            "system_memory": {
                "advisory": system_state.get("advisory", "unknown"),
                "ceiling_bytes": memory_ceiling_bytes,
                "static_ceiling_bytes": system_state.get("static_ceiling_bytes"),
                "available_bytes": system_state.get("available_bytes"),
                "compressor_bytes": system_state.get("compressor_bytes"),
                "other_processes_bytes": system_state.get("other_processes_bytes"),
                "pressure_level": system_state.get("pressure_level"),
                "source": system_state.get("source"),
                # Mira's OWN compressed bytes, and which signal the verdict came
                # from. `compressor_bytes` above is system-wide and says nothing
                # about whose pages they are; this one does.
                "self_compressed_bytes": system_state.get("self_compressed_bytes"),
                "eviction_signal": system_state.get("eviction_signal"),
                # Every per-process field considered for the verdict. Kept
                # because the pmap-based ones (`pmap_compressed_bytes`,
                # `resident_bytes`) are actively misleading for a Metal process
                # and someone will otherwise reach for them again.
                "self_memory": system_state.get("self_memory"),
            },
            # Reclaiming the model on the idle branch, so the decompression is
            # not paid by whoever asks next. Measured, never assumed.
            "decompress": {
                "events": decompress_events,
                "last_seconds": decompress_last_seconds,
                "last_reclaimed_bytes": decompress_last_bytes,
                "skipped_no_headroom": decompress_skipped,
                "failures": decompress_failures,
            },
            "boundary_snapshot": {
                "enabled": self.boundary_snapshot,
                "taken": snapshots_taken,
                "taken_by_kind": snapshots_by_kind,
                "skipped_too_short": snapshots_skipped_short,
                "failures": snapshot_failures,
                "last_seconds": snapshot_last_seconds,
                "last_bytes": snapshot_last_bytes,
            },
            # `enabled` follows the config, not whether the tower happens to be
            # resident: with lazy loading it is None most of the time, and
            # reporting that as disabled would make an idle release look like a
            # failure. `tower_resident` is the one that answers "is the 0.89GB
            # in memory right now".
            "vision": (
                {
                    "enabled": True,
                    "tower_resident": self.vision_tower is not None,
                    "tower_bytes": (
                        self.vision_tower.weight_bytes
                        if self.vision_tower is not None
                        else 0
                    ),
                    "max_pixels": (
                        self.vision_tower.max_pixels
                        if self.vision_tower is not None
                        else self.vision_max_pixels
                    ),
                    "idle_timeout_s": self.vision_tower_idle_timeout,
                    "tower_loads": tower_loads,
                    "tower_unloads": tower_unloads,
                    "tower_last_reclaimed_bytes": tower_reclaimed,
                    "images_embedded": image_count,
                    "image_tokens": image_tokens,
                }
                if self.vision
                else {"enabled": False, "error": self._vision_error}
                if self._vision_error
                else None
            ),
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

    def _expand_image_tokens(self, prompt_tokens: list, images: list) -> tuple:
        """Expand each image placeholder and build the prompt's embeddings.

        The chat template emits exactly one `<|image_pad|>` per image; the HF
        processor is what normally expands that to one token per merged patch,
        and we drive the tokenizer directly, so we do it here. Then the prompt is
        embedded through the model's own table and the image rows are overwritten
        with the tower's output.

        Returns (expanded_tokens, embeddings) where embeddings covers every
        prompt token, which is the shape `insert_segments` expects.
        """
        from core.inference.vision_tower import splice_image_embeddings

        # Loads on first use and after any idle release. Raising here rather
        # than dereferencing None keeps a failed load on the existing error path
        # in _start_job, which falls the turn back to OCR.
        tower = self._ensure_tower()
        if tower is None:
            raise ValueError(
                "vision was requested but the tower is unavailable; "
                f"{self._vision_error or 'vision is off'}"
            )
        image_token_id = tower.image_token_id
        placeholders = [i for i, t in enumerate(prompt_tokens) if t == image_token_id]
        if len(placeholders) != len(images):
            raise ValueError(
                f"chat template emitted {len(placeholders)} image placeholders for "
                f"{len(images)} images"
            )

        counts = [tower.num_image_tokens(img) for img in images]
        expanded: list = []
        next_image = 0
        for token in prompt_tokens:
            if token == image_token_id:
                expanded.extend([image_token_id] * counts[next_image])
                next_image += 1
            else:
                expanded.append(token)

        if self.max_kv_size is not None and len(expanded) >= self.max_kv_size:
            raise ValueError(
                f"prompt is {len(expanded)} tokens once the {len(images)} image(s) "
                f"expand ({sum(counts)} of them are image tokens), and this "
                f"machine's context ceiling is {self.max_kv_size} tokens"
            )

        _t = time.time()
        embeds = tower.embed(images)
        text_embeddings = _embed_tokens(self.model, mx.array(expanded))
        merged = splice_image_embeddings(
            text_embeddings, expanded, embeds, image_token_id
        )
        mx.eval(merged)
        logger.info(
            "vision: %d image(s) -> %d tokens in %.2fs",
            len(images),
            sum(counts),
            time.time() - _t,
        )
        with self._stats_lock:
            self._image_count += len(images)
            self._image_tokens += sum(counts)
        return expanded, merged

    def _maybe_snapshot_boundary(self, pr) -> None:
        """Cache the state at each planned boundary, on the way past.

        Consumes one planned boundary per end_of_segment, in prefill order.
        mlx-lm raises end_of_segment as each segment drains (`generate.py:2005`)
        and once more when the sequence moves to generation, so the queue is
        always exactly one shorter than the number of events and the trailing
        event correctly finds it empty. Order is what makes the pairing safe:
        the queue was built from the same sorted boundary list that produced the
        segments, so the Nth event is the Nth boundary.

        The cache the generator is actually filling is not the one handed to
        insert_segments — PromptProcessingBatch merges the supplied caches, so
        that object is never mutated and its offset stays 0. The live state
        comes out of the batch with extract_cache(idx), resolved by uid every
        time because sequences migrate to the generation batch as they finish.

        Failures are swallowed: this is an optimisation, and a request that
        loses its snapshot is merely as slow as it was before.
        """
        if not getattr(pr, "end_of_segment", False):
            return
        state = self._pending.get(pr.uid)
        if not state:
            return
        queue = state.get("snapshot_queue")
        if not queue:
            return
        kind, tokens = queue.pop(0)
        try:
            _t = time.time()
            batch = self.batch_generator._prompt_batch
            idx = batch.uids.index(pr.uid)
            snapshot = batch.extract_cache(idx)
            mx.eval([c.state for c in snapshot if hasattr(c, "state")])
            nbytes = sum(c.nbytes for c in snapshot)
            self.prompt_cache.insert_cache(
                self.model_path, tokens, snapshot,
                cache_type=SNAPSHOT_CACHE_TYPE[kind],
            )
            elapsed = time.time() - _t
            with self._stats_lock:
                self._snapshots_taken += 1
                self._snapshots_by_kind[kind] = self._snapshots_by_kind.get(kind, 0) + 1
                self._snapshot_last_seconds = round(elapsed, 3)
                self._snapshot_last_bytes = nbytes
            logger.info(
                "boundary snapshot [%s]: %d tokens, %.1f MB, %.3fs",
                kind, len(tokens), nbytes / (1024 ** 2), elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            with self._stats_lock:
                self._snapshot_failures += 1
            logger.warning("boundary snapshot failed (%s); continuing", exc)

    def _history_boundary(self, messages, tools, ckwargs, prompt_tokens):
        """Length of the prompt up to (not including) the generation prompt.

        Qwen3's generation prompt ends `<|im_start|>assistant\\n<think>\\n`, and
        the template never re-emits that scaffold when it later replays the
        assistant turn from history. So the full prompt of turn N is NOT a prefix
        of turn N+1, and since this model's cache cannot be trimmed (its linear
        layers hold recurrent state), nothing is reusable. The history *without*
        the scaffold is a prefix of every later turn — that is the sequence worth
        caching, and it can only be captured while prefill passes through it.

        Returns None when the boundary is unusable, and the caller then behaves
        exactly as before.
        """
        return self._render_boundary(messages, tools, ckwargs, prompt_tokens)

    def _system_boundary(self, messages, tools, ckwargs, prompt_tokens):
        """Length of the prompt up to the end of the leading system message(s).

        The history boundary only ever helps turn N+1 of the conversation that
        produced it. Every *opener* still re-prefills the system prompt plus
        project context from scratch: measured 2026-08-08 over an agentic bench,
        8 of 23 requests were full misses of 3,787-4,025 identical tokens, about
        4.8s each. That prefix is shared by every conversation on the machine,
        so one entry serves all of them, which makes it the cheapest entry in
        the cache by a wide margin.

        Staleness needs no special handling. The entry is keyed on the token ids
        actually processed, so a system prompt that changes with the project,
        the memories or the date produces a different key and simply misses,
        rather than serving a stale state.

        The system messages cannot simply be rendered on their own. Qwen3's
        template raises `TemplateError: No user query found in messages` for a
        list with no user turn, so the obvious implementation returns None on
        every request and the entry is never created. Measured 2026-08-11: that
        is exactly what the first version of this method did.

        So render the system messages twice, with two different throwaway user
        messages. Both renders contain the same system block and then diverge
        where the user content starts, so their common prefix is the system
        block plus the opening of the user turn, and it does not depend on what
        this request's user actually asked. Measured on the real system prompt
        with 15 tools: 3,625 tokens of a 3,641-token opener, 99.6%.

        Returns None when there is no leading system message, when the template
        refuses both renders, or when the result is not a verified prefix of
        this prompt.
        """
        lead = 0
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role != "system":
                break
            lead += 1
        if lead == 0:
            return None

        key = (lead, id(tools), repr(sorted(ckwargs.items())) if ckwargs else "",
               tuple(str(m) for m in messages[:lead]))
        probe = self._system_probe_cache.get(key)
        if probe is None:
            probes = []
            for filler in ("zzz alpha", "qqq beta"):
                try:
                    text = self.tokenizer.apply_chat_template(
                        list(messages[:lead]) + [{"role": "user", "content": filler}],
                        tools=tools, add_generation_prompt=True,
                        tokenize=False, **ckwargs
                    )
                    probes.append(self.tokenizer.encode(text, add_special_tokens=False))
                except Exception as exc:  # noqa: BLE001 - never break a request for this
                    logger.debug("system probe render failed (%s); skipping", exc)
                    return None
            n = 0
            for a, b in zip(*probes):
                if a != b:
                    break
                n += 1
            probe = probes[0][:n]
            # One system prompt per server in practice; bound it anyway so a
            # caller varying tools or template kwargs cannot grow this forever.
            if len(self._system_probe_cache) > 8:
                self._system_probe_cache.clear()
            self._system_probe_cache[key] = probe

        n = len(probe)
        if n == 0 or n >= len(prompt_tokens):
            return None
        if list(prompt_tokens[:n]) != list(probe):
            logger.warning(
                "system boundary is not a prefix of the prompt; skipping snapshot"
            )
            return None
        return n

    def _render_boundary(self, messages, tools, ckwargs, prompt_tokens):
        """Render `messages` without the generation prompt and prove the result
        really is a prefix of `prompt_tokens`, returning its length.

        Verifying rather than assuming is the whole safety argument for both
        callers. A template that renders the same messages differently in a
        partial list than in the full one, or differently with and without
        `add_generation_prompt`, would otherwise produce a cache entry keyed on
        tokens that were never processed. That does not slow generation down, it
        silently changes what the model says.
        """
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=False,
                tokenize=False, **ckwargs
            )
            boundary = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:  # noqa: BLE001 - never break a request for this
            logger.debug("boundary render failed (%s); skipping snapshot", exc)
            return None
        n = len(boundary)
        if n == 0 or n >= len(prompt_tokens):
            return None
        if list(prompt_tokens[:n]) != list(boundary):
            logger.warning(
                "boundary is not a prefix of the prompt; skipping snapshot "
                "(template renders these messages differently in isolation)"
            )
            return None
        return n

    def _start_job(self, job: ChatJob) -> None:
        """Admit one job to the batch; a failure here costs only that job.

        The guard used to cover the prepare block alone, so everything from
        fetch_nearest_cache to insert_segments ran bare. Anything raising there
        — a chat_template_kwargs the template rejects, a logits processor built
        wrong, a prompt cache entry that will not load — propagated out through
        _drain_inbox into the engine loop and killed the thread, turning one bad
        request into a server that hangs every request after it. Admission is
        per request and its failures are per request.
        """
        try:
            self._admit_job(job)
        except BaseException as exc:  # noqa: BLE001
            logger.exception("job %s rejected during admission", job.request_id)
            job.out_queue.put(exc)
            job.out_queue.put(DONE)

    def _admit_job(self, job: ChatJob) -> None:
        ckwargs = dict(job.chat_template_kwargs or {})
        # Not a template variable — the orchestrator nests it here because
        # that is the only extra_body channel the OpenAI client exposes.
        # Pop it: Qwen3.6's template ignores unknown kwargs, but a future
        # template that errors on them would break every request.
        thinking_budget = ckwargs.pop("thinking_budget", 0)
        messages, images = _prepare_messages(job.messages, vision=self.vision)
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages, tools=job.tools, add_generation_prompt=True,
            tokenize=True, **ckwargs
        )
        image_embeds = None
        if images:
            prompt_tokens, image_embeds = self._expand_image_tokens(
                prompt_tokens, images
            )

        # A single prompt at or beyond max_kv_size leaves no room for even one
        # generated token, and RotatingKVCache's behavior in that case is
        # undefined — reject clearly instead of letting it degrade silently.
        # ValueError specifically: the HTTP layer maps it to 400, not 500.
        if self.max_kv_size is not None and len(prompt_tokens) >= self.max_kv_size:
            raise ValueError(
                f"prompt is {len(prompt_tokens)} tokens, this machine's context "
                f"ceiling is {self.max_kv_size} tokens"
            )

        _t0 = time.time()
        if image_embeds is not None:
            # The prompt cache keys on token ids alone, and an image is N copies
            # of one placeholder id. Two different screenshots at the same size
            # therefore produce a byte-identical prefix, so a cache hit would
            # answer about the previous image. Skip the cache on image turns
            # rather than key it on pixels: it costs one prefill, and it cannot
            # be got subtly wrong later.
            cache, rest = None, prompt_tokens
        else:
            cache, rest = self.prompt_cache.fetch_nearest_cache(
                self.model_path, prompt_tokens
            )
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

        # Seeding is global to MLX, not per sequence, so this is deliberately
        # only done when sampling is stochastic: at temperature 0 make_sampler
        # is argmax and never draws, so seeding would be a no-op that still
        # perturbed the RNG other in-flight sequences are drawing from. It also
        # cannot promise reproducibility under load -- continuous batching
        # changes the arithmetic, so a pinned seed reproduces on an idle engine
        # only. What it does buy is that two identical requests stop coming back
        # byte-identical, which is what made "regenerate" useless.
        if job.temperature > 0:
            seed = job.seed if job.seed is not None else secrets.randbits(32)
            mx.random.seed(seed)
            logger.debug("sampling seed %d (%s)", seed,
                         "requested" if job.seed is not None else "drawn")
        sampler = make_sampler(temp=job.temperature, top_p=job.top_p, top_k=job.top_k)
        logits_processors = _build_logits_processors(
            thinking_budget=thinking_budget,
            think_start=tuple(getattr(self.tokenizer, "think_start_tokens", ()) or ()),
            think_end=tuple(getattr(self.tokenizer, "think_end_tokens", ()) or ()),
            prompt_tokens=prompt_tokens,
            enable_thinking=ckwargs.get("enable_thinking", True),
            penalties=job.penalties,
        )

        # Split prefill at the history boundary so the state there can be
        # snapshotted on the way past. Only worth a segment (and an LRU slot)
        # when the boundary is both ahead of what the cache already covered and
        # long enough that reusing it next turn beats re-prefilling it.
        segments = [rest]
        snapshot_queue = []
        if self.boundary_snapshot and image_embeds is None:
            # Two boundaries, and they pay off for different requests. The
            # system one is a prefix of every conversation's first turn; the
            # history one is a prefix of the next turn of this conversation.
            # An opener has only the first, a continuation usually has both.
            found = {}
            for kind, finder in (
                ("system", self._system_boundary),
                ("history", self._history_boundary),
            ):
                n = finder(messages, job.tools, ckwargs, prompt_tokens)
                if n is None:
                    continue
                if n < SNAPSHOT_MIN_BOUNDARY_TOKENS:
                    with self._stats_lock:
                        self._snapshots_skipped_short += 1
                    continue
                found[n] = kind  # same index from both finders: one segment, one entry
            segments, to_cache = plan_prefill_segments(
                rest, prompt_cache_count, list(found)
            )
            snapshot_queue = [(found[n], list(prompt_tokens[:n])) for n in to_cache]

        think_end = tuple(getattr(self.tokenizer, "think_end_tokens", ()) or ())
        preopened = _think_preopened(
            prompt_tokens,
            getattr(self.tokenizer, "think_start_tokens", ()) or (),
            think_end,
        )

        # Built before insert_segments, not after, and deliberately: once a job
        # is in the batch the generator decodes it whether or not _pending knows
        # about it, and _handle_response silently drops responses for a uid it
        # cannot find. An exception raised between the two — the detokenizer,
        # ToolCallFormatter on a malformed tool schema — would therefore leave a
        # job burning decode steps that nobody reads and a client waiting on a
        # DONE that nobody sends, which is a hang the per-job guard cannot undo
        # because the job is already inside the batch. So everything that can
        # raise happens first and only the dict store is left after the insert.
        state = {
            # One (kind, tokens) pair per planned boundary, in prefill order,
            # popped by _maybe_snapshot_boundary as each segment drains. Empty
            # when prefill was not split, which is what stops the trailing
            # end_of_segment (the single token insert_segments peels off for
            # generation) from caching the prompt minus its last token.
            "snapshot_queue": snapshot_queue,
            "job": job,
            # Usage accounting. prompt_tokens is the whole prompt, cached or not
            # — cached_tokens reports the reused share separately, as OpenAI does.
            "prompt_tokens": len(prompt_tokens),
            "cached_tokens": prompt_cache_count,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "in_reasoning": preopened,
            "think_end": think_end,
            "tail": [],
            "detokenizer": self.tokenizer.detokenizer,
            "tool_formatter": ToolCallFormatter(self.tokenizer.tool_parser, job.tools, streaming=job.stream),
            "tool_text": "",
            "tool_calls_raw": [],
            "made_tool_call": False,
            "in_tool_state": False,
            "prev_state": "normal",
            # Stamped after the insert below, so the latency window still starts
            # where it always did rather than silently growing by however long
            # insert_segments took.
            "created": 0,
            "start_time": 0.0,
            # Set on the first generated token, which is the only boundary the
            # engine can actually observe between prefill and decode. Stays None
            # if the job dies before generating anything, and the split is then
            # simply not recorded rather than recorded as a zero.
            "first_token_time": None,
        }

        _t2 = time.time()
        (uid,) = self.batch_generator.insert_segments(
            segments=[segments],
            max_tokens=[min(job.max_tokens, self.max_tokens)],
            caches=[cache],
            all_tokens=[prompt_tokens[:prompt_cache_count]],
            samplers=[sampler],
            logits_processors=[logits_processors],
            state_machines=[self.state_machine],
            input_embeddings=[image_embeds] if image_embeds is not None else None,
        )
        logger.info("insert_segments took %.2fs (rest=%d tokens)", time.time() - _t2, len(rest))
        state["created"] = int(time.time())
        state["start_time"] = time.time()
        self._pending[uid] = state

    def _handle_response(self, r) -> None:
        state = self._pending.get(r.uid)
        if state is None:
            return
        job = state["job"]
        # .get, not [], and deliberately: a diagnostics counter must never be
        # able to raise inside the generation loop. A state dict from before
        # this field existed simply starts its decode window here.
        if state.get("first_token_time") is None:
            state["first_token_time"] = time.time()
        state["detokenizer"].add_token(r.token)
        with self._stats_lock:
            self._total_generated_tokens += 1

        # Count every sampled token, reasoning included — that is what OpenAI's
        # completion_tokens means, and it is what actually cost compute.
        state["completion_tokens"] += 1
        if state["in_reasoning"] or r.current_state == "reasoning":
            state["reasoning_tokens"] += 1
        if state["in_reasoning"] and state["think_end"]:
            # The closing marker can be more than one token, so match on a
            # rolling tail rather than the current token alone.
            state["tail"].append(r.token)
            del state["tail"][: -len(state["think_end"])]
            if tuple(state["tail"]) == state["think_end"]:
                state["in_reasoning"] = False

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

            state["timing"] = self._record_timing(state)
            usage = self._usage(state)
            if job.stream:
                if text_delta:
                    job.out_queue.put(self._chunk(job, delta={"content": text_delta}, finish_reason=None))
                if tool_calls:
                    job.out_queue.put(self._chunk(job, delta={"tool_calls": tool_calls}, finish_reason=None))
                job.out_queue.put(self._chunk(job, delta={}, finish_reason=finish_reason))
                if job.include_usage:
                    # Separate trailing chunk with no choices, per OpenAI. Clients
                    # that don't ask never see it, so this can't shift the shape
                    # of a stream anyone is already parsing.
                    job.out_queue.put(self._chunk(job, delta=None, finish_reason=None, usage=usage))
            else:
                message = {"role": "assistant", "content": state.get("full_text", "") + text_delta}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                    message["content"] = message["content"] or None
                job.out_queue.put(self._response(job, message=message, finish_reason=finish_reason,
                                                 usage=usage))
            job.out_queue.put(DONE)

            if r.prompt_cache is not None and r.all_tokens is not None:
                # Explicitly "assistant" — already the default, but stated so the
                # three eviction classes are visible in one grep (SNAPSHOT_CACHE_TYPE).
                # A completed turn is the most replaceable entry in the pool and
                # should be the first thing evicted.
                self.prompt_cache.insert_cache(self.model_path, r.all_tokens, r.prompt_cache,
                                               cache_type="assistant")
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

    def _record_timing(self, state: dict) -> Optional[dict]:
        """Split one finished request into prefill and decode, and record it.

        Returns None when the split cannot be honestly computed, which is the
        point: a request that generated nothing has no decode window, and one
        that generated a single token has a window of zero. Reporting either as
        "0 tok/s" would poison the percentiles, so they are dropped instead.

        ``decode_tps`` divides by ``completion_tokens - 1`` because the first
        token *ends* prefill rather than starting decode — the decode window is
        the gaps between tokens, and there are n-1 of them.

        ``ttft_ms`` is wall clock from job start, so under continuous batching it
        includes time queued behind other requests. That is deliberate: it is
        what the caller waited. ``prefill_tps`` inherits that contamination and
        is therefore a floor on prefill speed, never a clean measurement of it.
        """
        start = state.get("start_time")
        first = state.get("first_token_time")
        completion = state.get("completion_tokens", 0)
        if not start or not first:
            return None

        now = time.time()
        ttft_ms = (first - start) * 1000
        decode_s = now - first
        uncached = max(0, state.get("prompt_tokens", 0) - state.get("cached_tokens", 0))

        decode_tps = None
        if completion >= 2 and decode_s > 0:
            decode_tps = (completion - 1) / decode_s
        prefill_tps = None
        if uncached > 0 and ttft_ms > 0:
            prefill_tps = uncached / (ttft_ms / 1000)

        with self._stats_lock:
            self._recent_ttft_ms.append(ttft_ms)
            self._recent_decode_ms.append(decode_s * 1000)
            if decode_tps is not None:
                self._recent_decode_tps.append(decode_tps)
            if prefill_tps is not None:
                self._recent_prefill_tps.append(prefill_tps)

        timing = {
            "ttft_ms": round(ttft_ms, 1),
            "decode_ms": round(decode_s * 1000, 1),
            "decode_tps": round(decode_tps, 2) if decode_tps is not None else None,
            "prefill_tps": round(prefill_tps, 1) if prefill_tps is not None else None,
            "uncached_prompt_tokens": uncached,
        }
        logger.info(
            "timing: ttft=%.0fms decode=%.0fms decode_tps=%s prefill_tokens=%d reasoning=%d/%d",
            ttft_ms, decode_s * 1000,
            f"{decode_tps:.1f}" if decode_tps is not None else "n/a",
            uncached, state.get("reasoning_tokens", 0), completion,
        )
        return timing

    @staticmethod
    def _usage(state: dict) -> dict:
        """OpenAI-shaped usage for one finished job.

        ``cached_tokens`` is the prompt-cache hit for this request specifically —
        /v1/stats only ever exposed a lifetime hit rate, which says nothing about
        the turn in front of you. ``reasoning_tokens`` is counted from the
        sequence state machine, not estimated from character length.
        """
        prompt = state["prompt_tokens"]
        completion = state["completion_tokens"]
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": state["cached_tokens"]},
            "completion_tokens_details": {"reasoning_tokens": state["reasoning_tokens"]},
        }
        # Mira extension, added only when there is something to report so the
        # block above stays byte-identical to OpenAI's shape for every client
        # that doesn't know about it. The orchestrator needs the prefill/decode
        # split per LLM call to attribute a whole turn.
        timing = state.get("timing")
        if timing is not None:
            usage["timing"] = timing
        return usage

    def _chunk(self, job: ChatJob, delta: Optional[dict], finish_reason: Optional[str],
               usage: Optional[dict] = None) -> dict:
        # `delta=None` builds the usage-only chunk: no choices at all, which is
        # how OpenAI marks it and how bc.normalize_oai_stream recognises it.
        return {
            "id": job.request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": ([] if delta is None
                        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]),
            "usage": usage,
        }

    def _response(self, job: ChatJob, message: dict, finish_reason: Optional[str],
                  usage: Optional[dict] = None) -> dict:
        return {
            "id": job.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }


def _penalties_from_body(body: dict) -> dict:
    """The repetition-penalty keyword arguments a request asked for.

    Absent and 0 are both "off" and produce no key at all, so an ordinary
    request builds the same empty processor list it always did. A context size
    is only carried when its penalty is actually set -- mlx-lm defaults each one
    to 20 and passing a bare size would change nothing while making the request
    look configured in the log.
    """
    penalties: dict = {}
    for name in ("repetition", "presence", "frequency"):
        value = body.get(f"{name}_penalty")
        if not value:
            continue
        penalties[f"{name}_penalty"] = float(value)
        size = body.get(f"{name}_context_size")
        if size:
            penalties[f"{name}_context_size"] = int(size)
    return penalties


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
            top_k=int(body.get("top_k", 0) or 0),
            seed=None if body.get("seed") is None else int(body["seed"]),
            penalties=_penalties_from_body(body),
            chat_template_kwargs=body.get("chat_template_kwargs"),
            include_usage=bool((body.get("stream_options") or {}).get("include_usage")),
        )
        loop = asyncio.get_event_loop()
        try:
            engine.submit(job)
        except EngineDead as exc:
            # 503, and with the original cause in the body: the engine thread is
            # gone and no amount of retrying this request brings it back. Same
            # reason as the non-streaming branch below — FastAPI's default 500
            # handler would throw the message away.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    # (docs/moe-offload-case-study.md). No-op on dense models; zero
    # overhead unless explicitly set. See core/inference/expert_profiler.py.
    parser.add_argument("--profile-experts", action="store_true")
    parser.add_argument("--expert-profile-path", default=None)
    # Opt-in MoE expert disk offloading (docs/offload-resident-sizing.md).
    # None (default) = every expert resident, today's behavior. No-op on dense
    # models. See core/inference/expert_offload.py.
    parser.add_argument("--resident-expert-fraction", type=float, default=None)
    # Off by default. On, this loads the checkpoint's own vision tower (about
    # 0.89 GB for Qwen3.6-35B-A3B) so screenshots are read as images rather than
    # run through OCR. Costs nothing when off: the tower is never imported.
    parser.add_argument("--vision", action="store_true")
    # Ceiling on an image's pixel count after Qwen's smart-resize, and the single
    # biggest lever on vision cost. The checkpoint asks for 16,777,216, which in
    # practice caps nothing: a 5712x4284 photo stays at 16,170 image tokens and
    # 243s in the tower. 1 MP holds any image to ~1k tokens and ~1.6s; 2 MP to
    # ~2k and ~4.4s, worth it only when fine visual detail matters (measured:
    # 1 MP keeps screenshot text but blurs attributes like armor vs skin).
    # Only ever lowers the checkpoint's own ceiling.
    parser.add_argument("--vision-max-pixels", type=int, default=1024 * 1024)
    # Seconds without an image before the 0.89GB tower is released. It reloads in
    # 0.14s page-cached (1.94s cold) and Metal kernels survive the round trip, so
    # the next image after an idle stretch pays about two seconds. 0 disables the
    # release and keeps the tower resident once loaded.
    parser.add_argument("--vision-tower-idle-timeout", type=float, default=300.0)
    # Fault the model back into RAM on the idle branch when another app has had
    # it compressed out, so the decompression is not paid by whoever asks next
    # (measured 17.60s against a warm 0.45s on a fully evicted model).
    parser.add_argument("--proactive-decompress", action="store_true")
    parser.add_argument("--boundary-snapshot", action="store_true")
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
        vision=args.vision,
        vision_max_pixels=args.vision_max_pixels,
        vision_tower_idle_timeout=args.vision_tower_idle_timeout,
        proactive_decompress=args.proactive_decompress,
        boundary_snapshot=args.boundary_snapshot,
    )
    engine.start()

    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
