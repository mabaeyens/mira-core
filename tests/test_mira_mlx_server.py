"""Unit tests for core/inference/mira_mlx_server.py's model-free helpers and the
oversized-single-prompt rejection path. No real model load needed."""
import asyncio
import json
import time
import types

import httpx
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("mlx.core")  # mlx is macOS-only (Apple Silicon), absent on Linux CI

from core.inference.disk_prompt_cache import DiskBackedPromptCache
from core.inference.mira_mlx_server import (
    ChatJob,
    DONE,
    GenerationEngine,
    create_app,
    _build_state_machine,
    _prepare_messages,
    _think_preopened,
)


# -- _prepare_messages --------------------------------------------------------

def test_prepare_messages_none_content_becomes_empty_string():
    out, images = _prepare_messages([{"role": "user", "content": None}])
    assert out[0]["content"] == ""
    assert images == []


def test_prepare_messages_leaves_normal_content_untouched():
    out, images = _prepare_messages([{"role": "user", "content": "hello"}])
    assert out[0]["content"] == "hello"
    assert images == []


def test_prepare_messages_parses_tool_call_argument_json_string():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}],
    }
    out, _ = _prepare_messages([msg])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}


def test_prepare_messages_malformed_tool_call_json_raises():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}],
    }
    with pytest.raises(json.JSONDecodeError):
        _prepare_messages([msg])


def test_prepare_messages_does_not_mutate_input():
    original = {"role": "user", "content": None}
    _prepare_messages([original])
    assert original["content"] is None  # _prepare_messages must copy, not mutate


def test_prepare_messages_rejects_image_content():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ],
    }
    with pytest.raises(ValueError, match="does not support image inputs"):
        _prepare_messages([msg])


def _png_data_url(width=64, height=48):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_prepare_messages_extracts_images_when_vision_on():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what's in this image?"},
            {"type": "image_url", "image_url": {"url": _png_data_url()}},
        ],
    }
    out, images = _prepare_messages([msg], vision=True)
    assert len(images) == 1
    assert images[0].size == (64, 48)
    # The image part is rewritten to the bare shape the chat template renders as
    # a single <|image_pad|>; the text part is untouched.
    assert out[0]["content"] == [
        {"type": "text", "text": "what's in this image?"},
        {"type": "image"},
    ]


def test_prepare_messages_preserves_image_order_across_messages():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _png_data_url(32, 32)}},
                {"type": "image_url", "image_url": {"url": _png_data_url(64, 16)}},
            ],
        },
    ]
    _, images = _prepare_messages(msgs, vision=True)
    assert [im.size for im in images] == [(32, 32), (64, 16)]


def test_prepare_messages_refuses_remote_image_urls():
    """Fetching a remote URL here would let a crafted conversation make the
    server issue arbitrary outbound requests."""
    msg = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
        ],
    }
    with pytest.raises(ValueError, match="data URLs"):
        _prepare_messages([msg], vision=True)


def test_prepare_messages_vision_off_still_rejects():
    msg = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": _png_data_url()}}],
    }
    with pytest.raises(ValueError, match="does not support image inputs"):
        _prepare_messages([msg], vision=False)


# -- _build_state_machine -----------------------------------------------------

class FakeTokenizer:
    eos_token_ids = [1]
    has_thinking = False
    has_tool_calling = False
    think_start_tokens = ()
    think_end_tokens = ()
    tool_call_start_tokens = ()
    tool_call_end_tokens = ()

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_build_state_machine_plain_model_only_has_normal_state():
    sm = _build_state_machine(FakeTokenizer())
    assert sm._initial == "normal"
    assert set(sm._states.keys()) == {"normal"}


def test_build_state_machine_thinking_model_adds_reasoning_state():
    class ThinkingTokenizer(FakeTokenizer):
        has_thinking = True
        think_start_tokens = (10,)
        think_end_tokens = (11,)

    sm = _build_state_machine(ThinkingTokenizer())
    assert "reasoning" in sm._states


def test_build_state_machine_tool_calling_model_adds_tool_state():
    class ToolTokenizer(FakeTokenizer):
        has_tool_calling = True
        tool_call_start_tokens = (20,)
        tool_call_end_tokens = ()  # Mistral-style: one-sided, no end marker

    sm = _build_state_machine(ToolTokenizer())
    assert "tool" in sm._states


# -- oversized single prompt --------------------------------------------------

class FixedLengthTokenizer:
    def __init__(self, n_tokens):
        self.n_tokens = n_tokens

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True):
        return list(range(self.n_tokens))


def _job(max_tokens=128):
    return ChatJob(messages=[{"role": "user", "content": "hi"}], tools=None, stream=False,
                   max_tokens=max_tokens, temperature=0.0, top_p=0.0)


def test_oversized_prompt_rejected_with_clear_error():
    engine = GenerationEngine(model_path="fake/model", max_kv_size=32)
    engine.tokenizer = FixedLengthTokenizer(50)
    job = _job()

    engine._start_job(job)

    err = job.out_queue.get_nowait()
    assert job.out_queue.get_nowait() is DONE
    assert isinstance(err, ValueError)
    assert "50" in str(err) and "32" in str(err)


def test_fitting_prompt_passes_the_ceiling_check():
    engine = GenerationEngine(model_path="fake/model", max_kv_size=1000)
    engine.tokenizer = FixedLengthTokenizer(50)
    job = _job()

    # prompt_cache/batch_generator are deliberately None: a fitting prompt must
    # get past the oversized-prompt check and fail later on real engine state,
    # not be blocked by the check itself.
    with pytest.raises(AttributeError):
        engine._start_job(job)


def test_no_max_kv_size_never_rejects():
    engine = GenerationEngine(model_path="fake/model", max_kv_size=None)
    engine.tokenizer = FixedLengthTokenizer(10_000_000)
    job = _job()

    with pytest.raises(AttributeError):
        engine._start_job(job)


# -- kv_bits/kv_group_size/quantized_kv_start CLI threading ------------------

def test_generation_engine_kv_bits_defaults_preserve_unquantized_behavior():
    """Default construction (no kv_bits passed, matching --kv-bits' argparse
    default of None) must produce the exact same engine state as before this
    parameter existed."""
    engine = GenerationEngine(model_path="fake/model")
    assert engine.kv_bits is None
    assert engine.kv_group_size == 64
    assert engine.quantized_kv_start == 0


def test_generation_engine_stores_kv_bits_settings():
    engine = GenerationEngine(model_path="fake/model", kv_bits=8, kv_group_size=32, quantized_kv_start=0)
    assert engine.kv_bits == 8
    assert engine.kv_group_size == 32
    assert engine.quantized_kv_start == 0


def test_main_cli_kv_bits_default_is_none(monkeypatch):
    import sys
    from core.inference import mira_mlx_server

    captured = {}

    class _StubEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            raise SystemExit("stop before uvicorn.run")

    monkeypatch.setattr(mira_mlx_server, "GenerationEngine", _StubEngine)
    monkeypatch.setattr(sys, "argv", ["mira_mlx_server", "--model", "fake/model"])

    with pytest.raises(SystemExit):
        mira_mlx_server.main()

    assert captured["kv_bits"] is None
    assert captured["kv_group_size"] == 64
    assert captured["quantized_kv_start"] == 0


def test_main_cli_kv_bits_threaded_through(monkeypatch):
    import sys
    from core.inference import mira_mlx_server

    captured = {}

    class _StubEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            raise SystemExit("stop before uvicorn.run")

    monkeypatch.setattr(mira_mlx_server, "GenerationEngine", _StubEngine)
    monkeypatch.setattr(
        sys, "argv",
        ["mira_mlx_server", "--model", "fake/model", "--kv-bits", "8", "--kv-group-size", "32"],
    )

    with pytest.raises(SystemExit):
        mira_mlx_server.main()

    assert captured["kv_bits"] == 8
    assert captured["kv_group_size"] == 32


# -- stats_snapshot (docs/architecture.md, "mira-mlx specifics") -------------

def test_stats_snapshot_counts_a_cache_miss():
    engine = GenerationEngine(model_path="fake/model", max_kv_size=None)
    engine.tokenizer = FixedLengthTokenizer(50)
    engine.prompt_cache = DiskBackedPromptCache(max_size=10, max_bytes=1, disk_store=None)
    job = _job()

    with pytest.raises(AttributeError):
        engine._start_job(job)  # fails later at batch_generator.insert_segments (None), as expected

    snap = engine.stats_snapshot()
    assert snap["cache_misses"] == 1
    assert snap["cache_hits"] == 0
    assert snap["cache_hit_rate"] == 0.0
    assert snap["total_prompt_tokens"] == 50
    assert snap["disk_cache_hits"] == 0


def test_stats_snapshot_empty_state_has_no_hit_rate_or_latency():
    engine = GenerationEngine(model_path="fake/model")
    snap = engine.stats_snapshot()
    assert snap["cache_hit_rate"] is None
    assert snap["latency_p50_ms"] is None
    assert snap["latency_sample_size"] == 0
    # The split must be absent-not-zero on an idle engine, for the same reason:
    # "0 tok/s" and "never measured" are different claims.
    assert snap["decode_tps_p50"] is None
    assert snap["ttft_p50_ms"] is None
    assert snap["decode_tps_sample_size"] == 0


# -- prefill/decode split (specs/decode-roofline.md) -------------------------

def _timed_state(engine, *, start, first, completion, prompt=1000, cached=0):
    """A finished-job state with the two clocks already set."""
    return {
        "start_time": start, "first_token_time": first,
        "completion_tokens": completion, "reasoning_tokens": 0,
        "prompt_tokens": prompt, "cached_tokens": cached,
    }


def test_record_timing_splits_ttft_from_decode():
    engine = GenerationEngine(model_path="fake/model")
    now = time.time()
    # 2.0s of prefill, then 100 tokens over the following ~1.0s.
    timing = engine._record_timing(
        _timed_state(engine, start=now - 3.0, first=now - 1.0, completion=100)
    )
    assert 1990 < timing["ttft_ms"] < 2010
    assert 990 < timing["decode_ms"] < 1010
    # 99 gaps between 100 tokens across ~1s, not 100.
    assert 97 < timing["decode_tps"] < 101
    # 1000 uncached prompt tokens in the 2s before the first token.
    assert 490 < timing["prefill_tps"] < 510


def test_record_timing_drops_a_single_token_generation():
    """One token has no decode window; reporting 0 tok/s would poison the p50."""
    engine = GenerationEngine(model_path="fake/model")
    now = time.time()
    timing = engine._record_timing(
        _timed_state(engine, start=now - 1.0, first=now, completion=1)
    )
    assert timing["decode_tps"] is None
    assert engine.stats_snapshot()["decode_tps_p50"] is None
    # The ttft half is still real and still recorded.
    assert engine.stats_snapshot()["ttft_p50_ms"] is not None


def test_record_timing_returns_none_when_nothing_was_generated():
    engine = GenerationEngine(model_path="fake/model")
    state = _timed_state(engine, start=time.time(), first=None, completion=0)
    assert engine._record_timing(state) is None
    assert engine.stats_snapshot()["decode_tps_sample_size"] == 0


def test_record_timing_ignores_cached_tokens_in_prefill_rate():
    """A cache hit did not prefill those tokens, so they must not inflate the rate."""
    engine = GenerationEngine(model_path="fake/model")
    now = time.time()
    timing = engine._record_timing(
        _timed_state(engine, start=now - 1.0, first=now, completion=10,
                     prompt=1000, cached=900)
    )
    assert timing["uncached_prompt_tokens"] == 100
    assert 95 < timing["prefill_tps"] < 105


def test_usage_carries_timing_only_when_measured():
    plain = GenerationEngine._usage({
        "prompt_tokens": 10, "completion_tokens": 5,
        "cached_tokens": 0, "reasoning_tokens": 0,
    })
    assert "timing" not in plain

    withtiming = GenerationEngine._usage({
        "prompt_tokens": 10, "completion_tokens": 5,
        "cached_tokens": 0, "reasoning_tokens": 0,
        "timing": {"ttft_ms": 1.0, "decode_ms": 2.0},
    })
    assert withtiming["timing"] == {"ttft_ms": 1.0, "decode_ms": 2.0}


# -- HTTP error surfacing (2026-07-09 live-verification finding) -------------
#
# A prior version put a bare string on the error payload / re-raised the raw
# exception, which the openai SDK's streaming client and FastAPI's default
# handler both swallow into a generic message ("An error occurred during
# streaming" / "Internal Server Error") — discarding the real detail (e.g. the
# oversized-prompt ValueError's actual token counts). Caught only by an actual
# live request through the real openai client, not by unit tests alone.

class FakeEngineRaisesValueError:
    model_path = "fake/model"
    max_tokens = 4096

    def submit(self, job):
        job.out_queue.put(ValueError("prompt is 200000 tokens, this machine's context ceiling is 65536 tokens"))
        job.out_queue.put(DONE)

    def stats_snapshot(self):
        return {}


def test_non_streaming_error_surfaces_real_message_not_generic_500():
    app = create_app(FakeEngineRaisesValueError())
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert resp.status_code == 400
    assert "200000 tokens" in resp.json()["detail"]


def test_streaming_error_payload_is_a_mapping_with_message_not_a_bare_string():
    # A fresh event loop via asyncio.run + httpx's ASGI transport, rather than
    # FastAPI's TestClient (whose threaded anyio portal conflicts with other
    # tests' event loops when run as part of the full suite). sse_starlette
    # also caches a should_exit_event on its first use, bound to whatever loop
    # was active then — reset it so it gets rebuilt on *this* test's loop.
    import sse_starlette.sse as sse_module
    sse_module.AppStatus.should_exit_event = None

    app = create_app(FakeEngineRaisesValueError())

    async def _post():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            return resp.text

    body = asyncio.run(_post())

    lines = [l for l in body.splitlines() if l.startswith("data: ") and "error" in l]
    assert lines, f"no error event found in SSE stream: {body!r}"
    payload = json.loads(lines[0][len("data: "):])
    assert isinstance(payload["error"], dict), "error must be a mapping (openai SDK requires error.message)"
    assert "200000 tokens" in payload["error"]["message"]


# -- lazy vision tower --------------------------------------------------------


def test_vision_tower_is_not_loaded_at_construction():
    """Startup must stay text-only. A session that never sends an image should
    never pay the 0.89GB, which is the whole point of loading it lazily."""
    engine = GenerationEngine(model_path="fake/model", vision=True)
    assert engine.vision is True
    assert engine.vision_tower is None


def test_ensure_tower_records_the_failure_and_disables_vision():
    """A bad checkpoint costs one load attempt, not one per image: the first
    failure flips vision off so later turns take the OCR path immediately."""
    engine = GenerationEngine(model_path="does/not/exist", vision=True)
    assert engine._ensure_tower() is None
    assert engine.vision is False
    assert engine._vision_error
    # Second call must not retry the load - vision is already off.
    assert engine._ensure_tower() is None


def test_stats_reports_vision_enabled_while_the_tower_is_released():
    """`enabled` follows config, not residency. An idle release must not look
    like a vision failure to anything reading /v1/stats."""
    engine = GenerationEngine(model_path="fake/model", vision=True)
    vision = engine.stats_snapshot()["vision"]
    assert vision["enabled"] is True
    assert vision["tower_resident"] is False
    assert vision["tower_bytes"] == 0


def test_idle_release_is_a_no_op_without_a_tower():
    engine = GenerationEngine(model_path="fake/model", vision=True)
    engine._release_idle_tower()  # must not raise on a None tower
    assert engine.vision_tower is None


def test_idle_release_respects_the_timeout_and_then_frees():
    class FakeTower:
        weight_bytes = 893142496

    engine = GenerationEngine(
        model_path="fake/model", vision=True, vision_tower_idle_timeout=60.0
    )
    engine.vision_tower = FakeTower()
    engine._tower_last_used = time.time()
    engine._release_idle_tower()
    assert engine.vision_tower is not None, "released while still fresh"

    engine._tower_last_used = time.time() - 61.0
    engine._release_idle_tower()
    assert engine.vision_tower is None, "not released after the idle timeout"
    assert engine._tower_unloads == 1


def test_idle_release_refreshes_the_memory_snapshot():
    """_check_memory_pressure only runs while jobs are in flight, so without
    this the stats stay frozen at their last active value and an idle release
    reports as having reclaimed nothing."""

    class FakeTower:
        weight_bytes = 893142496

    engine = GenerationEngine(
        model_path="fake/model", vision=True, vision_tower_idle_timeout=1.0
    )
    engine.vision_tower = FakeTower()
    engine._tower_last_used = time.time() - 2.0
    engine._active_memory_bytes = 123456789  # a stale sample from an earlier turn
    engine._release_idle_tower()
    assert engine._active_memory_bytes != 123456789, "memory snapshot not refreshed"
    assert engine.stats_snapshot()["vision"]["tower_last_reclaimed_bytes"] is not None


def test_zero_timeout_keeps_the_tower_resident():
    class FakeTower:
        weight_bytes = 893142496

    engine = GenerationEngine(
        model_path="fake/model", vision=True, vision_tower_idle_timeout=0
    )
    engine.vision_tower = FakeTower()
    engine._tower_last_used = time.time() - 100000
    engine._release_idle_tower()
    assert engine.vision_tower is not None


# -- usage accounting ---------------------------------------------------------

class _FakeDetokenizer:
    """Decodes each token id as a single character, like a toy BPE."""

    def __init__(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = chr(token)

    def finalize(self):
        self.last_segment = ""


def _usage_engine(job, *, prompt=100, cached=64, preopened=False, think_end=(), uid=7):
    engine = GenerationEngine(model_path="fake/model")
    engine._eos_ids = {1}
    engine.tokenizer = types.SimpleNamespace(tool_call_end=None)
    engine._pending[uid] = {
        "snapshot_tokens": None,
        "job": job,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "in_reasoning": preopened,
        "think_end": tuple(think_end),
        "tail": [],
        "detokenizer": _FakeDetokenizer(),
        "tool_formatter": lambda raw: None,
        "tool_text": "",
        "tool_calls_raw": [],
        "made_tool_call": False,
        "in_tool_state": False,
        "prev_state": "normal",
        "created": 0,
        "start_time": time.time(),
    }
    return engine


def _token(uid, token, state="normal", finish_reason=None):
    return types.SimpleNamespace(uid=uid, token=token, current_state=state,
                                 finish_reason=finish_reason,
                                 prompt_cache=None, all_tokens=None)


def _drain(job):
    out = []
    while True:
        item = job.out_queue.get_nowait()
        if item is DONE:
            return out
        out.append(item)


def _stream_job(include_usage=True):
    return ChatJob(messages=[{"role": "user", "content": "hi"}], tools=None, stream=True,
                   max_tokens=128, temperature=0.0, top_p=0.0, include_usage=include_usage)


def test_usage_block_has_the_openai_shape():
    usage = GenerationEngine._usage({
        "prompt_tokens": 1200, "completion_tokens": 300,
        "cached_tokens": 1024, "reasoning_tokens": 250,
    })
    assert usage == {
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 1024},
        "completion_tokens_details": {"reasoning_tokens": 250},
    }


def test_stream_emits_a_trailing_usage_only_chunk_when_asked():
    job = _stream_job(include_usage=True)
    engine = _usage_engine(job, prompt=42, cached=8)
    engine._handle_response(_token(7, ord("a")))
    engine._handle_response(_token(7, ord("b"), finish_reason="stop"))

    chunks = _drain(job)
    # The finish_reason chunk comes first and carries no usage; the usage chunk
    # follows it with an empty choices list, exactly as OpenAI specifies.
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-2]["usage"] is None
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["prompt_tokens"] == 42
    assert chunks[-1]["usage"]["completion_tokens"] == 2
    assert chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 8


def test_stream_stays_silent_about_usage_when_not_asked():
    job = _stream_job(include_usage=False)
    engine = _usage_engine(job)
    engine._handle_response(_token(7, ord("a")))
    engine._handle_response(_token(7, ord("b"), finish_reason="stop"))

    chunks = _drain(job)
    assert all(c["choices"] for c in chunks), "emitted a usage-only chunk unasked"
    assert all(c["usage"] is None for c in chunks)


def test_non_streaming_response_always_carries_usage():
    """Only streaming gates usage behind stream_options — a plain response never does."""
    job = ChatJob(messages=[{"role": "user", "content": "hi"}], tools=None, stream=False,
                  max_tokens=128, temperature=0.0, top_p=0.0)
    engine = _usage_engine(job, prompt=11, cached=0)
    engine._handle_response(_token(7, ord("x"), finish_reason="stop"))

    (response,) = _drain(job)
    assert response["usage"]["prompt_tokens"] == 11
    assert response["usage"]["completion_tokens"] == 1


def test_reasoning_tokens_counted_from_the_state_machine():
    job = _stream_job()
    engine = _usage_engine(job)
    engine._handle_response(_token(7, ord("a"), state="reasoning"))
    engine._handle_response(_token(7, ord("b"), state="reasoning"))
    engine._handle_response(_token(7, ord("c"), state="normal"))
    engine._handle_response(_token(7, ord("d"), state="normal", finish_reason="stop"))

    usage = _drain(job)[-1]["usage"]
    assert usage["completion_tokens"] == 4
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 2


def test_preopened_thinking_counts_despite_the_normal_state():
    """Qwen3 opens <think> in the prompt, so the machine reports "normal"
    throughout the reasoning stretch. Counting only "reasoning" would score 0."""
    job = _stream_job()
    engine = _usage_engine(job, preopened=True, think_end=(99,))
    for tok in (ord("a"), ord("b"), 99):  # two thoughts, then </think>
        engine._handle_response(_token(7, tok, state="normal"))
    engine._handle_response(_token(7, ord("y"), state="normal"))
    engine._handle_response(_token(7, ord("z"), state="normal", finish_reason="stop"))

    usage = _drain(job)[-1]["usage"]
    assert usage["completion_tokens"] == 5
    # The closing marker itself is generation the model paid for: 2 thoughts + </think>.
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 3


def test_multi_token_think_end_closes_the_reasoning_stretch():
    """The closing marker can span tokens; a single-token match would never fire."""
    job = _stream_job()
    engine = _usage_engine(job, preopened=True, think_end=(98, 99))
    for tok in (ord("a"), 98, 99, ord("y"), ord("z")):
        engine._handle_response(_token(7, tok, state="normal"))
    engine._handle_response(_token(7, ord("!"), state="normal", finish_reason="stop"))

    usage = _drain(job)[-1]["usage"]
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 3


def test_non_thinking_model_reports_zero_reasoning_not_none():
    job = _stream_job()
    engine = _usage_engine(job, preopened=False, think_end=())
    engine._handle_response(_token(7, ord("a")))
    engine._handle_response(_token(7, ord("b"), finish_reason="stop"))

    usage = _drain(job)[-1]["usage"]
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 0


def test_length_stop_still_reports_usage():
    """A truncated turn is exactly when the token count matters most."""
    job = _stream_job()
    engine = _usage_engine(job, prompt=5)
    engine._handle_response(_token(7, ord("a"), finish_reason="length"))

    chunks = _drain(job)
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["total_tokens"] == 6


# -- pre-opened thinking detection --------------------------------------------

def test_think_preopened_true_when_prompt_ends_with_the_marker():
    assert _think_preopened([5, 6, 151667], (151667,), (151668,)) is True


def test_think_preopened_true_when_a_newline_follows_the_marker():
    # The real Qwen3.6 tail: '<|im_start|>assistant\n<think>\n'. The marker is
    # second-to-last, so an exact end-of-prompt match misses it and every
    # reasoning token gets counted as answer text.
    assert _think_preopened([5, 6, 248068, 198], (248068,), (248069,)) is True


def test_think_preopened_false_when_the_template_pre_closed_the_block():
    # Thinking disabled emits '<think>\n\n</think>\n\n' — opened and closed, so
    # what the model generates next is answer text, not reasoning.
    assert _think_preopened([5, 248068, 198, 248069, 198], (248068,), (248069,)) is False


def test_think_preopened_false_for_a_model_without_thinking():
    assert _think_preopened([5, 6, 7], (), ()) is False


def test_think_preopened_handles_a_multi_token_marker():
    assert _think_preopened([1, 27, 271], (27, 271), (28, 271)) is True
    assert _think_preopened([271], (27, 271), (28, 271)) is False


def test_think_preopened_reads_the_last_marker_pair_not_the_first():
    # A closed block from an earlier turn must not mask the open one that the
    # generation prompt just added.
    tokens = [248068, 5, 248069, 9, 248068, 198]
    assert _think_preopened(tokens, (248068,), (248069,)) is True
