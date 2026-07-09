"""Unit tests for core/inference/mira_mlx_server.py's model-free helpers and the
oversized-single-prompt rejection path. No real model load needed."""
import asyncio
import json

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
)


# -- _prepare_messages --------------------------------------------------------

def test_prepare_messages_none_content_becomes_empty_string():
    out = _prepare_messages([{"role": "user", "content": None}])
    assert out[0]["content"] == ""


def test_prepare_messages_leaves_normal_content_untouched():
    out = _prepare_messages([{"role": "user", "content": "hello"}])
    assert out[0]["content"] == "hello"


def test_prepare_messages_parses_tool_call_argument_json_string():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}],
    }
    out = _prepare_messages([msg])
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


# -- oversized single prompt (specs/mira-mlx-oversized-prompt.md) ------------

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


# -- stats_snapshot (specs/mira-mlx-stats-endpoint.md) -----------------------

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
