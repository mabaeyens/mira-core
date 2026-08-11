"""`finish_reason` must survive the wire adapter and reach the orchestrator.

Before this, `normalize_oai_stream` collapsed every terminating chunk into
`done=True` and threw the reason away, so a reply cut off at `max_tokens`
("length") was indistinguishable from one the model chose to end ("stop").
Mira could not tell a truncated answer from a finished one, which is the
root of the unclosed-`</think>` symptom in specs/generation-runaway-guard.md.

These tests drive the *real* adapter — the chunks reaching the orchestrator are
whatever `normalize_oai_stream` produces from an OpenAI-shaped stream, not
hand-built stand-ins for it.
"""

import types

import pytest
from unittest.mock import patch

from core import backend_client as bc
from core.orchestrator import ChatOrchestrator


@pytest.fixture
def orchestrator():
    return ChatOrchestrator(verbose=False)


def _oai_chunk(content=None, finish_reason=None):
    """One chunk in the shape the OpenAI SDK yields."""
    delta = types.SimpleNamespace(content=content, tool_calls=None,
                                  reasoning=None, model_extra={})
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


def _events(orchestrator, oai_chunks, *, thinking_enabled=False):
    """Run the real adapter's output through the orchestrator's streaming loop."""
    orchestrator.backend = "mira-mlx"
    orchestrator.model = "Qwen3.6-35B-A3B"
    stream = bc.normalize_oai_stream(iter(oai_chunks))
    with patch.object(orchestrator, "_call_llm", return_value=stream):
        return list(orchestrator._stream_llm_with_thinking(thinking_enabled))


def _llm_done(events):
    return [e for e in events if e.get("type") == "llm_done"][0]


def test_length_reaches_the_orchestrator(orchestrator):
    events = _events(orchestrator, [
        _oai_chunk("A sentence that stops mid-"),
        _oai_chunk("way through", finish_reason="length"),
    ])
    assert _llm_done(events)["finish_reason"] == "length"


def test_stop_reaches_the_orchestrator(orchestrator):
    events = _events(orchestrator, [
        _oai_chunk("A complete sentence."),
        _oai_chunk("", finish_reason="stop"),
    ])
    assert _llm_done(events)["finish_reason"] == "stop"


def test_truncated_and_finished_turns_are_distinguishable(orchestrator):
    """The point of the change: same `done`, same visible text shape, different
    verdict. Anything downstream that needs to know now can."""
    truncated = _llm_done(_events(orchestrator, [
        _oai_chunk("half an answer", finish_reason="length")]))
    finished = _llm_done(_events(orchestrator, [
        _oai_chunk("a whole answer", finish_reason="stop")]))
    assert truncated["finish_reason"] != finished["finish_reason"]


def test_truncation_is_logged_loudly(orchestrator, caplog):
    """The warning is the instrumentation — it is the only record that a reply
    was cut off, and counting it in the log is how we size how often that
    happens. Losing it silently loses the measurement."""
    with caplog.at_level("WARNING"):
        _events(orchestrator, [_oai_chunk("cut off", finish_reason="length")])
    assert any("finish_reason=length" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING"):
        _events(orchestrator, [_oai_chunk("finished", finish_reason="stop")])
    assert not any("finish_reason=length" in (r.getMessage()) for r in caplog.records)


def test_backend_without_finish_reason_yields_none(orchestrator):
    """Ollama-shaped chunks carry no `finish_reason` attribute at all. That must
    read as "the backend didn't say", not crash the turn."""
    def _bare(content="", done=False):
        msg = types.SimpleNamespace(content=content, tool_calls=None, thinking=None)
        return types.SimpleNamespace(message=msg, done=done)

    orchestrator.backend = "mira-mlx"
    orchestrator.model = "Qwen3.6-35B-A3B"
    with patch.object(orchestrator, "_call_llm",
                      return_value=iter([_bare("hi"), _bare(done=True)])):
        events = list(orchestrator._stream_llm_with_thinking(False))
    assert _llm_done(events)["finish_reason"] is None
