"""Thinking tokens must be counted once, whoever counts them.

mira-mlx now reports `reasoning_tokens` from its sequence state machine, and its
`completion_tokens` already includes the thinking stream. The orchestrator's own
~3.5-chars-per-token estimate (core/context_manager.py) predates that and must
step aside whenever the backend reports a real number, or every thinking turn's
output-token readout is inflated by roughly the size of the reasoning itself.

These tests drive the *real* wire adapter, so what reaches the orchestrator is
whatever `normalize_oai_stream` builds from an OpenAI-shaped stream.
"""

import types

import pytest
from unittest.mock import patch

from core import backend_client as bc
from core import context_manager as ctxmgr
from core.orchestrator import ChatOrchestrator


THINKING = "The user is asking about X."
ANSWER = "The answer is 42."


@pytest.fixture
def orchestrator():
    o = ChatOrchestrator(verbose=False)
    o.backend = "mira-mlx"
    o.model = "Qwen3.6-35B-A3B"
    return o


def _oai_chunk(content=None, finish_reason=None, usage=None):
    delta = types.SimpleNamespace(content=content, tool_calls=None,
                                  reasoning=None, model_extra={})
    choices = [] if content is None and finish_reason is None else [
        types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    ]
    return types.SimpleNamespace(choices=choices, usage=usage)


def _usage(prompt, completion, reasoning=None, cached=None):
    return types.SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning),
    )


def _thinking_turn(usage):
    """Qwen3 emits only the closing tag — the template pre-opens `<think>`."""
    return [
        _oai_chunk(THINKING),
        _oai_chunk("</think>"),
        _oai_chunk(ANSWER),
        _oai_chunk("", finish_reason="stop"),
        _oai_chunk(usage=usage),  # trailing usage-only chunk
    ]


def _run(orchestrator, oai_chunks):
    stream = bc.normalize_oai_stream(iter(oai_chunks))
    with patch.object(orchestrator, "_call_llm", return_value=stream):
        events = list(orchestrator.stream_chat("Hello", thinking_enabled=True))
    return [e for e in events if e["type"] == "stats"][0]


def test_reported_reasoning_tokens_are_not_estimated_on_top(orchestrator):
    """completion_tokens=300 already covers the thinking; the readout says 300."""
    stats = _run(orchestrator, _thinking_turn(_usage(1200, 300, reasoning=250)))
    assert stats["output_tokens"] == 300


def test_backend_without_reasoning_tokens_still_gets_the_estimate(orchestrator):
    """The fallback has to keep working: a backend that counts only visible
    content would otherwise report a fraction of what the turn actually cost."""
    stats = _run(orchestrator, _thinking_turn(_usage(1200, 50)))
    assert stats["output_tokens"] == 50 + ctxmgr.thinking_tokens(len(THINKING))


def test_reasoning_tokens_zero_is_a_report_not_a_silence(orchestrator):
    """A non-thinking model reports 0, which still means "the backend counted".
    Treating 0 as missing would re-add the estimate for the one case where the
    backend is certain there was nothing to add."""
    stats = _run(orchestrator, _thinking_turn(_usage(1200, 300, reasoning=0)))
    assert stats["output_tokens"] == 300


def test_the_flag_does_not_leak_into_the_next_turn(orchestrator):
    """Backends can be switched mid-conversation. A turn whose backend reports
    reasoning_tokens must not suppress the estimate on a later turn whose
    backend doesn't — that would silently undercount instead of double-count."""
    _run(orchestrator, _thinking_turn(_usage(1200, 300, reasoning=250)))
    before = orchestrator.total_output_tokens

    stats = _run(orchestrator, _thinking_turn(_usage(1200, 50)))
    assert stats["output_tokens"] - before == 50 + ctxmgr.thinking_tokens(len(THINKING))


def test_prompt_tokens_reach_the_context_gauge(orchestrator):
    """context_pct is computed from last_prompt_tokens. With no usage chunk it
    stayed 0 on every mira-mlx turn, which is what the usage chunk fixes."""
    _run(orchestrator, _thinking_turn(_usage(1200, 300, reasoning=250, cached=1024)))
    assert orchestrator.last_prompt_tokens == 1200
