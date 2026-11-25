"""A tool call the turn did not offer is refused, not executed.

Found 2026-11-25: a GPQA run with `tools_enabled=False` handed the model no tools,
yet the engine's parser still turned the model's tool-call message into a call and
the orchestrator ran it (web searches, 9 calls, 600 s timeout, no answer).
"""
from types import SimpleNamespace

import pytest

from core.orchestrator import ChatOrchestrator


@pytest.fixture
def orchestrator():
    return ChatOrchestrator(verbose=False)


def _tool_call(name, args, tc_id="call_0"):
    return SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=args))


def _turn(content="", tool_calls=None):
    return {
        "type": "llm_done",
        "full_content": content,
        "final_message": SimpleNamespace(content=content, tool_calls=tool_calls or None),
        "thinking_chars": 0,
    }


def _script(orchestrator, turns):
    calls = []

    def fake_stream(thinking_enabled):
        i = len(calls)
        calls.append(thinking_enabled)
        if i >= len(turns):
            raise AssertionError("loop asked for more replies than scripted")
        yield turns[i]

    orchestrator._stream_llm_with_thinking = fake_stream
    return calls


def test_no_tools_turn_does_not_execute_a_search(orchestrator):
    ran = []
    orchestrator.search_engine = SimpleNamespace(
        search=lambda q: ran.append(q) or [], format_tool_result=lambda r: "")
    calls = _script(orchestrator, [
        _turn(tool_calls=[_tool_call("web_search", {"query": "answer", }, tc_id="tc-1")]),
        _turn(content="B"),
    ])
    events = list(orchestrator.stream_chat("question", tools_enabled=False))

    assert ran == [], "a search ran in a turn that offered no tools"
    assert len(calls) == 2
    assert [e for e in events if e.get("type") == "done"][0]["content"] == "B"
    refusal = [m for m in orchestrator.conversation_history if m["role"] == "tool"][0]
    assert refusal["tool_call_id"] == "tc-1"
    assert "Refused" in refusal["content"]
    assert not any(e.get("type") in ("search_start", "tool_start", "fetch_start") for e in events)


def test_offered_tool_in_a_normal_turn_still_runs(orchestrator):
    ran = []
    orchestrator.search_engine = SimpleNamespace(
        search=lambda q: ran.append(q) or [], format_tool_result=lambda r: "")
    _script(orchestrator, [
        _turn(tool_calls=[_tool_call("web_search", {"query": "x"})]),
        _turn(content="done"),
    ])
    list(orchestrator.stream_chat("question"))
    assert ran == ["x"]
