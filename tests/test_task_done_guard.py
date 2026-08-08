"""task_done may not be the entire user-visible output of a turn.

Bug this covers: on 2026-08-01 bench Q4 — "write a sqlite3 context manager",
`tools: false` — the model emitted zero token events and called `task_done` with
the summary "Provided a complete Python context manager for sqlite3 ...". The
user got the sentence and no context manager, roughly two runs in three.

`tools: false` not reaching the server was one half of that and is fixed
separately (d6a8a17, tests/test_tools_enabled.py). This is the other half, and it
holds even when tools are legitimately on: nothing checked that the work being
declared done had ever been shown to the user.

The guard keys on whether tools ran, not on empty content alone, because a
genuinely agentic turn whose answer *is* a summary ("deleted 3 files") is
correct and must still exit.
"""
import json
from types import SimpleNamespace

import pytest

from core.orchestrator import ChatOrchestrator


@pytest.fixture
def orchestrator():
    return ChatOrchestrator(verbose=False)


def _tool_call(name, args, tc_id="call_0"):
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


def _turn(content="", tool_calls=None):
    """One scripted LLM reply, in the shape _stream_llm_with_thinking yields."""
    return {
        "type": "llm_done",
        "full_content": content,
        "final_message": SimpleNamespace(content=content, tool_calls=tool_calls or None),
        "thinking_chars": 0,
    }


def _script(orchestrator, turns):
    """Replace the LLM with a fixed sequence of replies; record how many ran."""
    calls = []

    def fake_stream(thinking_enabled):
        i = len(calls)
        calls.append(thinking_enabled)
        if i >= len(turns):
            raise AssertionError(f"loop asked for reply {i + 1}; script has {len(turns)}")
        yield turns[i]

    orchestrator._stream_llm_with_thinking = fake_stream
    return calls


def _run(orchestrator, message="Write a sqlite3 context manager"):
    return list(orchestrator.stream_chat(message))


def _done(events):
    done = [e for e in events if e.get("type") == "done"]
    assert len(done) == 1, f"expected exactly one done event, got {len(done)}"
    return done[0]


ANSWER = "```python\nclass Db:\n    ...\n```"
SUMMARY = "Provided a complete Python context manager for sqlite3."


# ── the refusal ───────────────────────────────────────────────────────────────

def test_a_bare_task_done_is_refused_and_the_answer_arrives_instead(orchestrator):
    """The Q4 reproduction. The user must end up with the code, not the sentence."""
    calls = _script(orchestrator, [
        _turn(tool_calls=[_tool_call("task_done", {"summary": SUMMARY})]),
        _turn(content=ANSWER),
    ])
    events = _run(orchestrator)

    assert len(calls) == 2, "the model was never asked to actually answer"
    assert _done(events)["content"] == ANSWER
    assert SUMMARY not in _done(events)["content"]


def test_the_refusal_lands_before_the_done_event(orchestrator):
    """Edge case (d): once done is yielded the client has been told the turn
    ended, so a refusal after it changes nothing the user sees."""
    _script(orchestrator, [
        _turn(tool_calls=[_tool_call("task_done", {"summary": SUMMARY})]),
        _turn(content=ANSWER),
    ])
    events = _run(orchestrator)

    types = [e.get("type") for e in events]
    assert types.count("done") == 1
    assert _done(events).get("task_done") is not True


def test_the_refused_call_is_answered_in_history(orchestrator):
    """An assistant message carrying tool_calls with no matching tool reply
    breaks strict-alternation chat templates (Mistral family). The refusal has
    to be delivered as the tool result, not as a dangling user turn."""
    _script(orchestrator, [
        _turn(tool_calls=[_tool_call("task_done", {"summary": SUMMARY}, tc_id="tc-42")]),
        _turn(content=ANSWER),
    ])
    _run(orchestrator)

    roles = [m["role"] for m in orchestrator.conversation_history]
    assert roles[-3:] == ["assistant", "tool", "assistant"]
    refusal = orchestrator.conversation_history[-2]
    assert refusal["tool_call_id"] == "tc-42"
    assert refusal["name"] == "task_done"
    assert "Refused" in refusal["content"]


def test_every_call_in_the_batch_is_answered(orchestrator):
    """Two task_done calls in one batch leave two unanswered tool_call ids if the
    refusal only replies to the first."""
    _script(orchestrator, [
        _turn(tool_calls=[
            _tool_call("task_done", {"summary": SUMMARY}, tc_id="tc-a"),
            _tool_call("task_done", {"summary": SUMMARY}, tc_id="tc-b"),
        ]),
        _turn(content=ANSWER),
    ])
    _run(orchestrator)

    answered = {m["tool_call_id"] for m in orchestrator.conversation_history
                if m["role"] == "tool"}
    assert answered == {"tc-a", "tc-b"}


# ── what the guard must not touch ─────────────────────────────────────────────

def test_a_summary_after_real_tool_work_still_exits(orchestrator):
    """Edge case (a): the model did the work and the summary IS the answer.
    Banning empty-content task_done outright would break this."""
    orchestrator._execute_tools = lambda prepared, step, fetch_results: iter(())
    calls = _script(orchestrator, [
        _turn(tool_calls=[_tool_call("read_file", {"path": "x.py"}, tc_id="tc-r")]),
        _turn(tool_calls=[_tool_call("task_done", {"summary": "Deleted 3 files."})]),
    ])
    events = _run(orchestrator, "clean up the temp files")

    assert len(calls) == 2, "the model was made to answer a turn it had earned"
    done = _done(events)
    assert done["content"] == "Deleted 3 files."
    assert done["task_done"] is True


def test_task_done_alongside_visible_content_is_accepted(orchestrator):
    """The model answered and then declared itself finished. Nothing was hidden,
    so there is nothing to refuse."""
    calls = _script(orchestrator, [
        _turn(content=ANSWER, tool_calls=[_tool_call("task_done", {"summary": SUMMARY})]),
    ])
    events = _run(orchestrator)

    assert len(calls) == 1
    assert _done(events)["task_done"] is True


def test_whitespace_only_content_does_not_count_as_an_answer(orchestrator):
    """A newline is not a context manager."""
    _script(orchestrator, [
        _turn(content="   \n\n", tool_calls=[_tool_call("task_done", {"summary": SUMMARY})]),
        _turn(content=ANSWER),
    ])
    assert _done(_run(orchestrator))["content"] == ANSWER


# ── the latch ─────────────────────────────────────────────────────────────────

def test_the_refusal_does_not_recurse(orchestrator):
    """Edge case (c): a model that calls task_done again after being refused gets
    taken at its word the second time. One refusal per turn, then move on."""
    calls = _script(orchestrator, [
        _turn(tool_calls=[_tool_call("task_done", {"summary": SUMMARY})]),
        _turn(tool_calls=[_tool_call("task_done", {"summary": SUMMARY})]),
    ])
    events = _run(orchestrator)

    assert len(calls) == 2, "refused more than once"
    done = _done(events)
    assert done["content"] == SUMMARY
    assert done["task_done"] is True


def test_the_latch_resets_between_turns(orchestrator):
    """The orchestrator is pooled per conversation and reused. A turn that spent
    its refusal must not leave the next turn unguarded."""
    _script(orchestrator, [
        _turn(tool_calls=[_tool_call("task_done", {"summary": SUMMARY})]),
        _turn(content=ANSWER),
    ])
    _run(orchestrator)
    assert orchestrator._task_done_refused is True

    _script(orchestrator, [
        _turn(tool_calls=[_tool_call("task_done", {"summary": SUMMARY})]),
        _turn(content=ANSWER),
    ])
    assert _done(_run(orchestrator, "and again"))["content"] == ANSWER


# ── the missing summary key, unchanged by the guard ───────────────────────────

def test_a_task_done_without_a_summary_still_has_a_fallback(orchestrator):
    """Reached only on the second call, so the guard must not have eaten the
    default on the way past."""
    _script(orchestrator, [
        _turn(tool_calls=[_tool_call("task_done", {})]),
        _turn(tool_calls=[_tool_call("task_done", {})]),
    ])
    assert _done(_run(orchestrator))["content"] == "Task complete."
