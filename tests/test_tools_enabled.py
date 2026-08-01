"""Per-turn tool kill switch.

Bug this covers: `scripts/bench_questions.yaml` has declared `tools: false` per
question since the suite was written, `bench_compare.stream_chat()` accepted a
`tools` argument, and it never reached the server. Every question ran with the
full agentic toolset. On 2026-08-01 bench Q4 — "write a sqlite3 context manager",
`tools: false`, `expected_tool_calls: 0` — called `task_done` with a one-line
summary and emitted zero token events, so the answer was replaced by "Provided a
complete Python context manager for sqlite3 ...". A pure-generation question has
to reach the model with nothing to call.
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

import server
from core.orchestrator import ChatOrchestrator
from core.tools import TOOLS, GITHUB_TOOLS, _LOCAL_TOOLS


@pytest.fixture
def orchestrator():
    return ChatOrchestrator(verbose=False)


def _names(schemas):
    return {t["function"]["name"] for t in schemas}


# ── the switch itself ─────────────────────────────────────────────────────────

def test_default_is_on_so_existing_callers_are_unchanged(orchestrator):
    """Nothing that never heard of this flag may lose its tools."""
    assert orchestrator._tools_enabled is True
    assert orchestrator._active_tools


def test_disabled_hands_the_model_nothing(orchestrator):
    orchestrator._tools_enabled = False
    assert orchestrator._active_tools == []


def test_task_done_is_gone_when_disabled(orchestrator):
    """The specific escape hatch Q4 took. task_done is not a local tool, so a
    filter written in terms of _LOCAL_TOOLS would leave it reachable."""
    assert "task_done" in _names(orchestrator._active_tools)
    assert "task_done" not in _LOCAL_TOOLS  # guards the assumption above
    orchestrator._tools_enabled = False
    assert "task_done" not in _names(orchestrator._active_tools)


def test_github_tools_do_not_survive_the_switch(orchestrator):
    """github_tools_enabled is applied after the workspace branches; the kill
    switch has to win over it, not be overwritten by it."""
    orchestrator._github_tools_enabled = True
    assert _names(GITHUB_TOOLS) <= _names(orchestrator._active_tools)
    orchestrator._tools_enabled = False
    assert orchestrator._active_tools == []


def test_a_project_workspace_does_not_reopen_the_toolset(orchestrator, tmp_path):
    """workspace_root is the widest branch — it returns TOOLS whole."""
    orchestrator.project = {"local_path": str(tmp_path)}
    assert _names(orchestrator._active_tools) == _names(TOOLS)
    orchestrator._tools_enabled = False
    assert orchestrator._active_tools == []


def test_a_temp_workspace_does_not_reopen_the_toolset(orchestrator, tmp_path):
    orchestrator.project = None
    orchestrator._temp_workspace = str(tmp_path)
    assert orchestrator._active_tools
    orchestrator._tools_enabled = False
    assert orchestrator._active_tools == []


# ── the wire: stream_chat has to set it, and reset it ─────────────────────────

def test_stream_chat_accepts_the_flag_and_applies_it(orchestrator):
    """Consume one event so the generator body actually runs."""
    gen = orchestrator.stream_chat("hi", tools_enabled=False)
    next(gen, None)
    assert orchestrator._tools_enabled is False
    gen.close()


def test_the_flag_is_per_turn_not_sticky(orchestrator):
    """An orchestrator is pooled per conversation and reused across turns. A
    tools-off turn must not disarm every later turn in the same conversation."""
    gen = orchestrator.stream_chat("hi", tools_enabled=False)
    next(gen, None)
    gen.close()
    assert orchestrator._tools_enabled is False

    gen = orchestrator.stream_chat("hi again")  # default
    next(gen, None)
    gen.close()
    assert orchestrator._tools_enabled is True
    assert orchestrator._active_tools


# ── the wire, one step further out: POST /chat has to forward the form field ──
#
# This is the layer the original bug lived in. bench_compare.stream_chat() took a
# `tools` argument, used it to pick a project, and left it out of form_data — so
# every assertion about it held locally while the server never heard of it.

@pytest.fixture(scope="module")
def client():
    # sse_starlette keeps a PROCESS-GLOBAL `AppStatus.should_exit_event`, created
    # lazily the first time an SSE response waits on it and never reset. An
    # anyio/asyncio Event binds to the loop that first awaits it, so the second
    # test module to open an SSE stream — each TestClient runs its own loop —
    # dies with "bound to a different event loop" before the endpoint is even
    # reached. Today the first module is test_cancel.py. Clearing it lets
    # sse_starlette rebuild the event in this module's loop (sse.py:194).
    # Nothing to do with mira's code; drop this when sse_starlette scopes it.
    AppStatus.should_exit_event = None
    with TestClient(server.app, base_url="http://localhost") as c:
        yield c


def _capture_kwargs(client, form):
    seen = {}

    def fake_stream(message, attachments=None, **kwargs):
        seen.update(kwargs)
        yield {"type": "done", "content": "ok"}

    with patch.object(ChatOrchestrator, "stream_chat", side_effect=fake_stream):
        resp = client.post("/chat", data=form)
    assert resp.status_code == 200
    return seen


def test_chat_forwards_tools_enabled_false(client):
    seen = _capture_kwargs(client, {"message": "write me a function",
                                    "conversation_id": "__claude-test__",
                                    "tools_enabled": "false"})
    assert seen["tools_enabled"] is False


def test_chat_defaults_tools_enabled_to_true(client):
    """Omitting the field must not disarm tools for ordinary clients."""
    seen = _capture_kwargs(client, {"message": "hi",
                                    "conversation_id": "__claude-test__"})
    assert seen["tools_enabled"] is True
