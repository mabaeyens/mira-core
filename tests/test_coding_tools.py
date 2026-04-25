"""Orchestrator integration tests for coding tools (fs, shell, GitHub)."""

import json
import pytest
from unittest.mock import MagicMock, patch
from orchestrator import ChatOrchestrator


# ── Mock factories (match existing test_queries.py patterns) ──────────────────

def _make_chunk(content="", tool_calls=None, done=True):
    chunk = MagicMock()
    chunk.message.content = content
    chunk.message.tool_calls = tool_calls
    chunk.message.thinking = ""
    chunk.done = done
    return chunk


def _tool_stream(name: str, args: dict):
    """Mock _call_ollama for a single tool call."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return iter([_make_chunk(tool_calls=[tc], done=True)])


def _final_stream(content="Done."):
    return iter([_make_chunk(content=content, done=True)])


def _consume(gen):
    events = list(gen)
    done_events = [e for e in events if e["type"] == "done"]
    return events, (done_events[0]["content"] if done_events else "")


@pytest.fixture
def orc():
    return ChatOrchestrator(verbose=False)


# ── tool_start / tool_done events ─────────────────────────────────────────────

def test_read_file_emits_tool_events(orc, tmp_path):
    fake_result = {"path": "README.md", "content": "# Hello", "size": 7}
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("read_file", {"path": "README.md"}),
        _final_stream("The README says Hello."),
    ]), patch("fs_tools.read_file", return_value=fake_result):
        events, content = _consume(orc.stream_chat("What's in the README?"))

    types = [e["type"] for e in events]
    assert "tool_start" in types
    assert "tool_done" in types

    start = next(e for e in events if e["type"] == "tool_start")
    assert start["tool"] == "read_file"
    assert "README.md" in start["label"]

    done = next(e for e in events if e["type"] == "tool_done")
    assert done["tool"] == "read_file"
    assert "7" in done["label"]


def test_write_file_emits_tool_events(orc):
    fake_result = {"path": "out.txt", "bytes_written": 12, "action": "created"}
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("write_file", {"path": "out.txt", "content": "hello world\n"}),
        _final_stream("File written."),
    ]), patch("fs_tools.write_file", return_value=fake_result):
        events, _ = _consume(orc.stream_chat("Write hello world to out.txt"))

    start = next(e for e in events if e["type"] == "tool_start")
    assert "out.txt" in start["label"]
    done = next(e for e in events if e["type"] == "tool_done")
    assert "created" in done["label"]


def test_run_shell_emits_tool_events(orc):
    fake_result = {"command": "ls", "cwd": ".", "exit_code": 0, "stdout": "file.txt\n", "stderr": "", "truncated": False}
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("run_shell", {"command": "ls"}),
        _final_stream("Files: file.txt"),
    ]), patch("shell_tools.run_shell", return_value=fake_result):
        events, _ = _consume(orc.stream_chat("List the files"))

    start = next(e for e in events if e["type"] == "tool_start")
    assert "ls" in start["label"]
    done = next(e for e in events if e["type"] == "tool_done")
    assert "exit 0" in done["label"]


def test_github_list_repos_emits_tool_events(orc):
    fake_result = {"repos": [{"name": "user/repo"}], "count": 1}
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("github_list_repos", {}),
        _final_stream("You have 1 repo."),
    ]), patch("github_tools.github_list_repos", return_value=fake_result):
        events, _ = _consume(orc.stream_chat("List my repos"))

    start = next(e for e in events if e["type"] == "tool_start")
    assert start["tool"] == "github_list_repos"


# ── Tool result in conversation history ───────────────────────────────────────

def test_tool_result_added_to_history(orc):
    """Tool result must be injected as a 'tool' role message for the model."""
    fake_result = {"path": "a.py", "content": "x = 1", "size": 5}
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("read_file", {"path": "a.py"}),
        _final_stream("x equals 1."),
    ]), patch("fs_tools.read_file", return_value=fake_result):
        _consume(orc.stream_chat("What's in a.py?"))

    tool_msgs = [m for m in orc.conversation_history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert json.loads(tool_msgs[0]["content"]) == fake_result


# ── Confirmation flow ─────────────────────────────────────────────────────────

def test_confirmation_required_result_reaches_model(orc):
    """When a tool returns requires_confirmation, the model should receive it so it can ask the user."""
    confirmation_resp = {
        "requires_confirmation": True,
        "action": "delete_file",
        "path": "important.txt",
        "message": "This will permanently delete 'important.txt'. Ask the user to confirm.",
    }
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("delete_file", {"path": "important.txt"}),
        _final_stream("I need your confirmation before deleting important.txt."),
    ]), patch("fs_tools.delete_file", return_value=confirmation_resp):
        events, content = _consume(orc.stream_chat("Delete important.txt"))

    # Model must receive the confirmation_resp so it can surface it
    tool_msgs = [m for m in orc.conversation_history if m.get("role") == "tool"]
    parsed = json.loads(tool_msgs[0]["content"])
    assert parsed.get("requires_confirmation") is True

    # Final answer should be produced (model tells user to confirm)
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1


def test_shell_dangerous_command_confirmation_flow(orc):
    """run_shell returning requires_confirmation should reach the model."""
    confirmation_resp = {
        "requires_confirmation": True,
        "action": "run_shell",
        "command": "rm -rf .",
        "matched": "rm with -r/-f flag",
        "message": "Destructive operation. Ask user to confirm, then retry with force=true.",
    }
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("run_shell", {"command": "rm -rf ."}),
        _final_stream("That command is dangerous. Do you want me to proceed?"),
    ]), patch("shell_tools.run_shell", return_value=confirmation_resp):
        events, _ = _consume(orc.stream_chat("Run rm -rf ."))

    tool_msgs = [m for m in orc.conversation_history if m.get("role") == "tool"]
    parsed = json.loads(tool_msgs[0]["content"])
    assert parsed.get("requires_confirmation") is True
    assert parsed.get("matched") == "rm with -r/-f flag"


# ── Error propagation ─────────────────────────────────────────────────────────

def test_tool_error_result_reaches_model(orc):
    """Tool errors must be passed back to the model, not silently dropped."""
    error_result = {"error": "File not found: missing.txt"}
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("read_file", {"path": "missing.txt"}),
        _final_stream("That file doesn't exist."),
    ]), patch("fs_tools.read_file", return_value=error_result):
        events, content = _consume(orc.stream_chat("Read missing.txt"))

    tool_msgs = [m for m in orc.conversation_history if m.get("role") == "tool"]
    parsed = json.loads(tool_msgs[0]["content"])
    assert "error" in parsed

    done_ev = next(e for e in events if e["type"] == "tool_done")
    assert "Error" in done_ev["label"]


# ── Unknown tool fallback ─────────────────────────────────────────────────────

def test_unknown_tool_returns_error_to_model(orc):
    with patch.object(orc, '_call_ollama', side_effect=[
        _tool_stream("nonexistent_tool", {}),
        _final_stream("I couldn't do that."),
    ]):
        events, _ = _consume(orc.stream_chat("Do something unknown"))

    tool_msgs = [m for m in orc.conversation_history if m.get("role") == "tool"]
    parsed = json.loads(tool_msgs[0]["content"])
    assert "error" in parsed
    assert "nonexistent_tool" in parsed["error"]
