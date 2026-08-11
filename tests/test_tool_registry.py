"""Parity tests for the tool registry: the dispatchable schema set and the handler
set must be identical, so a schema/handler mismatch fails here instead of surfacing
as a swallowed runtime error."""
import pytest

from core import tools, tool_registry


# web_search and fetch_url have schemas but are handled inline in the agent loop,
# not via the registry — exclude them from the parity check.
_LOOP_HANDLED = {"web_search", "fetch_url"}


def _schema_names(*schema_lists):
    names = set()
    for lst in schema_lists:
        for t in lst:
            names.add(t["function"]["name"])
    return names


def test_schema_and_handler_sets_match():
    schema_names = _schema_names(tools.TOOLS, tools.GITHUB_TOOLS) - _LOOP_HANDLED
    handler_names = tool_registry.handler_names()
    assert schema_names == handler_names, (
        f"schema-only (no handler): {sorted(schema_names - handler_names)}; "
        f"handler-only (no schema): {sorted(handler_names - schema_names)}"
    )


def test_loop_handled_tools_have_schemas_but_no_handler():
    """web_search/fetch_url stay out of the registry by design."""
    for name in _LOOP_HANDLED:
        assert name not in tool_registry.handler_names()
    all_schema_names = _schema_names(tools.TOOLS, tools.GITHUB_TOOLS)
    assert _LOOP_HANDLED <= all_schema_names


def test_dispatch_unknown_tool_returns_error():
    ctx = tool_registry.ToolContext()
    result = tool_registry.dispatch("does_not_exist", {}, ctx)
    assert result["error"] == "Unknown tool: does_not_exist"


def test_dispatch_handler_exception_becomes_error_dict():
    """A handler raising (e.g. missing required arg) is caught and returned as error."""
    ctx = tool_registry.ToolContext()
    # github_read_file requires args["repo"] — omitting it raises KeyError, caught.
    result = tool_registry.dispatch("github_read_file", {}, ctx)
    assert "error" in result


def test_list_attachments_uses_context():
    ctx = tool_registry.ToolContext(attachments={
        "notes.txt": {"name": "notes.txt", "type": "text", "size": 12, "content": "hello world"},
    })
    out = tool_registry.dispatch("list_attachments", {}, ctx)
    assert "notes.txt" in out
    assert "text" in out


def test_read_attachment_offset_limit():
    ctx = tool_registry.ToolContext(attachments={
        "a.txt": {"name": "a.txt", "type": "text", "size": 11, "content": "0123456789X"},
    })
    out = tool_registry.dispatch("read_attachment", {"name": "a.txt", "offset": 2, "limit": 3}, ctx)
    assert "234" in out


def test_task_done_invokes_callback():
    captured = {}

    def _mark(summary):
        captured["summary"] = summary
        return {"done": True}

    ctx = tool_registry.ToolContext(mark_task_done=_mark)
    result = tool_registry.dispatch("task_done", {"summary": "all set"}, ctx)
    assert result == {"done": True}
    assert captured["summary"] == "all set"


@pytest.mark.parametrize("name, args, result", [
    ("list_attachments", {}, "No files attached to this conversation."),
    ("read_attachment", {"name": "a.txt"}, "[a.txt]\nhello"),
])
def test_a_string_result_still_gets_a_ui_label(name, args, result):
    """The label layer must not assume every tool hands back a dict.

    It did, and it cost three turns of a real conversation: the model called
    list_attachments, the done-label lambda ran .get on a str, and the stream
    died with "Internal error" after the tool_start event had already been sent.
    """
    from core.orchestrator import _tool_ui_labels

    start, done = _tool_ui_labels(name, args)
    assert start
    assert done(result) == "Done"
