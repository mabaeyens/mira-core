"""Central tool registry: pairs each tool's handler with its JSON schema name so the
two can never silently drift apart.

Schemas live in ``core/tools.py``; handlers are registered here by the same name via
the ``@tool`` decorator. ``tests/test_tool_registry.py`` asserts the dispatchable
schema set and the handler set are identical, so a schema with no handler (or a
handler with no schema) fails the suite instead of surfacing as a swallowed runtime
KeyError.

``web_search`` and ``fetch_url`` are intentionally NOT registered here — they are
handled inline in the orchestrator's agent loop (they emit progress events and read
orchestrator-owned engines), so they are excluded from the parity check.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from . import fs_tools, shell_tools, github_tools, db

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Per-turn state a handler may need. The orchestrator builds one per dispatch."""
    workspace_root: Optional[str] = None
    temp_workspace: Optional[str] = None
    attachments: Optional[Dict[str, dict]] = None          # name -> {name,type,size,content}
    mark_task_done: Optional[Callable[[str], dict]] = None

    @property
    def read_root(self) -> Optional[str]:
        """Root for read-only fs tools — falls back to the temp attachment workspace."""
        return self.workspace_root or self.temp_workspace


Handler = Callable[[ToolContext, dict], dict]

_HANDLERS: Dict[str, Handler] = {}


def tool(name: str):
    """Register ``fn`` as the handler for the tool schema named ``name``."""
    def deco(fn: Handler) -> Handler:
        if name in _HANDLERS:
            raise ValueError(f"Duplicate handler registered for tool '{name}'")
        _HANDLERS[name] = fn
        return fn
    return deco


def handler_names() -> set:
    return set(_HANDLERS.keys())


def dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    """Run the handler for ``name``. Mirrors the previous orchestrator behaviour:
    unknown tools and handler exceptions both return an ``{"error": ...}`` dict."""
    fn = _HANDLERS.get(name)
    if fn is None:
        logger.warning("Unknown tool: %s", name)
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(ctx, args)
    except Exception as e:
        logger.error("Tool %s raised: %s", name, e)
        return {"error": str(e)}


# ── Filesystem ────────────────────────────────────────────────────────────────

@tool("read_file")
def _read_file(ctx, a):
    return fs_tools.read_file(a.get("path", ""), root=ctx.read_root)


@tool("write_file")
def _write_file(ctx, a):
    return fs_tools.write_file(a.get("path", ""), a.get("content", ""), root=ctx.workspace_root)


@tool("edit_file")
def _edit_file(ctx, a):
    return fs_tools.edit_file(a.get("path", ""), a.get("old_str", ""), a.get("new_str", ""), root=ctx.workspace_root)


@tool("list_files")
def _list_files(ctx, a):
    return fs_tools.list_files(a.get("path", "."), a.get("recursive", False), root=ctx.read_root)


@tool("search_files")
def _search_files(ctx, a):
    return fs_tools.search_files(a.get("pattern", ""), a.get("path", "."), a.get("case_sensitive", False), root=ctx.read_root)


@tool("move_file")
def _move_file(ctx, a):
    return fs_tools.move_file(a.get("src", ""), a.get("dst", ""), root=ctx.workspace_root)


@tool("delete_file")
def _delete_file(ctx, a):
    return fs_tools.delete_file(a.get("path", ""), a.get("confirm", False), root=ctx.workspace_root)


@tool("run_shell")
def _run_shell(ctx, a):
    return shell_tools.run_shell(
        a.get("command", ""), a.get("cwd", "."), a.get("force", False),
        root=ctx.workspace_root, timeout=a.get("timeout", 30),
    )


# ── Attachments ───────────────────────────────────────────────────────────────

@tool("list_attachments")
def _list_attachments(ctx, a):
    reg = ctx.attachments or {}
    if not reg:
        return "No files attached to this conversation."
    lines = [
        f"- {info['name']} ({info['type']}, {info['size']:,} bytes)"
        for info in reg.values()
    ]
    return "Attached files:\n" + "\n".join(lines)


@tool("read_attachment")
def _read_attachment(ctx, a):
    reg = ctx.attachments or {}
    name = a.get("name", "")
    offset = a.get("offset", 0)
    limit = a.get("limit")
    info = reg.get(name)
    if info is None:
        available = ", ".join(reg.keys()) or "none"
        return f"No attachment named '{name}'. Available: {available}"
    content = info["content"]
    if not content:
        return f"Attachment '{name}' has no readable text content (type: {info['type']})."
    chunk = content[offset: offset + limit] if limit is not None else content[offset:]
    total = len(content)
    if offset or limit is not None:
        end = offset + len(chunk)
        return f"[{name} — chars {offset}–{end} of {total}]\n{chunk}"
    return f"[{name}]\n{chunk}"


# ── Memory / conversations ────────────────────────────────────────────────────

@tool("schedule_reminder")
def _schedule_reminder(ctx, a):
    import dateparser
    from dateutil import parser as dateutil_parser
    import datetime
    text = a.get("text", "")
    when = a.get("when", "")
    if not text.strip():
        return {"error": "Reminder text cannot be empty."}
    if not when.strip():
        return {"error": "Please specify when to deliver the reminder."}
    dt = dateparser.parse(when, settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": True})
    if dt is None:
        # Fall back to dateutil for ISO 8601 and absolute dates
        try:
            now = datetime.datetime.now().astimezone()
            dt = dateutil_parser.parse(when, default=now, fuzzy=True)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=now.tzinfo)
        except Exception:
            return {"error": f"Could not parse '{when}' as a date/time. Try 'tomorrow at 9am', 'in 2 hours', or '2026-06-04T09:00'."}
    scheduled_at = int(dt.timestamp())
    reminder_id = db.add_reminder(text.strip(), scheduled_at)
    return {
        "id": reminder_id,
        "text": text.strip(),
        "scheduled_at": scheduled_at,
        "scheduled_for": dt.strftime("%A, %B %-d at %-I:%M %p"),
    }


@tool("search_conversations")
def _search_conversations(ctx, a):
    query = a.get("query", "")
    limit = a.get("limit", 10)
    if not query.strip():
        return {"results": [], "message": "No query provided."}
    results = db.search_conversations(query, limit=min(int(limit), 50))
    if not results:
        return {"results": [], "message": f"No conversations found matching '{query}'."}
    return {"results": results, "count": len(results)}


# ── Agentic ───────────────────────────────────────────────────────────────────

@tool("task_done")
def _task_done(ctx, a):
    summary = a.get("summary", "Task complete.")
    if ctx.mark_task_done is not None:
        return ctx.mark_task_done(summary)
    return {"done": True}


# ── GitHub ────────────────────────────────────────────────────────────────────

@tool("github_clone_repo")
def _github_clone_repo(ctx, a):
    result = github_tools.github_clone_repo(a["repo"], a.get("dest", ""))
    if "error" in result:
        return result
    repo_name = a["repo"].split("/")[-1]
    project_name = a.get("project_name", "").strip() or repo_name
    project_id = db.create_project(project_name, local_path=result["cloned_to"], github_repo=a["repo"])
    result["project_id"] = project_id
    result["project_name"] = project_name
    return result


@tool("github_list_repos")
def _github_list_repos(ctx, a):
    return github_tools.github_list_repos(a.get("repo_type", "owner"))


@tool("github_read_file")
def _github_read_file(ctx, a):
    return github_tools.github_read_file(a["repo"], a["path"], a.get("ref", ""))


@tool("github_list_files")
def _github_list_files(ctx, a):
    return github_tools.github_list_files(a["repo"], a.get("path", ""), a.get("ref", ""))


@tool("github_list_issues")
def _github_list_issues(ctx, a):
    return github_tools.github_list_issues(a["repo"], a.get("state", "open"))


@tool("github_list_prs")
def _github_list_prs(ctx, a):
    return github_tools.github_list_prs(a["repo"], a.get("state", "open"))


@tool("github_search_code")
def _github_search_code(ctx, a):
    return github_tools.github_search_code(a["query"], a.get("repo", ""))


@tool("github_write_file")
def _github_write_file(ctx, a):
    return github_tools.github_write_file(a["repo"], a["path"], a["content"], a["message"], a.get("branch", ""), a.get("sha", ""))


@tool("github_create_repo")
def _github_create_repo(ctx, a):
    return github_tools.github_create_repo(a["name"], a.get("private", True), a.get("description", ""), a.get("auto_init", True))


@tool("github_create_issue")
def _github_create_issue(ctx, a):
    return github_tools.github_create_issue(a["repo"], a["title"], a.get("body", ""))


@tool("github_create_branch")
def _github_create_branch(ctx, a):
    return github_tools.github_create_branch(a["repo"], a["branch"], a.get("from_ref", ""))


@tool("github_create_pr")
def _github_create_pr(ctx, a):
    return github_tools.github_create_pr(a["repo"], a["title"], a.get("body", ""), a.get("head", ""), a.get("base", ""))


@tool("github_merge_pr")
def _github_merge_pr(ctx, a):
    return github_tools.github_merge_pr(a["repo"], a["pr_number"], a.get("merge_method", "merge"), a.get("confirm", False))


@tool("github_delete_file")
def _github_delete_file(ctx, a):
    return github_tools.github_delete_file(a["repo"], a["path"], a["message"], a.get("branch", ""), a.get("confirm", False))


@tool("github_delete_branch")
def _github_delete_branch(ctx, a):
    return github_tools.github_delete_branch(a["repo"], a["branch"], a.get("confirm", False))
