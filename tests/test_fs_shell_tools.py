"""Tests for fs_tools and shell_tools (sandboxed to a tmp directory)."""

import pytest
from pathlib import Path
from unittest.mock import patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def ws(tmp_path):
    """Patch WORKSPACE_ROOT in the workspace module (the single source of truth)."""
    with patch("core.workspace.WORKSPACE_ROOT", str(tmp_path)):
        yield tmp_path


# ── fs_tools: read_file ───────────────────────────────────────────────────────

def test_read_file_returns_content(ws):
    from core import fs_tools
    (ws / "hello.txt").write_text("world")
    result = fs_tools.read_file("hello.txt")
    assert result["content"] == "world"
    assert result["size"] == 5


def test_read_file_missing_returns_error(ws):
    from core import fs_tools
    result = fs_tools.read_file("nope.txt")
    assert "error" in result


def test_read_file_sandbox_escape_blocked(ws):
    from core import fs_tools
    result = fs_tools.read_file("../../etc/passwd")
    assert "error" in result


# ── fs_tools: write_file ──────────────────────────────────────────────────────

def test_write_file_creates_file(ws):
    from core import fs_tools
    result = fs_tools.write_file("new.txt", "content")
    assert result["action"] == "created"
    assert (ws / "new.txt").read_text() == "content"


def test_write_file_updates_existing(ws):
    from core import fs_tools
    (ws / "existing.txt").write_text("old")
    result = fs_tools.write_file("existing.txt", "new")
    assert result["action"] == "updated"
    assert (ws / "existing.txt").read_text() == "new"


def test_write_file_creates_parent_dirs(ws):
    from core import fs_tools
    result = fs_tools.write_file("a/b/c.txt", "deep")
    assert "error" not in result
    assert (ws / "a/b/c.txt").exists()


def test_write_file_sandbox_escape_blocked(ws):
    from core import fs_tools
    result = fs_tools.write_file("../outside.txt", "evil")
    assert "error" in result


# ── fs_tools: list_files ──────────────────────────────────────────────────────

def test_list_files_returns_entries(ws):
    from core import fs_tools
    (ws / "a.txt").write_text("x")
    (ws / "b.txt").write_text("y")
    result = fs_tools.list_files(".")
    names = [e["path"] for e in result["entries"]]
    assert "a.txt" in names
    assert "b.txt" in names


def test_list_files_recursive(ws):
    from core import fs_tools
    (ws / "sub").mkdir()
    (ws / "sub/deep.py").write_text("code")
    result = fs_tools.list_files(".", recursive=True)
    paths = [e["path"] for e in result["entries"]]
    assert any("deep.py" in p for p in paths)


def test_list_files_missing_dir_returns_error(ws):
    from core import fs_tools
    result = fs_tools.list_files("nonexistent")
    assert "error" in result


# ── fs_tools: search_files ────────────────────────────────────────────────────

def test_search_files_finds_match(ws):
    from core import fs_tools
    (ws / "code.py").write_text("def hello():\n    return 42\n")
    result = fs_tools.search_files("def hello")
    assert result["count"] == 1
    assert result["matches"][0]["line"] == 1


def test_search_files_case_insensitive_by_default(ws):
    from core import fs_tools
    (ws / "notes.txt").write_text("Hello World\n")
    result = fs_tools.search_files("hello world")
    assert result["count"] == 1


def test_search_files_case_sensitive(ws):
    from core import fs_tools
    (ws / "notes.txt").write_text("Hello World\n")
    result = fs_tools.search_files("hello world", case_sensitive=True)
    assert result["count"] == 0


def test_search_files_no_match(ws):
    from core import fs_tools
    (ws / "empty.txt").write_text("nothing here\n")
    result = fs_tools.search_files("xyz_not_found")
    assert result["count"] == 0


def test_search_files_invalid_regex_returns_error(ws):
    from core import fs_tools
    result = fs_tools.search_files("[unclosed")
    assert "error" in result


def test_search_files_excludes_dot_and_pycache_dirs(ws):
    from core import fs_tools
    (ws / "pkg").mkdir()
    (ws / "pkg" / "y.py").write_text("# TODO real\n")
    (ws / ".venv").mkdir()
    (ws / ".venv" / "x.py").write_text("# TODO vendored\n")
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "z.py").write_text("# TODO cached\n")
    result = fs_tools.search_files("TODO")
    files = {m["file"] for m in result["matches"]}
    assert files == {"pkg/y.py"}


# ── fs_tools: move_file ───────────────────────────────────────────────────────

def test_move_file_renames(ws):
    from core import fs_tools
    (ws / "old.txt").write_text("data")
    result = fs_tools.move_file("old.txt", "new.txt")
    assert "error" not in result
    assert not (ws / "old.txt").exists()
    assert (ws / "new.txt").read_text() == "data"


def test_move_file_missing_source_returns_error(ws):
    from core import fs_tools
    result = fs_tools.move_file("ghost.txt", "dest.txt")
    assert "error" in result


def test_move_file_sandbox_escape_blocked(ws):
    from core import fs_tools
    (ws / "src.txt").write_text("data")
    result = fs_tools.move_file("src.txt", "../../escape.txt")
    assert "error" in result


# ── fs_tools: delete_file ─────────────────────────────────────────────────────

def test_delete_file_without_confirm_returns_confirmation_request(ws):
    from core import fs_tools
    (ws / "target.txt").write_text("data")
    result = fs_tools.delete_file("target.txt")
    assert result.get("requires_confirmation") is True
    assert (ws / "target.txt").exists()  # not deleted yet


def test_delete_file_with_confirm_deletes(ws):
    from core import fs_tools
    (ws / "target.txt").write_text("data")
    result = fs_tools.delete_file("target.txt", confirm=True)
    assert "deleted" in result
    assert not (ws / "target.txt").exists()


def test_delete_file_missing_returns_error(ws):
    from core import fs_tools
    result = fs_tools.delete_file("ghost.txt", confirm=True)
    assert "error" in result


# ── shell_tools: run_shell ────────────────────────────────────────────────────

def test_run_shell_basic_command(ws):
    from core import shell_tools
    result = shell_tools.run_shell("echo hello")
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


def test_run_shell_cwd_is_within_workspace(ws):
    from core import shell_tools
    result = shell_tools.run_shell("pwd")
    assert str(ws) in result["stdout"]


def test_run_shell_non_zero_exit_code(ws):
    from core import shell_tools
    result = shell_tools.run_shell("exit 1", cwd=".")
    assert result["exit_code"] == 1


def test_run_shell_captures_stderr(ws):
    from core import shell_tools
    result = shell_tools.run_shell("ls /nonexistent_path_xyz_abc 2>&1 || true")
    # Either stderr or stdout should mention the path doesn't exist
    output = result["stdout"] + result["stderr"]
    assert len(output) > 0


def test_run_shell_cwd_sandbox_escape_blocked(ws):
    from core import shell_tools
    result = shell_tools.run_shell("pwd", cwd="../../..")
    assert "error" in result


def test_run_shell_allows_glob_exclusion_pattern(ws):
    from core import shell_tools
    # Glob-relative exclusion patterns must NOT be mistaken for absolute paths.
    result = shell_tools.run_shell("find . -path '*/build/*' -prune -o -name '*.py' -print")
    assert "error" not in result
    assert result["exit_code"] == 0


def test_run_shell_absolute_path_outside_workspace_still_blocked(ws):
    from core import shell_tools
    # Security regression: a true absolute path outside the workspace stays blocked.
    result = shell_tools.run_shell("cat /etc/hosts")
    assert "error" in result
    assert "absolute path" in result["error"].lower()


def test_run_shell_rm_rf_blocked_without_force(ws):
    from core import shell_tools
    result = shell_tools.run_shell("rm -rf .")
    assert result.get("requires_confirmation") is True
    assert result.get("matched") == "rm with -r/-f flag"


def test_run_shell_git_push_force_blocked(ws):
    from core import shell_tools
    result = shell_tools.run_shell("git push origin main --force")
    assert result.get("requires_confirmation") is True


def test_run_shell_git_reset_hard_blocked(ws):
    from core import shell_tools
    result = shell_tools.run_shell("git reset --hard HEAD")
    assert result.get("requires_confirmation") is True


def test_run_shell_sudo_blocked(ws):
    from core import shell_tools
    result = shell_tools.run_shell("sudo rm file.txt")
    assert result.get("requires_confirmation") is True


def test_run_shell_force_bypasses_guard(ws):
    from core import shell_tools
    # `force` is now an INTERNAL parameter set from a user approval token, never
    # from model output. Called directly it still bypasses the guard — that is
    # the mechanism the approval layer drives.
    (ws / "deleteme.txt").write_text("bye")
    result = shell_tools.run_shell("rm -rf deleteme.txt", force=True)
    assert result["exit_code"] == 0
    assert not (ws / "deleteme.txt").exists()


def test_model_cannot_self_approve_destructive_command(ws):
    """Regression guard: the model must not be able to authorise its own
    destructive command. `force` was previously a field in run_shell's JSON
    schema, so emitting force=true was enough to defeat the confirmation gate.
    """
    from core import tools, tool_registry

    # 1. The flag must not be offered to the model at all.
    shell_params = tools.RUN_SHELL_TOOL["function"]["parameters"]["properties"]
    assert "force" not in shell_params
    delete_params = tools.DELETE_FILE_TOOL["function"]["parameters"]["properties"]
    assert "confirm" not in delete_params

    # 2. Even if the model emits it anyway, dispatch must ignore it.
    (ws / "keepme.txt").write_text("still here")
    ctx = tool_registry.ToolContext(workspace_root=str(ws), approved=frozenset())
    result = tool_registry.dispatch(
        "run_shell", {"command": "rm -rf keepme.txt", "force": True}, ctx
    )
    assert result.get("requires_confirmation") is True
    assert (ws / "keepme.txt").exists(), "model-supplied force must not delete anything"

    # 3. A genuine user approval token does let it through.
    from core.approvals import approval_token
    ctx_ok = tool_registry.ToolContext(
        workspace_root=str(ws),
        approved=frozenset({approval_token("run_shell", "rm -rf keepme.txt")}),
    )
    result_ok = tool_registry.dispatch("run_shell", {"command": "rm -rf keepme.txt"}, ctx_ok)
    assert result_ok.get("exit_code") == 0
    assert not (ws / "keepme.txt").exists()

    # 4. An approval for a DIFFERENT command must not authorise this one.
    (ws / "other.txt").write_text("x")
    ctx_wrong = tool_registry.ToolContext(
        workspace_root=str(ws),
        approved=frozenset({approval_token("run_shell", "rm -rf something-else")}),
    )
    result_wrong = tool_registry.dispatch("run_shell", {"command": "rm -rf other.txt"}, ctx_wrong)
    assert result_wrong.get("requires_confirmation") is True
    assert (ws / "other.txt").exists()


def test_run_shell_timeout(ws):
    from core import shell_tools
    with patch("core.shell_tools.SHELL_TIMEOUT", 1):
        result = shell_tools.run_shell("sleep 10")
    assert "error" in result
    assert "Timed out" in result["error"]
