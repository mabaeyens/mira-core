"""Tests for fs_tools and shell_tools (sandboxed to a tmp directory)."""

import pytest
from pathlib import Path
from unittest.mock import patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def ws(tmp_path):
    """Patch WORKSPACE_ROOT in the workspace module (the single source of truth)."""
    with patch("workspace.WORKSPACE_ROOT", str(tmp_path)):
        yield tmp_path


# ── fs_tools: read_file ───────────────────────────────────────────────────────

def test_read_file_returns_content(ws):
    import fs_tools
    (ws / "hello.txt").write_text("world")
    result = fs_tools.read_file("hello.txt")
    assert result["content"] == "world"
    assert result["size"] == 5


def test_read_file_missing_returns_error(ws):
    import fs_tools
    result = fs_tools.read_file("nope.txt")
    assert "error" in result


def test_read_file_sandbox_escape_blocked(ws):
    import fs_tools
    result = fs_tools.read_file("../../etc/passwd")
    assert "error" in result


# ── fs_tools: write_file ──────────────────────────────────────────────────────

def test_write_file_creates_file(ws):
    import fs_tools
    result = fs_tools.write_file("new.txt", "content")
    assert result["action"] == "created"
    assert (ws / "new.txt").read_text() == "content"


def test_write_file_updates_existing(ws):
    import fs_tools
    (ws / "existing.txt").write_text("old")
    result = fs_tools.write_file("existing.txt", "new")
    assert result["action"] == "updated"
    assert (ws / "existing.txt").read_text() == "new"


def test_write_file_creates_parent_dirs(ws):
    import fs_tools
    result = fs_tools.write_file("a/b/c.txt", "deep")
    assert "error" not in result
    assert (ws / "a/b/c.txt").exists()


def test_write_file_sandbox_escape_blocked(ws):
    import fs_tools
    result = fs_tools.write_file("../outside.txt", "evil")
    assert "error" in result


# ── fs_tools: list_files ──────────────────────────────────────────────────────

def test_list_files_returns_entries(ws):
    import fs_tools
    (ws / "a.txt").write_text("x")
    (ws / "b.txt").write_text("y")
    result = fs_tools.list_files(".")
    names = [e["path"] for e in result["entries"]]
    assert "a.txt" in names
    assert "b.txt" in names


def test_list_files_recursive(ws):
    import fs_tools
    (ws / "sub").mkdir()
    (ws / "sub/deep.py").write_text("code")
    result = fs_tools.list_files(".", recursive=True)
    paths = [e["path"] for e in result["entries"]]
    assert any("deep.py" in p for p in paths)


def test_list_files_missing_dir_returns_error(ws):
    import fs_tools
    result = fs_tools.list_files("nonexistent")
    assert "error" in result


# ── fs_tools: search_files ────────────────────────────────────────────────────

def test_search_files_finds_match(ws):
    import fs_tools
    (ws / "code.py").write_text("def hello():\n    return 42\n")
    result = fs_tools.search_files("def hello")
    assert result["count"] == 1
    assert result["matches"][0]["line"] == 1


def test_search_files_case_insensitive_by_default(ws):
    import fs_tools
    (ws / "notes.txt").write_text("Hello World\n")
    result = fs_tools.search_files("hello world")
    assert result["count"] == 1


def test_search_files_case_sensitive(ws):
    import fs_tools
    (ws / "notes.txt").write_text("Hello World\n")
    result = fs_tools.search_files("hello world", case_sensitive=True)
    assert result["count"] == 0


def test_search_files_no_match(ws):
    import fs_tools
    (ws / "empty.txt").write_text("nothing here\n")
    result = fs_tools.search_files("xyz_not_found")
    assert result["count"] == 0


def test_search_files_invalid_regex_returns_error(ws):
    import fs_tools
    result = fs_tools.search_files("[unclosed")
    assert "error" in result


# ── fs_tools: move_file ───────────────────────────────────────────────────────

def test_move_file_renames(ws):
    import fs_tools
    (ws / "old.txt").write_text("data")
    result = fs_tools.move_file("old.txt", "new.txt")
    assert "error" not in result
    assert not (ws / "old.txt").exists()
    assert (ws / "new.txt").read_text() == "data"


def test_move_file_missing_source_returns_error(ws):
    import fs_tools
    result = fs_tools.move_file("ghost.txt", "dest.txt")
    assert "error" in result


def test_move_file_sandbox_escape_blocked(ws):
    import fs_tools
    (ws / "src.txt").write_text("data")
    result = fs_tools.move_file("src.txt", "../../escape.txt")
    assert "error" in result


# ── fs_tools: delete_file ─────────────────────────────────────────────────────

def test_delete_file_without_confirm_returns_confirmation_request(ws):
    import fs_tools
    (ws / "target.txt").write_text("data")
    result = fs_tools.delete_file("target.txt")
    assert result.get("requires_confirmation") is True
    assert (ws / "target.txt").exists()  # not deleted yet


def test_delete_file_with_confirm_deletes(ws):
    import fs_tools
    (ws / "target.txt").write_text("data")
    result = fs_tools.delete_file("target.txt", confirm=True)
    assert "deleted" in result
    assert not (ws / "target.txt").exists()


def test_delete_file_missing_returns_error(ws):
    import fs_tools
    result = fs_tools.delete_file("ghost.txt", confirm=True)
    assert "error" in result


# ── shell_tools: run_shell ────────────────────────────────────────────────────

def test_run_shell_basic_command(ws):
    import shell_tools
    result = shell_tools.run_shell("echo hello")
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


def test_run_shell_cwd_is_within_workspace(ws):
    import shell_tools
    result = shell_tools.run_shell("pwd")
    assert str(ws) in result["stdout"]


def test_run_shell_non_zero_exit_code(ws):
    import shell_tools
    result = shell_tools.run_shell("exit 1", cwd=".")
    assert result["exit_code"] == 1


def test_run_shell_captures_stderr(ws):
    import shell_tools
    result = shell_tools.run_shell("ls /nonexistent_path_xyz_abc 2>&1 || true")
    # Either stderr or stdout should mention the path doesn't exist
    output = result["stdout"] + result["stderr"]
    assert len(output) > 0


def test_run_shell_cwd_sandbox_escape_blocked(ws):
    import shell_tools
    result = shell_tools.run_shell("pwd", cwd="../../..")
    assert "error" in result


def test_run_shell_rm_rf_blocked_without_force(ws):
    import shell_tools
    result = shell_tools.run_shell("rm -rf .")
    assert result.get("requires_confirmation") is True
    assert result.get("matched") == "rm with -r/-f flag"


def test_run_shell_git_push_force_blocked(ws):
    import shell_tools
    result = shell_tools.run_shell("git push origin main --force")
    assert result.get("requires_confirmation") is True


def test_run_shell_git_reset_hard_blocked(ws):
    import shell_tools
    result = shell_tools.run_shell("git reset --hard HEAD")
    assert result.get("requires_confirmation") is True


def test_run_shell_sudo_blocked(ws):
    import shell_tools
    result = shell_tools.run_shell("sudo rm file.txt")
    assert result.get("requires_confirmation") is True


def test_run_shell_force_bypasses_guard(ws):
    import shell_tools
    # Use a safe destructive-looking command that won't actually damage anything
    (ws / "deleteme.txt").write_text("bye")
    result = shell_tools.run_shell("rm -rf deleteme.txt", force=True)
    assert result["exit_code"] == 0
    assert not (ws / "deleteme.txt").exists()


def test_run_shell_timeout(ws):
    import shell_tools
    with patch("shell_tools.SHELL_TIMEOUT", 1):
        result = shell_tools.run_shell("sleep 10")
    assert "error" in result
    assert "Timed out" in result["error"]
