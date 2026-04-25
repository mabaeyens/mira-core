"""Tests for github_tools — all network and subprocess calls are mocked."""

import base64
import json
import pytest
from unittest.mock import MagicMock, patch

import github_tools


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resp(status: int, body):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.text = json.dumps(body) if isinstance(body, (dict, list)) else str(body)
    return r


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


@pytest.fixture(autouse=True)
def mock_token():
    """Always return a fake token from gh CLI."""
    with patch("github_tools.subprocess.run") as m:
        proc = MagicMock()
        proc.stdout = "ghp_faketoken\n"
        m.return_value = proc
        yield m


# ── github_list_repos ─────────────────────────────────────────────────────────

def test_list_repos_returns_names():
    repos = [
        {"full_name": "user/repo-a", "private": False, "description": "", "updated_at": "2025-01-01", "default_branch": "main"},
        {"full_name": "user/repo-b", "private": True,  "description": "desc", "updated_at": "2025-01-02", "default_branch": "main"},
    ]
    with patch("github_tools._gh", return_value=_resp(200, repos)):
        result = github_tools.github_list_repos()
    assert result["count"] == 2
    assert result["repos"][0]["name"] == "user/repo-a"


def test_list_repos_api_error():
    with patch("github_tools._gh", return_value=_resp(401, {"message": "Bad credentials"})):
        result = github_tools.github_list_repos()
    assert "error" in result


# ── github_read_file ──────────────────────────────────────────────────────────

def test_read_file_decodes_base64():
    payload = {"sha": "abc123", "size": 5, "encoding": "base64", "content": _b64("hello")}
    with patch("github_tools._gh", return_value=_resp(200, payload)):
        result = github_tools.github_read_file("user/repo", "README.md")
    assert result["content"] == "hello"
    assert result["sha"] == "abc123"


def test_read_file_not_found():
    with patch("github_tools._gh", return_value=_resp(404, {"message": "Not Found"})):
        result = github_tools.github_read_file("user/repo", "missing.txt")
    assert "error" in result


# ── github_list_files ─────────────────────────────────────────────────────────

def test_list_files_directory():
    entries = [
        {"name": "README.md", "type": "file", "size": 100, "path": "README.md"},
        {"name": "src",       "type": "dir",  "size": None, "path": "src"},
    ]
    with patch("github_tools._gh", return_value=_resp(200, entries)):
        result = github_tools.github_list_files("user/repo")
    assert result["count"] == 2
    assert result["entries"][0]["name"] == "README.md"


def test_list_files_not_found():
    with patch("github_tools._gh", return_value=_resp(404, {})):
        result = github_tools.github_list_files("user/repo", "missing/")
    assert "error" in result


# ── github_write_file ─────────────────────────────────────────────────────────

def test_write_file_creates_new():
    existing_404 = _resp(404, {"message": "Not Found"})
    commit_resp = _resp(201, {
        "commit": {"sha": "newsha"},
        "content": {"html_url": "https://github.com/user/repo/blob/main/new.txt"},
    })
    with patch("github_tools._gh", side_effect=[existing_404, commit_resp]):
        result = github_tools.github_write_file("user/repo", "new.txt", "hello", "add file")
    assert result["action"] == "created"
    assert result["commit_sha"] == "newsha"


def test_write_file_updates_existing():
    existing_200 = _resp(200, {"sha": "oldsha", "size": 5, "encoding": "base64", "content": _b64("old")})
    commit_resp  = _resp(200, {
        "commit": {"sha": "updsha"},
        "content": {"html_url": "https://github.com/user/repo/blob/main/f.txt"},
    })
    with patch("github_tools._gh", side_effect=[existing_200, commit_resp]):
        result = github_tools.github_write_file("user/repo", "f.txt", "new content", "update")
    assert result["action"] == "updated"


def test_write_file_api_error():
    existing_404 = _resp(404, {})
    with patch("github_tools._gh", side_effect=[existing_404, _resp(422, {"message": "Validation Failed"})]):
        result = github_tools.github_write_file("user/repo", "f.txt", "x", "msg")
    assert "error" in result


# ── github_create_repo ────────────────────────────────────────────────────────

def test_create_repo_success():
    payload = {
        "full_name": "user/new-repo", "html_url": "https://github.com/user/new-repo",
        "ssh_url": "git@github.com:user/new-repo.git", "private": True, "default_branch": "main",
    }
    with patch("github_tools._gh", return_value=_resp(201, payload)):
        result = github_tools.github_create_repo("new-repo", private=True)
    assert result["created"] is True
    assert result["full_name"] == "user/new-repo"


def test_create_repo_duplicate_name():
    with patch("github_tools._gh", return_value=_resp(422, {"message": "name already exists"})):
        result = github_tools.github_create_repo("existing-repo")
    assert "error" in result
    assert "already exists" in result["error"]


# ── github_create_branch ──────────────────────────────────────────────────────

def test_create_branch_success():
    repo_resp   = _resp(200, {"default_branch": "main"})
    ref_resp    = _resp(200, {"object": {"sha": "basesha"}})
    create_resp = _resp(201, {"ref": "refs/heads/feature", "object": {"sha": "basesha"}})
    with patch("github_tools._gh", side_effect=[repo_resp, ref_resp, create_resp]):
        result = github_tools.github_create_branch("user/repo", "feature")
    assert result["created"] == "feature"
    assert result["sha"] == "basesha"


def test_create_branch_already_exists():
    repo_resp = _resp(200, {"default_branch": "main"})
    ref_resp  = _resp(200, {"object": {"sha": "basesha"}})
    with patch("github_tools._gh", side_effect=[repo_resp, ref_resp, _resp(422, {})]):
        result = github_tools.github_create_branch("user/repo", "existing-branch")
    assert "error" in result


# ── github_list_issues / create_issue ─────────────────────────────────────────

def test_list_issues_excludes_prs():
    items = [
        {"number": 1, "title": "Bug", "state": "open", "html_url": "https://github.com/..."},
        {"number": 2, "title": "PR",  "state": "open", "html_url": "https://github.com/...", "pull_request": {}},
    ]
    with patch("github_tools._gh", return_value=_resp(200, items)):
        result = github_tools.github_list_issues("user/repo")
    assert result["count"] == 1
    assert result["issues"][0]["number"] == 1


def test_create_issue_success():
    payload = {"number": 42, "html_url": "https://github.com/.../42", "title": "My issue"}
    with patch("github_tools._gh", return_value=_resp(201, payload)):
        result = github_tools.github_create_issue("user/repo", "My issue", "body text")
    assert result["number"] == 42


# ── github_list_prs ───────────────────────────────────────────────────────────

def test_list_prs():
    prs = [{"number": 5, "title": "Fix", "state": "open", "head": {"ref": "fix"}, "base": {"ref": "main"}, "html_url": "..."}]
    with patch("github_tools._gh", return_value=_resp(200, prs)):
        result = github_tools.github_list_prs("user/repo")
    assert result["count"] == 1
    assert result["prs"][0]["head"] == "fix"


# ── github_search_code ────────────────────────────────────────────────────────

def test_search_code():
    body = {
        "total_count": 1,
        "items": [{"repository": {"full_name": "user/repo"}, "path": "src/main.py", "html_url": "..."}],
    }
    with patch("github_tools._gh", return_value=_resp(200, body)):
        result = github_tools.github_search_code("def hello", repo="user/repo")
    assert result["total"] == 1
    assert result["items"][0]["path"] == "src/main.py"


# ── github_delete_file (destructive guard) ────────────────────────────────────

def test_delete_file_without_confirm_returns_confirmation_request():
    result = github_tools.github_delete_file("user/repo", "old.txt", "remove it")
    assert result.get("requires_confirmation") is True
    assert "confirm=true" in result["message"]


def test_delete_file_with_confirm_deletes():
    existing = _resp(200, {"sha": "filsha", "size": 3, "encoding": "base64", "content": _b64("hi")})
    delete   = _resp(200, {"commit": {"sha": "delsha"}})
    with patch("github_tools._gh", side_effect=[existing, delete]):
        result = github_tools.github_delete_file("user/repo", "old.txt", "remove it", confirm=True)
    assert result["deleted"] == "old.txt"
    assert result["commit_sha"] == "delsha"


# ── github_delete_branch (destructive guard) ──────────────────────────────────

def test_delete_branch_without_confirm_returns_confirmation_request():
    result = github_tools.github_delete_branch("user/repo", "old-branch")
    assert result.get("requires_confirmation") is True


def test_delete_branch_with_confirm_deletes():
    with patch("github_tools._gh", return_value=_resp(204, {})):
        result = github_tools.github_delete_branch("user/repo", "old-branch", confirm=True)
    assert result["deleted"] == "old-branch"


def test_delete_branch_api_error():
    with patch("github_tools._gh", return_value=_resp(422, {"message": "branch not found"})):
        result = github_tools.github_delete_branch("user/repo", "ghost", confirm=True)
    assert "error" in result
