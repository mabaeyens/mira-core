"""GitHub REST API tools — token sourced from gh CLI keyring (cached per session)."""

import base64
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

_cached_token: Optional[str] = None
_REPO_RE = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')


def _validate_repo(repo: str) -> Optional[Dict[str, Any]]:
    """Return an error dict if repo is not in owner/repo format, else None."""
    if not _REPO_RE.match(repo):
        return {"error": f"Invalid repo format '{repo}' — expected 'owner/repo'"}
    return None


def _token() -> str:
    global _cached_token
    if _cached_token:
        return _cached_token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("gh auth token returned empty — run 'gh auth login' first")
    _cached_token = result.stdout.strip()
    return _cached_token


def _gh(method: str, path: str, **kwargs) -> httpx.Response:
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=30) as client:
        return client.request(method, url, headers=headers, **kwargs)


# ── Clone ────────────────────────────────────────────────────────────────────

def github_clone_repo(repo: str, dest: str = "") -> Dict[str, Any]:
    """Clone a GitHub repo with gh CLI (SSH). Returns the local path on success."""
    if err := _validate_repo(repo):
        return err
    from .config import WORKSPACE_ROOT
    repo_name = repo.split("/")[-1]
    clone_path = Path(dest).expanduser() if dest else Path(WORKSPACE_ROOT).expanduser() / repo_name
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    if clone_path.exists():
        return {"error": f"Destination already exists: {clone_path}"}
    result = subprocess.run(
        ["gh", "repo", "clone", repo, str(clone_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "Clone failed"}
    return {"cloned_to": str(clone_path), "repo": repo}


# ── Read operations ──────────────────────────────────────────────────────────

def github_list_repos(repo_type: str = "owner") -> Dict[str, Any]:
    resp = _gh("GET", "/user/repos", params={"type": repo_type, "per_page": 50, "sort": "updated"})
    if resp.status_code != 200:
        return {"error": resp.text}
    data = resp.json()
    return {
        "repos": [
            {
                "name": r["full_name"],
                "private": r["private"],
                "description": r.get("description") or "",
                "updated_at": r["updated_at"],
                "default_branch": r["default_branch"],
            }
            for r in data
        ],
        "count": len(data),
    }


def github_read_file(repo: str, path: str, ref: str = "") -> Dict[str, Any]:
    if err := _validate_repo(repo):
        return err
    params = {"ref": ref} if ref else {}
    resp = _gh("GET", f"/repos/{repo}/contents/{path}", params=params)
    if resp.status_code == 404:
        return {"error": f"Not found: {repo}/{path}"}
    if resp.status_code != 200:
        return {"error": resp.text}
    data = resp.json()
    if data.get("encoding") == "base64":
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    else:
        content = data.get("content", "")
    return {"repo": repo, "path": path, "sha": data["sha"], "size": data["size"], "content": content}


def github_list_files(repo: str, path: str = "", ref: str = "") -> Dict[str, Any]:
    if err := _validate_repo(repo):
        return err
    params = {"ref": ref} if ref else {}
    url_path = f"/repos/{repo}/contents/{path}" if path else f"/repos/{repo}/contents"
    resp = _gh("GET", url_path, params=params)
    if resp.status_code == 404:
        return {"error": f"Not found: {repo}/{path or ''}"}
    if resp.status_code != 200:
        return {"error": resp.text}
    data = resp.json()
    if not isinstance(data, list):
        return {"repo": repo, "path": path, "type": data.get("type"), "sha": data.get("sha")}
    entries = [{"name": e["name"], "type": e["type"], "size": e.get("size"), "path": e["path"]} for e in data]
    return {"repo": repo, "path": path, "entries": entries, "count": len(entries)}


def github_list_issues(repo: str, state: str = "open") -> Dict[str, Any]:
    if err := _validate_repo(repo):
        return err
    resp = _gh("GET", f"/repos/{repo}/issues", params={"state": state, "per_page": 30})
    if resp.status_code != 200:
        return {"error": resp.text}
    issues = [i for i in resp.json() if "pull_request" not in i]
    return {
        "issues": [
            {"number": i["number"], "title": i["title"], "state": i["state"], "url": i["html_url"]}
            for i in issues
        ],
        "count": len(issues),
    }


def github_list_prs(repo: str, state: str = "open") -> Dict[str, Any]:
    if err := _validate_repo(repo):
        return err
    resp = _gh("GET", f"/repos/{repo}/pulls", params={"state": state, "per_page": 30})
    if resp.status_code != 200:
        return {"error": resp.text}
    prs = resp.json()
    return {
        "prs": [
            {
                "number": pr["number"],
                "title": pr["title"],
                "state": pr["state"],
                "head": pr["head"]["ref"],
                "base": pr["base"]["ref"],
                "url": pr["html_url"],
            }
            for pr in prs
        ],
        "count": len(prs),
    }


def github_search_code(query: str, repo: str = "") -> Dict[str, Any]:
    if repo and (err := _validate_repo(repo)):
        return err
    q = f"{query} repo:{repo}" if repo else query
    resp = _gh("GET", "/search/code", params={"q": q, "per_page": 20})
    if resp.status_code != 200:
        return {"error": resp.text}
    data = resp.json()
    return {
        "total": data["total_count"],
        "items": [
            {"repo": i["repository"]["full_name"], "path": i["path"], "url": i["html_url"]}
            for i in data["items"][:20]
        ],
    }


# ── Write operations ─────────────────────────────────────────────────────────

def github_write_file(
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str = "",
    sha: str = "",
) -> Dict[str, Any]:
    """Create or update a file. sha is auto-fetched if the file already exists."""
    if err := _validate_repo(repo):
        return err
    body: Dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
    }
    if branch:
        body["branch"] = branch
    # Auto-fetch sha for updates
    if not sha:
        existing = github_read_file(repo, path, ref=branch)
        if "sha" in existing:
            body["sha"] = existing["sha"]
    else:
        body["sha"] = sha

    resp = _gh("PUT", f"/repos/{repo}/contents/{path}", json=body)
    if resp.status_code not in (200, 201):
        return {"error": resp.text}
    data = resp.json()
    return {
        "action": "updated" if resp.status_code == 200 else "created",
        "repo": repo,
        "path": path,
        "commit_sha": data["commit"]["sha"],
        "url": data["content"]["html_url"],
    }


def github_create_repo(
    name: str,
    private: bool = True,
    description: str = "",
    auto_init: bool = True,
) -> Dict[str, Any]:
    resp = _gh("POST", "/user/repos", json={
        "name": name,
        "private": private,
        "description": description,
        "auto_init": auto_init,
    })
    if resp.status_code == 422:
        return {"error": "Repository name already exists or is invalid"}
    if resp.status_code != 201:
        return {"error": resp.text}
    data = resp.json()
    return {
        "created": True,
        "full_name": data["full_name"],
        "url": data["html_url"],
        "clone_url": data["ssh_url"],
        "private": data["private"],
        "default_branch": data["default_branch"],
    }


def github_create_issue(repo: str, title: str, body: str = "") -> Dict[str, Any]:
    if err := _validate_repo(repo):
        return err
    resp = _gh("POST", f"/repos/{repo}/issues", json={"title": title, "body": body})
    if resp.status_code != 201:
        return {"error": resp.text}
    data = resp.json()
    return {"number": data["number"], "url": data["html_url"], "title": data["title"]}


def github_create_branch(repo: str, branch: str, from_ref: str = "") -> Dict[str, Any]:
    if err := _validate_repo(repo):
        return err
    if not from_ref:
        r = _gh("GET", f"/repos/{repo}")
        if r.status_code != 200:
            return {"error": r.text}
        from_ref = r.json()["default_branch"]
    ref_resp = _gh("GET", f"/repos/{repo}/git/refs/heads/{from_ref}")
    if ref_resp.status_code != 200:
        return {"error": f"Source ref '{from_ref}' not found"}
    sha = ref_resp.json()["object"]["sha"]
    resp = _gh("POST", f"/repos/{repo}/git/refs", json={"ref": f"refs/heads/{branch}", "sha": sha})
    if resp.status_code == 422:
        return {"error": f"Branch '{branch}' already exists"}
    if resp.status_code != 201:
        return {"error": resp.text}
    return {"created": branch, "repo": repo, "from": from_ref, "sha": sha}


# ── PR operations ────────────────────────────────────────────────────────────

def github_create_pr(
    repo: str,
    title: str,
    body: str = "",
    head: str = "",
    base: str = "",
) -> Dict[str, Any]:
    """Open a pull request. head is the branch with changes; base is the target (default branch if omitted)."""
    if err := _validate_repo(repo):
        return err
    if not base:
        r = _gh("GET", f"/repos/{repo}")
        if r.status_code != 200:
            return {"error": r.text}
        base = r.json()["default_branch"]
    if not head:
        return {"error": "head branch is required"}
    resp = _gh("POST", f"/repos/{repo}/pulls", json={
        "title": title,
        "body": body,
        "head": head,
        "base": base,
    })
    if resp.status_code == 422:
        return {"error": resp.json().get("message", resp.text)}
    if resp.status_code != 201:
        return {"error": resp.text}
    data = resp.json()
    return {
        "number": data["number"],
        "title": data["title"],
        "url": data["html_url"],
        "head": data["head"]["ref"],
        "base": data["base"]["ref"],
        "state": data["state"],
    }


def github_merge_pr(
    repo: str,
    pr_number: int,
    merge_method: str = "merge",
    confirm: bool = False,
) -> Dict[str, Any]:
    """Merge a pull request. merge_method: merge | squash | rebase."""
    if err := _validate_repo(repo):
        return err
    if not confirm:
        return {
            "requires_confirmation": True,
            "action": "github_merge_pr",
            "repo": repo,
            "pr_number": pr_number,
            "message": (
                f"This will merge PR #{pr_number} in {repo} using '{merge_method}'. "
                "Ask the user to confirm, then call github_merge_pr with confirm=true."
            ),
        }
    resp = _gh("PUT", f"/repos/{repo}/pulls/{pr_number}/merge", json={"merge_method": merge_method})
    if resp.status_code == 405:
        return {"error": "PR is not mergeable (conflicts or already merged)"}
    if resp.status_code == 404:
        return {"error": f"PR #{pr_number} not found in {repo}"}
    if resp.status_code not in (200, 204):
        return {"error": resp.text}
    data = resp.json()
    return {
        "merged": True,
        "sha": data.get("sha"),
        "message": data.get("message"),
    }


# ── Destructive operations (require confirm=True) ────────────────────────────

def github_delete_file(
    repo: str,
    path: str,
    message: str,
    branch: str = "",
    confirm: bool = False,
) -> Dict[str, Any]:
    if err := _validate_repo(repo):
        return err
    if not confirm:
        return {
            "requires_confirmation": True,
            "action": "github_delete_file",
            "repo": repo,
            "path": path,
            "message": (
                f"This will permanently delete '{path}' from {repo}. "
                "Ask the user to confirm, then call github_delete_file with confirm=true."
            ),
        }
    existing = github_read_file(repo, path, ref=branch)
    if "error" in existing:
        return existing
    body: Dict[str, Any] = {"message": message, "sha": existing["sha"]}
    if branch:
        body["branch"] = branch
    resp = _gh("DELETE", f"/repos/{repo}/contents/{path}", json=body)
    if resp.status_code != 200:
        return {"error": resp.text}
    return {"deleted": path, "repo": repo, "commit_sha": resp.json()["commit"]["sha"]}


def github_delete_branch(repo: str, branch: str, confirm: bool = False) -> Dict[str, Any]:
    if err := _validate_repo(repo):
        return err
    if not confirm:
        return {
            "requires_confirmation": True,
            "action": "github_delete_branch",
            "repo": repo,
            "branch": branch,
            "message": (
                f"This will delete branch '{branch}' from {repo}. "
                "Ask the user to confirm, then call github_delete_branch with confirm=true."
            ),
        }
    resp = _gh("DELETE", f"/repos/{repo}/git/refs/heads/{branch}")
    if resp.status_code == 204:
        return {"deleted": branch, "repo": repo}
    return {"error": resp.text}
