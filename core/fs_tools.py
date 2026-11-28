"""Filesystem tool implementations — all paths sandboxed to the active workspace root."""

import itertools
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

_MAX_LIST_ENTRIES = 2_000

# Directories never worth searching/walking — dependency & VCS noise. Dot-dirs
# (.venv, .git, …) are handled separately via the leading-"." check.
_EXCLUDED_DIRS = {"__pycache__", "node_modules"}

from .workspace import safe_path, rel
from .approvals import approval_token


# A whole large file in one result is re-prefilled on every later tool step (hybrid
# models cannot reuse the conversation cache), so a single read of a 3000-line
# module cost two minutes of prefill. Reads return a window; the model asks for more.
READ_MAX_LINES = 400
READ_MAX_CHARS = 24_000


def read_file(path: str, root: Optional[str] = None,
              start_line: int = 1, max_lines: int = READ_MAX_LINES) -> Dict[str, Any]:
    try:
        p = safe_path(path, root)
    except ValueError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"File not found: {path}"}
    if not p.is_file():
        return {"error": f"Not a file: {path}"}
    text = p.read_text(encoding="utf-8", errors="replace")
    try:
        start = max(1, int(start_line))
        n = max(1, min(int(max_lines), READ_MAX_LINES))
    except (TypeError, ValueError):
        return {"error": "start_line and max_lines must be integers"}
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if start == 1 and total <= n and len(text) <= READ_MAX_CHARS:
        return {"path": rel(p, root), "content": text, "size": len(text)}
    chunk = "".join(lines[start - 1:start - 1 + n])
    if len(chunk) > READ_MAX_CHARS:
        chunk = chunk[:READ_MAX_CHARS]
    shown = chunk.count("\n") + (0 if chunk.endswith("\n") or not chunk else 1)
    end = start - 1 + shown
    out = {"path": rel(p, root), "content": chunk, "size": len(text),
           "start_line": start, "end_line": end, "total_lines": total}
    if end < total:
        out["truncated"] = True
        out["note"] = (f"Showing lines {start}-{end} of {total}. Use search_files to find "
                       f"what you need, or call read_file again with start_line={end + 1}.")
    return out


def write_file(path: str, content: str, root: Optional[str] = None) -> Dict[str, Any]:
    try:
        p = safe_path(path, root)
    except ValueError as e:
        return {"error": str(e)}
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    return {
        "path": rel(p, root),
        "bytes_written": len(content.encode()),
        "action": "updated" if existed else "created",
    }


def list_files(path: str = ".", recursive: bool = False, root: Optional[str] = None) -> Dict[str, Any]:
    try:
        p = safe_path(path, root)
    except ValueError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"Path not found: {path}"}
    if not p.is_dir():
        return {"error": f"Not a directory: {path}"}
    raw = p.rglob("*") if recursive else p.iterdir()
    # Cap before sorting to avoid materialising huge trees (node_modules, build artefacts).
    capped = list(itertools.islice(raw, _MAX_LIST_ENTRIES + 1))
    truncated = len(capped) > _MAX_LIST_ENTRIES
    entries = []
    for item in sorted(capped[:_MAX_LIST_ENTRIES]):
        if any(part.startswith(".") for part in item.parts[-3:]):
            continue
        entries.append({
            "path": rel(item, root),
            "type": "dir" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None,
        })
    return {"path": rel(p, root), "entries": entries, "count": len(entries), "truncated": truncated}


def search_files(pattern: str, path: str = ".", case_sensitive: bool = False, root: Optional[str] = None) -> Dict[str, Any]:
    try:
        p = safe_path(path, root)
    except ValueError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"Path not found: {path}"}
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {"error": f"Invalid pattern: {e}"}
    matches = []
    # A path that names one file searches that file. rglob on a file yields nothing, which
    # used to return an empty result indistinguishable from "no match" and sent the model
    # into a long loop of re-searching a file that did contain the pattern.
    candidates = [p] if p.is_file() else sorted(p.rglob("*"))
    for file_path in candidates:
        if not file_path.is_file():
            continue
        # Skip dependency/VCS dirs (.venv, .git, __pycache__, node_modules) so repo
        # searches return signal, not thousands of vendored matches that blow the cap.
        # An explicitly named file is searched even if it sits under such a dir.
        if file_path != p:
            rel_parts = file_path.relative_to(p).parts
            if any(part.startswith(".") or part in _EXCLUDED_DIRS for part in rel_parts):
                continue
        try:
            for i, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    matches.append({"file": rel(file_path, root), "line": i, "content": line.strip()[:200]})
                    if len(matches) >= 200:
                        return {"matches": matches, "count": len(matches), "truncated": True}
        except Exception:
            continue
    return {"matches": matches, "count": len(matches), "truncated": False}


def move_file(src: str, dst: str, root: Optional[str] = None) -> Dict[str, Any]:
    try:
        s = safe_path(src, root)
        d = safe_path(dst, root)
    except ValueError as e:
        return {"error": str(e)}
    if not s.exists():
        return {"error": f"Source not found: {src}"}
    d.parent.mkdir(parents=True, exist_ok=True)
    s.rename(d)
    return {"from": rel(s, root), "to": rel(d, root)}


def edit_file(path: str, old_str: str, new_str: str, root: Optional[str] = None) -> Dict[str, Any]:
    try:
        p = safe_path(path, root)
    except ValueError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"File not found: {path}"}
    if not p.is_file():
        return {"error": f"Not a file: {path}"}
    content = p.read_text(encoding="utf-8", errors="replace")
    count = content.count(old_str)
    if count == 0:
        return {"error": "old_str not found in file — no changes made"}
    if count > 1:
        return {"error": f"old_str matches {count} locations — make it more specific so the edit is unambiguous"}
    updated = content.replace(old_str, new_str, 1)
    p.write_text(updated, encoding="utf-8")
    line_no = content[: content.index(old_str)].count("\n") + 1
    return {"path": rel(p, root), "line": line_no, "action": "edited"}


def delete_file(path: str, confirm: bool = False, root: Optional[str] = None) -> Dict[str, Any]:
    try:
        p = safe_path(path, root)
    except ValueError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"Not found: {path}"}
    r = rel(p, root)
    if not confirm:
        kind = "directory" if p.is_dir() else "file"
        return {
            "requires_confirmation": True,
            "action": "delete_file",
            "path": r,
            # Token binds to the path the model actually asked for, so approving
            # one deletion does not approve a different one.
            "approval_token": approval_token("delete_file", path),
            "message": (
                f"This will permanently delete {kind} '{r}'. "
                "Relay this to the user for approval. You cannot approve it "
                "yourself — do not retry this deletion."
            ),
        }
    if p.is_dir():
        shutil.rmtree(p)
        return {"deleted": r, "type": "directory"}
    p.unlink()
    return {"deleted": r, "type": "file"}
