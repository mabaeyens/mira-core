"""Filesystem tool implementations — all paths sandboxed to the active workspace root."""

import itertools
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

_MAX_LIST_ENTRIES = 2_000

from .workspace import safe_path, rel


def read_file(path: str, root: Optional[str] = None) -> Dict[str, Any]:
    try:
        p = safe_path(path, root)
    except ValueError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"File not found: {path}"}
    if not p.is_file():
        return {"error": f"Not a file: {path}"}
    content = p.read_text(encoding="utf-8", errors="replace")
    return {"path": rel(p, root), "content": content, "size": len(content)}


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
    for file_path in sorted(p.rglob("*")):
        if not file_path.is_file():
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
            "message": (
                f"This will permanently delete {kind} '{r}'. "
                "Ask the user to confirm, then call delete_file again with confirm=true."
            ),
        }
    if p.is_dir():
        shutil.rmtree(p)
        return {"deleted": r, "type": "directory"}
    p.unlink()
    return {"deleted": r, "type": "file"}
