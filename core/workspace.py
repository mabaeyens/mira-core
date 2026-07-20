"""Sandbox path enforcement — all filesystem operations go through here."""

from pathlib import Path
from typing import Optional
from .config import WORKSPACE_ROOT


def safe_path(user_path: str, root: Optional[str] = None) -> Path:
    """Resolve path and verify it sits within root (defaults to WORKSPACE_ROOT)."""
    r = Path(root or WORKSPACE_ROOT).expanduser().resolve()
    resolved = (r / user_path).resolve()
    if not str(resolved).startswith(str(r) + "/") and resolved != r:
        raise ValueError(f"Path '{user_path}' is outside the workspace ({r})")
    return resolved


def safe_filename(name: Optional[str], fallback: str = "attachment") -> str:
    """Reduce an untrusted filename to a bare, writable basename.

    Upload filenames come straight from the Content-Disposition header and are
    never sanitized by Starlette. They are joined to a directory before writing,
    and `Path(dir) / "/etc/passwd"` discards the directory entirely — so an
    absolute name escapes the workspace with no traversal sequence at all.
    Taking `.name` strips both directory components and any absolute prefix.
    """
    base = Path(name or "").name.replace("\x00", "").strip()
    if not base or base in (".", ".."):
        return fallback
    return base


def rel(path: Path, root: Optional[str] = None) -> str:
    """Return a path relative to root (defaults to WORKSPACE_ROOT) as a string."""
    r = Path(root or WORKSPACE_ROOT).expanduser().resolve()
    try:
        return str(path.resolve().relative_to(r))
    except ValueError:
        return str(path)
