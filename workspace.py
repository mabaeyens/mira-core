"""Sandbox path enforcement — all filesystem operations go through here."""

from pathlib import Path
from config import WORKSPACE_ROOT


def safe_path(user_path: str) -> Path:
    """Resolve path and verify it sits within WORKSPACE_ROOT. Raises ValueError if not."""
    root = Path(WORKSPACE_ROOT).expanduser().resolve()
    resolved = (root / user_path).resolve()
    if not str(resolved).startswith(str(root) + "/") and resolved != root:
        raise ValueError(f"Path '{user_path}' is outside the workspace ({root})")
    return resolved


def rel(path: Path) -> str:
    """Return a path relative to WORKSPACE_ROOT as a string."""
    root = Path(WORKSPACE_ROOT).expanduser().resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
