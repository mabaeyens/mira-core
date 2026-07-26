"""Tests for workspace sandbox enforcement."""

import ast
import pytest
from pathlib import Path
from unittest.mock import patch

import core.workspace as workspace


def test_workspace_root_has_one_binding():
    """No module outside config.py may import WORKSPACE_ROOT by value.

    `from .config import WORKSPACE_ROOT` copies the string into the importing
    module's namespace at import time, so each importer ends up with a private
    sandbox root. Patching one then leaves the others pointing at the real
    workspace, which is exactly how the run_shell tests spent ~3 months
    executing against ~/workspace, `rm -rf .` included, while appearing
    sandboxed. Use `from . import config` and read `config.WORKSPACE_ROOT` so
    there is a single patchable binding.
    """
    core_dir = Path(__file__).resolve().parent.parent / "core"
    offenders = []
    for py in sorted(core_dir.rglob("*.py")):
        if py.name == "config.py":
            continue
        for node in ast.walk(ast.parse(py.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not (node.module or "").endswith("config"):
                continue
            if any(a.name == "WORKSPACE_ROOT" for a in node.names):
                offenders.append(f"{py.relative_to(core_dir)}:{node.lineno}")

    assert not offenders, (
        "import WORKSPACE_ROOT through the config module, not by value, so a "
        f"single patch covers every consumer. Offenders: {offenders}"
    )


@pytest.fixture(autouse=True)
def patch_workspace_root(tmp_path):
    """Use a real tmp directory so resolve() works correctly on macOS.

    Patches core.config, the single binding every consumer reads through.
    """
    with patch("core.config.WORKSPACE_ROOT", str(tmp_path)):
        yield tmp_path


def test_safe_path_relative_resolves_inside_root(patch_workspace_root):
    p = workspace.safe_path("subdir/file.txt")
    assert p == (patch_workspace_root / "subdir" / "file.txt").resolve()


def test_safe_path_dot_is_root(patch_workspace_root):
    p = workspace.safe_path(".")
    assert p == patch_workspace_root.resolve()


def test_safe_path_rejects_parent_traversal():
    with pytest.raises(ValueError, match="outside the workspace"):
        workspace.safe_path("../../etc/passwd")


def test_safe_path_rejects_absolute_escape():
    with pytest.raises(ValueError, match="outside the workspace"):
        workspace.safe_path("/etc/passwd")


def test_safe_path_rejects_deep_traversal():
    with pytest.raises(ValueError, match="outside the workspace"):
        workspace.safe_path("a/b/c/../../../../../../../../etc/hosts")


def test_safe_path_allows_nested_subdir(patch_workspace_root):
    p = workspace.safe_path("a/b/c/d.py")
    assert str(p).startswith(str(patch_workspace_root.resolve()))


def test_rel_strips_root_prefix(patch_workspace_root):
    p = patch_workspace_root / "src" / "main.py"
    assert workspace.rel(p) == "src/main.py"


def test_rel_returns_root_as_dot(patch_workspace_root):
    assert workspace.rel(patch_workspace_root) == "."
