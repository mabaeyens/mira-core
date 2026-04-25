"""Tests for workspace sandbox enforcement."""

import pytest
from pathlib import Path
from unittest.mock import patch

import workspace


@pytest.fixture(autouse=True)
def patch_workspace_root(tmp_path):
    """Use a real tmp directory so resolve() works correctly on macOS."""
    with patch("workspace.WORKSPACE_ROOT", str(tmp_path)):
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
