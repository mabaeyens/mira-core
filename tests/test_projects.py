"""Tests for project endpoints — deletion is gated on the backing folder being gone."""
import shutil

import pytest
from fastapi.testclient import TestClient

import server
from core import db


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


def _create_project(client, local_path) -> str:
    resp = client.post("/projects", json={"name": f"test-del-{local_path.name}",
                                          "local_path": str(local_path)})
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def test_delete_refused_while_folder_exists(client, tmp_path):
    """A project whose local_path still exists cannot be deleted (409)."""
    proj_dir = tmp_path / "live_repo"
    proj_dir.mkdir()
    pid = _create_project(client, proj_dir)
    try:
        resp = client.delete(f"/projects/{pid}")
        assert resp.status_code == 409
        assert "still exist" in resp.json()["detail"].lower()
        # The entry must survive the refused delete.
        assert db.get_project(pid) is not None
    finally:
        db.delete_project(pid)  # bypass the guard for test cleanup


def test_delete_allowed_after_folder_removed(client, tmp_path):
    """Once the local folder is gone, the entry can be removed (200)."""
    proj_dir = tmp_path / "gone_repo"
    proj_dir.mkdir()
    pid = _create_project(client, proj_dir)
    cleanup = True
    try:
        shutil.rmtree(proj_dir)  # user manually deletes the folder
        resp = client.delete(f"/projects/{pid}")
        assert resp.status_code == 200
        assert db.get_project(pid) is None
        cleanup = False
    finally:
        if cleanup:
            db.delete_project(pid)


def test_delete_missing_project_returns_404(client):
    resp = client.delete("/projects/does-not-exist-xyz")
    assert resp.status_code == 404
