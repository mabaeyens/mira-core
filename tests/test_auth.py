"""Tests for the shared-secret auth middleware (server._auth_middleware)."""
import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture(scope="module")
def client():
    # Module-scoped: see note in test_cancel.py — one lifespan per module keeps
    # DB init / scheduler start to once.
    with TestClient(server.app) as c:
        yield c


def test_no_token_leaves_routes_open(client, monkeypatch):
    """With AUTH_TOKEN unset, sensitive routes need no Authorization header."""
    monkeypatch.setattr(server, "AUTH_TOKEN", "")
    resp = client.get("/status")
    assert resp.status_code == 200


def test_token_required_rejects_missing_header(client, monkeypatch):
    """With a token set, a request without the header is rejected."""
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    resp = client.get("/status")
    assert resp.status_code == 401


def test_token_rejects_wrong_value(client, monkeypatch):
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    resp = client.get("/status", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_token_allows_correct_header(client, monkeypatch):
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    resp = client.get("/status", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200


def test_health_open_even_with_token(client, monkeypatch):
    """/health must stay reachable pre-auth so clients can poll readiness."""
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    resp = client.get("/health")
    assert resp.status_code != 401
