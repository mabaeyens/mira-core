"""Tests for the shared-secret auth middleware (server._auth_middleware):
source-IP allowlist (gate 1) + constant-time Bearer token (gate 2)."""
import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture(scope="module")
def client():
    # Module-scoped: see note in test_cancel.py — one lifespan per module keeps
    # DB init / scheduler start to once.
    with TestClient(server.app) as c:
        yield c


@pytest.fixture
def allow_source(monkeypatch):
    """Bypass the source-IP gate so token tests can exercise gate 2. The TestClient's
    default peer is "testclient" (not an IP), which the real allowlist rejects."""
    monkeypatch.setattr(server, "_source_allowed", lambda host: True)


# ── Gate 1: source-IP allowlist ────────────────────────────────────────────────

def test_source_allowed_unit():
    assert server._source_allowed("127.0.0.1")      # loopback
    assert server._source_allowed("100.64.0.1")     # tailnet CGNAT
    assert server._source_allowed("100.127.255.254")
    assert not server._source_allowed("192.168.0.5")  # plain LAN
    assert not server._source_allowed("8.8.8.8")
    assert not server._source_allowed("testclient")   # non-IP
    assert not server._source_allowed(None)


def test_offlist_source_rejected(client, monkeypatch):
    """With a token set, a peer outside the allowlist gets 403 — before any token
    check, and even on otherwise-open paths."""
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    monkeypatch.setattr(server, "_source_allowed", lambda host: False)
    assert client.get("/status").status_code == 403
    assert client.get("/health").status_code == 403          # open path still gated
    assert client.options("/status").status_code == 403       # OPTIONS gated too


# ── Gate 2: Bearer token ────────────────────────────────────────────────────────

def test_no_token_leaves_routes_open(client, monkeypatch):
    """With AUTH_TOKEN unset, both gates are inert (loopback-only mode)."""
    monkeypatch.setattr(server, "AUTH_TOKEN", "")
    resp = client.get("/status")
    assert resp.status_code == 200


def test_token_required_rejects_missing_header(client, monkeypatch, allow_source):
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    resp = client.get("/status")
    assert resp.status_code == 401


def test_token_rejects_wrong_value(client, monkeypatch, allow_source):
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    resp = client.get("/status", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_token_allows_correct_header(client, monkeypatch, allow_source):
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    resp = client.get("/status", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200


def test_health_open_even_with_token(client, monkeypatch, allow_source):
    """/health stays reachable pre-auth for an allowed peer so clients can poll."""
    monkeypatch.setattr(server, "AUTH_TOKEN", "s3cret")
    resp = client.get("/health")
    assert resp.status_code != 401
