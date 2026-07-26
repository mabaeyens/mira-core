"""PATCH /conversations/{id} does two things now, and the second one has a trap.

`project_id` has three meanings, not two: absent (leave it alone), null (unfile
it) and a string (move it). A plain `Optional[str] = None` collapses the first
two, so an app sending only a title would silently unfile the conversation.
`model_fields_set` is what keeps them apart, and these tests are what keep that
from being refactored away.
"""
import pytest
from fastapi.testclient import TestClient

import server
from core import db


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app, base_url="http://localhost") as c:
        yield c


@pytest.fixture
def conv():
    conv_id = db.create_conversation("test-model")
    yield conv_id
    db.delete_conversation(conv_id)


@pytest.fixture
def project():
    pid = db.create_project("test-patch-project", None, None)
    yield pid
    db.delete_project(pid)


def _project_of(conv_id):
    return db.get_conversation(conv_id)["project_id"]


# ── title, unchanged behaviour ────────────────────────────────────────────────

def test_title_only_still_works(client, conv):
    assert client.patch(f"/conversations/{conv}", json={"title": "Renamed"}).status_code == 200
    assert db.get_conversation(conv)["title"] == "Renamed"


def test_blank_title_is_refused(client, conv):
    assert client.patch(f"/conversations/{conv}", json={"title": "   "}).status_code == 400


def test_title_only_does_not_touch_the_project(client, conv, project):
    """The trap. A rename must not unfile the conversation."""
    client.patch(f"/conversations/{conv}", json={"project_id": project})
    assert _project_of(conv) == project

    client.patch(f"/conversations/{conv}", json={"title": "Just a rename"})
    assert _project_of(conv) == project


# ── project assignment ────────────────────────────────────────────────────────

def test_assigns_a_project(client, conv, project):
    assert client.patch(f"/conversations/{conv}", json={"project_id": project}).status_code == 200
    assert _project_of(conv) == project


def test_explicit_null_unfiles_it(client, conv, project):
    client.patch(f"/conversations/{conv}", json={"project_id": project})
    assert client.patch(f"/conversations/{conv}", json={"project_id": None}).status_code == 200
    assert _project_of(conv) is None


def test_empty_string_unfiles_it_too(client, conv, project):
    client.patch(f"/conversations/{conv}", json={"project_id": project})
    client.patch(f"/conversations/{conv}", json={"project_id": ""})
    assert _project_of(conv) is None


def test_unknown_project_is_refused(client, conv):
    resp = client.patch(f"/conversations/{conv}", json={"project_id": "no-such-project"})
    assert resp.status_code == 400
    assert _project_of(conv) is None


def test_title_and_project_in_one_call(client, conv, project):
    resp = client.patch(f"/conversations/{conv}",
                        json={"title": "Both", "project_id": project})
    assert resp.status_code == 200
    row = db.get_conversation(conv)
    assert (row["title"], row["project_id"]) == ("Both", project)


# ── guards ────────────────────────────────────────────────────────────────────

def test_empty_body_is_refused(client, conv):
    assert client.patch(f"/conversations/{conv}", json={}).status_code == 400


def test_unknown_conversation_is_404(client, project):
    resp = client.patch("/conversations/does-not-exist-xyz", json={"title": "x"})
    assert resp.status_code == 404


def test_deleting_a_project_unfiles_its_conversations_without_deleting_them(client, conv):
    """Otherwise the row keeps pointing at a project that is gone."""
    pid = db.create_project("test-dangling", None, None)
    client.patch(f"/conversations/{conv}", json={"project_id": pid})
    assert _project_of(conv) == pid

    db.delete_project(pid)

    assert db.get_conversation(conv) is not None      # conversation survives
    assert _project_of(conv) is None                  # but is no longer filed
