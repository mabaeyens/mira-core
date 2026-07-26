"""Tests for the stop/cancel feature (server.py + index.html stop button)."""
import json
import shutil
import tempfile
from pathlib import Path

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

import server
from core.orchestrator import ChatOrchestrator


def _parse_sse(response_text):
    """Parse SSE response text into a list of event dicts."""
    events = []
    for line in response_text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except Exception:
                pass
    return events


@pytest.fixture(autouse=True)
def clear_cancel():
    """Reset the active-cancel registry before and after each test."""
    server._active_cancels.clear()
    yield
    server._active_cancels.clear()


@pytest.fixture(scope="module")
def client():
    # Module-scoped: one TestClient (one event loop, one lifespan) per module keeps
    # DB init / scheduler start to once. Locks are now per-conversation and created
    # lazily in the running loop (SessionManager), so they're no longer the reason
    # for module scope — efficiency is.
    with TestClient(server.app, base_url="http://localhost") as c:
        yield c


def _set_all_active_cancels():
    """Set every registered request's cancel event (used by mocked streams that
    can't see their own per-request event handle)."""
    for _cid, ev in list(server._active_cancels.values()):
        ev.set()


def test_cancel_endpoint_scopes_to_conversation(client):
    """POST /cancel with a conversation_id sets only that conversation's event."""
    import threading
    ev_a, ev_b = threading.Event(), threading.Event()
    server._active_cancels["req-a"] = ("conv-a", ev_a)
    server._active_cancels["req-b"] = ("conv-b", ev_b)

    resp = client.post("/cancel", json={"conversation_id": "conv-a"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert ev_a.is_set()
    assert not ev_b.is_set(), "cancel for conv-a must not touch conv-b"


def test_cancel_no_body_cancels_all(client):
    """POST /cancel with no body cancels every in-flight turn (back-compat)."""
    import threading
    ev_a, ev_b = threading.Event(), threading.Event()
    server._active_cancels["req-a"] = ("conv-a", ev_a)
    server._active_cancels["req-b"] = ("conv-b", ev_b)

    resp = client.post("/cancel")
    assert resp.status_code == 200
    assert ev_a.is_set() and ev_b.is_set()


def test_cancel_unknown_conversation_is_noop(client):
    """Cancel for a conversation with no active turn returns ok and sets nothing."""
    import threading
    ev = threading.Event()
    server._active_cancels["req-a"] = ("conv-a", ev)

    resp = client.post("/cancel", json={"conversation_id": "conv-zzz"})
    assert resp.status_code == 200
    assert not ev.is_set()


def test_cancel_stops_event_stream(client):
    """Events emitted after cancel fires must not reach the client."""
    def cancelling_stream(message, attachments=None, **kwargs):
        yield {"type": "token", "content": "partial"}
        _set_all_active_cancels()               # cancel this request mid-stream
        yield {"type": "token", "content": "should_not_arrive"}
        yield {"type": "done", "content": "partial should_not_arrive"}

    # Patch at the class so the pool-built orchestrator for this turn uses the mock.
    with patch.object(ChatOrchestrator, 'stream_chat', side_effect=cancelling_stream):
        resp = client.post("/chat", data={"message": "cancel me"})

    contents = [e.get("content", "") for e in _parse_sse(resp.text)]
    assert not any("should_not_arrive" in c for c in contents), \
        "Events emitted after the request's cancel event was set must be dropped by produce()"


def test_history_rolled_back_on_cancel(client):
    """Partial conversation history from a cancelled turn is removed."""
    conv_id = "test-rollback-conv"

    def cancelling_stream(message, attachments=None, **kwargs):
        # Simulate what stream_chat does — write user message to history, then
        # get cancelled before the assistant turn is appended.
        orch = server.sessions.get(conv_id)
        orch.conversation_history.append({"role": "user", "content": message})
        yield {"type": "thinking"}
        _set_all_active_cancels()               # cancel before done event
        yield {"type": "done", "content": "never delivered"}

    with patch.object(ChatOrchestrator, 'stream_chat', side_effect=cancelling_stream):
        client.post("/chat", data={"message": "cancel me", "conversation_id": conv_id})

    orch = server.sessions.get(conv_id)
    non_system = [m for m in orch.conversation_history if m["role"] != "system"]
    assert non_system == [], \
        "Cancelled turn must not leave user/assistant entries in conversation history"


# ── /browse endpoint tests ────────────────────────────────────────────────────

@pytest.fixture()
def home_dir(tmp_path_factory):
    """A scratch directory inside $HOME.

    /browse resolves through server._safe_path, which 403s anything outside
    Path.home(). pytest's tmp_path lives under /private/var on macOS and /tmp
    on Linux, both outside home, so these tests need a root of their own.
    """
    d = Path(tempfile.mkdtemp(prefix=".mira-browse-test-", dir=Path.home()))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_browse_home_directory(client):
    """/browse on the user's home directory returns a listing."""
    resp = client.get(f"/browse?path={Path.home()}")
    assert resp.status_code == 200
    data = resp.json()
    assert "path" in data
    assert "entries" in data
    assert isinstance(data["entries"], list)


def test_browse_outside_home_is_forbidden(client):
    """The _safe_path guard must reject anything above $HOME."""
    resp = client.get("/browse?path=/etc")
    assert resp.status_code == 403


def test_browse_entries_have_required_fields(client, home_dir):
    """Each entry in /browse response has name, is_dir, ext, path fields."""
    (home_dir / "doc.pdf").write_text("x")
    (home_dir / "subdir").mkdir()

    resp = client.get(f"/browse?path={home_dir}")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 2  # subdir first (sorted), then doc.pdf
    for e in entries:
        assert "name" in e
        assert "is_dir" in e
        assert "ext" in e
        assert "path" in e


def test_browse_dirs_sorted_before_files(client, home_dir):
    """Directories must appear before files in /browse results."""
    (home_dir / "z_file.txt").write_text("x")
    (home_dir / "a_dir").mkdir()

    resp = client.get(f"/browse?path={home_dir}")
    entries = resp.json()["entries"]
    dir_indices  = [i for i, e in enumerate(entries) if e["is_dir"]]
    file_indices = [i for i, e in enumerate(entries) if not e["is_dir"]]
    assert max(dir_indices) < min(file_indices), "All dirs must come before files"


def test_browse_nonexistent_path_returns_error(client):
    """/browse with a non-existent path returns 4xx.

    The path is kept inside $HOME on purpose: outside it the _safe_path guard
    would 403 first and this would stop testing the not-a-directory branch.
    """
    resp = client.get(f"/browse?path={Path.home()}/nonexistent/path/xyz123")
    assert resp.status_code in (400, 404, 500)


def test_browse_ext_field_is_lowercase_with_dot(client, home_dir):
    """ext field must be lowercase with leading dot (e.g. '.pdf') or '' for no extension."""
    (home_dir / "Document.PDF").write_text("x")
    (home_dir / "Makefile").write_text("x")

    resp = client.get(f"/browse?path={home_dir}")
    entries = {e["name"]: e for e in resp.json()["entries"]}
    assert entries["Document.PDF"]["ext"] == ".pdf"
    assert entries["Makefile"]["ext"] == ""


# ── file_handler magic-byte detection tests ──────────────────────────────────

def test_pdf_with_wrong_extension_detected_and_warned(tmp_path):
    """A file with .bump extension whose bytes start with %PDF is detected as PDF."""
    import fitz
    from core.file_handler import load_file

    # Create a minimal real PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello magic bytes test")
    bump_path = tmp_path / "strangeextensions.Bump"
    doc.save(str(bump_path))
    doc.close()

    att = load_file(str(bump_path))
    assert att["type"] == "rag", "Detected-as-PDF file must use RAG path"
    assert att["warning"], "Must warn that extension does not match detected type"
    assert "PDF" in att["warning"]
    assert "strangeextensions.Bump" in att["warning"]


def test_genuine_text_with_unknown_extension_processed_as_text(tmp_path):
    """A .xyz file containing plain text is still read as text (no false positive)."""
    from core.file_handler import load_file

    p = tmp_path / "notes.xyz"
    p.write_text("Just some plain text here.", encoding="utf-8")

    att = load_file(str(p))
    assert att["type"] in ("text", "rag")   # small file → text
    assert att["warning"] is None            # no spurious warning
    assert "plain text" in att["content"]


def test_binary_file_with_unknown_extension_rejected(tmp_path):
    """A binary file with an unknown extension (e.g. .qvf) is rejected with a warning."""
    from core.file_handler import load_file

    # Build synthetic binary data: ~90% non-UTF-8 bytes, well above the 5% threshold
    binary_data = bytes(range(256)) * 40  # 10 240 bytes, many invalid UTF-8 sequences
    p = tmp_path / "dashboard.qvf"
    p.write_bytes(binary_data)

    att = load_file(str(p))
    assert att["type"] == "text"
    assert att["content"] == ""           # nothing injected / indexed
    assert att["warning"]
    assert "binary" in att["warning"].lower()
    assert "dashboard.qvf" in att["warning"]
