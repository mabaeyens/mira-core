"""The engine subprocess's stdout goes to a file, not to /dev/null.

Why this exists: mira-mlx logs the only evidence of several things nothing else
can see — whether a finished request was registered in the prompt cache
(`insert_cache` vs `insert_cache SKIPPED`), disk-cache hits, decompression
timings. All of it was routed to DEVNULL, so "the prompt cache reports zero
hits against 40GB on disk" sat unexplained with the answer being written and
discarded on every request.
"""
import subprocess
from pathlib import Path

import pytest

from core import backend_manager as bm


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    p = tmp_path / "sub" / "mira-mlx.log"
    monkeypatch.setattr(bm, "ENGINE_LOG_PATH", p)
    return p


def test_it_returns_an_appending_handle_and_creates_the_directory(log_path):
    h = bm._engine_log_handle()
    assert h is not subprocess.DEVNULL
    h.write("first\n")
    h.close()

    h = bm._engine_log_handle()
    h.write("second\n")
    h.close()
    assert log_path.read_text() == "first\nsecond\n", "a restart truncated the log"


def test_an_oversized_log_starts_fresh(log_path, monkeypatch):
    """Capped rather than rotated: an unbounded log on a laptop is worse than
    one that restarts."""
    monkeypatch.setattr(bm, "ENGINE_LOG_MAX_BYTES", 100)
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"x" * 101)

    h = bm._engine_log_handle()
    h.close()
    assert log_path.read_bytes() == b""


def test_a_log_under_the_cap_is_kept(log_path, monkeypatch):
    monkeypatch.setattr(bm, "ENGINE_LOG_MAX_BYTES", 100)
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"x" * 99)

    h = bm._engine_log_handle()
    h.close()
    assert len(log_path.read_bytes()) == 99


def test_an_unwritable_path_falls_back_instead_of_blocking_startup(tmp_path, monkeypatch):
    """Losing the log is acceptable; refusing to start the backend is not."""
    monkeypatch.setattr(bm, "ENGINE_LOG_PATH", Path("/proc/definitely/not/writable/x.log"))
    assert bm._engine_log_handle() is subprocess.DEVNULL
