"""The bench's isolated-server lifecycle, specifically its teardown.

The bug these exist for: `stop()` runs twice — once explicitly at the end of a
run and again from the atexit handler registered in `start()` — and its `pkill`
matches *any* mira-mlx engine, not just the one this script started. On
2026-08-08 the second call landed moments after production had been restored and
killed the engine production was loading, leaving Mira serving with no backend
and nothing to retry it. Guarding only the restore was not enough.

No mlx import: `bench_compare` pulls in requests and yaml only, so this runs on
Linux CI.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

BENCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_compare.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_compare_under_test", BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def server(bench, monkeypatch, tmp_path):
    """An IsolatedServer with every external effect recorded instead of run."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench, "_wait_for", lambda *a, **k: True)
    monkeypatch.setattr(bench, "_port_is_open", lambda port: True)
    monkeypatch.setattr(bench.shutil, "rmtree", lambda *a, **k: None)

    s = bench.IsolatedServer()
    s.agent_was_loaded = True
    s.data_dir = str(tmp_path)
    s.proc = None
    return s, calls


def _count(calls, program, *contains):
    return sum(
        1 for c in calls
        if c and c[0] == program and all(any(x in part for part in c) for x in contains)
    )


def test_stop_is_idempotent(server):
    """The regression. A second stop() must not kill an engine it does not own."""
    s, calls = server
    s.stop()
    after_first = _count(calls, "pkill")
    s.stop()
    s.stop()
    assert after_first == 1
    assert _count(calls, "pkill") == 1, "stop() killed an engine on a repeat call"


def test_stop_restores_production_exactly_once(server):
    s, calls = server
    s.stop()
    s.stop()
    assert _count(calls, "launchctl", "load") == 1


def test_stop_leaves_production_down_if_it_was_already_down(server):
    """A bench must not start a server the user had deliberately stopped."""
    s, calls = server
    s.agent_was_loaded = False
    s.stop()
    assert _count(calls, "launchctl", "load") == 0


def test_restore_waits_for_the_backend_not_just_the_port(bench, monkeypatch, capsys):
    """:8000 answers while the model is still loading, so the port alone would
    report success over a Mira that cannot answer anything."""
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(bench, "_port_is_open", lambda port: True)
    monkeypatch.setattr(bench, "_wait_for", lambda pred, **k: pred())

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "ok", "backend_ready": False}

    monkeypatch.setattr(bench.requests, "get", lambda *a, **k: Resp())

    s = bench.IsolatedServer()
    s.agent_was_loaded = True
    s._restore_production()
    out = capsys.readouterr().out
    assert "backend never became ready" in out
    assert "back up, backend loaded" not in out
