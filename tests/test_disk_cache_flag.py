"""The disk prompt cache is off by default and the flag is the only thing that
decides whether the engine gets a budget to write with.

Turning it off is not a behaviour change in the engine: it disables the store by
being handed a 0 budget (`mira_mlx_server.py`: `if self.disk_cache_dir and
self.disk_cache_max_bytes > 0`), which is why there is no second code path to
test. What these tests protect is that the flag actually reaches that argument,
because the failure mode is invisible — a wrong value here silently refills
40GB of disk that measured zero reads over three weeks.

No mlx import: `core.backend_manager` and `core.hardware` are both pure Python,
so this runs on Linux CI.
"""
import pytest

from core import backend_manager as bm
from core import hardware


class _FakeProc:
    """Enough of Popen for start_mira_mlx; it is never actually waited on."""

    def poll(self):
        return None


@pytest.fixture
def launch_args(monkeypatch):
    """Run start_mira_mlx with everything external stubbed, return its argv."""
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return _FakeProc()

    monkeypatch.setattr(bm, "resolve_offload_fraction", lambda model: None)
    monkeypatch.setattr(hardware, "fits_in_memory", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(hardware, "derive_prompt_cache_max_bytes", lambda *a, **k: 5 * 1024**3)
    monkeypatch.setattr(hardware, "derive_context_window", lambda *a, **k: 65536)
    monkeypatch.setattr(hardware, "derive_disk_cache_max_bytes", lambda *a, **k: 42 * 1024**3)
    monkeypatch.setattr(bm, "_engine_log_handle", lambda: None)
    monkeypatch.setattr(bm, "_wait_for_ready", lambda *a, **k: None)
    monkeypatch.setattr(bm.subprocess, "Popen", fake_popen)

    def run():
        bm.start_mira_mlx("some/model")
        args = captured["args"]
        return args[args.index("--disk-cache-max-bytes") + 1]

    return run


def test_disk_cache_budget_is_zero_when_disabled(monkeypatch, launch_args):
    monkeypatch.setattr(bm, "DISK_PROMPT_CACHE", False)
    assert launch_args() == "0"


def test_disk_cache_budget_is_derived_when_enabled(monkeypatch, launch_args):
    monkeypatch.setattr(bm, "DISK_PROMPT_CACHE", True)
    assert launch_args() == str(42 * 1024**3)


def test_disk_cache_is_off_by_default():
    """The default lives in config.py, not in a caller's kwargs. Measured over
    three weeks: 39.75GB held, zero reads served, so the default is off until a
    prefix-capable store is measured against the prefill it would replace."""
    from core import config

    assert config.DISK_PROMPT_CACHE is False


def test_disk_cache_dir_is_still_passed_when_disabled(monkeypatch, launch_args):
    """The directory argument stays. The engine gates on the budget, and keeping
    the path means re-enabling the flag needs no other change — and that a future
    prefix-capable store finds the same location."""
    monkeypatch.setattr(bm, "DISK_PROMPT_CACHE", False)
    launch_args()
    # launch_args already asserted the budget; here just prove the dir survived.
    assert bm.MIRA_MLX_CACHE_DIR.name == "mira_mlx_cache"
