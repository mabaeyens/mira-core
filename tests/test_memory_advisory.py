"""The eviction advisory and the per-process memory probe behind it.

`derive_dynamic_ceiling_bytes` shipped in e59551c with no unit tests at all; the
only coverage was of the notification watcher on top of it. These cover the
verdict itself, and in particular the thing the per-process signal was added to
fix: a machine where something OTHER than Mira is compressed used to read as
Mira being evicted, because the compressor is system-wide.

Pure functions plus one ctypes probe, so no model and no running server.
"""
import subprocess
import sys

import pytest

from core import hardware

GB = hardware.BYTES_PER_GB

# Measured on this 32GB Mac, 2026-08-08, against a 19.66GB MLX footprint.
MIRA_FOOTPRINT = 19_662_626_582
WARM_COMPRESSED = 350_000_000        # 1.8% - model fully resident
HALF_EVICTED_COMPRESSED = 9_900_000_000   # 50% - forced hog, next turn 3.38s
FULLY_EVICTED_COMPRESSED = 18_800_000_000  # 96% - unforced, next turn 17.60s


@pytest.fixture
def healthy_vm(monkeypatch):
    """A machine with room, nothing compressed, and macOS reporting normal."""
    monkeypatch.setattr(hardware, "read_vm_state", lambda: {
        "available_bytes": 12 * GB,
        "compressor_bytes": 400_000_000,
        "wired_bytes": 3 * GB,
    })
    monkeypatch.setattr(hardware, "read_memory_pressure_level",
                        lambda: hardware.PRESSURE_NORMAL)
    monkeypatch.setattr(hardware, "get_total_ram_bytes", lambda: 32 * GB)


def _advisory(**kwargs):
    _, diag = hardware.derive_dynamic_ceiling_bytes(**kwargs)
    return diag


# --- the per-process signal, which is the point of this file -----------------

def test_self_signal_is_preferred_over_the_system_wide_compressor(healthy_vm, monkeypatch):
    """Another app's 20GB of compressed pages is not Mira being evicted.

    This is the false positive the system-wide heuristic cannot avoid even in
    principle, because `compressor_bytes` does not say whose pages they are.
    """
    monkeypatch.setattr(hardware, "read_vm_state", lambda: {
        "available_bytes": 4 * GB,
        "compressor_bytes": 20 * GB,   # somebody else's
        "wired_bytes": 3 * GB,
    })
    diag = _advisory(mira_used_bytes=MIRA_FOOTPRINT,
                     self_compressed_bytes=WARM_COMPRESSED)
    assert diag["advisory"] == "ok"
    assert diag["eviction_signal"] == "self"
    assert diag["self_compressed_bytes"] == WARM_COMPRESSED


def test_same_machine_state_without_the_self_signal_reports_evicted(healthy_vm, monkeypatch):
    """The counterpart to the test above: the fallback DOES misfire here.

    Pinning it rather than pretending otherwise, so the reason the per-process
    reading exists stays visible if someone later removes it.
    """
    monkeypatch.setattr(hardware, "read_vm_state", lambda: {
        "available_bytes": 4 * GB,
        "compressor_bytes": 20 * GB,
        "wired_bytes": 3 * GB,
    })
    diag = _advisory(mira_used_bytes=MIRA_FOOTPRINT)
    assert diag["advisory"] == "evicted"
    assert diag["eviction_signal"].startswith("system-wide")


@pytest.mark.parametrize("compressed,expected", [
    (WARM_COMPRESSED, "ok"),
    (HALF_EVICTED_COMPRESSED, "evicted"),
    (FULLY_EVICTED_COMPRESSED, "evicted"),
])
def test_measured_states_land_on_the_right_verdict(healthy_vm, compressed, expected):
    """The three states actually observed on the machine, not invented numbers."""
    assert _advisory(mira_used_bytes=MIRA_FOOTPRINT,
                     self_compressed_bytes=compressed)["advisory"] == expected


def test_zero_footprint_never_reports_evicted(healthy_vm):
    """A model that is not loaded cannot be evicted, whatever the ratio says."""
    diag = _advisory(mira_used_bytes=0, self_compressed_bytes=5 * GB)
    assert diag["advisory"] != "evicted"


def test_probe_failure_reports_unknown_not_healthy(monkeypatch):
    monkeypatch.setattr(hardware, "read_vm_state", lambda: None)
    monkeypatch.setattr(hardware, "read_memory_pressure_level", lambda: None)
    monkeypatch.setattr(hardware, "get_total_ram_bytes", lambda: 32 * GB)
    diag = _advisory(mira_used_bytes=MIRA_FOOTPRINT)
    assert diag["advisory"] == "unknown"
    # Every return path carries the same keys, including the early one.
    assert "self_compressed_bytes" in diag and "eviction_signal" in diag


def test_pressure_alone_is_busy_not_evicted(healthy_vm, monkeypatch):
    """macOS's own level lags badly - it read 1 while 8.64GB was being taken -
    so it may warn but must never be what declares the model evicted."""
    monkeypatch.setattr(hardware, "read_memory_pressure_level",
                        lambda: hardware.PRESSURE_WARN)
    diag = _advisory(mira_used_bytes=MIRA_FOOTPRINT,
                     self_compressed_bytes=WARM_COMPRESSED)
    assert diag["advisory"] == "busy"


def test_ceiling_never_exceeds_the_static_one(healthy_vm):
    ceiling, diag = hardware.derive_dynamic_ceiling_bytes(
        mira_used_bytes=MIRA_FOOTPRINT, available_bytes=64 * GB)
    assert ceiling <= diag["static_ceiling_bytes"]


def test_ceiling_never_falls_below_the_floor(healthy_vm):
    ceiling, _ = hardware.derive_dynamic_ceiling_bytes(
        mira_used_bytes=0, available_bytes=0)
    assert ceiling == hardware.MIN_DYNAMIC_CEILING_BYTES


# --- read_self_memory_state --------------------------------------------------

@pytest.mark.skipif(sys.platform != "darwin", reason="task_info is macOS-only")
def test_compressed_comes_from_the_graphics_ledger_not_the_pmap_stats():
    """The single most important line in hardware.py, so it is pinned here.

    task_vm_info mixes two accounting systems. Its rev0 fields are pmap
    statistics (task.c:5303, `map->pmap->stats`), and the pmap does not account
    IOKit/Metal mappings, which is where all of Mira's memory lives. Measured on
    the live engine while the model was resident and replying in 0.40s, against
    vmmap reporting 292MB swapped: `compressed` read 15.46GB and `resident_size`
    read 4.65GB against an 18.9G reality. Deriving from either - including via
    `phys_footprint - resident_size`, which was also tried - makes the advisory a
    permanent false positive.

    The rev3 graphics ledger entries do account it: 19.76GB footprint (matching
    MLX's own 19.66GB) with 0 compressed when warm.
    """
    state = hardware.read_self_memory_state()
    assert state is not None
    expected = max(
        state["graphics_footprint_compressed_bytes"]
        + state["graphics_nofootprint_compressed_bytes"], 0)
    assert state["compressed_bytes"] == expected
    # The rejected candidates stay exposed so nobody reaches for them again.
    assert "pmap_compressed_bytes" in state
    assert "resident_bytes" in state and "footprint_bytes" in state


@pytest.mark.skipif(sys.platform != "darwin", reason="task_info is macOS-only")
def test_read_self_memory_state_agrees_with_footprint():
    """Cross-check the ctypes struct against a tool that reads the same kernel
    data. A wrong field offset would still return plausible-looking numbers, so
    "it returned something" is not evidence the layout is right."""
    state = hardware.read_self_memory_state()
    assert state is not None
    assert state["footprint_bytes"] > 0
    assert state["resident_bytes"] > 0
    assert state["compressed_bytes"] >= 0
    # This test process is small, busy and entirely resident, so essentially
    # nothing of it should read as paged out.
    assert state["compressed_bytes"] < 0.5 * state["footprint_bytes"]

    out = subprocess.run(["/usr/bin/footprint", "-p", str(__import__("os").getpid())],
                         capture_output=True, text=True)
    if out.returncode != 0 or "Footprint:" not in out.stdout:
        pytest.skip("footprint(1) unavailable")
    # "python [123]: 64-bit    Footprint: 19 GB" / "6157 MB" / "412 KB"
    field = out.stdout.split("Footprint:")[1].split()
    value, unit = float(field[0]), field[1]
    reported = value * {"KB": 1e3, "MB": 1e6, "GB": 1e9}[unit]
    ours = state["footprint_bytes"]
    # footprint(1) rounds hard (two significant figures at GB scale), so this is
    # an order-of-magnitude agreement check, not an equality one.
    assert 0.5 * reported <= ours <= 2.0 * reported, (ours, reported, out.stdout[:200])


def test_read_self_memory_state_returns_none_off_darwin(monkeypatch):
    """None, never zeros: a caller must not read "could not measure" as
    "nothing is compressed"."""
    monkeypatch.setattr(hardware, "_libsystem_tried", False)
    monkeypatch.setattr(hardware, "_libsystem_handle", None)
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    assert hardware.read_self_memory_state() is None


def test_libsystem_load_failure_is_remembered(monkeypatch):
    """The idle branch calls this every 30s; a failed load must not be retried
    on every call for the life of the process."""
    monkeypatch.setattr(hardware, "_libsystem_tried", False)
    monkeypatch.setattr(hardware, "_libsystem_handle", None)
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    calls = []

    def boom(*a, **kw):
        calls.append(1)
        raise OSError("nope")

    monkeypatch.setattr(hardware.ctypes, "CDLL", boom)
    assert hardware._libsystem() is None
    assert hardware._libsystem() is None
    assert len(calls) == 1


# --- on_battery --------------------------------------------------------------

@pytest.mark.parametrize("stdout,expected", [
    ("Now drawing from 'Battery Power'\n -InternalBattery-0\t80%", True),
    ("Now drawing from 'AC Power'\n -InternalBattery-0\t100%", False),
])
def test_on_battery_reads_pmset(monkeypatch, stdout, expected):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout, ""))
    assert hardware.on_battery() is expected


def test_on_battery_assumes_wall_power_when_unreadable(monkeypatch):
    """False on doubt. This gates optional work, so an unreadable pmset must not
    silently switch the feature off everywhere."""
    def boom(*a, **kw):
        raise OSError("no pmset")

    monkeypatch.setattr(subprocess, "run", boom)
    assert hardware.on_battery() is False
