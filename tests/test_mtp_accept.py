"""Unit tests for the pure MTP acceptance logic (spec §5.3, mtp_batch.py).

No MLX and no model — this is the one piece of the decode loop that is pure
control flow, so it is fully testable ahead of the live backbone run.
"""

import pytest

from core.inference.mtp.mtp_batch import (
    _record_cycle,
    accept_prefix,
    emitted_tokens,
    reset_stats,
    stats,
    truncate_at_stop,
)


def test_accept_all_when_every_draft_matches():
    assert accept_prefix([5, 6, 7], [5, 6, 7]) == 3


def test_accept_none_on_first_mismatch():
    assert accept_prefix([5, 6, 7], [9, 6, 7]) == 0


def test_accept_prefix_stops_at_first_mismatch():
    # d_1,d_2 match, d_3 diverges -> accept 2, ignore anything after.
    assert accept_prefix([5, 6, 7], [5, 6, 99]) == 2


def test_accept_handles_empty_drafts():
    assert accept_prefix([], []) == 0


def test_accept_is_bounded_by_draft_length():
    # verify may be longer (k+1 outputs); acceptance never exceeds the drafts.
    assert accept_prefix([5, 6], [5, 6, 7]) == 2


def test_emitted_is_main_plus_accepted_prefix():
    assert emitted_tokens(main_next=1, draft_tokens=[5, 6, 7], accepted=2) == [1, 5, 6]
    assert emitted_tokens(main_next=1, draft_tokens=[5, 6, 7], accepted=0) == [1]
    # emitted length is always accepted + 1
    assert len(emitted_tokens(9, [2, 3, 4], 3)) == 4


def test_truncate_passes_through_when_no_stop():
    kept, reason = truncate_at_stop([1, 2, 3], eos_ids=set(), is_stopped=lambda p: False)
    assert kept == [1, 2, 3] and reason is None


def test_truncate_cuts_at_eos_and_emits_it():
    kept, reason = truncate_at_stop([1, 2, 7, 3], eos_ids={7}, is_stopped=lambda p: False)
    assert kept == [1, 2, 7] and reason == "eos"


def test_truncate_cuts_at_first_stop_sequence():
    # stop fires once the running prefix ends in [2, 3]
    def is_stopped(prefix):
        return prefix[-2:] == [2, 3]

    kept, reason = truncate_at_stop([1, 2, 3, 4], eos_ids=set(), is_stopped=is_stopped)
    assert kept == [1, 2, 3] and reason == "stop"


def test_truncate_stop_takes_precedence_over_later_tokens():
    # A stop mid-run drops everything after it, so MTP never emits past the boundary.
    kept, reason = truncate_at_stop(
        [8, 8, 8], eos_ids=set(), is_stopped=lambda p: len(p) == 1
    )
    assert kept == [8] and reason == "stop"


def test_module_imports_without_mlx():
    # The pure logic must import even where mlx / mlx_lm are absent; the batch
    # base classes are only imported lazily by patch(). This asserts the public
    # API is reachable without mlx (patch installs it; stats feeds /v1/stats).
    import importlib

    mod = importlib.import_module("core.inference.mtp.mtp_batch")
    assert hasattr(mod, "patch")
    assert hasattr(mod, "stats")


# --- accept-rate accumulator (feeds GET /v1/stats) ------------------------- #


@pytest.fixture(autouse=True)
def _fresh_stats():
    # The accumulator is process-global; isolate every test that reads it.
    reset_stats()
    yield
    reset_stats()


def test_stats_empty_before_any_cycle():
    s = stats()
    assert s["cycles"] == 0
    assert s["accept_rate"] is None
    assert s["tokens_per_cycle"] is None
    assert s["depth_accept_rate"] == []


def test_stats_accumulates_accept_rate_and_tokens_per_cycle():
    # depth-3: a partial accept (2/3), a full accept (3/3), a full reject (0/3).
    _record_cycle([10, 20, 30], 2, 3)
    _record_cycle([11, 21, 31], 3, 3)
    _record_cycle([12, 22, 32], 0, 3)
    s = stats()
    assert s["cycles"] == 3
    assert s["drafted"] == 9
    assert s["accepted"] == 5
    assert s["accept_rate"] == round(5 / 9, 3)
    # emitted = (2+1) + (3+1) + (0+1) = 8 over 3 cycles.
    assert s["tokens_per_cycle"] == round(8 / 3, 3)
    # A cycle counts as a rejection whenever it accepts fewer than it drafted.
    assert s["rejections"] == 2


def test_stats_depth_accept_rate_falls_off_with_position():
    _record_cycle([10, 20, 30], 2, 3)  # positions 0,1 accepted
    _record_cycle([11, 21, 31], 3, 3)  # positions 0,1,2 accepted
    _record_cycle([12, 22, 32], 0, 3)  # none accepted
    # d1 and d2 confirmed 2/3, d3 only 1/3.
    assert stats()["depth_accept_rate"] == [round(2 / 3, 3), round(2 / 3, 3), round(1 / 3, 3)]


def test_stats_grows_depth_arrays_when_depth_increases():
    _record_cycle([10], 1, 1)
    _record_cycle([11, 21, 31], 3, 3)
    # First cycle only ever touched position 0; later positions still resolve.
    dar = stats()["depth_accept_rate"]
    assert len(dar) == 3
    assert dar[0] == 1.0
