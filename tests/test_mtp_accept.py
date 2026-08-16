"""Unit tests for the pure MTP acceptance logic (spec §5.3, mtp_batch.py).

No MLX and no model — this is the one piece of the decode loop that is pure
control flow, so it is fully testable ahead of the live backbone run.
"""

from core.inference.mtp.mtp_batch import (
    accept_prefix,
    emitted_tokens,
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
    # The pure logic must import even where mlx / mlx_lm are absent; the
    # CompletionBatch base is only imported lazily by the factory.
    import importlib

    mod = importlib.import_module("core.inference.mtp.mtp_batch")
    assert hasattr(mod, "make_mtp_completion_batch_class")
