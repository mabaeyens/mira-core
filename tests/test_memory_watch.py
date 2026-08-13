"""The notification fires on the transition into an *actionable* memory state,
and only then.

The failure mode that killed this feature was firing too often: 302 eviction
transitions in 70.5 hours, every one of them the idle compress/decompress
treadmill the user can do nothing about (spec: memory-advisory-cause). The cause
field lets the watcher tell that treadmill apart from a genuine shortage, so the
guards here are about *selectivity*: the treadmill stays silent, a real shortage
(external_pressure / critical) still interrupts, and neither a persisting state
nor a backend restart re-fires.
"""
import threading
from unittest.mock import patch

from core import hardware, memory_watch

# Readings as (advisory, cause) pairs, which is what _read_advisory now returns.
OK = ("ok", hardware.CAUSE_NONE)
BUSY = ("busy", hardware.CAUSE_NONE)
UNKNOWN = ("unknown", hardware.CAUSE_NONE)
EVICT_IDLE = ("evicted", hardware.CAUSE_IDLE_RECLAIM)          # the treadmill: silent
EVICT_PRESSURE = ("evicted", hardware.CAUSE_EXTERNAL_PRESSURE)  # actionable
CRITICAL = ("critical", hardware.CAUSE_EXTERNAL_PRESSURE)       # actionable
GONE = (None, None)                                            # no information


def _run_once(readings, min_interval=0, notifications=True):
    """Drive the poll loop through a fixed sequence of (advisory, cause) readings.

    `notifications` is patched explicitly rather than inherited from config: the
    shipped default is False, and a test of transition semantics that silently
    stopped exercising the notify path would keep passing while asserting nothing.
    """
    sent = []
    calls = iter(readings)

    def fake_wait(_interval):
        try:
            fake_wait.next_value = next(calls)
        except StopIteration:
            return True  # stop_event set -> loop exits
        return False

    def fake_read():
        return fake_wait.next_value

    stop = threading.Event()
    with patch.object(memory_watch, "_read_advisory", fake_read), \
         patch.object(memory_watch, "MEMORY_ADVISORY_NOTIFICATIONS", notifications), \
         patch.object(memory_watch.scheduler, "notify",
                      side_effect=lambda text: (sent.append(text), True)[1]), \
         patch.object(memory_watch, "_MIN_NOTIFY_INTERVAL_S", min_interval):
        stop.wait = fake_wait  # type: ignore[method-assign]
        memory_watch._poll_loop(stop)
    return sent


# ── the pure decision, tested as one (spec §5) ────────────────────────────────

def test_should_notify_matrix():
    S = memory_watch._should_notify
    P, C = hardware.CAUSE_EXTERNAL_PRESSURE, hardware.CAUSE_IDLE_RECLAIM
    # actionable transitions
    assert S("ok", "evicted", P) is True
    assert S("ok", "critical", P) is True
    assert S("evicted", "critical", P) is True
    # the treadmill and unknown causes stay silent
    assert S("ok", "evicted", C) is False
    assert S("ok", "evicted", None) is False       # missing cause is never actionable
    # not a transition / no baseline
    assert S("evicted", "evicted", P) is False     # persisting state
    assert S(None, "evicted", P) is False          # first reading / restart baseline
    # non-interrupt states
    assert S("ok", "busy", hardware.CAUSE_NONE) is False
    assert S("ok", "ok", hardware.CAUSE_NONE) is False


# ── the loop: selectivity ─────────────────────────────────────────────────────

def test_notifies_on_transition_into_actionable_eviction():
    assert len(_run_once([OK, EVICT_PRESSURE])) == 1


def test_the_treadmill_eviction_never_notifies():
    """The whole point: an idle_reclaim eviction is the 302-in-70h case and must
    not interrupt, even though it is still an 'evicted' advisory."""
    assert _run_once([OK, EVICT_IDLE]) == []
    assert _run_once([OK, EVICT_IDLE, OK, EVICT_IDLE, OK, EVICT_IDLE]) == []


def test_critical_pressure_notifies():
    assert len(_run_once([OK, CRITICAL])) == 1


def test_first_reading_is_a_baseline_not_a_transition():
    # Restarting the backend reports "evicted" on the first poll because the
    # outgoing process's memory is being reclaimed while the new one loads. With
    # `previous` starting at None that must not read as a transition.
    assert _run_once([EVICT_PRESSURE]) == []
    assert _run_once([EVICT_PRESSURE, EVICT_PRESSURE]) == []


def test_a_real_shortage_after_a_restart_still_notifies():
    assert len(_run_once([EVICT_PRESSURE, OK, EVICT_PRESSURE])) == 1


def test_does_not_repeat_while_still_evicted():
    sent = _run_once([OK, EVICT_PRESSURE, EVICT_PRESSURE, EVICT_PRESSURE])
    assert len(sent) == 1, "level-triggered notification would spam a quiet machine"


def test_notifies_again_after_recovery_and_a_second_shortage():
    assert len(_run_once([OK, EVICT_PRESSURE, OK, EVICT_PRESSURE])) == 2


def test_rate_limit_suppresses_a_rapid_second_event():
    sent = _run_once([OK, EVICT_PRESSURE, OK, EVICT_PRESSURE], min_interval=10_000)
    assert len(sent) == 1, "a thrashing machine must produce one notice, not a stream"


def test_backend_going_away_is_not_a_transition():
    # None means "no information", not "recovered". A backend restart between two
    # evicted readings must not re-notify for a state already reported.
    assert len(_run_once([OK, EVICT_PRESSURE, GONE, GONE, EVICT_PRESSURE])) == 1


def test_never_notifies_for_ok_or_busy_or_unknown():
    assert _run_once([OK, BUSY, UNKNOWN, OK]) == []


# Every one of these is a claim about what the *next* reply will cost. The
# engine's reclaim decision is made after this fires and depends on conditions it
# cannot see, so none can be made honestly — in either direction.
_PREDICTIONS = ("next reply", "will be slow", "will be fast", "will load")


def test_notification_does_not_predict_the_next_reply():
    sent = _run_once([OK, EVICT_PRESSURE])
    assert len(sent) == 1
    text = sent[0].lower()
    for claim in _PREDICTIONS:
        assert claim not in text, f"the notification cannot know this: {sent[0]!r}"
    assert "memory" in text, f"says nothing about what happened: {sent[0]!r}"


# ── the acceptance test: replay the treadmill, count the interrupts ────────────

def test_replayed_treadmill_collapses_to_single_digits():
    """Spec §5 acceptance, as a replay rather than a three-day wait.

    Reconstructs the measured 2026-08 shape: a long run of idle_reclaim evictions
    (each a real transition, since a request decompresses the model back to `ok`
    in between) with a handful of genuine shortages mixed in. The old trigger
    fired on all 302; the cause-aware decision fires only on the real ones.

    Grounded in real data: the current engine log holds 490 `-> evicted`
    transitions, all with >=5.1GB free (the treadmill), and exactly one
    `-> critical`; replaying it through _should_notify yields 1 notification.
    """
    readings = []
    for _ in range(302):              # the headline number from the spec
        readings += [OK, EVICT_IDLE]
    # three genuine shortages scattered across the window
    readings += [OK, CRITICAL, OK, EVICT_PRESSURE, OK, CRITICAL]

    # pure-function replay
    previous = None
    notifs = 0
    for advisory, cause in readings:
        if memory_watch._should_notify(previous, advisory, cause):
            notifs += 1
        if advisory is not None:
            previous = advisory
    assert notifs == 3, f"expected the 3 real shortages, got {notifs}"
    assert notifs < 10, "single digits, not 302"

    # and the same through the whole loop (rate limit off so each real one counts)
    sent = _run_once(readings)
    assert len(sent) == 3


# ── notifications off: still watching, just not interrupting ──────────────────

def test_notifications_off_sends_nothing():
    assert _run_once([OK, EVICT_PRESSURE], notifications=False) == []


def test_the_watcher_still_runs_when_notifications_are_off():
    """Was `return None` before, which killed the log with the notification. The
    record of these transitions is what made the 302 finding possible at all."""
    with patch.object(memory_watch, "MEMORY_ADVISORY_NOTIFICATIONS", False), \
         patch.object(memory_watch.threading, "Thread") as thread:
        stop = memory_watch.start()

    assert stop is not None, "the watcher stopped watching, not just notifying"
    assert thread.called, "no poll thread was started"


def test_records_every_transition_even_the_silent_treadmill(caplog):
    """The treadmill never notifies, but it must still be logged — the record is
    the point, the interrupt is the thing that was noise."""
    with caplog.at_level("WARNING", logger="core.memory_watch"):
        _run_once([OK, EVICT_IDLE, OK, EVICT_IDLE], notifications=False)

    evictions = [r for r in caplog.records if "evicted" in r.getMessage()]
    assert len(evictions) == 2, "a transition went unrecorded"


def test_does_not_re_log_a_persisting_state(caplog):
    with caplog.at_level("WARNING", logger="core.memory_watch"):
        _run_once([OK, EVICT_IDLE, EVICT_IDLE, EVICT_IDLE], notifications=False)

    evictions = [r for r in caplog.records if "evicted" in r.getMessage()]
    assert len(evictions) == 1, f"logged the same persisting state {len(evictions)} times"


def test_read_advisory_returns_none_pair_when_backend_unreachable():
    with patch.object(memory_watch.urllib.request, "urlopen", side_effect=OSError("down")):
        assert memory_watch._read_advisory() == (None, None)
