"""The notification must fire on the transition into eviction, and only then.

The failure modes worth guarding are all about firing too often: a level-
triggered notification repeats for as long as the user is not using Mira, and a
backend restart must not read as a fresh eviction.
"""
import threading
from unittest.mock import patch

from core import memory_watch


def _run_once(advisories, min_interval=0, notifications=True):
    """Drive the poll loop through a fixed sequence of readings, once each.

    `notifications` is patched explicitly rather than inherited from config: the
    shipped default is now False, and a test of transition semantics that
    silently stopped exercising the notify path would keep passing while
    asserting nothing.
    """
    sent = []
    calls = iter(advisories)

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


def test_notifies_on_transition_into_evicted():
    assert len(_run_once(["ok", "evicted"])) == 1


def test_first_reading_is_a_baseline_not_a_transition():
    # Regression, found live 2026-08-08: restarting the backend reports
    # "evicted" on the first poll because the outgoing process's memory is being
    # reclaimed while the new one loads. With `previous` starting at None that
    # read as a transition and notified the user, blaming another app for
    # Mira's own restart.
    assert _run_once(["evicted"]) == []
    assert _run_once(["evicted", "evicted"]) == []


def test_a_real_eviction_after_a_restart_still_notifies():
    # The baseline rule must not swallow the genuine case that follows it.
    assert len(_run_once(["evicted", "ok", "evicted"])) == 1


def test_does_not_repeat_while_still_evicted():
    sent = _run_once(["ok", "evicted", "evicted", "evicted"])
    assert len(sent) == 1, "level-triggered notification would spam a quiet machine"


def test_notifies_again_after_recovery_and_a_second_eviction():
    sent = _run_once(["ok", "evicted", "ok", "evicted"])
    assert len(sent) == 2


def test_rate_limit_suppresses_a_rapid_second_event():
    sent = _run_once(["ok", "evicted", "ok", "evicted"], min_interval=10_000)
    assert len(sent) == 1, "a thrashing machine must produce one notice, not a stream"


def test_backend_going_away_is_not_a_transition():
    # None means "no information", not "recovered". A backend restart between
    # two evicted readings must not re-notify for a state already reported.
    sent = _run_once(["ok", "evicted", None, None, "evicted"])
    assert len(sent) == 1


def test_never_notifies_for_ok_or_busy_or_unknown():
    assert _run_once(["ok", "busy", "critical", "unknown", "ok"]) == []


# Every one of these is a claim about what the *next* reply will cost. The
# engine's reclaim decision is made after this notification fires and depends on
# four preconditions it cannot see, so none of them can be made honestly — in
# either direction. "will load itself back in" is as much a prediction as "will
# be slow"; the fix for the original wording must not swing to the opposite.
_PREDICTIONS = ("next reply", "will be slow", "will be fast", "will load")


def test_notification_does_not_predict_the_next_reply():
    # Regression, 2026-08-08: the banner in mira-apps stopped predicting a slow
    # reply in 29159fc and this notification did not, so the same eviction was
    # described two ways — and the surface still making the claim is the one
    # that interrupts. Nothing asserted what either surface said, which is
    # exactly how they drifted: one was fixed, the other was not, nothing failed.
    sent = _run_once(["ok", "evicted"])
    assert len(sent) == 1
    text = sent[0].lower()
    for claim in _PREDICTIONS:
        assert claim not in text, f"the notification cannot know this: {sent[0]!r}"
    # Guards the other direction: an empty or vague string would pass the check
    # above while telling the user nothing about what happened.
    assert "memory" in text, f"says nothing about what happened: {sent[0]!r}"


# ── notifications off: still watching, just not interrupting ──────────────────
#
# The default flipped to off on 2026-08-11 after the log showed 302 eviction
# transitions in 70.5 hours. Watching and interrupting are different things and
# only the second one was noise, so the watcher must keep running.


def test_notifications_off_sends_nothing():
    assert _run_once(["ok", "evicted"], notifications=False) == []


def test_the_watcher_still_runs_when_notifications_are_off():
    """Was `return None` before, which killed the log with the notification. The
    record of these transitions is what made the 302 finding possible at all."""
    with patch.object(memory_watch, "MEMORY_ADVISORY_NOTIFICATIONS", False), \
         patch.object(memory_watch.threading, "Thread") as thread:
        stop = memory_watch.start()

    assert stop is not None, "the watcher stopped watching, not just notifying"
    assert thread.called, "no poll thread was started"


def test_notifications_off_still_logs_every_transition(caplog):
    with caplog.at_level("WARNING", logger="core.memory_watch"):
        _run_once(["ok", "evicted", "ok", "evicted"], notifications=False)

    evictions = [r for r in caplog.records if "evicted" in r.getMessage()]
    assert len(evictions) == 2, "a transition went unrecorded"


def test_notifications_off_does_not_re_fire_the_same_state(caplog):
    """The gate must not `continue` past the end of the loop body: `previous` is
    advanced there, and skipping it would re-log the same eviction on every poll
    for as long as it persisted -- swapping a notification storm for a log
    storm."""
    with caplog.at_level("WARNING", logger="core.memory_watch"):
        _run_once(["ok", "evicted", "evicted", "evicted"], notifications=False)

    evictions = [r for r in caplog.records if "evicted" in r.getMessage()]
    assert len(evictions) == 1, f"logged the same persisting state {len(evictions)} times"


def test_read_advisory_returns_none_when_backend_unreachable():
    with patch.object(memory_watch.urllib.request, "urlopen", side_effect=OSError("down")):
        assert memory_watch._read_advisory() is None
