"""The notification must fire on the transition into eviction, and only then.

The failure modes worth guarding are all about firing too often: a level-
triggered notification repeats for as long as the user is not using Mira, and a
backend restart must not read as a fresh eviction.
"""
import threading
from unittest.mock import patch

from core import memory_watch


def _run_once(advisories, min_interval=0):
    """Drive the poll loop through a fixed sequence of readings, once each."""
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
         patch.object(memory_watch.scheduler, "notify",
                      side_effect=lambda text: (sent.append(text), True)[1]), \
         patch.object(memory_watch, "_MIN_NOTIFY_INTERVAL_S", min_interval):
        stop.wait = fake_wait  # type: ignore[method-assign]
        memory_watch._poll_loop(stop)
    return sent


def test_notifies_on_transition_into_evicted():
    assert len(_run_once(["ok", "evicted"])) == 1


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


def test_start_returns_none_when_disabled():
    with patch.object(memory_watch, "MEMORY_ADVISORY_NOTIFICATIONS", False):
        assert memory_watch.start() is None


def test_read_advisory_returns_none_when_backend_unreachable():
    with patch.object(memory_watch.urllib.request, "urlopen", side_effect=OSError("down")):
        assert memory_watch._read_advisory() is None
