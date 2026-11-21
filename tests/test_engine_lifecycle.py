"""Engine on-demand lifecycle: state machine, single-flight wake, in-flight guard.

Pure Python: start/stop/is_up are fakes, the clock is injected. No mlx, no server.
"""
import threading
import time

import pytest

from core import engine_lifecycle as el


class Fake:
    def __init__(self, start_delay=0.0, fail_start=False, stays_up_after_stop=False):
        self.up = True
        self.starts = 0
        self.stops = 0
        self.start_delay = start_delay
        self.fail_start = fail_start
        self.stays_up_after_stop = stays_up_after_stop

    def start(self):
        self.starts += 1
        time.sleep(self.start_delay)
        if self.fail_start:
            raise RuntimeError("boom")
        self.up = True

    def stop(self):
        self.stops += 1
        if not self.stays_up_after_stop:
            self.up = False

    def is_up(self):
        return self.up


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(idle_min=10, **kw):
    f, c = Fake(**{k: v for k, v in kw.items() if k in ("start_delay", "fail_start", "stays_up_after_stop")}), Clock()
    lc = el.EngineLifecycle(idle_min, f.start, f.stop, f.is_up, clock=c,
                            wake_timeout_s=kw.get("wake_timeout_s", 5.0))
    return lc, f, c


def test_disabled_is_a_noop():
    lc, f, c = make(idle_min=0)
    c.t += 10_000
    assert lc.unload_if_idle() is False
    lc.ensure_awake()
    with lc.request():
        pass
    assert (f.starts, f.stops) == (0, 0) and lc.state == el.RUNNING


def test_unloads_only_after_the_idle_window():
    lc, f, c = make(idle_min=10)
    c.t += 599
    assert lc.unload_if_idle() is False
    c.t += 2
    assert lc.unload_if_idle() is True
    assert lc.state == el.SLEEPING and f.stops == 1


def test_never_unloads_with_a_request_in_flight():
    lc, f, c = make()
    with lc.request():
        c.t += 10_000
        assert lc.unload_if_idle() is False
        assert lc.unload_now() is False
    assert lc.state == el.RUNNING and f.stops == 0
    c.t += 10_000
    assert lc.unload_if_idle() is True


def test_request_wakes_a_sleeping_engine():
    lc, f, c = make()
    c.t += 10_000
    lc.unload_if_idle()
    assert lc.state == el.SLEEPING
    with lc.request():
        assert lc.state == el.RUNNING and f.starts == 1


def test_concurrent_wakes_start_the_engine_once():
    lc, f, c = make(start_delay=0.3)
    c.t += 10_000
    lc.unload_if_idle()
    errors = []

    def go():
        try:
            with lc.request():
                pass
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and f.starts == 1 and lc.state == el.RUNNING


def test_failed_wake_restores_sleeping_and_raises():
    lc, f, c = make(fail_start=True)
    c.t += 10_000
    lc.unload_if_idle()
    with pytest.raises(el.EngineWakeError):
        with lc.request():
            pass
    assert lc.state == el.SLEEPING
    f.fail_start = False
    with lc.request():  # the next request retries cleanly
        assert lc.state == el.RUNNING


def test_unload_that_leaves_the_port_up_is_not_reported_sleeping():
    lc, f, c = make(stays_up_after_stop=True)
    c.t += 10_000
    assert lc.unload_if_idle() is False
    assert lc.state == el.RUNNING


def test_request_during_unload_waits_then_wakes():
    lc, f, c = make()
    gate = threading.Event()
    real_stop = f.stop

    def slow_stop():
        gate.wait(2)
        real_stop()

    lc._stop = slow_stop
    c.t += 10_000
    t = threading.Thread(target=lc.unload_if_idle)
    t.start()
    while lc.state != el.UNLOADING:
        time.sleep(0.01)
    result = []
    r = threading.Thread(target=lambda: (lc.ensure_awake(), result.append(lc.state)))
    r.start()
    time.sleep(0.1)
    assert result == []  # still waiting on the unload
    gate.set()
    t.join()
    r.join()
    assert result == [el.RUNNING] and f.starts == 1


def test_guard_stream_holds_inflight_until_exhausted():
    lc, f, c = make()
    g = lc.guard_stream(lambda: iter([1, 2, 3]))
    assert next(g) == 1
    assert lc.snapshot()["inflight"] == 1
    c.t += 10_000
    assert lc.unload_if_idle() is False
    assert list(g) == [2, 3]
    assert lc.snapshot()["inflight"] == 0


def test_guard_stream_released_when_the_consumer_abandons_it():
    lc, f, c = make()
    g = lc.guard_stream(lambda: iter([1, 2, 3]))
    next(g)
    g.close()
    assert lc.snapshot()["inflight"] == 0


def test_a_switch_blocks_unload_and_resyncs_to_running():
    lc, f, c = make()
    c.t += 10_000
    lc.begin_switch()
    assert lc.unload_if_idle() is False
    lc.end_switch()
    assert lc.state == el.RUNNING


def test_a_switch_from_sleeping_resyncs_to_running():
    lc, f, c = make()
    c.t += 10_000
    lc.unload_if_idle()
    lc.begin_switch()
    lc.ensure_awake()  # a switch owns the engine: no wake
    assert f.starts == 0
    lc.end_switch()
    assert lc.state == el.RUNNING


def test_unload_now_works_when_idle_window_has_not_passed():
    lc, f, c = make()
    assert lc.unload_now() is True and lc.state == el.SLEEPING


def test_guard_stream_is_eager_and_released_if_never_iterated():
    lc, f, c = make()
    calls = []
    g = lc.guard_stream(lambda: calls.append(1) or iter([1]))
    assert calls == [1]  # the client call happens at call time, not on first next()
    assert lc.snapshot()["inflight"] == 1
    g.close()
    assert lc.snapshot()["inflight"] == 0


def test_guard_stream_factory_error_releases_and_propagates():
    lc, f, c = make()

    def boom():
        raise ConnectionError("nope")

    with pytest.raises(ConnectionError):
        lc.guard_stream(boom)
    assert lc.snapshot()["inflight"] == 0
