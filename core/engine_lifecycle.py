"""On-demand engine lifecycle: unload the inference engine when idle, wake it on request.

The mira-mlx engine is a child of server.py and used to stay resident (~20GB wired) from
startup until the server exited. With `engine_idle_unload_minutes > 0` this module stops
the engine after that many idle minutes and starts it again on the next request. Default
is 0 (off): behaviour is exactly as before.

States: running | unloading | sleeping | waking. A request that arrives while the engine
is sleeping wakes it (single flight: concurrent requests share one start). `unloading`
exists so a request that lands mid-stop waits for the stop to finish and then wakes,
instead of racing a second start against a dying process.

Hook point is the OpenAI client layer in the orchestrator (`request()` for one-shot calls,
`guard_stream()` for streamed ones), not individual routes, so title, summary and forced-
JSON turns are covered too. Polling endpoints (`/v1/stats`, `/health`) never count as
activity. See specs/engine-on-demand-lifecycle.md.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Callable, Iterator, Optional

logger = logging.getLogger("mira.engine_lifecycle")

RUNNING, UNLOADING, SLEEPING, WAKING = "running", "unloading", "sleeping", "waking"


class EngineWakeError(RuntimeError):
    """The engine could not be brought up for a request (wake failed or timed out)."""


class _GuardedStream:
    """Iterator wrapper that releases the lifecycle's in-flight hold exactly once."""

    def __init__(self, lifecycle: "EngineLifecycle", it: Iterator):
        self._lc, self._it, self._released = lifecycle, it, False

    def _release_once(self) -> None:
        if not self._released:
            self._released = True
            self._lc._release()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except BaseException:
            self._release_once()
            raise

    def close(self) -> None:
        try:
            closer = getattr(self._it, "close", None)
            if closer:
                closer()
        finally:
            self._release_once()

    def __del__(self):
        try:
            self._release_once()
        except Exception:  # noqa: BLE001 — interpreter shutdown
            pass


class EngineLifecycle:
    def __init__(
        self,
        idle_minutes: float,
        start_fn: Callable[[], None],
        stop_fn: Callable[[], None],
        is_up_fn: Callable[[], bool],
        *,
        wake_timeout_s: float = 260.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.idle_s = max(0.0, float(idle_minutes)) * 60.0
        self._start, self._stop, self._is_up = start_fn, stop_fn, is_up_fn
        self.wake_timeout_s = wake_timeout_s
        self._clock = clock
        self._cv = threading.Condition()
        self._state = RUNNING
        self._inflight = 0
        self._suspended = 0
        self._last = clock()
        self._thread: Optional[threading.Thread] = None
        self._stopping_watcher = threading.Event()

    # ── introspection ────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self.idle_s > 0

    @property
    def state(self) -> str:
        return self._state

    def snapshot(self) -> dict:
        with self._cv:
            return {
                "enabled": self.enabled,
                "state": self._state,
                "idle_unload_minutes": self.idle_s / 60.0,
                "idle_seconds": round(self._clock() - self._last, 1),
                "inflight": self._inflight,
            }

    def touch(self) -> None:
        self._last = self._clock()

    # ── wake ─────────────────────────────────────────────────────────────────
    def ensure_awake(self) -> None:
        """Return once the engine is running; start it if asleep. No-op when disabled
        or while a backend switch owns the lifecycle (suspended)."""
        if not self.enabled:
            return
        deadline = self._clock() + self.wake_timeout_s
        with self._cv:
            while True:
                if self._suspended or self._state == RUNNING:
                    return
                if self._state == SLEEPING:
                    self._state = WAKING
                    break  # this caller is the single flight; start outside the lock
                # waking or unloading: someone else is mid-transition, wait for it
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise EngineWakeError("timed out waiting for the engine to wake")
                self._cv.wait(timeout=min(remaining, 5.0))
        try:
            logger.info("engine wake: starting")
            t0 = self._clock()
            self._start()
            logger.info("engine wake: ready in %.1fs", self._clock() - t0)
        except Exception as exc:  # noqa: BLE001 — surfaced to the caller, state restored
            with self._cv:
                self._state = SLEEPING
                self._cv.notify_all()
            logger.error("engine wake failed: %s", exc)
            raise EngineWakeError(f"engine wake failed: {exc}") from exc
        with self._cv:
            self._state = RUNNING
            self._last = self._clock()
            self._cv.notify_all()

    # ── activity guards ──────────────────────────────────────────────────────
    @contextlib.contextmanager
    def request(self):
        """Wrap one engine call: wake if needed, block unloads while it runs."""
        self.ensure_awake()
        with self._cv:
            self._inflight += 1
        try:
            yield
        finally:
            with self._cv:
                self._inflight -= 1
                self._last = self._clock()

    def guard_stream(self, factory: Callable[[], Iterator]) -> Iterator:
        """Like request(), but held for the lifetime of a streamed response. Eager like the
        call it wraps: the engine is woken and `factory()` runs now, so connection errors
        surface at the call site exactly as before. The in-flight hold is released when the
        stream is exhausted, raises, is closed, or is garbage-collected unread."""
        self.ensure_awake()
        with self._cv:
            self._inflight += 1
        try:
            it = factory()
        except BaseException:
            self._release()
            raise
        return _GuardedStream(self, it)

    def _release(self) -> None:
        with self._cv:
            self._inflight -= 1
            self._last = self._clock()

    # ── unload ───────────────────────────────────────────────────────────────
    def unload_if_idle(self) -> bool:
        if not self.enabled:
            return False
        with self._cv:
            if (self._state != RUNNING or self._inflight or self._suspended
                    or self._clock() - self._last < self.idle_s):
                return False
        return self._unload()

    def unload_now(self) -> bool:
        """Manual unload. Refuses (False) when a request is in flight."""
        with self._cv:
            if self._state != RUNNING or self._inflight or self._suspended:
                return False
        return self._unload()

    def _unload(self) -> bool:
        with self._cv:
            # re-check under the lock: a request may have slipped in since the caller looked
            if self._state != RUNNING or self._inflight or self._suspended:
                return False
            self._state = UNLOADING
        try:
            logger.info("engine unload: stopping")
            self._stop()
            if self._is_up():
                raise RuntimeError("engine port still answering after stop")
        except Exception as exc:  # noqa: BLE001
            with self._cv:
                self._state = RUNNING  # still (partly) up: never claim sleeping when it may not be
                self._last = self._clock()
                self._cv.notify_all()
            logger.error("engine unload failed: %s", exc)
            return False
        with self._cv:
            self._state = SLEEPING
            self._cv.notify_all()
        return True

    # ── backend switch ownership ─────────────────────────────────────────────
    def begin_switch(self) -> None:
        """A backend/model switch owns the engine until end_switch(): no unload, no wake.
        Blocks while a transition is mid-flight, so call it off the event loop."""
        with self._cv:
            while self._state in (UNLOADING, WAKING):
                self._cv.wait(timeout=5.0)
            self._suspended += 1

    def end_switch(self) -> None:
        """Release a switch. A switch always leaves the engine up, so resync to RUNNING."""
        with self._cv:
            self._suspended = max(0, self._suspended - 1)
            self._state = RUNNING
            self._last = self._clock()
            self._cv.notify_all()

    @contextlib.contextmanager
    def suspend(self):
        self.begin_switch()
        try:
            yield
        finally:
            self.end_switch()

    # ── watcher ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        interval = max(5.0, min(30.0, self.idle_s / 4))

        def loop():
            while not self._stopping_watcher.wait(interval):
                try:
                    self.unload_if_idle()
                except Exception:  # noqa: BLE001 — a watcher must never die
                    logger.exception("engine lifecycle watcher error")

        self._thread = threading.Thread(target=loop, name="engine-lifecycle", daemon=True)
        self._thread.start()
        logger.info("engine lifecycle on: unload after %.1f min idle", self.idle_s / 60)

    def shutdown(self) -> None:
        self._stopping_watcher.set()


# Process-wide instance. Disabled (no-op) until server.py configures it.
_instance: EngineLifecycle = EngineLifecycle(0, lambda: None, lambda: None, lambda: False)


def get() -> EngineLifecycle:
    return _instance


def configure(lifecycle: EngineLifecycle) -> EngineLifecycle:
    global _instance
    _instance = lifecycle
    return lifecycle
