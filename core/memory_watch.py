"""Tell the user when something else on their Mac evicted Mira's model.

The inference backend re-derives its memory ceiling from real system state every
30s and reports an advisory (see `core/hardware.py` and `mira_mlx_server.py`),
but it writes to DEVNULL and the advisory only reaches anyone who polls
`/hardware`. This is the push half: a small poll loop in the main process that
watches for the state that actually costs the user something and says so.

Why this state and not general memory pressure: measured 2026-08-08 on an
M5/32GB, once macOS compressed the model out (17.03GB of compressor), the next
request took **15.37s against a warm 0.47s**. That single turn decompressed the
model and the advisory cleared by itself. So the user experiences one
unexplained slow reply and then normality, which is the least debuggable shape a
performance problem can have. A notification turns it into something they can
understand, and, if it keeps happening, act on.

**That last sentence is why notifications are now off by default (2026-08-11).**
It assumed evictions were rare enough that one message would be informative.
Measured over 70.5 hours they are not: 302 transitions, one every ~14 minutes,
round the clock. At that rate the notification is not an explanation, it is
noise on the channel that should carry the alerts worth reading, and there is
nothing a user can do about another process taking memory anyway.

So the watcher keeps watching and keeps logging — the log is where the 302 came
from — and the interruption is gated on `memory_advisory_notifications`.
"""
import json
import logging
import threading
import time
import urllib.request

from core import hardware, scheduler
from core.config import BACKEND_HOST, MEMORY_ADVISORY_NOTIFICATIONS

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 30  # matches the backend's own probe cadence; polling faster
                     # cannot see anything new, it just re-reads the same value.

# Never send a second notification within this window, even across separate
# eviction events. If the machine is oversubscribed enough to evict repeatedly,
# the user needs one message, not a stream of them saying the same thing.
_MIN_NOTIFY_INTERVAL_S = 15 * 60

# Describes the state; deliberately does not predict what the next reply costs.
# Since 13ba3db the engine can fault the model back in on its own idle branch,
# so by the time this is read the eviction may already be fixed. It is not
# reliably false either — that reclaim is skipped under critical pressure, on
# battery, without headroom, or when `proactive_decompress` is off, which is the
# default. Nothing here can tell which case the machine is in, so any sentence
# of the form "the next reply will X" is a prediction this process cannot make.
#
# Says "your Mac" where the mira-apps banner says "the Mac running Mira": this
# fires on the machine running the model, while the banner may be read on an
# iPhone talking to a remote Mac. The two agree on the claim, not the wording.
# Only sent for actionable states (external_pressure / critical), so it can name
# the one thing the user can do — free some memory — without the treadmill's
# false alarms. Deliberately makes no claim about what the next reply will cost:
# the reclaim decision is made after this fires and depends on conditions this
# process cannot see.
_TEXT = ("Your Mac is low on memory — other apps are competing for it, so Mira "
         "may be slow until some frees up.")


def _read_advisory() -> tuple[str | None, str | None]:
    """Current (advisory, cause) from the backend, or (None, None) if there isn't one.

    (None, None) covers every uninteresting case identically: backend down,
    backend still starting, a backend that is not mira-mlx, or a probe that
    failed. All of them mean "no information", and none of them mean "everything
    is fine".

    `cause` (spec: memory-advisory-cause) is what lets the loop interrupt for a
    real shortage while staying silent for the idle treadmill. A payload from an
    older backend without the field yields cause None, which the decision treats
    as "unknown" — never as actionable.
    """
    try:
        url = f"{BACKEND_HOST.rstrip('/')}/v1/stats"
        with urllib.request.urlopen(url, timeout=3) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - advisory only, never fatal
        logger.debug("memory watch: could not read backend stats (%s)", exc)
        return None, None
    system_memory = payload.get("system_memory")
    if not isinstance(system_memory, dict):
        return None, None
    return system_memory.get("advisory"), system_memory.get("cause")


def _should_notify(previous: str | None, current: str | None, cause: str | None) -> bool:
    """Does this (previous -> current, cause) transition warrant interrupting?

    Pure, so it is the acceptance test: replay a recorded transition sequence
    through it, count the notifications, no low-memory Mac required (spec §5).

    The single rule (spec §3): never interrupt for a state the user cannot change
    and did not cause.
    - Only on a real transition into the state — a persisting advisory must not
      re-fire, and `previous is None` (the first reading, and the eviction spike
      a backend restart reports as it reclaims the old process) is a baseline,
      never an accusation (edge (a)).
    - `critical` always interrupts: the machine is short of memory now and
      closing something helps (edge (b)).
    - `evicted` interrupts only when its cause is external_pressure. The idle
      treadmill (idle_reclaim), and any eviction whose cause is unknown/missing,
      stay silent (edges (d), (e)).
    - `ok`/`busy`/`unknown` never interrupt.
    """
    if previous is None or current == previous:
        return False
    if current == "critical":
        return True
    if current == "evicted":
        return cause == hardware.CAUSE_EXTERNAL_PRESSURE
    return False


def _poll_loop(stop_event: threading.Event) -> None:
    logger.info("Memory advisory watch started (poll interval: %ds)", _POLL_INTERVAL)
    previous: str | None = None
    last_notified_at = 0.0

    while not stop_event.wait(_POLL_INTERVAL):
        try:
            current, cause = _read_advisory()
        except Exception as exc:  # noqa: BLE001 - a watcher must not die
            logger.error("memory watch poll error: %s", exc)
            continue

        # Record every transition into an advisory state, regardless of whether
        # we interrupt for it — the log is the record that made the 302-in-70h
        # finding possible, and suppressing it is a different question (see
        # specs/idle-decompress-treadmill.md). A transition needs a prior state:
        # `previous is not None` guards the first reading and the eviction spike
        # a backend restart reports as it reclaims the old process.
        is_transition = (
            current is not None and previous is not None and current != previous
        )
        if is_transition and current in ("evicted", "critical"):
            logger.warning("memory advisory: %s (cause=%s)", current, cause)

        # Interrupt only for the states the user can actually act on. The idle
        # treadmill (evicted / idle_reclaim) is recorded above but never
        # surfaced here — that selectivity is the whole point of the cause field.
        if _should_notify(previous, current, cause):
            # Not gated by `continue`: the loop still has to advance `previous`
            # below, or the same transition re-fires on every poll while it lasts.
            if MEMORY_ADVISORY_NOTIFICATIONS:
                now = time.time()
                if now - last_notified_at >= _MIN_NOTIFY_INTERVAL_S:
                    if scheduler.notify(_TEXT):
                        last_notified_at = now
                        logger.info("memory advisory: notified user (%s)", current)
                else:
                    logger.info(
                        "memory advisory: notification suppressed (last notice %.0fs ago)",
                        now - last_notified_at,
                    )

        # Only advance `previous` on a real reading. Letting None overwrite it
        # would make every backend restart look like a fresh transition and
        # re-notify for a state the user already knows about.
        if current is not None:
            previous = current

    logger.info("Memory advisory watch stopped")


def start() -> threading.Event | None:
    """Start the watcher. Returns its stop event.

    The watcher runs whether or not notifications are enabled: watching and
    interrupting are different things, and only the second one is noise. With
    notifications off the loop still records every transition, which is where
    the useful history lives.
    """
    if not MEMORY_ADVISORY_NOTIFICATIONS:
        logger.info("Memory advisory notifications off; still recording transitions")
    stop_event = threading.Event()
    threading.Thread(
        target=_poll_loop, args=(stop_event,), name="mira-memory-watch", daemon=True
    ).start()
    return stop_event
