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
"""
import json
import logging
import threading
import time
import urllib.request

from core import scheduler
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
_TEXT = ("Something else on your Mac pushed Mira's model out of memory. "
         "Replies may be slow until it loads back in.")


def _read_advisory() -> str | None:
    """Current advisory from the backend, or None if there isn't one.

    None covers every uninteresting case identically: backend down, backend
    still starting, a backend that is not mira-mlx, or a probe that failed. All
    of them mean "no information", and none of them mean "everything is fine".
    """
    try:
        url = f"{BACKEND_HOST.rstrip('/')}/v1/stats"
        with urllib.request.urlopen(url, timeout=3) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - advisory only, never fatal
        logger.debug("memory watch: could not read backend stats (%s)", exc)
        return None
    system_memory = payload.get("system_memory")
    if not isinstance(system_memory, dict):
        return None
    return system_memory.get("advisory")


def _poll_loop(stop_event: threading.Event) -> None:
    logger.info("Memory advisory watch started (poll interval: %ds)", _POLL_INTERVAL)
    previous: str | None = None
    last_notified_at = 0.0

    while not stop_event.wait(_POLL_INTERVAL):
        try:
            current = _read_advisory()
        except Exception as exc:  # noqa: BLE001 - a watcher must not die
            logger.error("memory watch poll error: %s", exc)
            continue

        # Fire on the TRANSITION into eviction, not while it persists. The
        # advisory clears itself as soon as any request decompresses the model,
        # so a level-triggered notification would repeat for as long as the user
        # is not using Mira, which is exactly when they do not want to hear it.
        #
        # `previous is not None` is the load-bearing half: a transition needs a
        # prior state to be a transition at all. Without it the FIRST reading
        # always looks like one, and a backend restart reliably reports "evicted"
        # on that first poll — the outgoing process's memory is being reclaimed
        # while the new one loads, which spikes the compressor for a few seconds.
        # Observed live 2026-08-08: notified 30s after start, compressor back to
        # 0.77GB shortly after, with no other app involved. Restarting Mira must
        # never accuse another app of evicting it.
        if current == "evicted" and previous not in (None, "evicted"):
            now = time.time()
            if now - last_notified_at >= _MIN_NOTIFY_INTERVAL_S:
                if scheduler.notify(_TEXT):
                    last_notified_at = now
                    logger.warning("memory advisory: model evicted, notified user")
            else:
                logger.info(
                    "memory advisory: model evicted again, suppressed (last notice %.0fs ago)",
                    now - last_notified_at,
                )

        # Only advance `previous` on a real reading. Letting None overwrite it
        # would make every backend restart look like a fresh transition and
        # re-notify for a state the user already knows about.
        if current is not None:
            previous = current

    logger.info("Memory advisory watch stopped")


def start() -> threading.Event | None:
    """Start the watcher. Returns its stop event, or None if disabled."""
    if not MEMORY_ADVISORY_NOTIFICATIONS:
        logger.info("Memory advisory notifications disabled by config")
        return None
    stop_event = threading.Event()
    threading.Thread(
        target=_poll_loop, args=(stop_event,), name="mira-memory-watch", daemon=True
    ).start()
    return stop_event
