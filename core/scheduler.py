import logging
import subprocess
import threading
import time

from . import db

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 30  # seconds between reminder checks


def _fire_reminder(reminder: dict) -> None:
    text = reminder["text"]
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{text}" with title "Mira"'],
            check=True,
            capture_output=True,
        )
        logger.info("Reminder fired (id=%s): %s", reminder["id"], text)
    except Exception as e:
        logger.error("Failed to deliver reminder id=%s: %s", reminder["id"], e)
    finally:
        db.mark_reminder_fired(reminder["id"])


def _poll_loop(stop_event: threading.Event) -> None:
    logger.info("Reminder scheduler started (poll interval: %ds)", _POLL_INTERVAL)
    while not stop_event.wait(_POLL_INTERVAL):
        try:
            due = db.get_pending_reminders()
            for reminder in due:
                _fire_reminder(reminder)
        except Exception as e:
            logger.error("Scheduler poll error: %s", e)
    logger.info("Reminder scheduler stopped")


def start() -> threading.Event:
    """Start the background scheduler. Returns the stop event (call .set() to stop)."""
    stop_event = threading.Event()
    # Fire any reminders that came due while the server was stopped
    try:
        due = db.get_pending_reminders()
        for reminder in due:
            logger.info("Firing missed reminder on startup (id=%s)", reminder["id"])
            _fire_reminder(reminder)
    except Exception as e:
        logger.error("Startup reminder check failed: %s", e)
    t = threading.Thread(target=_poll_loop, args=(stop_event,), daemon=True)
    t.start()
    return stop_event
