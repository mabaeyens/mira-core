import logging
import subprocess
import threading
import time

from . import db

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 30  # seconds between reminder checks

# `osascript -e` evaluates its argument as AppleScript, so reminder text must
# never be interpolated into the script body — a double quote in the text closes
# the string literal and everything after it parses as code. An arg list is not
# enough here: the boundary that matters is inside osascript, not the shell.
# Pass the text as a run argument instead, where AppleScript only ever sees an
# inert string.
_NOTIFY_SCRIPT = (
    "on run argv",
    'display notification (item 1 of argv) with title "Mira"',
    "end run",
)


def _notify_argv(text: str) -> list[str]:
    """Build the osascript call for a notification, with `text` as data.

    The trailing `--` is load-bearing: without it, a reminder whose text starts
    with a dash is parsed as an osascript option and delivery fails with a
    syntax error instead of showing the notification.
    """
    argv = ["osascript"]
    for line in _NOTIFY_SCRIPT:
        argv += ["-e", line]
    return argv + ["--", text]


def notify(text: str) -> bool:
    """Deliver a macOS Notification Center alert. Returns whether it went out.

    Extracted from _fire_reminder so callers other than reminders can use the
    same hardened path — the argv construction above is the load-bearing part
    and re-implementing it elsewhere is how the quoting bug comes back.
    """
    try:
        subprocess.run(_notify_argv(text), check=True, capture_output=True)
        return True
    except Exception as e:
        logger.error("Failed to deliver notification: %s", e)
        return False


def _fire_reminder(reminder: dict) -> None:
    text = reminder["text"]
    try:
        if notify(text):
            logger.info("Reminder fired (id=%s): %s", reminder["id"], text)
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
