"""Reminder text is data, never AppleScript.

`_fire_reminder` used to build its command as
`["osascript", "-e", f'display notification "{text}" with title "Mira"']`.
The arg list is the right instinct at the wrong boundary: `subprocess` is not
what parses this. `osascript -e` evaluates the string it is handed AS
AppleScript, so a double quote inside `text` closes the literal and everything
after it is executed, `do shell script` included — outside the `sandbox-exec`
confinement that covers `run_shell`.

`text` reaches here from the `schedule_reminder` tool, so it is model-controlled,
and the reminder fires from a background thread with no approval gate.

These tests pin the shape of the command rather than running osascript, so they
hold on Linux CI where osascript does not exist.
"""
import core.scheduler as scheduler


PAYLOAD = (
    'x" with title "y\n'
    'do shell script "touch /tmp/mira_scheduler_injection"\n'
    'display notification "z'
)


def test_reminder_text_never_lands_in_the_script_body():
    argv = scheduler._notify_argv(PAYLOAD)
    scripts = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]

    # Every -e fragment is a fixed line of the template. None of them carries
    # any part of the reminder text.
    assert scripts == list(scheduler._NOTIFY_SCRIPT)
    for fragment in scripts:
        assert "do shell script" not in fragment
        assert PAYLOAD not in fragment


def test_reminder_text_is_passed_as_a_run_argument():
    argv = scheduler._notify_argv(PAYLOAD)
    assert argv[-1] == PAYLOAD
    assert argv[-2] == "--"


def test_leading_dash_text_survives_option_parsing():
    """Without the `--`, osascript parses this as its own flag and the
    notification fails with a syntax error instead of being delivered."""
    argv = scheduler._notify_argv("-e display notification \"nope\"")
    assert argv[-2] == "--"
    assert argv[-1] == '-e display notification "nope"'


def test_shell_is_never_invoked():
    argv = scheduler._notify_argv(PAYLOAD)
    assert argv[0] == "osascript"
    assert not any(a in ("sh", "bash", "-c") for a in argv)


def test_benign_text_still_builds_a_working_call():
    argv = scheduler._notify_argv("Stand up and stretch")
    assert argv[0] == "osascript"
    assert "on run argv" in argv
    assert argv[-1] == "Stand up and stretch"
