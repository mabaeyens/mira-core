"""run_shell OS sandbox (step 7).

Asserts the FILESYSTEM EFFECT, not the return string: each bypass must fail to
write outside the workspace, and legitimate work inside must still succeed.
Skips entirely if sandbox-exec is unavailable (non-macOS / removed).
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from core import shell_tools

pytestmark = pytest.mark.skipif(
    not shell_tools._sandbox_available(),
    reason="sandbox-exec not available on this platform")


@pytest.fixture
def ws(tmp_path, monkeypatch):
    # Ensure the sandbox is on for these tests regardless of local mira.yaml.
    monkeypatch.setattr(shell_tools, "SHELL_SANDBOX", True)
    monkeypatch.setattr(shell_tools, "SHELL_SANDBOX_ALLOW_NETWORK", True)
    return tmp_path


@pytest.fixture
def outside():
    # A canary UNDER $HOME — a genuinely protected location the profile does NOT
    # make writable (unlike TMPDIR, which the spec requires writable). Referenced
    # via $HOME so the command carries no literal outside-absolute-path token and
    # thus slips PAST run_shell's abs-path check — so a write here proves the
    # SANDBOX (not the abs-path check) is what stops it.
    name = f".mira-sbx-canary-{os.getpid()}"
    target = Path.home() / name
    target.unlink(missing_ok=True)
    yield target, f"$HOME/{name}"
    target.unlink(missing_ok=True)


# --- writes outside the workspace must be denied (filesystem effect) --------
# Each payload defeats the regex/abs-path layer (constructed path, quote removal,
# command substitution, interpreter) and must be stopped by the OS sandbox alone.

@pytest.mark.parametrize("payload_name", ["home_var", "quote_removal", "cmd_subst",
                                          "python_open", "base64_pipe", "perl_open"])
def test_write_outside_workspace_is_denied(ws, outside, payload_name):
    target, ref = outside                       # ref is "$HOME/.mira-sbx-canary-<pid>"
    payloads = {
        "home_var":      f"printf x > {ref}",
        "quote_removal": f'pr""intf x > {ref}',
        "cmd_subst":     f"$(echo printf) x > {ref}",
        "python_open":   f'python3 -c "import os;open(os.path.expanduser(\\"{ref}\\"),\\"w\\").write(\\"x\\")" 2>/dev/null',
        "base64_pipe":   f"echo {ref} | xargs -I@ sh -c 'printf x > @' 2>/dev/null",
        "perl_open":     f'perl -e "open(F,\\">\\",\\"{target}\\");print F 1" 2>/dev/null',
    }
    shell_tools.run_shell(payloads[payload_name], cwd=".", root=str(ws), force=True)
    assert not target.exists(), (
        f"SANDBOX BREACH: {payload_name} wrote outside the workspace to {target}")


# --- writes inside the workspace must succeed -------------------------------

def test_write_inside_workspace_succeeds(ws):
    res = shell_tools.run_shell("printf hi > inside.txt", cwd=".", root=str(ws))
    assert (ws / "inside.txt").read_text() == "hi", res


def test_legitimate_tools_work(ws):
    # git must still work inside the sandbox (reads broad, writes to ws + TMPDIR).
    res = shell_tools.run_shell("git init -q . && git status --porcelain && echo OK",
                                cwd=".", root=str(ws))
    assert res.get("exit_code") == 0, res
    assert "OK" in res.get("stdout", ""), res


def test_symlink_out_does_not_grant_write(ws, outside):
    # A symlink inside the workspace pointing to a PROTECTED location ($HOME) must
    # NOT gain write access — the sandbox matches the resolved path, so writing
    # through the relative symlink path still lands outside the writable subpaths.
    target, _ = outside
    link = ws / "escape"
    os.symlink(str(Path.home()), str(link))
    shell_tools.run_shell(f"printf x > escape/{target.name}",
                          cwd=".", root=str(ws), force=True)
    assert not target.exists(), "SANDBOX BREACH via symlink"


# --- fail-closed when sandbox unavailable ----------------------------------

def test_fails_closed_when_sandbox_missing(ws, monkeypatch):
    monkeypatch.setattr(shell_tools, "_sandbox_available", lambda: False)
    res = shell_tools.run_shell("echo hi", cwd=".", root=str(ws))
    assert "error" in res and "sandbox" in res["error"].lower()
    assert "exit_code" not in res  # nothing ran


def test_unsafe_workspace_path_refused(ws, monkeypatch):
    # A workspace path with a quote cannot be embedded in the profile → refuse.
    with pytest.raises(ValueError):
        shell_tools._sandbox_profile('/tmp/ev"il', True)
