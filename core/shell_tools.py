"""Shell command execution — cwd confined and wrapped in an OS sandbox.

Commands run under macOS `sandbox-exec` with a deny-by-default profile that
confines *writes* to the workspace and temp dirs. This is the structural control:
regex prefiltering runs on the literal string and the shell re-interprets it
afterwards (quote removal, command substitution, base64|sh), so the denylist
below cannot contain a shell — the OS sandbox can. Layers, outermost first:
  1. sandbox-exec confines writes to the workspace + TMPDIR (fails closed).
  2. CWD is confined to the workspace root via safe_path().
  3. Commands referencing absolute paths outside the workspace are rejected.
  4. Known-destructive patterns require explicit out-of-band approval.
Reads stay broad on purpose — the tool is for inspecting code, and narrowing
reads breaks git/python/compilers. The threat closed here is destruction and
exfiltration-by-write outside the workspace.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .config import (
    SHELL_TIMEOUT,
    WORKSPACE_ROOT,
    SHELL_SANDBOX,
    SHELL_SANDBOX_ALLOW_NETWORK,
)
from .workspace import safe_path, rel
from .approvals import approval_token

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def _sandbox_available() -> bool:
    return os.path.exists(_SANDBOX_EXEC) and os.access(_SANDBOX_EXEC, os.X_OK)


def _sandbox_profile(workspace_root: str, allow_network: bool) -> str:
    """Build a deny-by-default sbpl profile confining writes to the workspace.

    Raises ValueError if the resolved workspace path cannot be safely embedded
    in the profile language (contains a quote or newline) — better to refuse
    than to emit a profile an attacker-influenced path could break out of.
    """
    root = str(Path(workspace_root).expanduser().resolve())
    if '"' in root or "\n" in root or "\\" in root:
        raise ValueError(f"workspace path not sandbox-safe: {root!r}")
    net = "" if allow_network else "(deny network*)\n"
    return (
        "(version 1)\n"
        "(deny default)\n"
        "(allow process-exec process-fork signal)\n"
        "(allow sysctl-read mach-lookup)\n"
        "(allow file-read*)\n"
        "(allow file-write*\n"
        f'  (subpath "{root}")\n'
        '  (subpath "/private/tmp")\n'
        '  (subpath "/private/var/folders")\n'
        '  (subpath "/tmp"))\n'
        "(allow file-write-data\n"
        '  (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr")\n'
        '  (literal "/dev/dtracehelper") (literal "/dev/tty"))\n'
        + net
    )


def _normalize(cmd: str) -> str:
    """Strip leading backslash-escapes (e.g. \\rm → rm) and collapse whitespace."""
    cmd = re.sub(r'\\([A-Za-z])', r'\1', cmd)
    cmd = re.sub(r'\s+', ' ', cmd)
    return cmd


_DANGEROUS = [
    # rm with -r or -f: bare name, absolute path, via 'command'/'env' wrappers
    (re.compile(r"\brm\s+.*-[rRf]", re.I),                  "rm with -r/-f flag"),
    (re.compile(r"/[a-z/]*\brm\s+.*-[rRf]", re.I),          "rm via absolute path"),
    (re.compile(r"\b(command|env)\s+rm\s+.*-[rRf]", re.I),  "rm via command/env wrapper"),
    (re.compile(r"\bxargs\s+.*\brm\b", re.I),               "xargs rm"),
    # Git destructive ops
    (re.compile(r"\bgit\s+push\b.*--force", re.I),           "git push --force"),
    (re.compile(r"\bgit\s+reset\s+--hard\b", re.I),          "git reset --hard"),
    (re.compile(r"\bgit\s+clean\s+-[fFdx]", re.I),           "git clean -f/-d/-x"),
    # System-level destructive ops
    (re.compile(r"\bdd\s+if=", re.I),                        "dd disk write"),
    (re.compile(r"\bsudo\b", re.I),                          "sudo"),
    (re.compile(r"\bmkfs\b", re.I),                          "mkfs"),
    (re.compile(r">\s*/dev/", re.I),                         "write to /dev/"),
    (re.compile(r"\bdrop\s+table\b", re.I),                  "DROP TABLE"),
]


def _abs_outside_ws_pattern(workspace_root: str) -> re.Pattern:
    ws = str(Path(workspace_root).expanduser().resolve())
    # Match /word-start sequences that are absolute paths outside the workspace.
    # Require a letter/digit after / so sed s/^/- / and similar regex delimiters
    # aren't falsely flagged (metacharacters like ^ $ . [ are not path components).
    # The lookbehind also excludes a preceding '*' so glob-relative exclusion
    # patterns (e.g. find . -path '*/build/*', grep --exclude '*/dist/*') are not
    # mistaken for absolute paths — a shell glob expands relative to cwd and cannot
    # escape the workspace, whereas a true absolute path ('/etc', ' /etc') still matches.
    # /tmp/ is explicitly allowed as a safe OS temp directory.
    return re.compile(
        r'(?<![\w*])/(?=[a-zA-Z0-9])'
        r'(?!tmp(?:/|$))'
        r'(?!' + re.escape(ws.lstrip('/')) + r'(?:/|$))',
    )


def run_shell(command: str, cwd: str = ".", force: bool = False, root: Optional[str] = None, timeout: int = SHELL_TIMEOUT) -> Dict[str, Any]:
    timeout = max(1, min(timeout, 300))
    effective_root = root or WORKSPACE_ROOT
    try:
        work_dir = safe_path(cwd, effective_root)
    except ValueError as e:
        return {"error": str(e)}

    normalized = _normalize(command)

    if _abs_outside_ws_pattern(effective_root).search(normalized):
        return {
            "error": (
                "Command references an absolute path outside the workspace. "
                "Use relative paths only (e.g. '.' or 'subdir/file'). "
                f"Workspace root: {effective_root}"
            )
        }

    for pattern, label in _DANGEROUS:
        if pattern.search(normalized):
            if not force:
                return {
                    "requires_confirmation": True,
                    "action": "run_shell",
                    "command": command,
                    "matched": label,
                    # The client shows this to the user and, on approval, sends the
                    # token back on the next request. The model cannot mint it.
                    "approval_token": approval_token("run_shell", command),
                    "message": (
                        f"Command contains a potentially destructive operation ({label}). "
                        "Relay this to the user for approval. You cannot approve it "
                        "yourself — do not retry this command."
                    ),
                }
            break  # user confirmed — allow it

    # Wrap in the OS sandbox unless explicitly disabled in config. If the sandbox
    # is enabled but unavailable, fail closed — never silently drop to an
    # unsandboxed shell.
    if SHELL_SANDBOX:
        if not _sandbox_available():
            return {
                "error": (
                    "Shell sandbox is enabled but sandbox-exec is unavailable; "
                    "refusing to run the command unsandboxed. Set shell_sandbox: "
                    "false in mira.yaml only if you accept an unsandboxed shell."
                ),
                "command": command,
            }
        try:
            profile = _sandbox_profile(effective_root, SHELL_SANDBOX_ALLOW_NETWORK)
        except ValueError as e:
            return {"error": str(e), "command": command}
        argv = [_SANDBOX_EXEC, "-p", profile, "/bin/sh", "-c", command]
        use_shell = False
    else:
        argv = command
        use_shell = True

    try:
        result = subprocess.run(
            argv,
            shell=use_shell,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "cwd": rel(work_dir, effective_root),
            "exit_code": result.returncode,
            "stdout": result.stdout[:8000],
            "stderr": result.stderr[:2000],
            "truncated": len(result.stdout) > 8000 or len(result.stderr) > 2000,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Timed out after {timeout}s", "command": command}
    except Exception as e:
        return {"error": str(e), "command": command}
