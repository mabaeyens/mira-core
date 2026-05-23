"""Shell command execution — cwd always sandboxed to the active workspace root.

Security note: commands are executed with shell=True so that pipelines and
redirects work.  This means a prompt-injection attack could, in principle,
smuggle arbitrary shell code through model output.  Mitigations in place:
  1. CWD is always confined to the workspace root via safe_path().
  2. Commands referencing absolute paths outside the workspace are rejected.
  3. Known-destructive patterns require explicit user confirmation (force=True).
The denylist below is defence-in-depth; it is not a complete sandbox.
"""

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .config import SHELL_TIMEOUT, WORKSPACE_ROOT
from .workspace import safe_path, rel


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
    return re.compile(
        r'(?<!\w)/'
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
                    "message": (
                        f"Command contains a potentially destructive operation ({label}). "
                        "Ask the user to confirm, then call run_shell again with force=true."
                    ),
                }
            break  # user confirmed — allow it

    try:
        result = subprocess.run(
            command,
            shell=True,
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
