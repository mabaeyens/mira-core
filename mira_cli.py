#!/usr/bin/env python3
"""Mira command-line entry point.

Thin dispatcher exposed as the `mira` console script (see pyproject.toml):

    mira setup [flags]   run the installer / re-run setup (scripts/setup.sh)
    mira serve           start the web server (server.py) — web UI + SSE
    mira chat            start the interactive CLI (main.py)
    mira doctor          health check the install and running backends

`serve` and `chat` shell out to the existing entry-point files so their
__main__ behaviour (pkill, caffeinate, SSL) is preserved verbatim. `doctor`
is deliberately self-contained (stdlib only) so it can diagnose a half-built
install without importing the heavy core stack.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Kept in sync with core/backend_manager.py and server.py.
OMLX_HOST = "http://localhost:8080"
OLLAMA_HOST = "http://localhost:11434"
SERVER_URL = "http://localhost:8000"
OMLX_APP = Path("/Applications/oMLX.app")
OMLX_MODEL = "Qwen3.6-35B-A3B"
OMLX_RELEASES = "https://github.com/jundot/omlx/releases"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _repo_root() -> Path:
    """Locate the mira-core checkout.

    Priority: $MIRA_HOME → the directory holding this file (source / editable
    install) → the default clone location ~/mira-core.
    """
    env = os.environ.get("MIRA_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    if (here / "server.py").exists():
        return here
    default = Path.home() / "mira-core"
    if (default / "server.py").exists():
        return default
    sys.exit(
        f"{RED}Cannot locate the mira-core repo.{RESET} "
        "Set MIRA_HOME=/path/to/mira-core, or run from inside the checkout."
    )


def _run_in_repo(script: str, extra: list[str]) -> int:
    """Run a repo script with the repo's venv python, from the repo dir."""
    root = _repo_root()
    py = root / ".venv" / "bin" / "python"
    python = str(py) if py.exists() else sys.executable
    return subprocess.run([python, str(root / script), *extra], cwd=root).returncode


# ── doctor ──────────────────────────────────────────────────────────────────

def _omlx_api_key() -> str:
    try:
        cfg = json.loads((Path.home() / ".omlx" / "settings.json").read_text())
        return cfg["auth"]["api_key"]
    except Exception:
        return ""


def _http_ok(url: str, headers: dict | None = None, timeout: int = 2) -> bool:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def _line(ok: bool, label: str, fix: str = "") -> bool:
    mark = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
    tail = "" if ok or not fix else f"  {DIM}→ {fix}{RESET}"
    print(f"  {mark} {label}{tail}")
    return ok


def doctor() -> int:
    root = _repo_root()
    print(f"\n{YELLOW}Mira doctor{RESET}  {DIM}({root}){RESET}\n")

    ok = True
    ok &= _line(_which("uv"), "uv installed",
                "curl -LsSf https://astral.sh/uv/install.sh | sh")
    ok &= _line((root / ".venv" / "bin" / "python").exists(),
                ".venv built", "make install")
    ok &= _line((root / "mira.yaml").exists(),
                "mira.yaml present", "cp mira.yaml.example mira.yaml")
    ok &= _line(OMLX_APP.exists(), "oMLX app installed",
                f"download from {OMLX_RELEASES}, drag to /Applications")

    # Optional dep (informational — not counted toward exit status).
    _line(_which("tesseract"), "tesseract installed (optional — scanned-PDF OCR)",
          "mira setup --with-ocr")

    # Runtime checks (informational — not counted toward exit status).
    print(f"\n{DIM}  runtime (start a backend / server to light these up){RESET}")
    omlx_up = _http_ok(OMLX_HOST + "/v1/models",
                       {"Authorization": f"Bearer {_omlx_api_key()}"})
    _line(omlx_up, f"oMLX reachable on :8080 (model: {OMLX_MODEL})",
          f"open oMLX, load {OMLX_MODEL} in its model library")
    _line(_http_ok(OLLAMA_HOST + "/api/version"),
          "ollama reachable on :11434 (optional)", "mira setup --with-ollama")
    _line(_http_ok(SERVER_URL + "/"),
          "Mira server reachable on :8000", "mira serve")

    print()
    if not ok:
        print(f"{RED}Some prerequisites are missing — see fixes above.{RESET}")
        return 1
    print(f"{GREEN}Core install looks good.{RESET}")
    return 0


def _which(name: str) -> bool:
    from shutil import which
    return which(name) is not None


# ── dispatch ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="mira", description="Mira local AI assistant")
    sub = parser.add_subparsers(dest="cmd")

    p_setup = sub.add_parser("setup", help="install / re-run setup")
    p_setup.add_argument("rest", nargs=argparse.REMAINDER,
                         help="flags passed through to scripts/setup.sh")
    sub.add_parser("serve", help="start the web server (port 8000)")
    sub.add_parser("chat", help="start the interactive CLI")
    sub.add_parser("doctor", help="health check the install")

    args = parser.parse_args()

    if args.cmd == "setup":
        root = _repo_root()
        sys.exit(subprocess.run(
            ["bash", str(root / "scripts" / "setup.sh"), *args.rest], cwd=root
        ).returncode)
    elif args.cmd == "serve":
        sys.exit(_run_in_repo("server.py", []))
    elif args.cmd == "chat":
        sys.exit(_run_in_repo("main.py", []))
    elif args.cmd == "doctor":
        sys.exit(doctor())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
