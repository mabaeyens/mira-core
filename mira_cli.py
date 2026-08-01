#!/usr/bin/env python3
"""Mira command-line entry point.

Thin dispatcher exposed as the `mira` console script (see pyproject.toml):

    mira setup [flags]   run the installer / re-run setup (scripts/setup.sh)
    mira serve           start the web server (server.py) — web UI + SSE
    mira chat            start the interactive CLI (main.py)
    mira doctor          health check the install and running backends
    mira preflight       estimate disk + check free space / memory before install

`serve` and `chat` shell out to the existing entry-point files so their
__main__ behaviour (pkill, caffeinate, SSL) is preserved verbatim. `doctor`
is deliberately self-contained (stdlib only) so it can diagnose a half-built
install without importing the heavy core stack.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Kept in sync with core/backend_manager.py and server.py.
OMLX_HOST = "http://localhost:8080"
SERVER_URL = "http://localhost:8000"
OMLX_APP = Path("/Applications/oMLX.app")
OMLX_MODEL = "Qwen3.6-35B-A3B"
OMLX_RELEASES = "https://github.com/jundot/omlx/releases"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

GB = 1024 ** 3

# Approximate on-disk sizes (GB) for a full install. Models the installer can't
# fetch itself (oMLX is GUI-gated; HF models pull on first use) are still counted
# so the disk budget reflects where the user ends up — not just what setup.sh runs.
# id -> (label, approx GB, category)
COMPONENTS: "dict[str, tuple[str, float, str]]" = {
    "venv":         (".venv (uv sync)",                  1.5,  "auto"),
    "rag":          ("embedding + reranker models",      1.0,  "auto"),
    "omlx-qwen3":   ("Qwen3.6-35B-A3B (oMLX, default)",  19.0, "omlx"),
    "omlx-gemma4":  ("Gemma 4 26B (oMLX)",               15.0, "omlx"),
}
AUTO = ("venv", "rag")            # always installed by setup.sh
DEFAULT_SELECTION = ("omlx-qwen3",)  # picked under -y / non-interactive
OPTIONAL = ("omlx-qwen3", "omlx-gemma4")
LARGE_MODELS = ("omlx-qwen3", "omlx-gemma4")
BREATHING_GB = 15.0               # keep this much free after install


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
    print(f"  {DIM}ℹ {_free_gb():.0f} GB free on disk{RESET}")

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
    _line(_http_ok(SERVER_URL + "/"),
          "Mira server reachable on :8000", "mira serve")

    print()
    if not ok:
        print(f"{RED}Some prerequisites are missing — see fixes above.{RESET}")
        return 1
    print(f"{GREEN}Core install looks good.{RESET}")
    return 0


def _which(name: str) -> bool:
    return shutil.which(name) is not None


# ── preflight (disk + memory) ─────────────────────────────────────────────────

def _free_gb(path: "Path | None" = None) -> float:
    """Free space (GB) on the volume where models/venv land."""
    candidates = [path, Path.home() / ".cache" / "huggingface", Path.home()]
    for p in candidates:
        if p is None:
            continue
        target = p if p.exists() else p.parent
        try:
            return shutil.disk_usage(target).free / GB
        except Exception:
            continue
    return 0.0


def _ram_gb() -> float:
    """Physical RAM (GB) via sysctl; 0.0 if unavailable (non-Darwin)."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip()) / GB
    except Exception:
        return 0.0


def _ask(question: str, default: bool) -> bool:
    mark = "Y/n" if default else "y/N"
    try:
        ans = input(f"{question} [{mark}] ").strip().lower()
    except EOFError:
        return default
    if ans == "":
        return default
    return ans in ("y", "yes")


def preflight(include: list, assume_yes: bool, force: bool) -> int:
    print(f"\n{YELLOW}Mira preflight{RESET}  {DIM}(disk + memory check){RESET}\n")

    selected = set(AUTO)
    selected.update(include)

    if assume_yes:
        selected.update(DEFAULT_SELECTION)
    else:
        print(f"{DIM}  Pick the models to count toward the disk budget. The oMLX models")
        print(f"  download later via the oMLX app — counted here so you don't run out")
        print(f"  of space mid-install. Press Enter to accept each default.{RESET}\n")
        for cid in OPTIONAL:
            label, gb, _ = COMPONENTS[cid]
            default = cid in selected or cid in DEFAULT_SELECTION
            if _ask(f"  Include {label} (~{gb:.0f} GB)?", default):
                selected.add(cid)
            else:
                selected.discard(cid)
        print()

    total = sum(COMPONENTS[c][1] for c in selected)
    free = _free_gb()
    ram = _ram_gb()

    print(f"{DIM}  Component                              Size{RESET}")
    for cid, (label, gb, _) in COMPONENTS.items():
        if cid in selected:
            print(f"  {GREEN}✓{RESET} {label:<36} {gb:>5.1f} GB")
    print(f"  {DIM}{'─' * 48}{RESET}")
    print(f"    {'Total required':<36} {total:>5.1f} GB")
    print(f"    {'Free on disk now':<36} {free:>5.1f} GB")
    print(f"    {'Free after install':<36} {free - total:>5.1f} GB")
    print(f"    {DIM}(keep ≥ {BREATHING_GB:.0f} GB breathing room free){RESET}\n")

    needed = total + BREATHING_GB
    if free < needed:
        short = needed - free
        print(f"  {RED}✗{RESET} Not enough disk: need ~{needed:.0f} GB "
              f"(incl. {BREATHING_GB:.0f} GB headroom), {free:.0f} GB free "
              f"— short ~{short:.0f} GB")
        print(f"    {DIM}→ free up space, deselect models, or re-run with --force{RESET}")
        if not force:
            print(f"\n{RED}Aborting.{RESET}")
            return 1
    elif free < needed + BREATHING_GB:
        print(f"  {YELLOW}!{RESET} Disk is tight: only ~{free - total:.0f} GB would "
              f"remain free after install.")
    else:
        print(f"  {GREEN}✓{RESET} Disk OK ({free - total:.0f} GB free after install)")

    big = [c for c in selected if c in LARGE_MODELS]
    if ram:
        if ram <= 32 and len(big) >= 2:
            print(f"  {YELLOW}!{RESET} {ram:.0f} GB RAM: the large models can't be "
                  f"resident at once — Mira loads one at a time (switching unloads "
                  f"the other).")
        if ram < 24:
            print(f"  {YELLOW}!{RESET} {ram:.0f} GB RAM is below the recommended 32 GB "
                  f"— the default model may OOM at large context. "
                  f"See docs/model-comparison-m5-macbook.md.")

    if not assume_yes:
        print()
        if not _ask("  Proceed with install?", True):
            print(f"{YELLOW}Cancelled.{RESET}")
            return 1
    return 0


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

    p_pre = sub.add_parser("preflight", help="check disk + memory before install")
    p_pre.add_argument("--include", action="append", default=[], choices=list(COMPONENTS),
                       metavar="COMPONENT", help="add a component to the disk budget")
    p_pre.add_argument("-y", "--yes", action="store_true", help="non-interactive")
    p_pre.add_argument("--force", action="store_true", help="proceed despite low disk")

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
    elif args.cmd == "preflight":
        sys.exit(preflight(args.include, args.yes, args.force))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
