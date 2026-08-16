#!/usr/bin/env python3
"""Mira command-line entry point.

Thin dispatcher exposed as the `mira` console script (see pyproject.toml):

    mira setup [flags]   run the installer / re-run setup (scripts/setup.sh)
    mira serve           start the web server (server.py) — web UI + SSE
    mira chat            start the interactive CLI (main.py)
    mira doctor          health check the install and running backends
    mira fetch-model     pre-download the default model into the HF cache
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
import re
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

# The default backend (core/config.py) and the model it pulls from the HF cache
# on first run. doctor/preflight read the live values from mira.yaml; these are
# the fallbacks when mira.yaml is absent or silent.
DEFAULT_BACKEND = "mira-mlx"
DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

GB = 1024 ** 3

# Approximate on-disk sizes (GB) for a full install. Models the installer can't
# fetch during setup (the default model pulls from the HF cache on first run;
# oMLX models download via the GUI app) are still counted so the disk budget
# reflects where the user ends up — not just what setup.sh runs.
# id -> (label, approx GB, category)
COMPONENTS: "dict[str, tuple[str, float, str]]" = {
    "venv":         (".venv (uv sync)",                       1.5,  "auto"),
    "rag":          ("embedding + reranker models",           1.0,  "auto"),
    "mira-qwen3":   ("Qwen3.6-35B-A3B-4bit (default model)", 19.0, "mira-mlx"),
    "omlx-qwen3":   ("Qwen3.6-35B-A3B (oMLX backend)",       19.0, "omlx"),
    "omlx-gemma4":  ("Gemma 4 26B (oMLX)",                    15.0, "omlx"),
}
AUTO = ("venv", "rag")            # always installed by setup.sh
DEFAULT_SELECTION = ("mira-qwen3",)  # the default backend's model, picked under -y
OPTIONAL = ("mira-qwen3", "omlx-qwen3", "omlx-gemma4")
LARGE_MODELS = ("mira-qwen3", "omlx-qwen3", "omlx-gemma4")
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


def _warn(label: str, fix: str = "") -> None:
    """A finding that is worth reporting but is not a broken install."""
    tail = "" if not fix else f"\n     {DIM}→ {fix}{RESET}"
    print(f"  {YELLOW}⚠️{RESET}  {label}{tail}")


def _orphaned_prompt_cache_line() -> None:
    """Report dead disk-prompt-cache files, if the store is off and left some.

    Imported lazily and behind a bare except because this module is deliberately
    stdlib-only: `doctor` has to run on a half-built install, which is exactly
    when `core` may not import at all. A missing import here costs one advisory
    line, not the health check.
    """
    try:
        sys.path.insert(0, str(_repo_root()))
        from core.backend_manager import MIRA_MLX_CACHE_DIR
        from core.config import DISK_PROMPT_CACHE
        from core.hardware import format_bytes, orphaned_prompt_cache
    except Exception:
        return
    if DISK_PROMPT_CACHE:
        return
    count, nbytes = orphaned_prompt_cache(MIRA_MLX_CACHE_DIR)
    if not count:
        return
    _warn(
        f"{format_bytes(nbytes)} of dead prompt-cache files "
        f"({count} of them) in {MIRA_MLX_CACHE_DIR}",
        f"the disk cache is off and nothing reads them — rm -rf {MIRA_MLX_CACHE_DIR}",
    )


def _config_path() -> Path:
    """The active mira.yaml — MIRA_CONFIG if set (benches, Homebrew), else repo root.

    Mirrors core.config._load_yaml_config so doctor/setup report on the same file
    the server actually loads.
    """
    override = os.environ.get("MIRA_CONFIG")
    return Path(override).expanduser() if override else _repo_root() / "mira.yaml"


def _yaml_scalar(key: str, default: str) -> str:
    """Read a top-level `key: value` scalar from mira.yaml, no YAML dependency.

    doctor/preflight are stdlib-only (they run before .venv exists), so we can't
    import pyyaml. mira.yaml keeps `backend:` and `model:` as flat top-level lines,
    which is all this needs to route the health check — not a general parser.
    """
    try:
        for line in _config_path().read_text().splitlines():
            m = re.match(rf"^{re.escape(key)}:\s*(.+)$", line)
            if m:
                val = m.group(1).split("#", 1)[0].strip().strip("\"'")
                if val:
                    return val
    except Exception:
        pass
    return default


def _configured_backend() -> str:
    return _yaml_scalar("backend", DEFAULT_BACKEND)


def _configured_model() -> str:
    return _yaml_scalar("model", DEFAULT_MODEL)


def _hf_hub_cache() -> Path:
    """The HuggingFace hub cache dir, honoring HF_HUB_CACHE / HF_HOME overrides."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _hf_model_cached(repo_id: str) -> bool:
    """True if `repo_id` has a populated snapshot in the HF hub cache."""
    snap = _hf_hub_cache() / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    try:
        return snap.is_dir() and any(any(d.iterdir()) for d in snap.iterdir() if d.is_dir())
    except Exception:
        return False


def doctor() -> int:
    root = _repo_root()
    print(f"\n{YELLOW}Mira doctor{RESET}  {DIM}({root}){RESET}\n")
    print(f"  {DIM}ℹ {_free_gb():.0f} GB free on disk{RESET}")

    ok = True
    ok &= _line(_which("uv"), "uv installed",
                "curl -LsSf https://astral.sh/uv/install.sh | sh")
    ok &= _line((root / ".venv" / "bin" / "python").exists(),
                ".venv built", "make install")
    cfg = _config_path()
    ok &= _line(cfg.exists(),
                f"mira.yaml present ({cfg})",
                f"cp mira.yaml.example {cfg}")

    # Backend-specific prerequisites. Only oMLX needs a separate GUI app; the
    # default (mira-mlx) and the other in-process backends do not, so requiring
    # oMLX.app on those paths would fail a perfectly good install.
    backend = _configured_backend()
    if backend == "omlx":
        ok &= _line(OMLX_APP.exists(), "oMLX app installed",
                    f"download from {OMLX_RELEASES}, drag to /Applications")
    else:
        _line(True, f"backend: {backend} (no separate app to install)")
        model = _configured_model()
        if _hf_model_cached(model):
            _line(True, f"default model cached ({model})")
        else:
            _warn(f"default model not downloaded yet ({model})",
                  "it pulls into the HF cache on first `mira serve` (~19 GB)")

    # Optional dep (informational — not counted toward exit status).
    _line(_which("tesseract"), "tesseract installed (optional — scanned-PDF OCR)",
          "mira setup --with-ocr")

    # Runtime checks (informational — not counted toward exit status).
    print(f"\n{DIM}  runtime (start a backend / server to light these up){RESET}")
    if backend == "omlx":
        omlx_up = _http_ok(OMLX_HOST + "/v1/models",
                           {"Authorization": f"Bearer {_omlx_api_key()}"})
        _line(omlx_up, f"oMLX reachable on :8080 (model: {OMLX_MODEL})",
              f"open oMLX, load {OMLX_MODEL} in its model library")
    _line(_http_ok(SERVER_URL + "/"),
          "Mira server reachable on :8000", "mira serve")

    # Advisory, and deliberately not counted toward exit status: leftover cache
    # files waste disk but nothing about the install is broken, and `doctor`
    # returning 1 for them would fail scripts that gate on it.
    _orphaned_prompt_cache_line()

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
        print(f"{DIM}  Pick the models to count toward the disk budget. The default")
        print(f"  model downloads into the HF cache on first run; oMLX models download")
        print(f"  via the oMLX app. Counted here so you don't run out of space")
        print(f"  mid-install. Press Enter to accept each default.{RESET}\n")
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


# ── fetch-model ───────────────────────────────────────────────────────────────

def fetch_model(model: "str | None") -> int:
    """Download the configured (or given) mira-mlx model into the HF cache.

    The default backend pulls its ~19 GB model on first `mira serve`, which
    otherwise looks like a hang. Running this ahead of time makes the download
    explicit, with progress, and is idempotent (a cached model returns at once).
    """
    repo = model or _configured_model()
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        # doctor/preflight are stdlib-only, but this needs the venv. Re-exec there.
        root = _repo_root()
        py = root / ".venv" / "bin" / "python"
        if py.exists() and Path(sys.executable).resolve() != py.resolve():
            return subprocess.run(
                [str(py), str(root / "mira_cli.py"), "fetch-model", "--model", repo],
                cwd=root,
            ).returncode
        print(f"{RED}huggingface_hub is not available — run `make install` first.{RESET}")
        return 1

    if _hf_model_cached(repo):
        print(f"{GREEN}Already cached:{RESET} {repo}")
        return 0

    print(f"{YELLOW}Downloading{RESET} {repo}")
    print(f"{DIM}  → {_hf_hub_cache()}   (one-time; progress below){RESET}")
    try:
        snapshot_download(repo_id=repo)  # default tqdm bars stream to stderr
    except Exception as e:  # noqa: BLE001 — surface any HF/network/auth error plainly
        print(f"{RED}Download failed:{RESET} {e}")
        print(f"{DIM}  → check your network / disk space and re-run `mira fetch-model`{RESET}")
        return 1
    print(f"{GREEN}✅ {repo} is ready.{RESET}")
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

    p_fetch = sub.add_parser("fetch-model",
                             help="download the default model into the HF cache")
    p_fetch.add_argument("--model", default=None,
                         help="repo id to fetch (default: mira.yaml's model)")

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
    elif args.cmd == "fetch-model":
        sys.exit(fetch_model(args.model))
    elif args.cmd == "preflight":
        sys.exit(preflight(args.include, args.yes, args.force))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
