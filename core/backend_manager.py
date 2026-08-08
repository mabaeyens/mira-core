"""Manages start/stop of inference server processes for runtime backend switching."""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from core.config import CONTEXT_WINDOW, DB_PATH
from core.config import MLX_LM_CLI as _MLX_LM_CLI_PATH
from core.config import (
    MIRA_MLX_KV_BITS,
    MIRA_MLX_KV_GROUP_SIZE,
    MIRA_MLX_VISION,
    MIRA_MLX_VISION_MAX_PIXELS,
    MIRA_MLX_VISION_TOWER_IDLE_TIMEOUT,
)
from core.config import (
    MIRA_MLX_PROFILE_EXPERTS,
    MIRA_MLX_EXPERT_PROFILE_PATH,
    MIRA_MLX_TRUST_REMOTE_CODE,
    MIRA_MLX_ENABLE_TF32,
    BOUNDARY_SNAPSHOT,
    PROACTIVE_DECOMPRESS,
)
from core.config import resolve_offload_fraction
from core.config import OMLX_CLI as _OMLX_CLI_PATH, PREFILL_STEP_SIZE
from core.config import VLLM_MLX_CLI as _VLLM_MLX_CLI_PATH

logger = logging.getLogger(__name__)

MLX_LM_CLI = _MLX_LM_CLI_PATH
MLX_LM_PORT = 8080
MLX_LM_HOST = f"http://localhost:{MLX_LM_PORT}"
MLX_LM_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
MLX_LM_CONTEXT = 65536

# mira-mlx: Mira's own thin server over mlx-lm's primitives (core/inference/mira_mlx_server.py).
# In-repo module, not an external binary — launched via `python -m`, no `paths:` entry needed.
MIRA_MLX_PORT = 8080
MIRA_MLX_HOST = f"http://localhost:{MIRA_MLX_PORT}"
# Where the engine subprocess's stdout/stderr lands. Capped rather than rotated:
# this is a diagnostic tail, and a log that can grow without bound on a laptop
# is worse than one that starts fresh when it gets big.
ENGINE_LOG_PATH = Path(DB_PATH).parent / "mira-mlx.log"
ENGINE_LOG_MAX_BYTES = 32 * 1024 * 1024
MIRA_MLX_MODEL = "mlx-community/Ministral-3-14B-Instruct-2512-4bit"
# Follows mira.yaml's `context_window:` (config.CONTEXT_WINDOW) so the requested
# value actually reaches --max-kv-size below, instead of silently staying at a
# stale flat default (found 2026-07-18 while testing a context_window bump).
MIRA_MLX_CONTEXT = CONTEXT_WINDOW
MIRA_MLX_CACHE_DIR = DB_PATH.parent / "mira_mlx_cache"

OMLX_CLI = _OMLX_CLI_PATH
OMLX_PORT = 8080
OMLX_HOST = f"http://localhost:{OMLX_PORT}"
OMLX_MODEL = "Qwen3.6-35B-A3B"
OMLX_CONTEXT = 131072

VLLM_MLX_CLI = _VLLM_MLX_CLI_PATH
VLLM_MLX_PORT = 8080
VLLM_MLX_HOST = f"http://localhost:{VLLM_MLX_PORT}"
VLLM_MLX_MODEL = "mlx-community/Ministral-3-14B-Instruct-2512-4bit"
VLLM_MLX_CONTEXT = 65536

PRESETS = {
    # mlx-lm is benched out (architecture gap) — not offered in the default picker.
    # The backend code path remains for anyone who configures it explicitly in mira.yaml.
    "omlx": {
        "backend": "omlx",
        "model": OMLX_MODEL,
        "host": OMLX_HOST,
        "context_window": OMLX_CONTEXT,
        "vision": True,
    },
    "vllm-mlx": {
        "backend": "vllm-mlx",
        "model": VLLM_MLX_MODEL,
        "host": VLLM_MLX_HOST,
        "context_window": VLLM_MLX_CONTEXT,
        "vision": False,
    },
    "mira-mlx": {
        "backend": "mira-mlx",
        "model": MIRA_MLX_MODEL,
        "host": MIRA_MLX_HOST,
        "context_window": MIRA_MLX_CONTEXT,
        # Tracks the config flag rather than being hardcoded, so the capability
        # the orchestrator reads stays honest: with vision off it must keep
        # routing images to OCR, and with it on it must stop.
        "vision": MIRA_MLX_VISION,
    },
}

_mlx_lm_proc = None
_omlx_proc = None
_vllm_mlx_proc = None
_mira_mlx_proc = None


def _engine_log_handle():
    """Append-mode handle for the engine's stdout, or DEVNULL if it can't be
    opened. A backend that refuses to start because its log file is unwritable
    would be a worse failure than losing the log."""
    try:
        ENGINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if ENGINE_LOG_PATH.exists() and ENGINE_LOG_PATH.stat().st_size > ENGINE_LOG_MAX_BYTES:
            ENGINE_LOG_PATH.unlink()
        return open(ENGINE_LOG_PATH, "a", buffering=1)
    except OSError as exc:
        logger.warning("engine log unavailable (%s); falling back to DEVNULL", exc)
        return subprocess.DEVNULL


def start_mlx_lm(model: str = MLX_LM_MODEL) -> None:
    global _mlx_lm_proc
    _mlx_lm_proc = subprocess.Popen(
        [
            MLX_LM_CLI,
            "--model", model,
            "--host", "127.0.0.1",
            "--port", str(MLX_LM_PORT),
            "--max-tokens", "4096",
            "--prompt-cache-bytes", "3G",
            "--decode-concurrency", "1",
            "--prefill-step-size", str(PREFILL_STEP_SIZE),
            "--trust-remote-code",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_ready(MLX_LM_HOST + "/v1/models", timeout=120)


def start_mira_mlx(model: str = MIRA_MLX_MODEL) -> None:
    global _mira_mlx_proc
    from core import hardware

    # Per-model offload decision (auto mode: on only if the model won't otherwise
    # fit). Offload has a ~5x decode cost, so a model that fits runs resident.
    resident_expert_fraction = resolve_offload_fraction(model)
    logger.info(
        "mira-mlx offload for %s: %s",
        model,
        f"ON (resident fraction {resident_expert_fraction})"
        if resident_expert_fraction is not None else "OFF (fully resident)",
    )

    ok, reason = hardware.fits_in_memory(
        model, kv_bits=MIRA_MLX_KV_BITS, kv_group_size=MIRA_MLX_KV_GROUP_SIZE,
        resident_expert_fraction=resident_expert_fraction,
    )
    if not ok:
        raise RuntimeError(f"mira-mlx preflight check failed: {reason}")

    # Derived per-machine, not a flat constant: a fixed cache-pool size that's
    # comfortable on a 32GB Mac can itself exceed total RAM on an 8-16GB one
    # (found 2026-07-09 — a fixed 3GB cap was smaller than a single long
    # conversation's ~3.3GB KV cache entry, silently evicting it every time).
    prompt_cache_max_bytes = hardware.derive_prompt_cache_max_bytes(
        model, kv_bits=MIRA_MLX_KV_BITS, kv_group_size=MIRA_MLX_KV_GROUP_SIZE,
        resident_expert_fraction=resident_expert_fraction,
    )
    context_window = hardware.derive_context_window(
        model,
        requested_context=MIRA_MLX_CONTEXT,
        kv_bits=MIRA_MLX_KV_BITS,
        kv_group_size=MIRA_MLX_KV_GROUP_SIZE,
        resident_expert_fraction=resident_expert_fraction,
    )
    disk_cache_max_bytes = hardware.derive_disk_cache_max_bytes(MIRA_MLX_CACHE_DIR)
    # A single response can't usefully exceed the machine's own derived context
    # ceiling — on a smaller machine than this one, a flat 4096 could exceed
    # what --max-kv-size (context_window, below) actually allows.
    max_tokens = min(4096, context_window)

    args = [
        sys.executable, "-m", "core.inference.mira_mlx_server",
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(MIRA_MLX_PORT),
        "--max-tokens", str(max_tokens),
        "--prefill-step-size", str(PREFILL_STEP_SIZE),
        "--prompt-cache-max-bytes", str(prompt_cache_max_bytes),
        # Bounds a single conversation's own KV cache regardless of length —
        # without this, a very long conversation can alone exhaust RAM on
        # tight machines even with a well-sized cache pool.
        "--max-kv-size", str(context_window),
        # Mistral-family tokenizers need this regex fix (mlx-lm doesn't
        # default it on); harmless no-op for models that don't need it.
        "--fix-mistral-regex",
        # Entries evicted from the in-memory cache overflow here instead of
        # being discarded, surviving both memory-pressure trims and process
        # restarts (core/inference/disk_prompt_cache.py).
        "--disk-cache-dir", str(MIRA_MLX_CACHE_DIR),
        "--disk-cache-max-bytes", str(disk_cache_max_bytes),
    ]
    if MIRA_MLX_KV_BITS is not None:
        args += ["--kv-bits", str(MIRA_MLX_KV_BITS), "--kv-group-size", str(MIRA_MLX_KV_GROUP_SIZE)]
    if MIRA_MLX_PROFILE_EXPERTS:
        args += ["--profile-experts"]
        if MIRA_MLX_EXPERT_PROFILE_PATH:
            args += ["--expert-profile-path", MIRA_MLX_EXPERT_PROFILE_PATH]
    if resident_expert_fraction is not None:
        args += ["--resident-expert-fraction", str(resident_expert_fraction)]
    if MIRA_MLX_TRUST_REMOTE_CODE:
        args += ["--trust-remote-code"]
    if PROACTIVE_DECOMPRESS:
        args += ["--proactive-decompress"]
    if BOUNDARY_SNAPSHOT:
        args += ["--boundary-snapshot"]
    if MIRA_MLX_VISION:
        args += ["--vision"]
        args += ["--vision-max-pixels", str(MIRA_MLX_VISION_MAX_PIXELS)]
        args += [
            "--vision-tower-idle-timeout",
            str(MIRA_MLX_VISION_TOWER_IDLE_TIMEOUT),
        ]

    # Popen inherits this process's environment, so an unset MLX_ENABLE_TF32
    # silently takes MLX's own default. State it instead: the flag changes both
    # numerics and throughput, and the reasoning for the value is in config.py.
    env = os.environ.copy()
    env["MLX_ENABLE_TF32"] = "1" if MIRA_MLX_ENABLE_TF32 else "0"

    # The engine logs the things only it can see — prompt-cache insert/skip
    # decisions, disk-cache hits, decompression timings — and every one of them
    # went to DEVNULL, which is why "the prompt cache reports zero hits" had no
    # evidence attached for weeks. Keep them on disk instead, next to the DB.
    _mira_mlx_proc = subprocess.Popen(
        args,
        stdout=_engine_log_handle(),
        stderr=subprocess.STDOUT,
        env=env,
    )
    # GenerationEngine.start() (mira_mlx_server.py) itself waits up to 180s for
    # its own engine thread to finish loading before raising — this external
    # readiness poll must allow strictly more than that, or a legitimately-slow
    # but successful cold load (large model, memory pressure right after
    # killing a previous backend) raises TimeoutError here while the process
    # keeps loading in the background and becomes healthy moments later,
    # leaving callers (server.py's /models/switch) reporting failure for a
    # switch that actually succeeded. Confirmed 2026-07-18: a Qwen3.6->Gemma4
    # switch hit exactly this — /v1/models came up healthy well within 3
    # minutes of the process starting, after this 120s timeout had already
    # fired and left server.py's _rt state stale (see server.py's
    # `_reconcile_stale_switch_failure`).
    _wait_for_ready(MIRA_MLX_HOST + "/v1/models", timeout=200)


def stop_mira_mlx() -> None:
    global _mira_mlx_proc
    if _mira_mlx_proc and _mira_mlx_proc.poll() is None:
        _mira_mlx_proc.terminate()
        try:
            _mira_mlx_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _mira_mlx_proc.kill()
        _mira_mlx_proc = None


def start_vllm_mlx(model: str = VLLM_MLX_MODEL) -> None:
    global _vllm_mlx_proc
    _vllm_mlx_proc = subprocess.Popen(
        [
            VLLM_MLX_CLI, "serve", model,
            "--host", "127.0.0.1",
            "--port", str(VLLM_MLX_PORT),
            # Mistral-family models (Ministral 3, Devstral) need this parser to
            # correctly extract tool calls from their [TOOL_CALLS]/[ARGS] format.
            "--enable-auto-tool-choice",
            "--tool-call-parser", "mistral",
            # Default (300s) is too short for multi-step agentic turns; align
            # with the orchestrator's own MAX_AGENT_STEPS budget plus margin.
            "--timeout", "900",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_ready(VLLM_MLX_HOST + "/v1/models", timeout=120)


def stop_vllm_mlx() -> None:
    global _vllm_mlx_proc
    if _vllm_mlx_proc and _vllm_mlx_proc.poll() is None:
        _vllm_mlx_proc.terminate()
        try:
            _vllm_mlx_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _vllm_mlx_proc.kill()
        _vllm_mlx_proc = None


def stop_mlx_lm() -> None:
    global _mlx_lm_proc
    if _mlx_lm_proc and _mlx_lm_proc.poll() is None:
        _mlx_lm_proc.terminate()
        try:
            _mlx_lm_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _mlx_lm_proc.kill()
        _mlx_lm_proc = None


def _omlx_api_key() -> str:
    try:
        cfg = json.loads((Path.home() / ".omlx" / "settings.json").read_text())
        return cfg["auth"]["api_key"]
    except Exception:
        return ""


def _omlx_request(path: str, timeout: int = 2):
    """urlopen to an oMLX endpoint with Bearer auth."""
    req = urllib.request.Request(
        OMLX_HOST + path,
        headers={"Authorization": f"Bearer {_omlx_api_key()}"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


class BackendIdentityError(RuntimeError):
    """A process is answering the backend port but is not the model we expect.

    Raised instead of silently adopting an unverified listener. The listener may
    be a stale backend from a previous run serving a different model, or a local
    process that has squatted the port and would otherwise inherit every prompt.
    """


# Opt-in escape hatch for the bench workflow, which sometimes hand-starts a
# backend on the shared port and wants ensure_backend_running() to adopt it
# without an identity match. Off by default so production fails closed.
_ADOPT_UNVERIFIED = os.getenv("MIRA_ADOPT_UNVERIFIED_BACKEND") == "1"


def _model_basename(model_id: str) -> str:
    """Normalise a model id for comparison: drop any repo prefix, lowercase.

    Backends report the served model differently — mira-mlx echoes the full
    `mlx-community/Qwen3.6-35B-A3B-4bit` verbatim (verified), others may return
    only the short name. Comparing on the basename tolerates both without
    accepting an outright different model.
    """
    return (model_id or "").rsplit("/", 1)[-1].strip().lower()


def _model_matches(served: str, expected: str) -> bool:
    if not served or not expected:
        return False
    return _model_basename(served) == _model_basename(expected)


def _served_model(url: str, *, omlx: bool = False) -> Optional[str]:
    """Return data[0].id from /v1/models, or None if unreachable/unparseable.

    None means "nobody home or not a model server"; "" means it answered but the
    body had no model id (still not a match).
    """
    try:
        resp = _omlx_request("/v1/models") if omlx else urllib.request.urlopen(url, timeout=2)
        with resp:
            body = json.loads(resp.read())
        items = body.get("data") or []
        return items[0].get("id", "") if items else ""
    except Exception:
        return None


def _verify_or_adopt(url: str, expect_model: Optional[str], *, omlx: bool = False) -> bool:
    """True if the port serves the expected model (safe to adopt).

    False if nobody is listening. Raises BackendIdentityError if something IS
    listening but serves a different model — that is the case we must never
    silently adopt (stale backend or port squatter). The bench escape hatch
    downgrades the raise to adoption.
    """
    served = _served_model(url, omlx=omlx)
    if served is None:
        return False  # nothing there — caller should start the backend
    if not expect_model or _model_matches(served, expect_model):
        return True
    if _ADOPT_UNVERIFIED:
        logger.warning("adopting unverified listener on %s: serves %r, expected %r "
                       "(MIRA_ADOPT_UNVERIFIED_BACKEND=1)", url, served, expect_model)
        return True
    raise BackendIdentityError(
        f"{url} is serving {served!r} but {expect_model!r} was expected. "
        "Refusing to route prompts to an unverified process. If this is a "
        "hand-started backend you trust, set MIRA_ADOPT_UNVERIFIED_BACKEND=1.")


def _wait_for_ready(url: str, timeout: int = 60, *, omlx: bool = False,
                    expect_model: Optional[str] = None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if omlx:
                _omlx_request("/v1/models")
            else:
                urllib.request.urlopen(url, timeout=2)
        except Exception:
            time.sleep(1)
            continue
        # It answered. If we know which model to expect, prove identity before
        # declaring ready — a squatter that 200s on /v1/models must not pass.
        if expect_model is not None:
            served = _served_model(url, omlx=omlx)
            if not _model_matches(served or "", expect_model):
                raise BackendIdentityError(
                    f"{url} became reachable but serves {served!r}, "
                    f"expected {expect_model!r}")
        return
    raise TimeoutError(f"Server did not become ready at {url} after {timeout}s")


def is_backend_ready(backend: str) -> bool:
    """Return True if the inference backend is reachable and has the model available."""
    try:
        if backend == "omlx":
            _omlx_request("/v1/models")
        elif backend == "mlx-lm":
            urllib.request.urlopen(MLX_LM_HOST + "/v1/models", timeout=2)
        elif backend == "vllm-mlx":
            urllib.request.urlopen(VLLM_MLX_HOST + "/v1/models", timeout=2)
        elif backend == "mira-mlx":
            urllib.request.urlopen(MIRA_MLX_HOST + "/v1/models", timeout=2)
        else:
            return False
        return True
    except Exception:
        return False


def _warmup_model(model: str, host: str = MLX_LM_HOST, api_key: str = "") -> None:
    from core.prompts import build_system_prompt
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": "Hi"},
        ],
        "max_tokens": 1,
        "stream": False,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        host + "/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        logger.info("warming up model %s…", model)
        urllib.request.urlopen(req, timeout=60)
        logger.info("model warm")
    except Exception as exc:
        logger.warning("warmup failed (non-fatal): %s", exc)


def ensure_backend_running(backend: str, model: Optional[str] = None) -> None:
    """Start the backend if not already reachable. Safe to call on every startup.

    `model` overrides the backend's own hardcoded default constant (e.g.
    MIRA_MLX_MODEL) — pass core.config.MODEL_NAME (mira.yaml's configured
    model) here, or the backend silently ignores mira.yaml and starts with
    whatever placeholder model that constant happens to hold (confirmed
    2026-07-18: server.py startup was calling this with no model, so every
    fresh process cold-started mira-mlx into MIRA_MLX_MODEL's hardcoded
    Ministral default instead of mira.yaml's configured Qwen3.6, silently
    diverging from the configured model on every restart)."""
    # Identity gate: the "already running" short-circuits below are the actual
    # attack surface — they run before any spawn, so a squatter (or a stale
    # different-model backend) is met here first. _verify_or_adopt proves the
    # listener serves the expected model BEFORE _warmup_model, which is itself
    # the first disclosure (it POSTs the system prompt). A mismatch raises rather
    # than adopting, so no prompt is ever sent to an unverified process.
    if backend == "mlx-lm":
        if not _verify_or_adopt(MLX_LM_HOST + "/v1/models", MLX_LM_MODEL):
            start_mlx_lm()
        else:
            logger.info("mlx-lm already running")
        _warmup_model(MLX_LM_MODEL)
    elif backend == "omlx":
        if not _verify_or_adopt(OMLX_HOST + "/v1/models", OMLX_MODEL, omlx=True):
            start_omlx()
        else:
            logger.info("oMLX already running")
        _warmup_model(OMLX_MODEL, host=OMLX_HOST, api_key=_omlx_api_key())
    elif backend == "vllm-mlx":
        if not _verify_or_adopt(VLLM_MLX_HOST + "/v1/models", VLLM_MLX_MODEL):
            start_vllm_mlx()
        else:
            logger.info("vllm-mlx already running")
        _warmup_model(VLLM_MLX_MODEL, host=VLLM_MLX_HOST)
    elif backend == "mira-mlx":
        target_model = model or MIRA_MLX_MODEL
        # Compare the live listener against the TARGET model, not the default —
        # a switch to the same backend/different model must restart, not adopt
        # whatever is already up (see feedback_backend_switch_self_stop).
        if not _verify_or_adopt(MIRA_MLX_HOST + "/v1/models", target_model):
            start_mira_mlx(target_model)
        else:
            logger.info("mira-mlx already running")
        _warmup_model(target_model, host=MIRA_MLX_HOST)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Must be one of: {list(KNOWN_BACKENDS)}"
        )


def stop_omlx() -> None:
    global _omlx_proc
    if _omlx_proc and _omlx_proc.poll() is None:
        _omlx_proc.terminate()
        try:
            _omlx_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _omlx_proc.kill()
        _omlx_proc = None


def start_omlx() -> None:
    global _omlx_proc
    _omlx_proc = subprocess.Popen(
        [OMLX_CLI, "serve", "--port", str(OMLX_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_ready(OMLX_HOST + "/v1/models", timeout=60, omlx=True)


def _stop_all_backends() -> None:
    """Stop every Popen-managed backend, including the one being switched to.

    They all share port 8080 by design, so exactly one may run at a time; and
    stopping the target too is deliberate, since a same-backend different-model
    switch that skips it orphans the old process and then misreports which model
    is live (feedback_backend_switch_self_stop).
    """
    stop_mlx_lm()
    stop_omlx()
    stop_vllm_mlx()
    stop_mira_mlx()


_STARTERS = {
    "mlx-lm": lambda *a: start_mlx_lm(*a),
    "omlx": lambda *a: start_omlx(),
    "vllm-mlx": lambda *a: start_vllm_mlx(*a),
    "mira-mlx": lambda *a: start_mira_mlx(*a),
}


def switch_to(target: str) -> dict:
    """Stop the running inference server and start the target one.

    Returns the preset config dict; caller is responsible for updating the
    orchestrator and the server's runtime state.

    Raises ValueError for unknown backends, TimeoutError if the new server
    does not respond within its startup window.
    """
    if target not in PRESETS:
        raise ValueError(f"Unknown backend {target!r}. Must be one of: {list(PRESETS)}")
    logger.info("Switching backend to %s", target)
    _stop_all_backends()
    _STARTERS[target]()
    logger.info("Backend switch to %s complete", target)
    return PRESETS[target]


def get_preset_for(backend: str, model_id: str) -> dict:
    """Compute the runtime preset dict for an already-running backend/model,
    without starting or stopping anything. Used both by switch_to_model()'s own
    return value and by server.py's reconciliation path (a switch that raised
    from a slow-but-successful readiness check can still call this once the
    backend is confirmed healthy, to resync without repeating the stop/start)."""
    if backend == "mlx-lm":
        return {"backend": "mlx-lm", "model": model_id, "host": MLX_LM_HOST, "context_window": MLX_LM_CONTEXT}
    elif backend == "mira-mlx":
        from core import hardware
        actual_context = hardware.derive_context_window(
            model_id,
            requested_context=MIRA_MLX_CONTEXT,
            kv_bits=MIRA_MLX_KV_BITS,
            kv_group_size=MIRA_MLX_KV_GROUP_SIZE,
            resident_expert_fraction=resolve_offload_fraction(model_id),
        )
        return {"backend": "mira-mlx", "model": model_id, "host": MIRA_MLX_HOST, "context_window": actual_context}
    else:
        preset = dict(PRESETS[backend])
        preset["model"] = model_id
        return preset


def switch_to_model(backend: str, model_id: str) -> dict:
    """Switch to a specific model on the given backend.

    Every backend except omlx takes its model at launch, so the switch is a
    stop-and-restart. omlx serves a whole library from one process and picks the
    model per request instead. Returns an updated preset dict.
    """
    if backend not in _STARTERS:
        raise ValueError(
            f"Unknown backend {backend!r}. Must be one of: {list(_STARTERS)}"
        )
    logger.info("Switching to model %s on %s", model_id, backend)
    # Stops the target's own process too, not just the others: switching to the
    # same backend with a different model must restart it, or the old process is
    # orphaned and the reported model diverges from the live one
    # (feedback_backend_switch_self_stop).
    _stop_all_backends()
    if backend == "omlx":
        # omlx serves its whole model library from one process, so the model is
        # selected per request rather than at launch.
        _STARTERS[backend]()
    else:
        _STARTERS[backend](model_id)
    preset = get_preset_for(backend, model_id)
    logger.info("Model switch to %s/%s complete", backend, model_id)
    return preset


# Every backend Mira can start. PRESETS omits mlx-lm (benched out of the default
# picker) but the code path exists, so the library view still reports it.
KNOWN_BACKENDS = ("mira-mlx", "omlx", "mlx-lm", "vllm-mlx")


def list_models() -> dict:
    """Return locally available models across all backends.

    `backends` is the real answer: one entry per backend Mira knows how to
    start, saying whether it is installed and what it has. The flat `mlx_lm` and
    `ollama` keys are the original wire shape and are kept so older clients keep
    decoding. `ollama` is now always empty: the backend was retired on
    2026-08-01, and the key stays only so an app build that still decodes it
    does not fail on a missing field.
    """
    from core.models_api import list_backend_status
    from dataclasses import asdict

    statuses = list_backend_status(KNOWN_BACKENDS)
    by_name = {s.backend: s for s in statuses}

    return {
        "backends": [asdict(s) for s in statuses],
        # Back-compat. mlx_lm and mira-mlx serve the same HuggingFace cache, so
        # reporting mlx-lm's view here loses nothing.
        "mlx_lm": [asdict(m) for m in by_name["mlx-lm"].models],
        "ollama": [],
    }


def _default_backends() -> list:
    """Generate a preset list from the hardcoded PRESETS dict (fallback when mira.yaml has no `backends:`)."""
    label_map = {
        "omlx": "Qwen3.6 35B (omlx)",
        "mlx-lm": "Qwen3.6 35B (mlx-lm)",
    }
    return [
        {
            "id": k,
            "label": label_map.get(k, v["model"].split("/")[-1]),
            "backend": v["backend"],
            "model": v["model"],
            "context_window": v["context_window"],
        }
        for k, v in PRESETS.items()
    ]


def get_backends(active_backend: str, active_model: str) -> list:
    """Return the configured presets, annotated with whether each can be selected.

    Two things beyond the raw mira.yaml list:

    - Every entry carries `available` and, when false, a `detail` saying why.
      A preset whose backend is not installed, or whose model is not on disk,
      cannot be switched to; the client should not offer it as if it could.
      `mira.yaml` is a wish list, and nothing was checking it against reality.

    - The running (backend, model) pair is ALWAYS present and flagged active,
      even when no preset declares it. It routinely does not: the default has
      been mira-mlx + Qwen3.6-35B-A3B-4bit since 2026-07-09 and no preset pairs
      those two, so every entry came back active=False and clients had no row
      for the model actually serving them, nor any way back to it.
    """
    from core.config import BACKENDS
    from core.models_api import backend_status

    entries = BACKENDS if BACKENDS else _default_backends()

    # One probe per distinct backend, not one per preset — four of the seven
    # presets share a backend.
    probes = {}

    def _probe(backend: str):
        if backend not in probes:
            probes[backend] = backend_status(backend)
        return probes[backend]

    result = []
    found_active = False
    for p in entries:
        entry = dict(p)
        is_active = (p["backend"] == active_backend and p["model"] == active_model)
        found_active = found_active or is_active
        entry["active"] = is_active

        status = _probe(p["backend"])
        if not status.available:
            entry["available"], entry["detail"] = False, status.detail
        elif any(_model_matches(m.model_id, p["model"]) for m in status.models):
            entry["available"], entry["detail"] = True, ""
        elif is_active:
            # It is serving right now, so whatever the scan thinks, it exists.
            entry["available"], entry["detail"] = True, ""
        else:
            entry["available"] = False
            entry["detail"] = status.detail or f"{p['model']} is not installed for {p['backend']}"
        result.append(entry)

    if not found_active and active_backend and active_model:
        result.insert(0, {
            "id": f"{active_backend}-{_model_basename(active_model)}",
            "label": _model_basename(active_model),
            "backend": active_backend,
            "model": active_model,
            "context_window": PRESETS.get(active_backend, {}).get("context_window", CONTEXT_WINDOW),
            "active": True,
            "available": True,
            "detail": "",
        })

    return result


def pull_mlx_model(model_id: str, progress_cb=None) -> None:
    """Download a model from HuggingFace into the local mlx cache.

    progress_cb(downloaded_gb, total_gb, percent) is called periodically.
    Raises on network or auth errors.
    """
    from huggingface_hub import snapshot_download
    import os

    def _tqdm_callback(tqdm_obj):
        if progress_cb is None:
            return
        try:
            downloaded = tqdm_obj.n / (1024 ** 3)
            total = (tqdm_obj.total or 0) / (1024 ** 3)
            pct = int(tqdm_obj.n / tqdm_obj.total * 100) if tqdm_obj.total else 0
            progress_cb(downloaded, total, pct)
        except Exception:
            pass

    snapshot_download(
        repo_id=model_id,
        tqdm_class=None,
        local_files_only=False,
    )
    if progress_cb:
        progress_cb(0, 0, 100)
