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
from core.config import DFLASH_CLI as _DFLASH_CLI_PATH
from core.config import DFLASH_DIAGNOSTICS, MLX_LM_CLI as _MLX_LM_CLI_PATH
from core.config import MIRA_MLX_KV_BITS, MIRA_MLX_KV_GROUP_SIZE
from core.config import MIRA_MLX_PROFILE_EXPERTS, MIRA_MLX_EXPERT_PROFILE_PATH
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
MIRA_MLX_MODEL = "mlx-community/Ministral-3-14B-Instruct-2512-4bit"
# Follows mira.yaml's `context_window:` (config.CONTEXT_WINDOW) so the requested
# value actually reaches --max-kv-size below, instead of silently staying at a
# stale flat default (found 2026-07-18 while testing a context_window bump).
MIRA_MLX_CONTEXT = CONTEXT_WINDOW
MIRA_MLX_CACHE_DIR = DB_PATH.parent / "mira_mlx_cache"

DFLASH_CLI = _DFLASH_CLI_PATH
DFLASH_PORT = 8080
DFLASH_HOST = f"http://localhost:{DFLASH_PORT}"
DFLASH_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
DFLASH_CONTEXT = 65536
# dflash's prefix/KV cache (regenerable). dflash is a secondary large-context fallback
# here, so we trim this on stop rather than let it grow to tens of GB.
DFLASH_CACHE_DIR = Path.home() / ".cache" / "dflash"

# Validated target → draft pairings from the dflash-mlx README
DFLASH_DRAFT_MODELS = {
    "mlx-community/Qwen3.6-35B-A3B-4bit": "z-lab/Qwen3.6-35B-A3B-DFlash",
    "mlx-community/gemma-4-26b-a4b-it-4bit": "z-lab/gemma-4-26B-A4B-it-DFlash",
}

OMLX_CLI = _OMLX_CLI_PATH
OMLX_PORT = 8080
OMLX_HOST = f"http://localhost:{OMLX_PORT}"
OMLX_MODEL = "Qwen3.6-35B-A3B"
OMLX_CONTEXT = 131072

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:26b"
OLLAMA_CONTEXT = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", 262144))

VLLM_MLX_CLI = _VLLM_MLX_CLI_PATH
VLLM_MLX_PORT = 8080
VLLM_MLX_HOST = f"http://localhost:{VLLM_MLX_PORT}"
VLLM_MLX_MODEL = "mlx-community/Ministral-3-14B-Instruct-2512-4bit"
VLLM_MLX_CONTEXT = 65536

PRESETS = {
    # mlx-lm is benched out (architecture gap) — not offered in the default picker.
    # The backend code path remains for anyone who configures it explicitly in mira.yaml.
    "dflash": {
        "backend": "dflash",
        "model": DFLASH_MODEL,
        "host": DFLASH_HOST,
        "context_window": DFLASH_CONTEXT,
        "vision": False,
    },
    "omlx": {
        "backend": "omlx",
        "model": OMLX_MODEL,
        "host": OMLX_HOST,
        "context_window": OMLX_CONTEXT,
        "vision": True,
    },
    "ollama": {
        "backend": "ollama",
        "model": OLLAMA_MODEL,
        "host": OLLAMA_HOST,
        "context_window": OLLAMA_CONTEXT,
        "vision": False,
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
        "vision": False,
    },
}

_mlx_lm_proc = None
_dflash_proc = None
_omlx_proc = None
_vllm_mlx_proc = None
_mira_mlx_proc = None


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

    ok, reason = hardware.fits_in_memory(model, kv_bits=MIRA_MLX_KV_BITS, kv_group_size=MIRA_MLX_KV_GROUP_SIZE)
    if not ok:
        raise RuntimeError(f"mira-mlx preflight check failed: {reason}")

    # Derived per-machine, not a flat constant: a fixed cache-pool size that's
    # comfortable on a 32GB Mac can itself exceed total RAM on an 8-16GB one
    # (found 2026-07-09 — a fixed 3GB cap was smaller than a single long
    # conversation's ~3.3GB KV cache entry, silently evicting it every time).
    prompt_cache_max_bytes = hardware.derive_prompt_cache_max_bytes(
        model, kv_bits=MIRA_MLX_KV_BITS, kv_group_size=MIRA_MLX_KV_GROUP_SIZE
    )
    context_window = hardware.derive_context_window(
        model,
        requested_context=MIRA_MLX_CONTEXT,
        kv_bits=MIRA_MLX_KV_BITS,
        kv_group_size=MIRA_MLX_KV_GROUP_SIZE,
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
        # restarts (specs/mira-mlx-cache-persistence.md).
        "--disk-cache-dir", str(MIRA_MLX_CACHE_DIR),
        "--disk-cache-max-bytes", str(disk_cache_max_bytes),
    ]
    if MIRA_MLX_KV_BITS is not None:
        args += ["--kv-bits", str(MIRA_MLX_KV_BITS), "--kv-group-size", str(MIRA_MLX_KV_GROUP_SIZE)]
    if MIRA_MLX_PROFILE_EXPERTS:
        args += ["--profile-experts"]
        if MIRA_MLX_EXPERT_PROFILE_PATH:
            args += ["--expert-profile-path", MIRA_MLX_EXPERT_PROFILE_PATH]

    _mira_mlx_proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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


def start_dflash(model: str = DFLASH_MODEL) -> None:
    global _dflash_proc
    draft_model = DFLASH_DRAFT_MODELS.get(model, DFLASH_DRAFT_MODELS[DFLASH_MODEL])
    args = [
        DFLASH_CLI, "serve",
        "--model", model,
        "--draft-model", draft_model,
        "--host", "127.0.0.1",
        "--port", str(DFLASH_PORT),
        "--max-tokens", "16384",
        "--prefix-cache",
        "--prefill-step-size", str(PREFILL_STEP_SIZE),
    ]
    # Qwen3 requires thinking mode disabled via chat template args
    if "Qwen3" in model or "qwen3" in model.lower():
        args += ["--chat-template-args", '{"enable_thinking": false}']
    if DFLASH_DIAGNOSTICS != "off":
        args += ["--diagnostics", DFLASH_DIAGNOSTICS]
    _dflash_proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_ready(DFLASH_HOST + "/v1/models", timeout=120)


def _clear_dflash_cache() -> None:
    """Delete dflash's prefix/KV cache (regenerable). Trades a cold first-prefill on the
    next dflash session for not letting the cache grow to tens of GB. Appropriate because
    dflash is a secondary/large-context fallback here, not the primary server."""
    cache = DFLASH_CACHE_DIR / "prefix_l2"
    if cache.exists():
        try:
            shutil.rmtree(cache, ignore_errors=True)
            logger.info("Cleared dflash prefix cache at %s", cache)
        except Exception as e:
            logger.warning("Failed to clear dflash cache %s: %s", cache, e)


def stop_dflash() -> None:
    global _dflash_proc
    if _dflash_proc and _dflash_proc.poll() is None:
        _dflash_proc.terminate()
        try:
            _dflash_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _dflash_proc.kill()
        _dflash_proc = None
        # Reclaim the prefix cache now that dflash is down (it rebuilds on next use).
        _clear_dflash_cache()


def restart_dflash_if_dead(model: str = DFLASH_MODEL) -> None:
    """Restart dflash if the process has exited or the HTTP endpoint is unreachable.

    Called before each LLM request so OOM crashes are recovered transparently.
    """
    global _dflash_proc
    process_dead = _dflash_proc is None or _dflash_proc.poll() is not None
    if not process_dead:
        try:
            urllib.request.urlopen(DFLASH_HOST + "/v1/models", timeout=2)
            return  # alive and reachable
        except Exception:
            process_dead = True
    logger.warning("dflash process dead or unreachable — restarting with model %s", model)
    _dflash_proc = None
    start_dflash(model)


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


def _wait_for_ready(url: str, timeout: int = 60, *, omlx: bool = False) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if omlx:
                _omlx_request("/v1/models")
            else:
                urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Server did not become ready at {url} after {timeout}s")


def is_backend_ready(backend: str) -> bool:
    """Return True if the inference backend is reachable and has the model available."""
    try:
        if backend == "omlx":
            _omlx_request("/v1/models")
        elif backend == "mlx-lm":
            urllib.request.urlopen(MLX_LM_HOST + "/v1/models", timeout=2)
        elif backend == "dflash":
            urllib.request.urlopen(DFLASH_HOST + "/v1/models", timeout=2)
        elif backend == "vllm-mlx":
            urllib.request.urlopen(VLLM_MLX_HOST + "/v1/models", timeout=2)
        elif backend == "mira-mlx":
            urllib.request.urlopen(MIRA_MLX_HOST + "/v1/models", timeout=2)
        else:
            urllib.request.urlopen(OLLAMA_HOST + "/api/version", timeout=2)
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
    if backend == "dflash":
        try:
            urllib.request.urlopen(DFLASH_HOST + "/v1/models", timeout=2)
            logger.info("dflash already running")
        except Exception:
            start_dflash()
        _warmup_model(DFLASH_MODEL, host=DFLASH_HOST)
    elif backend == "mlx-lm":
        try:
            urllib.request.urlopen(MLX_LM_HOST + "/v1/models", timeout=2)
            logger.info("mlx-lm already running")
        except Exception:
            start_mlx_lm()
        _warmup_model(MLX_LM_MODEL)
    elif backend == "omlx":
        try:
            _omlx_request("/v1/models")
            logger.info("oMLX already running")
        except Exception:
            start_omlx()
        _warmup_model(OMLX_MODEL, host=OMLX_HOST, api_key=_omlx_api_key())
    elif backend == "vllm-mlx":
        try:
            urllib.request.urlopen(VLLM_MLX_HOST + "/v1/models", timeout=2)
            logger.info("vllm-mlx already running")
        except Exception:
            start_vllm_mlx()
        _warmup_model(VLLM_MLX_MODEL, host=VLLM_MLX_HOST)
    elif backend == "mira-mlx":
        target_model = model or MIRA_MLX_MODEL
        try:
            urllib.request.urlopen(MIRA_MLX_HOST + "/v1/models", timeout=2)
            logger.info("mira-mlx already running")
        except Exception:
            start_mira_mlx(target_model)
        _warmup_model(target_model, host=MIRA_MLX_HOST)
    else:
        try:
            urllib.request.urlopen(OLLAMA_HOST + "/api/version", timeout=2)
            logger.info("Ollama already running")
            return
        except Exception:
            pass
        start_ollama()


def stop_ollama() -> None:
    subprocess.run(["osascript", "-e", 'quit app "Ollama"'], capture_output=True)
    time.sleep(2)


def start_ollama() -> None:
    subprocess.run(["open", "-a", "Ollama"])
    _wait_for_ready(OLLAMA_HOST + "/api/version", timeout=30)


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
    if target == "dflash":
        stop_ollama()
        stop_omlx()
        stop_mlx_lm()
        stop_vllm_mlx()
        stop_mira_mlx()
        start_dflash()
    elif target == "mlx-lm":
        stop_ollama()
        stop_omlx()
        stop_dflash()
        stop_vllm_mlx()
        stop_mira_mlx()
        start_mlx_lm()
    elif target == "omlx":
        stop_ollama()
        stop_mlx_lm()
        stop_dflash()
        stop_vllm_mlx()
        stop_mira_mlx()
        start_omlx()
    elif target == "vllm-mlx":
        stop_ollama()
        stop_mlx_lm()
        stop_dflash()
        stop_omlx()
        stop_mira_mlx()
        start_vllm_mlx()
    elif target == "mira-mlx":
        stop_ollama()
        stop_mlx_lm()
        stop_dflash()
        stop_omlx()
        stop_vllm_mlx()
        start_mira_mlx()
    else:
        stop_mlx_lm()
        stop_dflash()
        stop_omlx()
        stop_vllm_mlx()
        stop_mira_mlx()
        start_ollama()
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
        )
        return {"backend": "mira-mlx", "model": model_id, "host": MIRA_MLX_HOST, "context_window": actual_context}
    else:
        preset = dict(PRESETS[backend])
        preset["model"] = model_id
        return preset


def switch_to_model(backend: str, model_id: str) -> dict:
    """Switch to a specific model on the given backend.

    For mlx-lm: stops the current server and starts a new one with model_id.
    For ollama: switches to ollama and uses model_id as the active model.
    Returns an updated preset dict.
    """
    if backend not in PRESETS and backend not in ("mlx-lm", "mira-mlx"):
        raise ValueError(f"Unknown backend {backend!r}.")
    logger.info("Switching to model %s on %s", model_id, backend)
    if backend == "dflash":
        stop_ollama()
        stop_omlx()
        stop_mlx_lm()
        stop_vllm_mlx()
        stop_mira_mlx()
        stop_dflash()
        start_dflash(model_id)
    elif backend == "mlx-lm":
        stop_ollama()
        stop_omlx()
        stop_dflash()
        stop_vllm_mlx()
        stop_mira_mlx()
        stop_mlx_lm()
        start_mlx_lm(model_id)
    elif backend == "omlx":
        stop_ollama()
        stop_mlx_lm()
        stop_dflash()
        stop_vllm_mlx()
        stop_mira_mlx()
        stop_omlx()
        start_omlx()
    elif backend == "vllm-mlx":
        stop_ollama()
        stop_mlx_lm()
        stop_dflash()
        stop_omlx()
        stop_mira_mlx()
        stop_vllm_mlx()
        start_vllm_mlx(model_id)
    elif backend == "mira-mlx":
        stop_ollama()
        stop_mlx_lm()
        stop_dflash()
        stop_omlx()
        stop_vllm_mlx()
        stop_mira_mlx()
        start_mira_mlx(model_id)
    else:
        stop_mlx_lm()
        stop_dflash()
        stop_omlx()
        stop_vllm_mlx()
        stop_mira_mlx()
        start_ollama()
    preset = get_preset_for(backend, model_id)
    logger.info("Model switch to %s/%s complete", backend, model_id)
    return preset


def list_models() -> dict:
    """Return locally available models across all backends."""
    from core.models_api import list_mlx_models, list_ollama_models
    from dataclasses import asdict

    mlx = [asdict(m) for m in list_mlx_models()]
    oll = [asdict(m) for m in list_ollama_models()]
    return {"mlx_lm": mlx, "ollama": oll}


def _default_backends() -> list:
    """Generate a preset list from the hardcoded PRESETS dict (fallback when mira.yaml has no `backends:`)."""
    label_map = {
        "omlx": "Qwen3.6 35B (omlx)",
        "dflash": "Qwen3.6 35B (dFlash)",
        "mlx-lm": "Qwen3.6 35B (mlx-lm)",
        "ollama": "Gemma 4 26B (ollama)",
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
    """Return the configured preset list with an `active` flag on the matching entry."""
    from core.config import BACKENDS
    entries = BACKENDS if BACKENDS else _default_backends()
    result = []
    for p in entries:
        entry = dict(p)
        entry["active"] = (p["backend"] == active_backend and p["model"] == active_model)
        result.append(entry)
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
