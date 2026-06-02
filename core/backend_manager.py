"""Manages start/stop of inference server processes for runtime backend switching."""

import json
import logging
import os
import subprocess
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

MLX_LM_CLI = "/Users/miguel/.local/bin/mlx_lm.server"
MLX_LM_PORT = 8080
MLX_LM_HOST = f"http://localhost:{MLX_LM_PORT}"
MLX_LM_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
MLX_LM_CONTEXT = 65536

DFLASH_CLI = "/Users/miguel/Documents/Projects/mira-core/.venv/bin/dflash"
DFLASH_PORT = 8080
DFLASH_HOST = f"http://localhost:{DFLASH_PORT}"
DFLASH_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
DFLASH_DRAFT_MODEL = "z-lab/Qwen3.6-35B-A3B-DFlash"  # validated pair from dflash-mlx README
DFLASH_CONTEXT = 65536

OMLX_CLI = "/Applications/oMLX.app/Contents/MacOS/omlx-cli"
OMLX_PORT = 8080
OMLX_HOST = f"http://localhost:{OMLX_PORT}"
OMLX_MODEL = "Qwen3.6-35B-A3B"
OMLX_CONTEXT = 262144

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:26b"
OLLAMA_CONTEXT = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", 262144))

PRESETS = {
    "mlx-lm": {
        "backend": "mlx-lm",
        "model": MLX_LM_MODEL,
        "host": MLX_LM_HOST,
        "context_window": MLX_LM_CONTEXT,
    },
    "dflash": {
        "backend": "dflash",
        "model": DFLASH_MODEL,
        "host": DFLASH_HOST,
        "context_window": DFLASH_CONTEXT,
    },
    "omlx": {
        "backend": "omlx",
        "model": OMLX_MODEL,
        "host": OMLX_HOST,
        "context_window": OMLX_CONTEXT,
    },
    "ollama": {
        "backend": "ollama",
        "model": OLLAMA_MODEL,
        "host": OLLAMA_HOST,
        "context_window": OLLAMA_CONTEXT,
    },
}

_mlx_lm_proc = None
_dflash_proc = None
_omlx_proc = None


def start_mlx_lm(model: str = MLX_LM_MODEL) -> None:
    global _mlx_lm_proc
    _mlx_lm_proc = subprocess.Popen(
        [
            MLX_LM_CLI,
            "--model", model,
            "--host", "127.0.0.1",
            "--port", str(MLX_LM_PORT),
            "--max-tokens", "4096",
            "--chat-template-args", '{"enable_thinking": false}',
            "--prompt-cache-bytes", "3G",
            "--decode-concurrency", "1",
            "--prefill-step-size", "512",
            "--trust-remote-code",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_ready(MLX_LM_HOST + "/v1/models", timeout=120)


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
    _dflash_proc = subprocess.Popen(
        [
            DFLASH_CLI, "serve",
            "--model", model,
            "--draft-model", DFLASH_DRAFT_MODEL,
            "--host", "127.0.0.1",
            "--port", str(DFLASH_PORT),
            "--max-tokens", "16384",
            "--chat-template-args", '{"enable_thinking": false}',
            "--prefix-cache",
            "--prefill-step-size", "512",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_ready(DFLASH_HOST + "/v1/models", timeout=120)


def stop_dflash() -> None:
    global _dflash_proc
    if _dflash_proc and _dflash_proc.poll() is None:
        _dflash_proc.terminate()
        try:
            _dflash_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _dflash_proc.kill()
        _dflash_proc = None


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
        else:
            urllib.request.urlopen(OLLAMA_HOST + "/api/version", timeout=2)
        return True
    except Exception:
        return False


def _warmup_model(model: str, host: str = MLX_LM_HOST) -> None:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        host + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        logger.info("warming up model %s…", model)
        urllib.request.urlopen(req, timeout=60)
        logger.info("model warm")
    except Exception as exc:
        logger.warning("warmup failed (non-fatal): %s", exc)


def ensure_backend_running(backend: str) -> None:
    """Start the backend if not already reachable. Safe to call on every startup."""
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
            return
        except Exception:
            pass
        start_omlx()
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
        start_dflash()
    elif target == "mlx-lm":
        stop_ollama()
        stop_omlx()
        stop_dflash()
        start_mlx_lm()
    elif target == "omlx":
        stop_ollama()
        stop_mlx_lm()
        stop_dflash()
        start_omlx()
    else:
        stop_mlx_lm()
        stop_dflash()
        stop_omlx()
        start_ollama()
    logger.info("Backend switch to %s complete", target)
    return PRESETS[target]


def switch_to_model(backend: str, model_id: str) -> dict:
    """Switch to a specific model on the given backend.

    For mlx-lm: stops the current server and starts a new one with model_id.
    For ollama: switches to ollama and uses model_id as the active model.
    Returns an updated preset dict.
    """
    if backend not in PRESETS:
        raise ValueError(f"Unknown backend {backend!r}.")
    logger.info("Switching to model %s on %s", model_id, backend)
    if backend == "dflash":
        stop_ollama()
        stop_omlx()
        stop_mlx_lm()
        start_dflash(model_id)
    elif backend == "mlx-lm":
        stop_ollama()
        stop_omlx()
        stop_dflash()
        start_mlx_lm(model_id)
    elif backend == "omlx":
        stop_ollama()
        stop_mlx_lm()
        stop_dflash()
        start_omlx()
    else:
        stop_mlx_lm()
        stop_dflash()
        stop_omlx()
        start_ollama()
    preset = dict(PRESETS[backend])
    preset["model"] = model_id
    logger.info("Model switch to %s/%s complete", backend, model_id)
    return preset


def list_models() -> dict:
    """Return locally available models across all backends."""
    from core.models_api import list_mlx_models, list_ollama_models
    from dataclasses import asdict

    mlx = [asdict(m) for m in list_mlx_models()]
    oll = [asdict(m) for m in list_ollama_models()]
    return {"mlx_lm": mlx, "ollama": oll}


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
