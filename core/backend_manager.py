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
MLX_LM_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
MLX_LM_CONTEXT = 65536

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
        "embed_backend": "ollama",
        "embed_host": OLLAMA_HOST,
        "context_window": MLX_LM_CONTEXT,
    },
    "omlx": {
        "backend": "omlx",
        "model": OMLX_MODEL,
        "host": OMLX_HOST,
        "embed_backend": "omlx",
        "embed_host": OMLX_HOST,
        "context_window": OMLX_CONTEXT,
    },
    "ollama": {
        "backend": "ollama",
        "model": OLLAMA_MODEL,
        "host": OLLAMA_HOST,
        "embed_backend": "ollama",
        "embed_host": OLLAMA_HOST,
        "context_window": OLLAMA_CONTEXT,
    },
}

_mlx_lm_proc = None
_omlx_proc = None


def start_mlx_lm() -> None:
    global _mlx_lm_proc
    _mlx_lm_proc = subprocess.Popen(
        [
            MLX_LM_CLI,
            "--model", MLX_LM_MODEL,
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
        else:
            urllib.request.urlopen(OLLAMA_HOST + "/api/version", timeout=2)
        return True
    except Exception:
        return False


def ensure_backend_running(backend: str) -> None:
    """Start the backend if not already reachable. Safe to call on every startup."""
    if backend == "mlx-lm":
        try:
            urllib.request.urlopen(MLX_LM_HOST + "/v1/models", timeout=2)
            logger.info("mlx-lm already running")
            return
        except Exception:
            pass
        start_mlx_lm()
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
        raise ValueError(f"Unknown backend {target!r}. Must be 'ollama', 'mlx-lm', or 'omlx'.")
    logger.info("Switching backend to %s", target)
    if target == "mlx-lm":
        stop_ollama()
        stop_omlx()
        start_mlx_lm()
    elif target == "omlx":
        stop_ollama()
        stop_mlx_lm()
        start_omlx()
    else:
        stop_mlx_lm()
        stop_omlx()
        start_ollama()
    logger.info("Backend switch to %s complete", target)
    return PRESETS[target]
