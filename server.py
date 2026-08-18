"""FastAPI server for Mira's web and app clients."""

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import core.db as db
import core.file_handler as file_handler
from core.config import VERBOSE_DEFAULT, COMPRESS_THRESHOLD, COMPRESS_KEEP_RECENT, MODEL_NAME, BACKEND, BACKEND_HOST, CONTEXT_WINDOW, AUTH_TOKEN, ALLOWED_SOURCE_CIDRS, ALLOWED_HOSTS, MIN_TOKEN_LENGTH, DISK_PROMPT_CACHE
from core.orchestrator import ChatOrchestrator
from core.session_manager import SessionManager
from core import backend_manager as _bm
from core.backend_manager import KNOWN_BACKENDS
from core import workspace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Per-conversation orchestrator pool. Replaces the old single global orchestrator so
# concurrent turns on different conversations no longer share mutable state; turns on
# the same conversation serialize through a per-conversation lock (see SessionManager).
sessions: SessionManager = None
_init_lock: asyncio.Lock = asyncio.Lock()
# request_id -> (conversation_id, cancel_event). The conv_id lets /cancel target a
# single conversation so one device's cancel never aborts another device's turn.
_active_cancels: Dict[str, tuple] = {}
_initialized = False
_backend_ready = False

def _detect_hardware() -> str:
    import subprocess as _sp, json as _json, re as _re
    try:
        out = _sp.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True, text=True, timeout=5
        ).stdout
        hw   = _json.loads(out)["SPHardwareDataType"][0]
        chip = hw.get("chip_type") or hw.get("cpu_type") or ""
        mem  = hw.get("physical_memory", "")
        if chip:
            return f"{chip} · {mem}" if mem else chip
    except Exception:
        pass
    try:
        out = _sp.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            m = _re.match(r"\s+(?:Chip|Processor Name):\s+(.+)", line)
            if m:
                chip  = m.group(1).strip()
                mem_m = _re.search(r"Memory:\s+(\S+\s*\S*)", out)
                mem   = mem_m.group(1).strip() if mem_m else ""
                return f"{chip} · {mem}" if mem else chip
    except Exception:
        pass
    return "Apple Silicon"

_HARDWARE = _detect_hardware()

# Runtime state — updated on every backend switch
_rt: Dict = {
    "backend": BACKEND,
    "model": MODEL_NAME,
    "host": BACKEND_HOST,
    "context_window": CONTEXT_WINDOW,
}


def _warn_orphaned_prompt_cache() -> None:
    """Say so, once per process, when the disabled disk cache has left files behind.

    Deliberately not a deletion. These are pure cache files with no user data in
    them, but they are also gigabytes inside the user's data directory, and Mira
    removing gigabytes on its own at startup is not a thing a server should do
    unasked. So it reports the size and the exact command, and stops there.

    Backend-independent on purpose: the leftovers are mira-mlx's, but they are
    just as dead when the configured backend is omlx, and that is precisely the
    case where nothing else would ever mention them.
    """
    if DISK_PROMPT_CACHE:
        return
    try:
        from core import hardware as hw

        count, nbytes = hw.orphaned_prompt_cache(_bm.MIRA_MLX_CACHE_DIR)
        if count:
            logger.warning(
                "Disk prompt cache is off, but %d files (%s) are still in %s. "
                "Nothing reads or deletes them. Remove with:  rm -rf %s",
                count, hw.format_bytes(nbytes), _bm.MIRA_MLX_CACHE_DIR, _bm.MIRA_MLX_CACHE_DIR,
            )
    except Exception as exc:  # noqa: BLE001 — advisory only, never blocks startup
        logger.debug("could not check for orphaned prompt cache (%s)", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sessions, _initialized, _backend_ready
    async with _init_lock:
        if not _initialized:
            _initialized = True
            db.init_db()
            _warn_orphaned_prompt_cache()
            
            # Per-conversation orchestrators are created lazily on first use (no
            # conversation is preloaded). Heavy RAG models are process-wide shared,
            # so each session is cheap.
            # Seed the pool with the SERVED context window, not the raw configured
            # request. derive_context_window caps a 65536 request to what actually
            # fits under the Metal wired ceiling (~39k on a 32GB Mac for the MTP
            # MoE); the engine already launches with that as --max-kv-size, and the
            # app-side compression trigger keys off _preset["context_window"], so
            # it must match — otherwise the engine caps the prompt while the app
            # keeps growing context past it. get_preset_for starts nothing.
            try:
                _served = _bm.get_preset_for(BACKEND, MODEL_NAME)
                _served_ctx = int(_served.get("context_window", CONTEXT_WINDOW))
            except Exception as exc:  # never block startup on a sizing estimate
                logger.warning("could not derive served context window at startup "
                               "(%s); using configured %d", exc, CONTEXT_WINDOW)
                _served_ctx = CONTEXT_WINDOW
            _rt["context_window"] = _served_ctx
            if _served_ctx != CONTEXT_WINDOW:
                logger.info("served context window %d (configured request %d capped to fit RAM)",
                            _served_ctx, CONTEXT_WINDOW)
            sessions = SessionManager(verbose=VERBOSE_DEFAULT, context_window=_served_ctx)
            logger.info(f"Initialized session pool — backend: {BACKEND}, model: {MODEL_NAME}")
            logger.info(f"{BACKEND} backend — model {MODEL_NAME} at {BACKEND_HOST}")
            _backend_ready = True
            # Auto-start the inference backend in a background thread so the app is
            # usable immediately (health returns 200) even while the model loads.
            # Skipped under tests: the warm-up would try (and time out) reaching a
            # backend that isn't running, leaving a lingering thread + noisy warning.
            if not os.getenv("MIRA_TESTING"):
                threading.Thread(
                    target=_bm.ensure_backend_running,
                    args=(BACKEND, MODEL_NAME),
                    daemon=True,
                ).start()
            from core import scheduler as _scheduler
            _scheduler.start()
            # Same guard as the backend thread above: without a backend running
            # there is nothing to watch, and under pytest it would poll a dead
            # port for the life of the suite.
            if not os.getenv("MIRA_TESTING"):
                from core import memory_watch as _memory_watch
                _memory_watch.start()

    yield

    # Under pytest, reset init state on shutdown so each module-scoped TestClient
    # (its own event loop) starts a fresh pool. Harmless in production (shutdown
    # only happens at process exit).
    if os.getenv("MIRA_TESTING"):
        _initialized = False
        sessions = None


app = FastAPI(title="Mira", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routes reachable without a token even when auth is enabled: liveness probe
# (clients poll it before they can authenticate) and the static web UI shell.
_AUTH_OPEN_PATHS = ("/health", "/")

_ALLOWED_HOST_NAMES = {h.strip().lower() for h in ALLOWED_HOSTS if h and h.strip()}

# Parsed once: the source-IP allowlist (defense-in-depth behind the off-host bind).
_ALLOWED_NETWORKS = []
for _cidr in ALLOWED_SOURCE_CIDRS:
    try:
        _ALLOWED_NETWORKS.append(ipaddress.ip_network(_cidr, strict=False))
    except ValueError:
        logger.warning("Ignoring invalid CIDR in ALLOWED_SOURCE_CIDRS: %r", _cidr)


# Host values accepted regardless of config. Anything else must be either an IP
# inside ALLOWED_SOURCE_CIDRS (so the tailnet address works without naming it) or
# listed explicitly in ALLOWED_HOSTS.
_ALWAYS_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _split_host(header: str) -> str:
    """Hostname from a Host header, port and IPv6 brackets removed."""
    h = (header or "").strip().lower()
    if h.startswith("["):                      # [::1]:8443
        return h[: h.index("]") + 1] if "]" in h else h
    return h.rsplit(":", 1)[0] if ":" in h else h


def _host_allowed(header: str | None) -> bool:
    """True when the Host header names this server by an address we expect.

    Blocks DNS rebinding: the attacker's domain resolves to 127.0.0.1, so the
    connection looks local, but the Host header still carries their domain.
    """
    if not header:
        return True          # no Host (HTTP/1.0, some native clients) — nothing to spoof
    name = _split_host(header)
    if name in _ALWAYS_ALLOWED_HOSTS or name in _ALLOWED_HOST_NAMES:
        return True
    # Bare IPs: accept any address we would already accept as a source, so the
    # discovered tailnet address works without being configured by hand.
    try:
        ip = ipaddress.ip_address(name.strip("[]"))
    except ValueError:
        return False
    return any(ip in net for net in _ALLOWED_NETWORKS)


def _origin_allowed(origin: str | None, host_header: str | None) -> bool:
    """True when a cross-site request may perform a state-changing call.

    Browsers set Origin on state-changing requests and scripts cannot forge it.
    Native apps and CLIs send none, which is why absence is allowed — they are
    not subject to the browser's ambient-credential problem in the first place.
    """
    if not origin:
        return True
    if origin == "null":         # sandboxed iframe / file:// — never ours
        return False
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https"):
        return False
    return _host_allowed(parsed.netloc) and (
        not host_header or _split_host(parsed.netloc) == _split_host(host_header)
    )


def _source_allowed(host: str | None) -> bool:
    """True if the peer IP falls in the allowlist. We use the real socket peer
    (request.client.host) and never trust X-Forwarded-For — there is no proxy."""
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(ip in net for net in _ALLOWED_NETWORKS)


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Two gates, in order: (1) source-IP allowlist — applies to every request
    including OPTIONS and the otherwise-open paths, so nothing leaks off the
    allowed networks; (2) shared-secret Bearer token on sensitive routes when a
    token is configured. The token check is OPTIONS-exempt and skips the open
    paths; the comparison is constant-time."""
    # Gate 0 — Host and Origin. Enforced ALWAYS, including when no token is set,
    # because this is the gate that protects the tokenless-loopback default: a
    # web page the user visits can reach 127.0.0.1 with the browser's ambient
    # credentials, and neither the source-IP allowlist nor a bearer token the
    # page cannot read will stop it. Host pinning blocks DNS rebinding; the
    # Origin check blocks ordinary cross-site form/fetch posts.
    if not _host_allowed(request.headers.get("host")):
        logger.warning("Rejected request with unexpected Host: %r",
                       request.headers.get("host"))
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS") and not _origin_allowed(
        request.headers.get("origin"), request.headers.get("host")
    ):
        logger.warning("Rejected cross-origin %s %s from Origin: %r",
                       request.method, request.url.path, request.headers.get("origin"))
        return JSONResponse({"detail": "Forbidden"}, status_code=403)

    # Gate 1 — source IP. Only enforced when a token is set (token-set ⇒ off-host).
    if AUTH_TOKEN and not _source_allowed(request.client.host if request.client else None):
        return JSONResponse({"detail": "Forbidden"}, status_code=403)

    # Gate 2 — Bearer token.
    if AUTH_TOKEN and request.method != "OPTIONS":
        path = request.url.path
        # Trailing slash matters: "/static" alone would also open "/staticfoo".
        is_open = path in _AUTH_OPEN_PATHS or path.startswith("/static/")
        presented = request.headers.get("authorization", "")
        # Encode before comparing: uvicorn decodes headers as latin-1, so a
        # non-ASCII byte in Authorization yields a str that compare_digest
        # rejects with TypeError — an unauthenticated way to raise a 500.
        if not is_open and not hmac.compare_digest(
            presented.encode("utf-8", "replace"), f"Bearer {AUTH_TOKEN}".encode("utf-8")
        ):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


def _ready():
    """FastAPI dependency — returns 503 until the session pool is initialised."""
    if sessions is None:
        raise HTTPException(status_code=503, detail="Server is starting up")


def _safe_path(path: str) -> Path:
    """Resolve path and raise 403 if it escapes the user's home directory."""
    resolved = Path(path).expanduser().resolve()
    home = Path.home()
    if resolved != home and home not in resolved.parents:
        raise HTTPException(status_code=403, detail="Path outside allowed root")
    return resolved


class VerboseRequest(BaseModel):
    enabled: bool


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    if not _backend_ready:
        return JSONResponse({"status": "starting"}, status_code=503)
    backend_ready = await asyncio.get_event_loop().run_in_executor(
        None, _bm.is_backend_ready, _rt["backend"]
    )
    return {"status": "ok", "backend_ready": backend_ready}


@app.get("/info")
async def info():
    """Return server/model metadata for display in the client app."""
    hardware = _HARDWARE
    result = {
        "model": _rt["model"],
        "backend": _rt["backend"],
        "host": _rt["host"],
        "context_window": _rt["context_window"],
        "compress_threshold": COMPRESS_THRESHOLD,
        "compress_keep_recent": COMPRESS_KEEP_RECENT,
        "hardware": hardware,
    }
    return result


_SYSTEM_MEMORY_TTL_S = 5.0
_system_memory_cache: dict = {"at": 0.0, "value": None}


async def _backend_system_memory() -> dict | None:
    """Relay the inference backend's system-memory advisory.

    The backend re-derives its memory ceiling from real system state every 30s on
    its idle loop, but it writes to DEVNULL and nothing here ever read its
    `/v1/stats`, so the advisory could not reach a client. This is that missing
    hop.

    Best-effort in every direction: only mira-mlx serves this, a backend that is
    down or older simply has no advisory, and a failure here must never affect
    the rest of `/hardware`. Cached briefly so polling this endpoint cannot turn
    into a flood against the backend.
    """
    now = time.time()
    if now - _system_memory_cache["at"] < _SYSTEM_MEMORY_TTL_S:
        return _system_memory_cache["value"]

    value = None
    try:
        import httpx

        base = BACKEND_HOST.rstrip("/")
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base}/v1/stats")
        if resp.status_code == 200:
            value = resp.json().get("system_memory")
    except Exception as exc:  # noqa: BLE001 - advisory only, never fatal
        logger.debug("could not read backend system memory (%s)", exc)

    _system_memory_cache["at"] = now
    _system_memory_cache["value"] = value
    return value


@app.get("/hardware")
async def hardware_info(_=Depends(_ready)):
    """RAM-aware sizing for the active model — why a given machine got the
    context window / cache budget it did (see core/hardware.py).

    `system_memory` is the live half: the other fields describe what this machine
    could do in principle, that one describes what it can do right now given
    whatever else is running.
    """
    from core import hardware as hw
    from core.config import MIRA_MLX_KV_BITS, MIRA_MLX_KV_GROUP_SIZE, PREFILL_STEP_SIZE

    total_ram = hw.get_total_ram_bytes()
    model = _rt["model"]
    kv_bytes_per_token = hw.estimate_kv_bytes_per_token(
        model, kv_bits=MIRA_MLX_KV_BITS, kv_group_size=MIRA_MLX_KV_GROUP_SIZE)
    model_bytes = hw.estimate_model_weight_bytes(model)
    return {
        "total_ram_gb": round(total_ram / hw.BYTES_PER_GB, 1),
        "model": model,
        "model_weight_gb": round(model_bytes / hw.BYTES_PER_GB, 1) if model_bytes else None,
        "kv_bytes_per_token": kv_bytes_per_token,
        "derived_context_window": hw.derive_context_window(
            model, total_ram, _rt["context_window"],
            kv_bits=MIRA_MLX_KV_BITS, kv_group_size=MIRA_MLX_KV_GROUP_SIZE,
            prefill_step_size=PREFILL_STEP_SIZE),
        "derived_prompt_cache_max_gb": round(
            hw.derive_prompt_cache_max_bytes(model, total_ram) / hw.BYTES_PER_GB, 1
        ),
        "active_context_window": _rt["context_window"],
        # 0 when the disk store is off, which it is by default — reporting what
        # the volume *could* afford while nothing is being written there reads as
        # a 40GB budget in use.
        "derived_disk_cache_max_gb": round(
            hw.derive_disk_cache_max_bytes(_bm.MIRA_MLX_CACHE_DIR) / hw.BYTES_PER_GB, 1
        ) if DISK_PROMPT_CACHE else 0.0,
        # None when the backend does not report one (not mira-mlx, still
        # starting, or an older build). A client must treat absence as "no
        # information", never as "everything is fine".
        "system_memory": await _backend_system_memory(),
    }


@app.get("/backend")
async def get_backend(_=Depends(_ready)):
    return {
        "backend": _rt["backend"],
        "model": _rt["model"],
        "host": _rt["host"],
        "context_window": _rt["context_window"],
    }


@app.get("/backends")
async def list_backends(_=Depends(_ready)):
    """Return configured backend presets (from mira.yaml `backends:` list) with active flag."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _bm.get_backends, _rt["backend"], _rt["model"])


@app.post("/backend")
async def switch_backend(request: Request, _=Depends(_ready)):
    global _backend_ready
    body = await request.json()
    target = body.get("backend", "")
    if target not in KNOWN_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"backend must be one of: {', '.join(KNOWN_BACKENDS)}",
        )
    if target == _rt["backend"]:
        return {"status": "ok", "backend": target, "message": "already active"}
    _backend_ready = False
    try:
        loop = asyncio.get_event_loop()
        preset = await loop.run_in_executor(None, _bm.switch_to, target)
        await sessions.reinitialize_all(
            backend=preset["backend"],
            model=preset["model"],
            host=preset["host"],
            context_window=preset["context_window"],
        )
        _rt.update(preset)
    except Exception as e:
        logger.error("Backend switch failed: %s", e)
        _backend_ready = True
        # Generic detail on purpose: backend errors embed absolute venv/HF-cache
        # paths and the username. The full error is in the server log.
        raise HTTPException(status_code=500, detail="Backend switch failed — see server logs")
    _backend_ready = True
    return {"status": "ok", "backend": _rt["backend"], "model": _rt["model"]}


@app.get("/models")
async def list_models():
    """Return all locally available models grouped by backend."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _bm.list_models)
    result["active"] = {"backend": _rt["backend"], "model_id": _rt["model"]}
    return result


async def _reconcile_stale_switch_failure(backend: str, model_id: str, grace_seconds: int = 60) -> Optional[dict]:
    """A switch can raise from a slow-but-successful readiness check — the
    target backend's own internal load can outlast backend_manager's external
    wait_for_ready timeout while still coming up healthy moments later
    (confirmed 2026-07-18: a Qwen3.6->Gemma4 switch under memory pressure hit
    this — see core/backend_manager.py's start_mira_mlx comment). Rather than
    leave `_rt` permanently pointed at the old model while the new one is
    actually live, poll briefly for the target to become ready before giving
    up. Returns the preset dict if it recovers within grace_seconds, else None."""
    loop = asyncio.get_event_loop()
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        try:
            ready = await loop.run_in_executor(None, _bm.is_backend_ready, backend)
        except Exception:
            ready = False
        if ready:
            return await loop.run_in_executor(None, _bm.get_preset_for, backend, model_id)
        await asyncio.sleep(2)
    return None


@app.post("/models/switch")
async def switch_model(request: Request, _=Depends(_ready)):
    global _backend_ready
    body = await request.json()
    backend = body.get("backend", "")
    model_id = body.get("model_id", "")
    if backend not in KNOWN_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"backend must be one of: {', '.join(KNOWN_BACKENDS)}",
        )
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if backend == _rt["backend"] and model_id == _rt["model"]:
        return {"status": "ok", "backend": backend, "model": model_id, "message": "already active"}
    _backend_ready = False
    try:
        loop = asyncio.get_event_loop()
        preset = await loop.run_in_executor(None, _bm.switch_to_model, backend, model_id)
        await sessions.reinitialize_all(
            backend=preset["backend"],
            model=preset["model"],
            host=preset["host"],
            context_window=preset["context_window"],
        )
        _rt.update(preset)
    except Exception as e:
        logger.error("Model switch to %s/%s failed: %s — checking whether it actually succeeded", backend, model_id, e)
        recovered_preset = await _reconcile_stale_switch_failure(backend, model_id)
        if recovered_preset is not None:
            logger.warning(
                "Model switch to %s/%s reported failure but the backend is actually healthy "
                "(slow cold load outlasted the readiness timeout) — resyncing instead of "
                "leaving state stale", backend, model_id,
            )
            await sessions.reinitialize_all(
                backend=recovered_preset["backend"], model=recovered_preset["model"],
                host=recovered_preset["host"], context_window=recovered_preset["context_window"],
            )
            _rt.update(recovered_preset)
            _backend_ready = True
            return {
                "status": "ok", "backend": _rt["backend"], "model": _rt["model"],
                "message": "recovered from a slow readiness check",
            }
        _backend_ready = True
        logger.error("Model switch failed: %s", e)
        raise HTTPException(status_code=500, detail="Model switch failed — see server logs")
    _backend_ready = True
    return {"status": "ok", "backend": _rt["backend"], "model": _rt["model"]}


@app.post("/models/pull")
async def pull_model(request: Request):
    """SSE stream that downloads a model from HuggingFace into the local cache."""
    body = await request.json()
    model_id = body.get("model_id", "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    async def event_stream():
        import queue as _queue
        q: _queue.Queue = _queue.Queue()

        def progress_cb(downloaded_gb, total_gb, pct):
            q.put({"downloaded_gb": round(downloaded_gb, 2), "total_gb": round(total_gb, 2), "percent": pct})

        def run_pull():
            try:
                _bm.pull_mlx_model(model_id, progress_cb=progress_cb)
                q.put(None)  # sentinel: done
            except Exception as exc:
                q.put({"error": str(exc)})

        loop = asyncio.get_event_loop()
        pull_future = loop.run_in_executor(None, run_pull)

        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=1))
            except Exception:
                if pull_future.done():
                    break
                yield {"data": json.dumps({"type": "progress", "percent": 0})}
                continue

            if item is None:
                yield {"data": json.dumps({"type": "done", "model_id": model_id})}
                break
            if "error" in item:
                yield {"data": json.dumps({"type": "error", "message": item["error"]})}
                break
            yield {"data": json.dumps({"type": "progress", **item})}

    return EventSourceResponse(event_stream())


@app.post("/cancel")
async def cancel(request: Request):
    """Abort in-progress turn(s). With a `conversation_id` in the JSON body, only
    that conversation's turns are cancelled; with no body, all are cancelled
    (back-compat for older clients)."""
    conv_id = ""
    try:
        body = await request.json()
        conv_id = (body or {}).get("conversation_id", "") or ""
    except Exception:
        conv_id = ""
    cancelled = 0
    for cid, ev in list(_active_cancels.values()):
        if not conv_id or cid == conv_id:
            ev.set()
            cancelled += 1
    return {"status": "cancelled", "count": cancelled}


def _rollback_to_last_user(history: List[Dict]) -> int:
    """Cut the in-memory history back to just before the last user message.

    Mutates in place: the orchestrator holds this list, and rebinding it here
    would leave the orchestrator prompting from the old one. Returns how many
    entries were removed. Cuts from the last `user` entry rather than a fixed
    count because a turn can leave tool messages behind it.
    """
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            removed = len(history) - i
            del history[i:]
            return removed
    return 0


@app.post("/chat")
async def chat(
    message: str = Form(...),
    conversation_id: str = Form(default=""),
    files: List[UploadFile] = File(default=[]),
    paths: List[str] = Form(default=[]),
    thinking_enabled: Optional[bool] = Form(default=None),
    github_tools_enabled: bool = Form(default=False),
    tools_enabled: bool = Form(default=True),
    # Set by a client that is re-sending a question because the last answer was
    # broken. Without it a retry appends: the question ends up in the database
    # twice with the failed reply between them, and every later turn is built on
    # top of both. The client cannot fix this itself — it cannot un-save what it
    # already sent — so the flag is the client telling the server which turn it
    # means to replace. Off by default: asking the same question twice on purpose
    # is a legitimate thing to do and must not silently delete history.
    retry: bool = Form(default=False),
    # Approval tokens for destructive actions the user confirmed in the client.
    # These come from the USER via the client UI, never from model output — that
    # is what makes the confirmation gate unforgeable by the model.
    approved_tokens: List[str] = Form(default=[]),
    _: None = Depends(_ready),
):
    """SSE endpoint — streams typed events from stream_chat() to the browser."""
    request_id = str(uuid.uuid4())
    cancel_event = threading.Event()

    # Resolve the target conversation and ensure its DB row exists, then get (or
    # lazily create) its dedicated orchestrator + lock from the pool.
    project = None
    if not conversation_id:
        # No ID supplied — always start a fresh conversation so callers without an
        # explicit ID never inherit another session.
        conv_id = db.create_conversation(sessions.model)
        fresh = True
    else:
        conv_id = conversation_id
        conv = db.get_conversation(conv_id)
        if conv:
            project = db.get_project(conv["project_id"]) if conv.get("project_id") else None
            fresh = False
        else:
            db.create_conversation(sessions.model, conv_id=conv_id)
            fresh = True
    orch, lock = await sessions.acquire(conv_id, project=project, fresh=fresh)
    # Register the cancel event against the resolved conversation so /cancel can
    # target this turn without touching other conversations' in-flight turns.
    _active_cancels[request_id] = (conv_id, cancel_event)

    attachments = []
    for upload in files:
        data = await upload.read()
        try:
            att = file_handler.load_file_bytes(workspace.safe_filename(upload.filename), data)
            attachments.append(att)
        except Exception as e:
            logger.warning(f"Could not process uploaded file '{upload.filename}': {e}")
            attachments.append({
                "type": "text", "name": upload.filename, "content": "",
                "warning": f"Could not process '{upload.filename}': {e}"
            })

    for path in paths:
        try:
            safe = _safe_path(path)
            att = file_handler.load_file(str(safe))
            attachments.append(att)
        except HTTPException as e:
            logger.warning(f"Rejected path '{path}': {e.detail}")
            attachments.append({
                "type": "text", "name": path, "content": "",
                "warning": f"Access denied: '{path}'"
            })
        except Exception as e:
            logger.warning(f"Could not load file at path '{path}': {e}")
            attachments.append({
                "type": "text", "name": path, "content": "",
                "warning": f"Could not load '{path}': {e}"
            })

    async def event_stream():
        # Hold the per-conversation lock for the whole turn: turns on the same
        # conversation serialize (no history corruption), different conversations
        # stream concurrently through their own locks.
        async with lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            # A retry replaces the previous turn instead of stacking on it. Done
            # under the lock and before the snapshot, so a turn in flight on this
            # conversation cannot have its history cut from under it, and so the
            # cancel rollback below measures the history the turn actually starts
            # from. Both stores are trimmed: the database is what a later session
            # reloads, and conversation_history is what this turn prompts with.
            if retry and not fresh:
                removed = db.drop_last_turn(conv_id)
                cut = _rollback_to_last_user(orch.conversation_history)
                if removed or cut:
                    logger.info(
                        "retry on %s: dropped %d saved message(s), %d in memory",
                        conv_id, removed, cut,
                    )

            snapshot = {"len": len(orch.conversation_history)}

            def produce():
                snapshot["len"] = len(orch.conversation_history)
                was_new_conv = orch._is_new_conv
                done_content = None
                thinking_content = None

                try:
                    for event in orch.stream_chat(message, attachments=attachments or None, thinking_enabled=thinking_enabled, github_tools_enabled=github_tools_enabled, approved_tokens=frozenset(approved_tokens), tools_enabled=tools_enabled):
                        if cancel_event.is_set():
                            break
                        if event.get("type") == "done":
                            done_content = event.get("content", "")
                        elif event.get("type") == "thinking" and event.get("content"):
                            thinking_content = (thinking_content or "") + event["content"]
                        loop.call_soon_threadsafe(queue.put_nowait, event)

                    if not cancel_event.is_set() and orch.conv_id and done_content is not None:
                        db.save_messages(orch.conv_id, [
                            {"role": "user",      "content": message},
                            {"role": "assistant", "content": done_content,
                             "thinking_content": thinking_content,
                             "finish_reason": getattr(orch, "last_finish_reason", None)},
                        ])

                        if was_new_conv:
                            orch._is_new_conv = False
                            title = orch.generate_title(message)
                            db.update_title(orch.conv_id, title)
                            loop.call_soon_threadsafe(queue.put_nowait, {
                                "type": "title",
                                "conv_id": orch.conv_id,
                                "title": title,
                            })

                        if orch.context_pct >= COMPRESS_THRESHOLD:
                            summary = orch.compress_history()
                            if summary:
                                compressed = [
                                    m for m in orch.conversation_history
                                    if m["role"] != "system"
                                ]
                                db.replace_messages(orch.conv_id, compressed)
                                loop.call_soon_threadsafe(queue.put_nowait, {
                                    "type": "compress",
                                    "message": "Earlier conversation summarised to free up context.",
                                })

                except Exception:
                    # Log it. The message below tells the user to check the
                    # server logs, and until 2026-08-11 this handler bound the
                    # exception and then dropped it, so the logs it points at
                    # held nothing. Three turns died this way during a corpus
                    # run with no traceback anywhere to say why.
                    logger.exception("chat stream failed for conversation %s",
                                     getattr(orch, "conv_id", "?"))
                    if not cancel_event.is_set():
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"type": "error", "message": "Internal error — see server logs"},
                        )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=produce, daemon=True).start()

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        yield {"data": json.dumps({"type": "heartbeat"})}
                        continue
                    if event is None:
                        if cancel_event.is_set():
                            orch.conversation_history = \
                                orch.conversation_history[:snapshot["len"]]
                        break
                    if event.get("type") == "search_done":
                        event = {**event, "results": [
                            {"title": r["title"], "url": r["url"]}
                            for r in event.get("results", [])
                        ]}
                    logger.debug("SSE → %s", event.get("type"))
                    yield {"data": json.dumps(event)}
            finally:
                _active_cancels.pop(request_id, None)

    return EventSourceResponse(event_stream())


@app.post("/reset")
async def reset(conversation_id: str = Form(default=""), _: None = Depends(_ready)):
    """Start a fresh conversation (old one stays in DB). If a `conversation_id` is
    given, the new conversation inherits its project."""
    project = None
    project_id = None
    if conversation_id:
        prev = db.get_conversation(conversation_id)
        if prev and prev.get("project_id"):
            project_id = prev["project_id"]
            project = db.get_project(project_id)
    conv_id = db.create_conversation(sessions.model, project_id=project_id)
    await sessions.acquire(conv_id, project=project, fresh=True)
    return {"status": "ok", "conv_id": conv_id, "title": "New conversation"}


@app.post("/compact")
async def compact(conversation_id: str = Form(...), _: None = Depends(_ready)):
    """Manually compress a conversation's history. Returns compress event JSON."""
    orch, lock = await sessions.acquire(conversation_id)
    async with lock:
        non_system = [m for m in orch.conversation_history if m["role"] != "system"]
        if len(non_system) == 0:
            return {"type": "compact_noop", "message": "Nothing to compact yet."}

        summary = orch.compress_history()
        if summary is None:
            return {"type": "compact_noop", "message": "Nothing to compact yet."}

        compressed = [m for m in orch.conversation_history if m["role"] != "system"]
        db.replace_messages(orch.conv_id, compressed)
    return {"type": "compress", "message": "Earlier conversation summarised to free up context."}


# ── Project endpoints ─────────────────────────────────────────────────────────

class ProjectRequest(BaseModel):
    # Bounded so an unbounded body cannot be written straight to SQLite. Sizes are
    # generous versus any real project name/path.
    name: str = Field(max_length=200)
    local_path: str = Field(default="", max_length=4096)
    github_repo: str = Field(default="", max_length=200)


@app.get("/projects")
async def list_projects():
    return {"projects": db.list_projects()}


@app.post("/projects")
async def create_project(body: ProjectRequest):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    local_path = body.local_path.strip() or None
    github_repo = body.github_repo.strip() or None
    if not local_path and not github_repo:
        raise HTTPException(status_code=400, detail="at least one of local_path or github_repo is required")
    if local_path:
        # A project's local_path becomes the orchestrator's workspace_root, which
        # is the sandbox root for every filesystem tool AND for run_shell. So an
        # unconstrained value here does not live inside the sandbox — it defines
        # it. local_path="/" would make every containment check downstream pass
        # for the whole filesystem.
        #
        # The guard is deliberately about WIDENING, not about location: a repo on
        # an external volume is a legitimate project, so we allow any directory
        # except the ones that would hand over everything at once.
        resolved = Path(local_path).expanduser().resolve()
        if not resolved.is_dir():
            raise HTTPException(status_code=400, detail=f"Path does not exist or is not a directory: {local_path}")
        _forbidden_roots = {Path(resolved.anchor), Path.home(), Path("/Users"), Path("/Volumes"),
                            Path("/etc"), Path("/usr"), Path("/var"), Path("/System"), Path("/Library")}
        if resolved in _forbidden_roots:
            raise HTTPException(
                status_code=400,
                detail=("Project path is too broad — pick a specific project folder, "
                        "not a filesystem, home, or system root"),
            )
        local_path = str(resolved)
    project_id = db.create_project(name, local_path, github_repo)
    return db.get_project(project_id)


@app.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    # A project entry can only be removed once its backing files are gone. While the
    # local folder still exists, deletion is refused — the user must delete the local
    # folder (or its GitHub repo) manually first. This prevents an accidental tap in the
    # app from dropping a live project and its conversation history.
    local_path = (project.get("local_path") or "").strip()
    if local_path and Path(local_path).expanduser().exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"Project files still exist at {local_path}. Delete the local folder "
                "(or its GitHub repo) manually first, then remove the project entry."
            ),
        )
    db.delete_project(project_id)
    return {"status": "ok"}


# ── Memory endpoints ──────────────────────────────────────────────────────────

class MemoryRequest(BaseModel):
    # Memories are injected into EVERY system prompt, so an unbounded one
    # permanently consumes context on every turn of every conversation.
    text: str = Field(max_length=2000)


@app.get("/memories")
async def list_memories():
    return {"memories": db.get_memories()}


@app.post("/memories")
async def add_memory(body: MemoryRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    if len(db.get_memories()) >= 30:
        raise HTTPException(status_code=400, detail="memory limit reached (30 max)")
    memory_id = db.add_memory(text)
    return {"id": memory_id, "text": text, "created_at": int(__import__("time").time())}


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    db.delete_memory(memory_id)
    return {"status": "ok"}


# ── Reminders endpoints ───────────────────────────────────────────────────────

@app.get("/reminders")
async def list_reminders():
    return {"reminders": db.list_reminders()}


@app.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: int):
    db.delete_reminder(reminder_id)
    return {"status": "ok"}


# ── Conversation endpoints ────────────────────────────────────────────────────

@app.get("/conversations/search")
async def search_conversations(q: str = "", limit: int = 10):
    if not q.strip():
        return {"results": []}
    results = db.search_conversations(q.strip(), limit=min(limit, 50))
    return {"results": results}


@app.get("/conversations")
async def list_conversations():
    return {"conversations": db.list_conversations()}


class CreateConversationRequest(BaseModel):
    project_id: str = ""


@app.post("/conversations")
async def create_conversation(
    body: Optional[CreateConversationRequest] = None,
    _: None = Depends(_ready),
):
    project_id = (body.project_id.strip() if body else "") or None
    project = None
    if project_id:
        project = db.get_project(project_id)
        if not project:
            raise HTTPException(status_code=400, detail=f"Project not found: {project_id}")
    conv_id = db.create_conversation(sessions.model, project_id=project_id)
    # The session is created lazily on first /chat; just register the DB row here.
    return {"id": conv_id, "title": "New conversation", "project_id": project_id}


class ConversationUpdate(BaseModel):
    """Partial update. Every field optional, at least one required.

    `project_id` distinguishes three cases, which is why it cannot just default
    to None: absent means "leave the project alone", null means "remove from its
    project", and a string means "move it there". Pydantic's
    `model_fields_set` is what tells absent from explicit null apart.
    """
    title: Optional[str] = Field(default=None, max_length=200)
    project_id: Optional[str] = None


# Kept as an alias: `RenameRequest` was this endpoint's only shape until the
# endpoint learned to move conversations between projects.
RenameRequest = ConversationUpdate


@app.patch("/conversations/{conv_id}")
async def update_conversation(conv_id: str, body: ConversationUpdate):
    supplied = body.model_fields_set
    if not supplied:
        raise HTTPException(status_code=400, detail="nothing to update")

    if not db.get_conversation(conv_id):
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    if "title" in supplied:
        title = (body.title or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="title required")
        db.update_title(conv_id, title)

    if "project_id" in supplied:
        project_id = (body.project_id or "").strip() or None
        if project_id and not db.get_project(project_id):
            raise HTTPException(status_code=400, detail=f"Project not found: {project_id}")
        db.update_project(conv_id, project_id)

    return {"status": "ok"}


@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, _: None = Depends(_ready)):
    db.delete_conversation(conv_id)
    await sessions.remove(conv_id)        # drop its cached orchestrator, if any
    # Report a sensible next conversation for clients that auto-select one.
    convs = db.list_conversations()
    active = convs[0]["id"] if convs else db.create_conversation(sessions.model)
    return {"status": "ok", "active_conv_id": active}


@app.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str):
    return {"messages": db.load_messages(conv_id)}


# ── Existing endpoints ────────────────────────────────────────────────────────

@app.get("/rag/documents")
async def rag_list(conversation_id: str = "", _: None = Depends(_ready)):
    # The RAG index is per-conversation; without an id there's nothing to list.
    if not conversation_id:
        return {"documents": []}
    orch, _l = await sessions.acquire(conversation_id)
    return {"documents": orch.rag_engine.list_documents()}


@app.delete("/rag/documents/{name:path}")
async def rag_remove(name: str, conversation_id: str = "", _: None = Depends(_ready)):
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    orch, _l = await sessions.acquire(conversation_id)
    orch.rag_engine.remove(name)
    return {"documents": orch.rag_engine.list_documents()}


@app.post("/verbose")
async def set_verbose(request: VerboseRequest, _: None = Depends(_ready)):
    sessions.set_verbose(request.enabled)
    return {"verbose": sessions.verbose}


@app.get("/status")
async def status(conversation_id: str = "", _: None = Depends(_ready)):
    """Per-conversation runtime stats. Without a `conversation_id`, returns the
    server-level defaults (no session is materialised)."""
    if not conversation_id:
        return {
            "model": sessions.model,
            "verbose": sessions.verbose,
            "history_length": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "context_pct": 0.0,
            "home_dir": str(Path.home()),
            "conv_id": None,
            "project": None,
            "workspace_root": None,
        }
    orch, _l = await sessions.acquire(conversation_id)
    return {
        "model": orch.model,
        "verbose": orch.verbose,
        "history_length": len(orch.conversation_history),
        "input_tokens": orch.total_input_tokens,
        "output_tokens": orch.total_output_tokens,
        "context_pct": orch.context_pct,
        "home_dir": str(Path.home()),
        "conv_id": orch.conv_id,
        "project": orch.project,
        "workspace_root": orch.workspace_root,
    }


@app.get("/browse")
async def browse(path: str = "/"):
    """List directory contents for the folder browser UI."""
    try:
        resolved = _safe_path(path)
        if not resolved.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

        try:
            names = sorted(
                os.listdir(resolved),
                key=lambda n: (not os.path.isdir(os.path.join(resolved, n)), n.lower())
            )
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

        entries = []
        for name in names:
            full = os.path.join(resolved, name)
            is_dir = os.path.isdir(full)
            _, ext = os.path.splitext(name)
            entries.append({
                "name": name,
                "is_dir": is_dir,
                "ext": ext.lower(),
                "path": full,
            })

        parent = str(Path(resolved).parent)
        home = str(Path.home())
        return {
            "path": str(resolved),
            "parent": parent if parent != str(resolved) and str(resolved) != home else None,
            "entries": entries,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("browse error: %s", e)
        raise HTTPException(status_code=500, detail="Internal error — see server logs")


class AskRequest(BaseModel):
    prompt: str = Field(max_length=100_000)
    system: str = Field(default="", max_length=20_000)


@app.post("/ask")
async def ask(body: AskRequest, _: None = Depends(_ready)):
    """One-shot ephemeral query — no conversation saved, no tools, no DB writes."""
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt required")
    messages = []
    if body.system.strip():
        messages.append({"role": "system", "content": body.system.strip()})
    messages.append({"role": "user", "content": body.prompt.strip()})
    try:
        text = await asyncio.get_running_loop().run_in_executor(
            None, lambda: sessions.llm_chat_sync(messages)
        )
        return {"response": text}
    except Exception as e:
        logger.error("ask error: %s", e)
        raise HTTPException(status_code=502, detail="Internal error — see server logs")


if __name__ == "__main__":
    import re
    import signal
    import subprocess
    import sys
    import time

    # Clear a stale instance of *this same deployment* only — scope the pattern to this
    # checkout's interpreter (sys.executable is unique per venv), so a second checkout or a
    # production server from another path is never killed. SIGTERM is ignored across the call
    # so pkill matching our own process is a no-op for us but still reaps the old instance.
    _self_py = re.escape(sys.executable)
    _old_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.run(["/usr/bin/pkill", "-f", rf"{_self_py}.*server\.py"], capture_output=True)
    signal.signal(signal.SIGTERM, _old_sigterm)
    time.sleep(0.4)

    # Prevent macOS idle sleep while the server is running.
    # caffeinate -i prevents idle system sleep (battery + AC); -s additionally
    # prevents sleep on AC power. -w <pid> ties the assertion to this process —
    # caffeinate exits automatically when the server exits, so no orphan is left.
    if sys.platform == "darwin":
        subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-s", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info("Power assertion active: idle sleep prevented while server is running")

    ssl_certfile = os.environ.get("SSL_CERTFILE")
    ssl_keyfile  = os.environ.get("SSL_KEYFILE")

    def _discover_tailnet_ip():
        """Return this Mac's Tailscale (100.64.0.0/10) IPv4 *only if it is actually
        bound to a local interface and bindable right now*, else None (fail closed).

        We scan interface addresses (`ifconfig`) rather than `tailscale ip -4`: the CLI
        returns the node's assigned identity IP even when the tunnel is DOWN, and that
        address is not on any interface, so binding it raises OSError. An address present
        in `ifconfig` is assigned; we also test-bind it to be certain before committing.
        Local-only, subprocess arg-list (never shell=True)."""
        cgnat = ipaddress.ip_network("100.64.0.0/10")

        def _bindable(ip: str) -> bool:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind((ip, 0))  # port 0 = any free port; just proving the addr is ours
                return True
            except OSError:
                return False
            finally:
                s.close()

        try:
            out = subprocess.run(["/sbin/ifconfig"], capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None

        # Parse PER INTERFACE and only accept the tunnel. 100.64.0.0/10 is the
        # shared CGNAT range, not a Tailscale-exclusive one: mobile hotspots and
        # CGNAT ISPs hand out addresses from it on real interfaces. Scanning
        # every token in ifconfig would happily bind :8443 to an ISP-facing
        # address that also satisfies the source-IP allowlist — i.e. exactly the
        # off-tailnet exposure this function exists to prevent.
        iface = None
        for line in out.stdout.splitlines():
            if line and not line[0].isspace():          # "utun4: flags=..."
                iface = line.split(":", 1)[0].strip().lower()
                continue
            if not iface or not (iface.startswith("utun") or iface.startswith("tailscale")):
                continue
            parts = line.split()
            if len(parts) < 2 or parts[0] != "inet":
                continue
            tok = parts[1]
            try:
                if ipaddress.ip_address(tok) not in cgnat:
                    continue
            except ValueError:
                continue
            if _bindable(tok):
                logger.info("Tailnet address %s found on interface %s", tok, iface)
                return tok
        return None

    # ── Secret hygiene warnings ──────────────────────────────────────────────
    if AUTH_TOKEN:
        if len(AUTH_TOKEN) < MIN_TOKEN_LENGTH:
            logger.warning(
                "Auth token is only %d chars — use at least %d (e.g. `openssl rand -hex 32`).",
                len(AUTH_TOKEN), MIN_TOKEN_LENGTH,
            )
        _yaml = Path(__file__).parent / "mira.yaml"
        try:
            if _yaml.exists() and (_yaml.stat().st_mode & 0o077):
                # Tighten it rather than only warning: this file holds the sole
                # credential for off-host access, and a warning in a log nobody
                # reads leaves the token readable by every local account.
                os.chmod(_yaml, 0o600)
                logger.warning(
                    "%s held auth_token but was group/other-readable — permissions "
                    "tightened to 600.", _yaml,
                )
        except OSError as e:
            logger.warning("Could not check/fix permissions on %s: %s", _yaml, e)

    # ── Bind policy ──────────────────────────────────────────────────────────
    # HTTP :8000 is loopback-only by default — no plaintext token/payload ever leaves
    # the machine. MIRA_HOST can opt :8000 back onto the LAN (plaintext — see
    # docs/remote-access.md), but only with a token; off-host without a token is
    # downgraded to loopback.
    _loopback = {"127.0.0.1", "localhost", "::1"}
    http_host = os.environ.get("MIRA_HOST", "127.0.0.1")
    if http_host not in _loopback and not AUTH_TOKEN:
        logger.warning(
            "Refusing to bind %s without a token — an open server with shell/filesystem "
            "tools must not be exposed off-host. Binding 127.0.0.1 instead.", http_host,
        )
        http_host = "127.0.0.1"

    # HTTPS :8443 binds the Tailscale interface only, so the socket exists solely on the
    # tailnet. Fail closed when the tailnet is down — off-host HTTPS is simply not served
    # (never 0.0.0.0, never a useless loopback HTTPS). HTTP :8000 (loopback) always runs.
    https_enabled = bool(ssl_certfile and ssl_keyfile and AUTH_TOKEN)
    logger.info("Binding HTTP :8000 to %s (auth %s)", http_host,
                "enabled" if AUTH_TOKEN else "disabled — loopback only")

    async def _run():
        http_server = uvicorn.Server(
            uvicorn.Config("server:app", host=http_host, port=8000, log_level="info")
        )

        async def _serve_https():
            # Tailscale may still be connecting when the server starts (e.g. right after
            # login/boot) — poll instead of checking once, so a late tailnet connection is
            # picked up without requiring a manual `/mira-server restart`. An HTTPS
            # bind/serve failure must NEVER take down the HTTP listener.
            retry_interval = 15
            warned = False
            https_host = await asyncio.to_thread(_discover_tailnet_ip)
            while not https_host:
                if not warned:
                    logger.warning(
                        "Tailscale not up (no bindable 100.64.0.0/10 address) — off-host "
                        "HTTPS disabled for now; retrying every %ds until Tailscale connects.",
                        retry_interval,
                    )
                    warned = True
                await asyncio.sleep(retry_interval)
                https_host = await asyncio.to_thread(_discover_tailnet_ip)

            logger.info("HTTPS :8443 will bind tailnet address %s", https_host)
            https_server = uvicorn.Server(uvicorn.Config(
                "server:app", host=https_host, port=8443,
                ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile, log_level="info",
            ))
            https_server.install_signal_handlers = lambda: None
            try:
                await https_server.serve()
            except Exception as exc:  # noqa: BLE001 — degrade gracefully, keep HTTP up
                logger.error("HTTPS :8443 failed (%s) — continuing with HTTP only.", exc)

        coros = [http_server.serve()]
        if https_enabled:
            coros.append(_serve_https())
        await asyncio.gather(*coros)

    asyncio.run(_run())
