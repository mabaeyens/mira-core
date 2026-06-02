"""FastAPI server for the ollama Search Tool web interface."""

import asyncio
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import ollama
import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import core.db as db
import core.file_handler as file_handler
from core.config import VERBOSE_DEFAULT, COMPRESS_THRESHOLD, COMPRESS_KEEP_RECENT, MODEL_NAME, BACKEND, OLLAMA_HOST, CONTEXT_WINDOW
from core.orchestrator import ChatOrchestrator
from core import backend_manager as _bm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

orchestrator: ChatOrchestrator = None
_orch_lock: asyncio.Lock = None
_init_lock: asyncio.Lock = asyncio.Lock()
_active_cancels: Dict[str, threading.Event] = {}
_initialized = False
_ollama_ready = False

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
    "host": OLLAMA_HOST,
    "context_window": CONTEXT_WINDOW,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, _orch_lock, _initialized, _ollama_ready
    async with _init_lock:
        if not _initialized:
            _initialized = True
            _orch_lock = asyncio.Lock()
            db.init_db()
            
            # Verify configured model is installed; warn clearly if not.
            # Uses list() (all installed models), not ps() (only loaded-in-memory models).
            if BACKEND == "ollama":
                try:
                    client = ollama.Client(host=OLLAMA_HOST)
                    installed = {m.model for m in client.list().models}
                    model_found = any(
                        MODEL_NAME == name or name.startswith(MODEL_NAME + ":")
                        for name in installed
                    )
                    if not model_found:
                        logger.warning(
                            f"Model '{MODEL_NAME}' is not installed. "
                            f"Run: ollama pull {MODEL_NAME}  "
                            f"(installed: {sorted(installed)})"
                        )
                    else:
                        logger.info(f"Model '{MODEL_NAME}' confirmed installed — warming up")
                        client.generate(model=MODEL_NAME, prompt="", keep_alive="24h")
                        logger.info(f"Model '{MODEL_NAME}' loaded and ready")
                except Exception as e:
                    logger.warning(f"Could not check Ollama models: {e}")

            orchestrator = ChatOrchestrator(verbose=VERBOSE_DEFAULT)
            convs = db.list_conversations()
            if convs:
                project = db.get_project(convs[0]["project_id"]) if convs[0].get("project_id") else None
                orchestrator.load_conversation(convs[0]["id"], project=project)
            else:
                conv_id = db.create_conversation(orchestrator.model)
                orchestrator.new_conversation(conv_id)
            logger.info(f"Initialized orchestrator — backend: {BACKEND}, model: {orchestrator.model}, conv: {orchestrator.conv_id}")
            if BACKEND != "ollama":
                logger.info(f"{BACKEND} backend — model {MODEL_NAME} at {OLLAMA_HOST}")
            _ollama_ready = True
            # Auto-start the inference backend in a background thread so the app is
            # usable immediately (health returns 200) even while oMLX/Ollama loads.
            threading.Thread(
                target=_bm.ensure_backend_running,
                args=(BACKEND,),
                daemon=True,
            ).start()

    yield


app = FastAPI(title="ollama Search Tool", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


def _ready():
    """FastAPI dependency — returns 503 until the orchestrator is initialised."""
    if orchestrator is None:
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
    if not _ollama_ready:
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


@app.get("/backend")
async def get_backend(_=Depends(_ready)):
    return {
        "backend": _rt["backend"],
        "model": _rt["model"],
        "host": _rt["host"],
        "context_window": _rt["context_window"],
    }


@app.post("/backend")
async def switch_backend(request: Request, _=Depends(_ready)):
    global _ollama_ready
    body = await request.json()
    target = body.get("backend", "")
    if target not in ("ollama", "omlx", "mlx-lm", "dflash"):
        raise HTTPException(status_code=400, detail="backend must be 'ollama', 'omlx', 'mlx-lm', or 'dflash'")
    if target == _rt["backend"]:
        return {"status": "ok", "backend": target, "message": "already active"}
    _ollama_ready = False
    try:
        async with _orch_lock:
            loop = asyncio.get_event_loop()
            preset = await loop.run_in_executor(None, _bm.switch_to, target)
            orchestrator.reinitialize_client(
                backend=preset["backend"],
                model=preset["model"],
                host=preset["host"],
                context_window=preset["context_window"],
            )
            _rt.update(preset)
    except Exception as e:
        logger.error("Backend switch failed: %s", e)
        _ollama_ready = True
        raise HTTPException(status_code=500, detail=str(e))
    _ollama_ready = True
    return {"status": "ok", "backend": _rt["backend"], "model": _rt["model"]}


@app.get("/models")
async def list_models():
    """Return all locally available models grouped by backend."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _bm.list_models)
    result["active"] = {"backend": _rt["backend"], "model_id": _rt["model"]}
    return result


@app.post("/models/switch")
async def switch_model(request: Request, _=Depends(_ready)):
    global _ollama_ready
    body = await request.json()
    backend = body.get("backend", "")
    model_id = body.get("model_id", "")
    if backend not in ("ollama", "mlx-lm", "omlx", "dflash"):
        raise HTTPException(status_code=400, detail="backend must be 'ollama', 'mlx-lm', 'omlx', or 'dflash'")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if backend == _rt["backend"] and model_id == _rt["model"]:
        return {"status": "ok", "backend": backend, "model": model_id, "message": "already active"}
    _ollama_ready = False
    try:
        async with _orch_lock:
            loop = asyncio.get_event_loop()
            preset = await loop.run_in_executor(None, _bm.switch_to_model, backend, model_id)
            orchestrator.reinitialize_client(
                backend=preset["backend"],
                model=preset["model"],
                host=preset["host"],
                context_window=preset["context_window"],
            )
            _rt.update(preset)
    except Exception as e:
        logger.error("Model switch failed: %s", e)
        _ollama_ready = True
        raise HTTPException(status_code=500, detail=str(e))
    _ollama_ready = True
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
async def cancel():
    for ev in list(_active_cancels.values()):
        ev.set()
    return {"status": "cancelled"}


@app.post("/chat")
async def chat(
    message: str = Form(...),
    conversation_id: str = Form(default=""),
    files: List[UploadFile] = File(default=[]),
    paths: List[str] = Form(default=[]),
    thinking_enabled: bool = Form(default=False),
    _: None = Depends(_ready),
):
    """SSE endpoint — streams typed events from stream_chat() to the browser."""
    request_id = str(uuid.uuid4())
    cancel_event = threading.Event()
    _active_cancels[request_id] = cancel_event

    async with _orch_lock:
        if not conversation_id:
            # No ID supplied — always start a fresh conversation so callers without
            # an explicit ID never inherit whatever session is currently loaded.
            conv_id = db.create_conversation(orchestrator.model)
            orchestrator.new_conversation(conv_id)
        elif conversation_id != orchestrator.conv_id:
            conv = db.get_conversation(conversation_id)
            if conv:
                project = db.get_project(conv["project_id"]) if conv.get("project_id") else None
                orchestrator.load_conversation(conversation_id, project=project)
            else:
                orchestrator.new_conversation(conversation_id)

    attachments = []
    for upload in files:
        data = await upload.read()
        try:
            att = file_handler.load_file_bytes(upload.filename, data)
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
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        snapshot = {"len": len(orchestrator.conversation_history)}

        def produce():
            snapshot["len"] = len(orchestrator.conversation_history)
            was_new_conv = orchestrator._is_new_conv
            done_content = None
            thinking_content = None

            try:
                for event in orchestrator.stream_chat(message, attachments=attachments or None, thinking_enabled=thinking_enabled):
                    if cancel_event.is_set():
                        break
                    if event.get("type") == "done":
                        done_content = event.get("content", "")
                    elif event.get("type") == "thinking" and event.get("content"):
                        thinking_content = (thinking_content or "") + event["content"]
                    loop.call_soon_threadsafe(queue.put_nowait, event)

                if not cancel_event.is_set() and orchestrator.conv_id and done_content is not None:
                    db.save_messages(orchestrator.conv_id, [
                        {"role": "user",      "content": message},
                        {"role": "assistant", "content": done_content,
                         "thinking_content": thinking_content},
                    ])

                    if was_new_conv:
                        orchestrator._is_new_conv = False
                        title = orchestrator.generate_title(message)
                        db.update_title(orchestrator.conv_id, title)
                        loop.call_soon_threadsafe(queue.put_nowait, {
                            "type": "title",
                            "conv_id": orchestrator.conv_id,
                            "title": title,
                        })

                    if orchestrator.context_pct >= COMPRESS_THRESHOLD:
                        summary = orchestrator.compress_history()
                        if summary:
                            compressed = [
                                m for m in orchestrator.conversation_history
                                if m["role"] != "system"
                            ]
                            db.replace_messages(orchestrator.conv_id, compressed)
                            loop.call_soon_threadsafe(queue.put_nowait, {
                                "type": "compress",
                                "message": "Earlier conversation summarised to free up context.",
                            })

            except Exception as e:
                if not cancel_event.is_set():
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {"type": "error", "message": str(e)}
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
                        orchestrator.conversation_history = \
                            orchestrator.conversation_history[:snapshot["len"]]
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
async def reset(_: None = Depends(_ready)):
    """Start a fresh conversation (old one stays in DB). Preserves active project."""
    project = orchestrator.project
    project_id = project["id"] if project else None
    orchestrator.reset_conversation()
    conv_id = db.create_conversation(orchestrator.model, project_id=project_id)
    orchestrator.new_conversation(conv_id, project=project)
    return {"status": "ok", "conv_id": conv_id, "title": "New conversation"}


@app.post("/compact")
async def compact(_: None = Depends(_ready)):
    """Manually compress conversation history. Returns compress event JSON."""
    non_system = [m for m in orchestrator.conversation_history if m["role"] != "system"]
    if len(non_system) == 0:
        return {"type": "compact_noop", "message": "Nothing to compact yet."}

    summary = orchestrator.compress_history()
    if summary is None:
        return {"type": "compact_noop", "message": "Nothing to compact yet."}

    compressed = [m for m in orchestrator.conversation_history if m["role"] != "system"]
    db.replace_messages(orchestrator.conv_id, compressed)
    return {"type": "compress", "message": "Earlier conversation summarised to free up context."}


# ── Project endpoints ─────────────────────────────────────────────────────────

class ProjectRequest(BaseModel):
    name: str
    local_path: str = ""
    github_repo: str = ""


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
    if local_path and not Path(local_path).expanduser().is_dir():
        raise HTTPException(status_code=400, detail=f"Path does not exist or is not a directory: {local_path}")
    project_id = db.create_project(name, local_path, github_repo)
    return db.get_project(project_id)


@app.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    if not db.get_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete_project(project_id)
    return {"status": "ok"}


# ── Memory endpoints ──────────────────────────────────────────────────────────

class MemoryRequest(BaseModel):
    text: str


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
    conv_id = db.create_conversation(orchestrator.model, project_id=project_id)
    orchestrator.new_conversation(conv_id, project=project)
    return {"id": conv_id, "title": "New conversation", "project_id": project_id}


class RenameRequest(BaseModel):
    title: str


@app.patch("/conversations/{conv_id}")
async def rename_conversation(conv_id: str, body: RenameRequest):
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title required")
    db.update_title(conv_id, title)
    return {"status": "ok"}


@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, _: None = Depends(_ready)):
    db.delete_conversation(conv_id)
    if orchestrator.conv_id == conv_id:
        convs = db.list_conversations()
        if convs:
            project = db.get_project(convs[0]["project_id"]) if convs[0].get("project_id") else None
            orchestrator.load_conversation(convs[0]["id"], project=project)
        else:
            new_id = db.create_conversation(orchestrator.model)
            orchestrator.new_conversation(new_id)
    return {"status": "ok", "active_conv_id": orchestrator.conv_id}


@app.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str):
    return {"messages": db.load_messages(conv_id)}


# ── Existing endpoints ────────────────────────────────────────────────────────

@app.get("/rag/documents")
async def rag_list(_: None = Depends(_ready)):
    return {"documents": orchestrator.rag_engine.list_documents()}


@app.delete("/rag/documents/{name:path}")
async def rag_remove(name: str, _: None = Depends(_ready)):
    orchestrator.rag_engine.remove(name)
    return {"documents": orchestrator.rag_engine.list_documents()}


@app.post("/verbose")
async def set_verbose(request: VerboseRequest, _: None = Depends(_ready)):
    orchestrator.verbose = request.enabled
    return {"verbose": orchestrator.verbose}


@app.get("/status")
async def status(_: None = Depends(_ready)):
    return {
        "model": orchestrator.model,
        "verbose": orchestrator.verbose,
        "history_length": len(orchestrator.conversation_history),
        "input_tokens": orchestrator.total_input_tokens,
        "output_tokens": orchestrator.total_output_tokens,
        "context_pct": orchestrator.context_pct,
        "home_dir": str(Path.home()),
        "conv_id": orchestrator.conv_id,
        "project": orchestrator.project,
        "workspace_root": orchestrator.workspace_root,
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
    prompt: str
    system: str = ""


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
            None, lambda: orchestrator._llm_chat_sync(messages)
        )
        return {"response": text}
    except Exception as e:
        logger.error("ask error: %s", e)
        raise HTTPException(status_code=502, detail="Internal error — see server logs")


if __name__ == "__main__":
    import signal
    import subprocess
    import sys
    import time

    _old_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.run(["/usr/bin/pkill", "-f", "python.*server\\.py"], capture_output=True)
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

    async def _run():
        http_server = uvicorn.Server(
            uvicorn.Config("server:app", host="0.0.0.0", port=8000, log_level="info")
        )
        if ssl_certfile and ssl_keyfile:
            https_cfg = uvicorn.Config(
                "server:app", host="0.0.0.0", port=8443,
                ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
                log_level="info",
            )
            https_server = uvicorn.Server(https_cfg)
            https_server.install_signal_handlers = lambda: None
            await asyncio.gather(http_server.serve(), https_server.serve())
        else:
            await http_server.serve()

    asyncio.run(_run())
