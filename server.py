"""FastAPI server for the ollama Search Tool web interface."""

import asyncio
import json
import logging
import os
import socket
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

try:
    from zeroconf import ServiceInfo, Zeroconf as _Zeroconf
    _HAS_ZEROCONF = True
except ImportError:
    _HAS_ZEROCONF = False

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile, File

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import core.db as db
import core.file_handler as file_handler
from core.config import VERBOSE_DEFAULT, COMPRESS_THRESHOLD, MODEL_NAME
from core.orchestrator import ChatOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

orchestrator: ChatOrchestrator = None
cancel_event = threading.Event()
_initialized = False
_ollama_ready = False   # True once Ollama warm-up succeeds (or exhausts retries)
_zc = None       # Zeroconf instance kept alive for the process lifetime
_zc_info = None  # ServiceInfo kept alive for clean unregistration


def _local_ip() -> str:
    """Return the primary LAN IP (used for Bonjour advertisement)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, _initialized, _ollama_ready, _zc, _zc_info
    # Guard: when running dual HTTP+HTTPS servers in the same process both
    # servers share this module's globals, so only initialise once.
    if not _initialized:
        _initialized = True
        db.init_db()
        orchestrator = ChatOrchestrator(verbose=VERBOSE_DEFAULT)
        # Resume the most recent conversation, or create a fresh one
        convs = db.list_conversations()
        if convs:
            project = db.get_project(convs[0]["project_id"]) if convs[0].get("project_id") else None
            orchestrator.load_conversation(convs[0]["id"], project=project)
        else:
            conv_id = db.create_conversation(orchestrator.model)
            orchestrator.new_conversation(conv_id)
        logger.info(f"Initialized orchestrator with model: {orchestrator.model}, conv: {orchestrator.conv_id}")
        # Pre-warm with retry — Ollama may still be starting up when launchd fires
        # server.py on login. Retry up to 5 times with a 5-second gap so the model
        # is loaded before the first user request arrives.
        for _attempt in range(1, 6):
            try:
                running = {m.model for m in orchestrator._ollama.ps().models}
                if MODEL_NAME in running:
                    logger.info(f"Model {MODEL_NAME} already loaded in Ollama — reusing")
                else:
                    logger.info(f"Model {MODEL_NAME} not loaded — warming up (attempt {_attempt}/5)...")
                    orchestrator._ollama.generate(model=MODEL_NAME, prompt="", keep_alive="24h")
                    logger.info(f"Model {MODEL_NAME} ready")
                break
            except Exception as e:
                if _attempt < 5:
                    logger.warning(f"Ollama not ready (attempt {_attempt}/5): {e} — retrying in 5s")
                    await asyncio.sleep(5)
                else:
                    logger.warning(f"Could not pre-warm model {MODEL_NAME} after 5 attempts: {e}")
        _ollama_ready = True
        # Register Bonjour so iOS (and macOS thin-client) can discover this server
        # on the LAN. zeroconf is synchronous, so we run it off the event loop.
        if _HAS_ZEROCONF:
            try:
                ip = _local_ip()
                info = ServiceInfo(
                    "_ollamasearch._tcp.local.",
                    "Mira._ollamasearch._tcp.local.",
                    addresses=[socket.inet_aton(ip)],
                    port=8000,
                )
                zc = _Zeroconf()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: zc.register_service(info))
                _zc, _zc_info = zc, info
                logger.info(f"Bonjour registered on {ip}:8000")
            except Exception as e:
                logger.warning(f"Bonjour registration failed: {e}")

    yield

    # Unregister Bonjour on shutdown (the _zc = None swap prevents double-close
    # when both the HTTP and HTTPS lifespans tear down).
    if _zc is not None:
        zc, _zc = _zc, None
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: (zc.unregister_all_services(), zc.close()))
        except Exception:
            pass


app = FastAPI(title="ollama Search Tool", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class VerboseRequest(BaseModel):
    enabled: bool


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    """Liveness probe — used by the macOS native app to know when the server is ready.
    Returns 503 while Ollama is still warming up so the native app splash keeps polling."""
    from fastapi.responses import JSONResponse
    if not _ollama_ready:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok"}


@app.post("/cancel")
async def cancel():
    cancel_event.set()
    return {"status": "cancelled"}


@app.post("/chat")
async def chat(
    message: str = Form(...),
    conversation_id: str = Form(default=""),
    files: List[UploadFile] = File(default=[]),
    paths: List[str] = Form(default=[]),
):
    """SSE endpoint — streams typed events from stream_chat() to the browser."""
    cancel_event.clear()

    # Switch conversation if the client is on a different one
    if conversation_id and conversation_id != orchestrator.conv_id:
        conv = db.get_conversation(conversation_id)
        if conv:
            project = db.get_project(conv["project_id"]) if conv.get("project_id") else None
            orchestrator.load_conversation(conversation_id, project=project)
        else:
            orchestrator.new_conversation(conversation_id)
    elif not orchestrator.conv_id:
        conv_id = db.create_conversation(orchestrator.model)
        orchestrator.new_conversation(conv_id)

    # Read uploaded files before entering the background thread
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
            att = file_handler.load_file(path)
            attachments.append(att)
        except Exception as e:
            logger.warning(f"Could not load file at path '{path}': {e}")
            attachments.append({
                "type": "text", "name": path, "content": "",
                "warning": f"Could not load '{path}': {e}"
            })

    async def event_stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        # Shared mutable so produce() can write the snapshot after load_conversation
        snapshot = {"len": len(orchestrator.conversation_history)}

        def produce():
            # Capture snapshot AFTER any conversation switch (already done above)
            snapshot["len"] = len(orchestrator.conversation_history)
            was_new_conv = orchestrator._is_new_conv
            done_content = None

            try:
                for event in orchestrator.stream_chat(message, attachments=attachments or None):
                    if cancel_event.is_set():
                        break
                    if event.get("type") == "done":
                        done_content = event.get("content", "")
                    loop.call_soon_threadsafe(queue.put_nowait, event)

                # ── Post-turn: save, title, compress ─────────────────────────
                if not cancel_event.is_set() and orchestrator.conv_id and done_content is not None:
                    # Save the clean user + assistant turn to DB
                    db.save_messages(orchestrator.conv_id, [
                        {"role": "user",      "content": message},
                        {"role": "assistant", "content": done_content},
                    ])

                    # Title generation on the very first turn of a new conversation
                    if was_new_conv:
                        orchestrator._is_new_conv = False
                        title = orchestrator.generate_title(message)
                        db.update_title(orchestrator.conv_id, title)
                        loop.call_soon_threadsafe(queue.put_nowait, {
                            "type": "title",
                            "conv_id": orchestrator.conv_id,
                            "title": title,
                        })

                    # Summarize-and-compress when context is filling up
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

    return EventSourceResponse(event_stream())


@app.post("/reset")
async def reset():
    """Start a fresh conversation (old one stays in DB). Preserves active project."""
    project = orchestrator.project
    project_id = project["id"] if project else None
    orchestrator.reset_conversation()
    conv_id = db.create_conversation(orchestrator.model, project_id=project_id)
    orchestrator.new_conversation(conv_id, project=project)
    return {"status": "ok", "conv_id": conv_id, "title": "New conversation"}


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


# ── Conversation endpoints ────────────────────────────────────────────────────

@app.get("/conversations")
async def list_conversations():
    return {"conversations": db.list_conversations()}


class CreateConversationRequest(BaseModel):
    project_id: str = ""


@app.post("/conversations")
async def create_conversation(body: Optional[CreateConversationRequest] = None):
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
async def delete_conversation(conv_id: str):
    db.delete_conversation(conv_id)
    # If we just deleted the active conversation, start a new one
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
async def rag_list():
    return {"documents": orchestrator.rag_engine.list_documents()}


@app.delete("/rag/documents/{name:path}")
async def rag_remove(name: str):
    orchestrator.rag_engine.remove(name)
    return {"documents": orchestrator.rag_engine.list_documents()}


@app.post("/verbose")
async def set_verbose(request: VerboseRequest):
    orchestrator.verbose = request.enabled
    return {"verbose": orchestrator.verbose}


@app.get("/status")
async def status():
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
        resolved = os.path.realpath(path)
        if not os.path.isdir(resolved):
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
        return {
            "path": resolved,
            "parent": parent if parent != resolved else None,
            "entries": entries,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AskRequest(BaseModel):
    prompt: str
    system: str = ""


@app.post("/ask")
async def ask(body: AskRequest):
    """One-shot ephemeral query — no conversation saved, no tools, no DB writes.
    Returns {"response": "..."}. Intended for orchestrator delegation from Claude Code."""
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt required")
    messages = []
    if body.system.strip():
        messages.append({"role": "system", "content": body.system.strip()})
    messages.append({"role": "user", "content": body.prompt.strip()})
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: orchestrator._ollama.chat(
                model=orchestrator.model,
                messages=messages,
                stream=False,
            ),
        )
        return {"response": (resp.message.content or "").strip()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    import asyncio
    import signal
    import subprocess
    import time

    # Kill any stale server.py from a previous run or Xcode dev session.
    # Temporarily ignore SIGTERM so pkill doesn't kill this process when it
    # matches our own argv — old instances die, we continue.
    _old_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.run(["/usr/bin/pkill", "-f", "python.*server\\.py"], capture_output=True)
    signal.signal(signal.SIGTERM, _old_sigterm)
    time.sleep(0.4)

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
            # Only one server should install signal handlers; let http_server own them.
            https_server.install_signal_handlers = lambda: None
            await asyncio.gather(http_server.serve(), https_server.serve())
        else:
            await http_server.serve()

    asyncio.run(_run())
