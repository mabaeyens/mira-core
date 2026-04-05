"""FastAPI server for the ollama Search Tool web interface."""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager

from typing import List

import uvicorn
from fastapi import FastAPI, Form, UploadFile, File

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import file_handler
from config import VERBOSE_DEFAULT
from orchestrator import ChatOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

orchestrator: ChatOrchestrator = None
cancel_event = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = ChatOrchestrator(verbose=VERBOSE_DEFAULT)
    logger.info(f"Initialized orchestrator with model: {orchestrator.model}")
    yield


app = FastAPI(title="ollama Search Tool", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class VerboseRequest(BaseModel):
    enabled: bool


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/cancel")
async def cancel():
    cancel_event.set()
    return {"status": "cancelled"}


@app.post("/chat")
async def chat(
    message: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    paths: List[str] = Form(default=[]),
):
    """SSE endpoint — streams typed events from stream_chat() to the browser."""
    cancel_event.clear()
    history_snapshot_len = len(orchestrator.conversation_history)

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

    # Load server-side file paths (typed directly in the web UI)
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

        def produce():
            try:
                for event in orchestrator.stream_chat(message, attachments=attachments or None):
                    if cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, event)
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
                # Send a real data event so the browser's ReadableStream reader
                # wakes up — SSE comment pings (ping=N) are invisible to fetch()
                # streams in some browsers and the connection silently drops.
                yield {"data": json.dumps({"type": "heartbeat"})}
                continue
            if event is None:
                # Produce thread finished — rollback history if cancelled
                if cancel_event.is_set():
                    orchestrator.conversation_history = orchestrator.conversation_history[:history_snapshot_len]
                break
            # Strip snippets — browser only needs title + url for the sources list
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
    orchestrator.reset_conversation()
    return {"status": "ok"}


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
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000)
