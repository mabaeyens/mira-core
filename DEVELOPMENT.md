# Development Log

> **Note (2026-05-30):** The backend has evolved since this log was written. Ollama is now an optional inference backend; mlx-lm handles all inference by default. RAG embeddings use `sentence-transformers` (`nomic-ai/nomic-embed-text-v1.5`) — Ollama is no longer required at runtime. See `docs/architecture.md` for the current state.

This document captures the development history of Mira: the decisions made, the problems encountered, and how they were resolved. It is written after the fact, reconstructed from the full build session.

## Tooling

The code for this project was written and debugged collaboratively with **[Claude Code](https://claude.ai/code)** (Anthropic's CLI coding assistant), running the `claude-sonnet-4-6` model. Claude Code was used for:

- Designing the overall architecture and event-based streaming model
- Writing all source files from scratch
- Diagnosing and fixing bugs in the Ollama integration
- Designing and implementing the FastAPI + SSE web interface
- Implementing file attachment support (Phase 2)
- Designing and implementing the RAG pipeline (Phase 3)
- Writing unit tests and all documentation

The user directed all goals, reviewed all output, and made final decisions on every design tradeoff.

---

## Phase 1 — Core CLI + Web Interface

### Project setup

The project uses `uv` (not `pip` or `venv`) for dependency management. `uv` was chosen for fast resolution and reproducible lockfiles. The first issue encountered was a `hatchling` build error on `uv sync` caused by a `[build-system]` section in `pyproject.toml`. Fix: remove `[build-system]` and add `package = false` under `[tool.uv]` to signal this is an application, not a library.

The repository was initialised locally and pushed to `github.com/mabaeyens/mira-core`. GitHub had auto-created a README and `.gitignore` on the remote, causing the initial push to be rejected. Resolved with `--allow-unrelated-histories` and kept our versions with `git checkout --ours`.

The package `duckduckgo-search` was renamed to `ddgs` upstream. Dependency and import updated accordingly.

### Model and search

The LLM is **Gemma 4:26b** running via **Ollama** (`http://localhost:11434`). Gemma 4 is a mixture-of-experts model with only 4b active parameters per inference step, making it fast on consumer hardware despite its 26b parameter count.

**Ollama native search was disabled.** Ollama's built-in `web_search` tool requires a paid Ollama subscription and was producing `Authorization header with Bearer token is required` warnings. `USE_NATIVE_SEARCH = False` in `config.py`; **DuckDuckGo** (`ddgs`) is used instead — free, no API key, no external dependency.

**Search is autonomous (ReAct pattern).** The model decides when to call the `web_search` tool based on the system prompt. No hardcoded heuristics.

**Problem:** The model was not searching for recent events like "Super Bowl 2026". Root cause: the system prompt didn't include today's date, so the model reasoned that these events were in the future relative to its training cutoff (April 2024). Fix: replaced the static `SYSTEM_PROMPT` constant with a `build_system_prompt()` function that injects the current date dynamically.

**Problem:** `AttributeError: 'ChatResponse' object has no attribute 'tool_calls'`. The Ollama Python client puts tool calls on `response.message.tool_calls`, not `response.tool_calls`. This also caused empty responses (the model was calling a tool but the code silently fell through to printing the empty `message.content`). Fixed in the orchestrator and tests.

### Architecture: event-based streaming

The core architectural decision was to make `ChatOrchestrator` **display-agnostic**. Rather than printing to the terminal or writing to a response object, `stream_chat()` is a generator that yields typed event dicts:

```
{"type": "thinking"}
{"type": "token", "content": "..."}
{"type": "search_start", "query": "..."}
{"type": "search_done", "query": "...", "count": N, "results": [...]}
{"type": "done", "content": "..."}
{"type": "error", "message": "..."}
```

The CLI (`_render_stream` in `main.py`) and the web server (`/chat` SSE endpoint in `server.py`) consume the same events differently. This kept the orchestrator testable in isolation and made the web interface a clean addition rather than a refactor.

`_call_ollama()` was extracted as the single mockable boundary in tests — all unit tests mock this method and require no live Ollama instance.

### CLI

The CLI uses the **Rich** library for spinners, formatted output, and markdown rendering. Arrow key support and input history were enabled with a single `import readline` (no configuration needed on macOS/Linux).

**CLI markdown rendering decision:** The original implementation streamed tokens character-by-character (typewriter effect). This was later changed to buffer all tokens and render the complete response as `rich.markdown.Markdown` after the model finishes. The streaming version was simpler but produced raw unformatted text; the buffered version renders headers, code blocks, and lists correctly. The tradeoff (no live streaming feedback in CLI) was accepted in favour of readable output.

### Web interface

The web interface is a single-page app (`static/index.html`) using vanilla HTML/CSS/JS with no build step. **Markdown rendering** uses `marked.js` loaded from CDN.

**Key implementation detail — SSE from a POST request:** The browser's native `EventSource` API only supports GET requests. Since the chat endpoint needs to receive a message body, the web UI uses the `fetch` API with a `ReadableStream` reader and parses SSE lines manually. This required a line buffer to handle chunk boundaries correctly.

**Async/sync bridge:** `stream_chat()` is a synchronous generator. FastAPI's `/chat` endpoint is async. The solution is a background thread running `stream_chat()` that feeds events into an `asyncio.Queue`, which the async SSE generator drains. This pattern avoids blocking the event loop without requiring a rewrite of the orchestrator.

---

## Phase 2 — File Attachments

### Supported file types

| Type | Extraction method |
|------|------------------|
| PDF (text-based) | PyMuPDF (`fitz`) |
| PDF (scanned) | Detected; user warned; no text extracted |
| HTML | BeautifulSoup — tags stripped, readable text extracted |
| Images (PNG, JPG, GIF, WEBP, …) | Read as bytes, base64-encoded |
| Text, code, and other files | Read as UTF-8 |

**PyMuPDF over pdfplumber:** The original plan listed `pdfplumber`. Switched to `pymupdf` after reviewing the existing RAG project at `github.com/mabaeyens/RAG`, which already used PyMuPDF in production. PyMuPDF is faster and handles a wider range of PDF variants. Reusing a validated dependency was the right call.

**Scanned PDFs:** PyMuPDF extracts an empty string when a PDF has no text layer (i.e., it is a scan). Rather than silently sending an empty attachment, the handler detects this and emits a `warning` event directing the user to try attaching individual pages as images instead.

**Context window guard:** Extracted text is capped at 80,000 characters before being injected into the conversation. Gemma 4:26b has a 128k context window, but large documents leave insufficient room for the conversation itself. If a file is truncated, a `warning` event is emitted. For very large documents, RAG (Phase 3) is the correct path.

### Attachment persistence

Attachments are **single-turn**: the file content is prepended to the user's message for that turn and is not stored separately in `conversation_history`. This keeps the implementation simple and avoids unbounded context growth. The full prepended message is stored in history (the model can refer back to it), but no special attachment tracking is needed.

### Ollama multimodal API

Images cannot be appended as text. The Ollama API expects a separate `images` key on the message dict:

```python
{"role": "user", "content": "describe this", "images": ["<base64>"]}
```

`stream_chat()` was extended to accept an optional `attachments` list. Text attachments are prepended to `user_message` under `[File: name]` headers. Image attachments populate the `images` field on the message dict.

### Web upload

The `/chat` endpoint was changed from accepting `{"message": str}` JSON to **multipart form data** (`FormData` on the client, FastAPI `Form` + `UploadFile` on the server). This required adding `python-multipart` as a dependency (FastAPI does not include it by default). File bytes are read in the async context before the background thread is spawned, avoiding any cross-thread I/O.

The web UI adds a 📎 button that opens a file picker. Selected files are shown as chips with individual remove buttons. On send, file chips are also shown in the user's chat bubble for context.

### New event type: `warning`

A `{"type": "warning", "message": "..."}` event was added for non-fatal issues: scanned PDFs, truncated content, and file processing errors. CLI renders it in yellow; web renders it as an amber chip. This keeps the existing `error` event reserved for fatal failures.

---

## Phase 3 — RAG

### Embedding model choice

Rather than using `sentence-transformers` for both embedding and reranking (as in the original RAG repo), Phase 3 uses **`nomic-embed-text` via Ollama** for embeddings. This keeps embeddings consistent with the existing Ollama interface and eliminates the ~1.5GB `sentence-transformers`+PyTorch download for that role. The CrossEncoder reranker still uses `sentence-transformers` because no reranker model is available through Ollama. Net result: `sentence-transformers` is a dependency but only for ~100MB of reranker weights, not the full embedding stack.

### ChromaDB version and in-memory store

The original RAG repo was built against ChromaDB 0.5.15 and had extensive defensive code (batch sleeps, recovery loops, telemetry-dir bugs) caused by SQLite persistence issues. Phase 3 uses the latest stable ChromaDB (1.5.5) with `EphemeralClient()` — fully in-memory, no SQLite, no file locking. Almost all the original defensive code became unnecessary.

**Discovered issue:** ChromaDB 1.x uses a shared Rust backend per process. Two `EphemeralClient()` instances in the same process share the backend, so creating a collection named `"docs"` in a second instance raises `InternalError: Collection [docs] already exists`. Fix: UUID suffix per collection name (`docs_{uuid4().hex[:8]}`). This affected tests, where the `orchestrator` pytest fixture creates a new `ChatOrchestrator` (and thus a new `RagEngine`) for each test.

### RAG repo integration decision

The existing `github.com/mabaeyens/RAG` repo was reviewed as a potential base. Decision: **rewrite from scratch** in `mira-core`. Reasons:
1. The original had complexity specific to `PersistentClient` that doesn't apply to `EphemeralClient`.
2. The `_log_memory_usage` method referenced `self.telemetry_dir` which was never assigned in `__init__` (latent bug).
3. Copying the code would have imported problems that don't exist with the new stack.
4. The RAG repo remains useful as a standalone bulk ingestor; no cross-repo dependency was created.

### RAG trigger strategy

Decision: **PDFs always go through RAG** regardless of size. HTML/text files above 80k chars also go through RAG. Rationale: PDFs are structurally long-form documents; consistent behaviour is more important than the marginal overhead of indexing a 2-page PDF. Local models have no token cost, so the tradeoff calculus differs from cloud APIs.

The previous 80k-char truncation + warning behaviour is replaced: `file_handler._guard()` now upgrades the attachment type to `"rag"` instead of truncating.

### Cross-turn retrieval

When the RAG index is non-empty, `stream_chat()` queries the index on every message — no re-attachment required. Chunks are injected only if their CrossEncoder score exceeds 0.0 (negative scores indicate irrelevance). For questions unrelated to any indexed document, the CrossEncoder naturally filters all chunks out.

### Memory management

`RagEngine.clear()` recreates the `EphemeralClient` and collection rather than calling `collection.delete()`. ChromaDB does not reliably release vector memory through collection deletion alone in CPython. Recreating the client object allows the garbage collector to free the memory.

---

---

## Phase 4 — RAG Chunk Sourcing

### Goal

When the model answers using RAG context, the user should be able to see which document sections were injected — the source filename, CrossEncoder score, and a text preview. This closes the "black box" problem where the model gives an answer but the user can't verify where it came from.

### Implementation

A new `rag_context` event was added to `ChatOrchestrator.stream_chat()`. It is emitted right before the `done` event whenever `rag_chunks` is non-empty:

```python
yield {
    "type": "rag_context",
    "chunks": [
        {"source": c["source"], "score": round(c["score"], 2), "preview": c["text"][:150] + "…"}
        for c in rag_chunks
    ],
}
```

The event is emitted at the end of the turn (after model tokens, before `done`) so the UI can attach it cleanly below the completed answer bubble.

**UI pattern:** In `index.html`, the `rag_context` event is buffered in `pendingRagContext`. The `done` handler consumes it by calling `appendRagContext(chunks, doneWrapper)`, which appends a collapsible chip directly to the assistant message wrapper element (not inside the bubble). The chip reads "📚 N document section(s) used ▴" and is expanded by default; clicking it collapses/expands and toggles the arrow. Each row shows the document name, score, and a 150-char italic preview.

The green color palette (`#f0fdf4 / #86efac / #166534`) was reused from the existing RAG document chips for visual consistency.

### Tests added

- `test_rag_context_event_emitted_when_chunks_retrieved` — mocks `rag_engine.query()` returning 2 chunks, verifies `rag_context` event payload (source, score, preview) and ordering before `done`.
- `test_rag_context_not_emitted_when_no_chunks` — verifies the event is absent when `query()` returns empty.

---

## Known limitations

- **Scanned PDFs**: No OCR. The user must provide a text-based PDF or attach individual pages as images.
- **Image reasoning quality**: Depends entirely on Gemma 4's multimodal capabilities.
- **RAG index not persisted**: The in-memory index is lost on server restart. Documents must be re-attached each session. A persistent option (ChromaDB `PersistentClient`) is a potential future addition.
