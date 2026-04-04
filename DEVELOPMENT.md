# Development Log

This document captures the development history of the ollama Search Tool: the decisions made, the problems encountered, and how they were resolved. It is written after the fact, reconstructed from the full build session.

## Tooling

The code for this project was written and debugged collaboratively with **[Claude Code](https://claude.ai/code)** (Anthropic's CLI coding assistant), running the `claude-sonnet-4-6` model. Claude Code was used for:

- Designing the overall architecture and event-based streaming model
- Writing all source files from scratch
- Diagnosing and fixing bugs in the Ollama integration
- Designing and implementing the FastAPI + SSE web interface
- Implementing file attachment support (Phase 2)
- Writing unit tests and all documentation

The user directed all goals, reviewed all output, and made final decisions on every design tradeoff.

---

## Phase 1 — Core CLI + Web Interface

### Project setup

The project uses `uv` (not `pip` or `venv`) for dependency management. `uv` was chosen for fast resolution and reproducible lockfiles. The first issue encountered was a `hatchling` build error on `uv sync` caused by a `[build-system]` section in `pyproject.toml`. Fix: remove `[build-system]` and add `package = false` under `[tool.uv]` to signal this is an application, not a library.

The repository was initialised locally and pushed to `github.com/mabaeyens/ollama-web-search`. GitHub had auto-created a README and `.gitignore` on the remote, causing the initial push to be rejected. Resolved with `--allow-unrelated-histories` and kept our versions with `git checkout --ours`.

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

## Known limitations and Phase 3

- **Large documents:** Files above 80k characters are truncated. RAG (chunking + vector search + reranking) is the correct solution for full-document reasoning. An existing implementation is available at `github.com/mabaeyens/RAG` and is the planned basis for Phase 3.
- **Scanned PDFs:** No OCR. The user must provide a text-based PDF or attach individual pages as images.
- **Image reasoning quality:** Depends entirely on Gemma 4's multimodal capabilities for the specific image type.
- **Attachment persistence:** Files are single-turn only. There is no mechanism to "keep this document in context for the whole session" without re-attaching.
