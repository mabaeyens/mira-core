# Architecture Reference

Detailed reference for subsystems. Read this file when working on events, RAG, cancel, config, or model quirks.

## Module map

```
main.py (CLI)     server.py (FastAPI + SSE)
        └─────────────────┘
                 │
      core/orchestrator.py → ChatOrchestrator
            │  stream_chat(user_message, attachments=None, thinking_enabled=True) → yields events
            ├── _call_llm() → openai.chat.completions.create() (mlx-lm or ollama, OpenAI-compatible)
            ├── core/search_engine.py → SearchEngine → ddgs.text()
            └── core/rag_engine.py → RagEngine
                      ├── SentenceTransformer (nomic-ai/nomic-embed-text-v1.5, local, 768 dims)
                      ├── chromadb.EphemeralClient (in-memory)
                      └── CrossEncoder (sentence-transformers)

core/config.py       — all tunables; loads overrides from mira.yaml (git-ignored)
core/file_handler.py — load_file() / load_file_bytes(): PDF→RAG, HTML→text, image→base64
core/tools.py        — OpenAI-compatible tool schema for web_search and all tools
core/prompts.py      — build_system_prompt() injects today's date + search rules
core/formatter.py    — Rich console helpers (CLI only)
core/db.py           — SQLite conversation persistence
core/workspace.py    — sandbox path enforcement
core/fs_tools.py     — filesystem tool implementations
core/shell_tools.py  — shell execution tool
core/github_tools.py — GitHub API tool
static/index.html    — single-page web UI (vanilla HTML/CSS/JS + marked.js)
```

## Event protocol

`ChatOrchestrator.stream_chat()` yields typed dicts consumed by both CLI (`main.py`) and web server (`server.py`):

| Event | Payload | Meaning |
|-------|---------|---------|
| `thinking` | `content` (optional) | Model is processing; thinking text arrives in `content` (Ollama backend only — disabled on mlx-lm) |
| `token` | `content` | Answer token (buffered by CLI, streamed by web) |
| `search_start` | `query` | Web search beginning |
| `search_done` | `query, count, results` | Search complete; `results` is `[{title, url}]` (snippet stripped before SSE) |
| `fetch_start` | `url` | Page fetch beginning |
| `fetch_done` | `url, chars` | Fetch complete; `chars` is length of text returned |
| `fetch_context` | `fetches` | All pages fetched this turn; `[{url, chars, preview}]` — emitted right before `done` |
| `rag_indexing` | `name` | Document being indexed |
| `rag_done` | `name, chunks` | Indexing complete |
| `rag_context` | `chunks` | RAG chunks injected this turn; `[{source, score, preview}]` — emitted right before `done` |
| `stats` | `input_tokens, output_tokens, context_pct` | Cumulative session token counts — emitted just before `done` |
| `warning` | `message` | Non-fatal issue (scanned PDF, chunk limit) |
| `done` | `content` | Turn complete, full answer |
| `error` | `message` | Fatal error |
| `title` | `conv_id, title` | New conversation title generated — emitted after `done`, only on first turn |
| `compress` | `message` | Context window compressed — emitted after `done` when `context_pct` exceeded threshold |
| `heartbeat` | — | Keepalive — emitted periodically during long tool calls to prevent connection timeout |

**Thinking toggle:** `stream_chat()` accepts `thinking_enabled: bool = True`. On the **Ollama backend**, passes `think=thinking_enabled`; Gemma4 yields thinking text in `chunk.message.thinking`. On **mlx-lm**, thinking is suppressed server-side via `--chat-template-args '{"enable_thinking": false}'` (template-level); `max_thinking_tokens` is not yet exposed as an API param in 0.31.3, so the UI toggle is disabled while on mlx-lm. The server form field `thinking_enabled` (default `true`) flows from the app's thinking toggle button in the input bar.

**Mockable boundary:** `_call_llm()` is the single point tests mock — returns an iterable of stream chunks with `.message.content`, `.message.tool_calls`, `.done`.

## RAG internals

**Flow:** PDFs always go through RAG. HTML/text > 80k chars also go through RAG. On every turn where the RAG index is non-empty, `rag_engine.query()` retrieves and reranks chunks; those scoring > `RAG_SCORE_THRESHOLD` (default 0.0) are prepended to the user message as `[Relevant document sections]`.

**Score threshold bypass (anti-hallucination):** When the user attaches a file in the current turn (`rag_indexed_this_turn = True` in `orchestrator.py`), `query()` is called with `score_threshold=float('-inf')`, retrieving top-K chunks unconditionally. On subsequent turns the normal threshold applies.

Why this matters: meta-instructions like "summarize this" or "translate this to English" embed nothing like document content, so all chunks score negative and get dropped without the bypass. The model then receives zero context and hallucinates that no file was attached ("No has adjuntado ningún artículo"). Do not remove this bypass.

## File loading

Two paths into `stream_chat(attachments=[...])`:
- CLI: `/attach <path>` → `file_handler.load_file(path)`
- Web upload: multipart → `load_file_bytes(name, data)`
- Web path field: `paths[]` form field → `load_file(path)` server-side (same machine, absolute path)

## Cancel / Stop mechanism

- `cancel_event = threading.Event()` is module-level in `server.py`
- Each `/chat` request clears it; `POST /cancel` sets it
- `produce()` thread checks `cancel_event.is_set()` before each `queue.put_nowait` — breaks out immediately when set
- `event_stream()` receives the `None` sentinel from `produce()`, checks cancel_event, and truncates `orchestrator.conversation_history` back to pre-turn length — as if the turn never happened
- Client JS awaits `POST /cancel` first, then calls `reader.cancel()` to close the SSE stream

**Key constraint:** History rollback relies on `event_stream()` receiving the `None` sentinel before the SSE connection closes. If the connection drops before `None` arrives (network issue, browser crash), history is left with the partial user message — not catastrophic, user can hit Reset.

## Context compression

When `context_pct` exceeds `COMPRESS_THRESHOLD` (default 70%), the orchestrator compresses conversation history at the end of the turn:

- Keeps the system prompt and the most recent `COMPRESS_KEEP_RECENT` messages (default 6) verbatim
- Summarises older messages into a single assistant message using the LLM
- Emits a `compress` event after `done` so clients can show a notice
- The compressed history is written back to `orchestrator.conversation_history` and saved to the database

This is transparent to clients — the next turn proceeds normally with a shorter history. Token counts in `stats` reflect the compressed window going forward.

## Endpoint reference

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat` | Stream a turn (multipart: `message`, `conversation_id`, `files[]`, `paths[]`) |
| `POST` | `/cancel` | Abort in-progress response; triggers history rollback |
| `POST` | `/reset` | New conversation (preserves active project) |
| `GET` | `/health` | `200` ready, `503` starting, other → unavailable |
| `GET` | `/status` | Model name, cumulative tokens, `context_pct`, `workspace_root` |
| `GET` | `/browse` | Directory listing (sandboxed to `$HOME`); query param `path` |
| `POST` | `/ask` | One-shot ephemeral query — no tools, no DB writes |
| `GET/POST` | `/projects` | List / create projects |
| `DELETE` | `/projects/{id}` | Delete project |
| `GET/POST` | `/conversations` | List / create conversations |
| `PATCH/DELETE` | `/conversations/{id}` | Rename / delete conversation |
| `GET` | `/conversations/{id}/messages` | Full message history |
| `GET` | `/info` | Model name, backend, host, context_window, hardware |
| `GET` | `/backend` | Current backend/model/host/context_window |
| `POST` | `/backend` | Switch inference backend (`{"backend": "ollama"}`); blocks until ready |
| `GET` | `/rag/documents` | List indexed RAG documents |
| `DELETE` | `/rag/documents/{name}` | Remove a RAG document |

## iOS/macOS client integration

The native clients (mira-apps) connect to this server over HTTP/HTTPS. Key integration points:

- **Discovery:** macOS connects to `localhost:8000`; iOS discovers the server via Bonjour (`_ollamasearch._tcp`) or a user-configured URL (Tailscale).
- **SSE streaming:** `SSEClient.swift` opens `POST /chat` as an `AsyncThrowingStream<ServerEvent>`, parsing each `data:` line as JSON.
- **Event mapping:** All events in the table above have a corresponding `ServerEvent` Swift enum case consumed by `ChatViewModel`.
- **Cancel:** iOS/macOS send `POST /cancel` then discard the stream; the server rolls back history.
- **File uploads:** Sent as multipart form-data, same schema as the web UI.
- **`title` and `compress` events** arrive after `done`; clients must keep the SSE connection open until the server closes it (signalled by the absence of further events, not by a sentinel).
- **Model switcher:** toolbar label button opens `ModelPickerView` sheet. Tapping an inactive model shows a confirmation step (warns about 30–60 s pause), then calls `POST /backend`. `ChatViewModel.switchBackend(to:)` drives `switchStatusMessage` through timed stages ("Stopping…", "Starting…", "Loading weights…", "Almost ready…") which `ModelPickerView` displays during the switch.
- **Thinking toggle:** brain icon in `InputBar` toggles `thinkingEnabled`; passed as `thinking_enabled` form field to `POST /chat`. Works on both backends.

See `mira-apps/OllamaSearch/Shared/Networking/` for client implementation.

## Why DDGS over Ollama native search

Ollama offers a free-tier web search API, but signup requires a phone number and Ollama's docs contain no privacy disclosures — queries are tied to an authenticated account and can be logged. DuckDuckGo's core value proposition is no-tracking search. This project is local-first and privacy-conscious, so DDGS is the deliberate choice. Additionally, `gemma4:26b` is not listed as a supported model for Ollama native search.

`USE_NATIVE_SEARCH` is `False` in `config.py` — do not change it.

## Backend startup

On Mira startup, `backend_manager.ensure_backend_running()` launches mlx-lm automatically if not already running (binary at `~/.local/bin/mlx_lm.server`, port 8080). The app's `/health` endpoint returns `backend_ready: false` until the inference server is reachable with the configured model, and the iOS/macOS chat view shows a banner with a "Start" button during that time. Ollama (port 11434) must be started manually when RAG embeddings are needed — it is not auto-started.

## Configuration reference

**External config (preferred):** copy `mira.yaml.example` → `mira.yaml` (git-ignored) and edit:

| Field | Default | Notes |
|-------|---------|-------|
| `backend` | `mlx-lm` | `mlx-lm` (default) or `ollama` |
| `model` | `mlx-community/gemma-4-26b-a4b-it-4bit` | Model identifier |
| `host` | `http://localhost:8080` | LLM server URL |
| `embed_model` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model for RAG (sentence-transformers) |
| `context_window` | `65536` | Token context window |

**RAG / search knobs in `core/config.py`** (no `mira.yaml` equivalent — edit directly):
`USE_NATIVE_SEARCH`, `MAX_SEARCH_RESULTS`, `SEARCH_TIMEOUT`, `MAX_RETRIES`, `MAX_TOOL_STEPS`, `VERBOSE_DEFAULT`, `RERANK_MODEL`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_RETRIEVE_K`, `RAG_RERANK_TOP_K`, `RAG_SCORE_THRESHOLD`, `RAG_MAX_CHUNKS`

## Model quirks

Behaviours that are intentional and must not be removed:

- **Gemma4 (mlx-lm):** emits `tool_calls` in an intermediate chunk (`done=False`) — handled by `accumulated_tool_calls` in `orchestrator.py`.
- **Gemma4:** occasionally emits LaTeX (e.g. `$\rightarrow$`) — `preprocessLatex()` in `index.html` converts to Unicode.
- **Gemma4 (Ollama) thinking:** when `think=True`, Ollama yields thinking text in `chunk.message.thinking` (separate from `chunk.message.content`). The streaming loop checks this field and emits `thinking` events. Not applicable on mlx-lm — thinking is disabled at the template level.
- **mlx-lm thinking suppression:** server started with `--chat-template-args '{"enable_thinking": false}'`. The `max_thinking_tokens` API parameter is not yet exposed in mlx-lm 0.31.3; re-enable the UI toggle when it is.

## Test patterns

`tests/test_queries.py` — mocks `_call_llm` via `patch.object`:
```python
patch.object(orchestrator, '_call_llm', return_value=iter([chunk]))
# chunk is a MagicMock with .message.content, .message.tool_calls, .done
```
Covers: search trigger, search_done payload, fetch_url dispatch, Gemma4 intermediate tool calls (`accumulated_tool_calls`), RAG threshold bypass, verbose toggle, conversation reset.

`tests/test_cancel.py` — uses FastAPI `TestClient`; mocks `orchestrator.stream_chat` directly.
Covers: cancel endpoint, cancel cleared on new chat, events dropped after cancel, history rollback.
