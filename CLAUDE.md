# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**ollama Search Tool** — a local AI assistant using Gemma 4:26b via Ollama with autonomous web search, file attachments (PDF/HTML/images/text), and RAG for large documents. Available as a CLI and a local web interface.

## Commands

```bash
# First-time setup
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Pull required Ollama models
ollama pull gemma4:26b
ollama pull nomic-embed-text   # used for RAG embeddings

# Start the Ollama daemon (required before running CLI or web server)
# macOS: Ollama.app in menu bar starts it automatically; or run manually:
ollama serve

# Run the CLI
python main.py

# Run the web interface (open http://localhost:8000)
python server.py

# Add a dependency
uv add <package>

# Run all tests (no Ollama needed — model calls are mocked)
uv run pytest

# Run a single test
uv run pytest tests/test_queries.py::test_toggle_verbose
```

Requires Ollama running locally (`http://localhost:11434` by default) with `gemma4:26b` and `nomic-embed-text` pulled. Override host with `OLLAMA_HOST` env var.

## Architecture

```
main.py (CLI loop + _render_stream)     server.py (FastAPI + SSE)
         └──────────────────────────────────────┘
                         │
              ChatOrchestrator (orchestrator.py)
                    │  stream_chat(user_message, attachments=None) → yields events
                    ├── _call_ollama() → ollama.chat(stream=True)
                    ├── SearchEngine (search_engine.py) → ddgs.text()
                    └── RagEngine (rag_engine.py)
                              ├── ollama.embed() → nomic-embed-text
                              ├── chromadb.EphemeralClient (in-memory)
                              └── CrossEncoder (sentence-transformers)

file_handler.py  — load_file() / load_file_bytes(): PDF→RAG, HTML→text, image→base64
config.py        — all tunables including RAG_* knobs
tools.py         — Ollama tool schema for web_search
prompts.py       — build_system_prompt() injects today's date + search rules
formatter.py     — Rich console helpers (CLI only)
static/index.html — single-page web UI (vanilla HTML/CSS/JS + marked.js)
```

**Event-based streaming flow:** `ChatOrchestrator.stream_chat()` yields typed events consumed by both CLI and web server:

| Event | Payload | Meaning |
|-------|---------|---------|
| `thinking` | — | Model is processing |
| `token` | `content` | Answer token (buffered by CLI, streamed by web) |
| `search_start` | `query` | Web search beginning |
| `search_done` | `query, count, results` | Search complete; `results` carries `{title, url}` per result (snippet stripped before SSE) |
| `fetch_start` | `url` | Page fetch beginning |
| `fetch_done` | `url, chars` | Fetch complete; `chars` is length of text returned |
| `fetch_context` | `fetches` | All pages fetched this turn; list of `{url, chars, preview}` — emitted right before `done` so the UI can render a "pages read" panel |
| `rag_indexing` | `name` | Document being indexed into RAG |
| `rag_done` | `name, chunks` | Indexing complete |
| `rag_context` | `chunks` | RAG chunks injected this turn; `chunks` is a list of `{source, score, preview}` — emitted right before `done` so the UI can render a "document sections used" panel |
| `warning` | `message` | Non-fatal issue (scanned PDF, chunk limit) |
| `done` | `content` | Turn complete, full answer |
| `error` | `message` | Fatal error |

**Mockable boundary:** `_call_ollama()` is the single point tests mock — returns an iterable of stream chunks.

**RAG flow:** PDFs always go through RAG. HTML/text > 80k chars also go through RAG. On every turn, if the RAG index is non-empty, `rag_engine.query()` retrieves and reranks chunks; those scoring > 0.0 are prepended to the user message as `[Relevant document sections]`.

**RAG score threshold bypass (anti-hallucination):** `rag_engine.query()` normally drops chunks scoring ≤ `RAG_SCORE_THRESHOLD` (default 0.0) after CrossEncoder reranking. This prevents injecting irrelevant context — but it creates a silent failure when the user's message is a meta-instruction like "summarize this" or "translate this to English". Those phrases embed nothing like document content, so all chunks score negative and get dropped. The model receives zero context and hallucinates that no file was attached.

Fix in `orchestrator.py`: when any RAG file was indexed in the current turn (`rag_indexed_this_turn = True`), `query()` is called with `score_threshold=float('-inf')`, retrieving top-K chunks unconditionally. On subsequent turns the normal threshold applies. The intent: the user explicitly attached a file this turn, so we must always give the model *something* from it, even if the query is oblique.

Do not remove this bypass. The hallucination it prevents ("No has adjuntado ningún artículo") is worse than occasionally injecting a lower-quality chunk.

**Search:** Ollama native search disabled (`USE_NATIVE_SEARCH = False`). DuckDuckGo (`ddgs`) is active.

**Why DDGS over Ollama native search:** Ollama does offer a free-tier web search API (free account + API key), but signup requires a phone number and Ollama's docs contain no privacy disclosures — queries are tied to an authenticated account and can be logged. DuckDuckGo's core value proposition is no-tracking search. Given this project is local-first and privacy-conscious, DDGS is the deliberate choice. Additionally, `gemma4:26b` is not listed as a supported model for Ollama native search, so compatibility is unverified.

## Configuration

All tunables in `config.py` (not git-ignored, no `.env`):
- `MODEL_NAME`, `OLLAMA_HOST`
- `USE_NATIVE_SEARCH`, `MAX_SEARCH_RESULTS`, `SEARCH_TIMEOUT`, `MAX_RETRIES`, `MAX_TOOL_STEPS`
- `VERBOSE_DEFAULT`
- `EMBED_MODEL` — `nomic-embed-text` (Ollama)
- `RERANK_MODEL` — `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers)
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_RETRIEVE_K`, `RAG_RERANK_TOP_K`, `RAG_SCORE_THRESHOLD`, `RAG_MAX_CHUNKS`

## Cancel / Stop button

**How it works (single-user app):**
- `cancel_event = threading.Event()` is module-level in `server.py`
- Each `/chat` request clears it; `POST /cancel` sets it
- `produce()` thread checks `cancel_event.is_set()` before each `queue.put_nowait` — breaks out immediately when set
- `event_stream()` receives the `None` sentinel from produce, checks cancel_event, and truncates `orchestrator.conversation_history` back to the length it was before the turn started — leaving history clean as if the turn never happened
- Client JS calls `POST /cancel` first (awaits), then `reader.cancel()` to close the SSE stream
- Stop button replaces Send button during streaming; swaps back on completion or cancel

**Key constraint:** History rollback relies on `event_stream()` receiving the `None` sentinel from `produce()` before the SSE connection closes. This works because the client awaits `POST /cancel` (which causes `produce()` to stop quickly) before calling `reader.cancel()`. If the connection drops before `None` arrives (network issue, browser crash), history is left with the partial user message — not catastrophic; user can hit Reset.

## Tests

`tests/test_queries.py` — all model and search calls mocked via `unittest.mock`. No Ollama instance needed.

Mock pattern: `patch.object(orchestrator, '_call_ollama', return_value=iter([chunk]))` where each chunk is a `MagicMock` with `.message.content`, `.message.tool_calls`, `.done`.

Tests cover: search trigger behaviour, search_done event payload, `fetch_url` dispatch, Gemma4 intermediate-chunk tool calls (`accumulated_tool_calls`), RAG threshold bypass for same-turn attachments, verbose toggle, conversation reset.

`tests/test_cancel.py` — uses FastAPI `TestClient`; mocks `orchestrator.stream_chat` directly (no `_call_ollama` mock needed). Tests cover: cancel endpoint, cancel cleared on new chat, events dropped after cancel, history rollback on cancel.

## Constraints

- Always use `uv` — never `pip` or `venv`.
- Keep all models and search local — no cloud APIs or API keys.
- Do not change `MODEL_NAME` from `gemma4:26b` unless explicitly requested.
- Do not set `USE_NATIVE_SEARCH = True` — Ollama native search is free-tier but requires an account (phone number at signup) and has no privacy guarantees; DDGS was chosen deliberately for privacy.
- Do not change `EMBED_MODEL` from `nomic-embed-text` unless explicitly requested.
- `ChatOrchestrator` must remain display-agnostic — no print/console calls in `orchestrator.py`.
- Tests mock `_call_ollama`, not higher-level methods. Mock chunks need `.message.content`, `.message.tool_calls`, `.done`.
- ChromaDB `EphemeralClient` in `rag_engine.py` uses UUID collection names — required because ChromaDB 1.x shares a Rust backend per process; fixed collection names cause collisions in tests.

## Working style

**Proceed without confirmation** for file edits and git operations (add, commit, push) on this repo. It is a private local project with no destructive risk.

**After any non-trivial fix, write a test before closing the session.** The web search tool-call bug (accumulated_tool_calls) had no offline reproduction path — a mock-based test would have validated it in the same session.

**Distinguish bug classes early:**
- Empty/wrong response → likely a code bug (streaming, tool call handling, SSE)
- Model gives a poor answer despite having data → prompt engineering, not code

**Prompt engineering debugging loop** — when a model ignores an instruction:
1. Observe the exact failure ("model hedges instead of searching")
2. Find what the prompt said that allowed it ("If you didn't search, explain why")
3. Close the loophole with a direct rule ("Never say 'I recommend checking'")

That loop is more effective than any technique from a course.

**Gemma4 quirks to keep in mind:**
- Emits `tool_calls` in an intermediate streaming chunk (`done=False`), not in the final `done=True` chunk — the `accumulated_tool_calls` pattern in `orchestrator.py` handles this.
- Occasionally emits LaTeX math notation (e.g. `$\rightarrow$`) — `preprocessLatex()` in `index.html` converts common commands to Unicode before rendering.
