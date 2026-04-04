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
| `search_done` | `query, count, results` | Search complete |
| `rag_indexing` | `name` | Document being indexed into RAG |
| `rag_done` | `name, chunks` | Indexing complete |
| `warning` | `message` | Non-fatal issue (scanned PDF, chunk limit) |
| `done` | `content` | Turn complete, full answer |
| `error` | `message` | Fatal error |

**Mockable boundary:** `_call_ollama()` is the single point tests mock — returns an iterable of stream chunks.

**RAG flow:** PDFs always go through RAG. HTML/text > 80k chars also go through RAG. On every turn, if the RAG index is non-empty, `rag_engine.query()` retrieves and reranks chunks; those scoring > 0.0 are prepended to the user message as `[Relevant document sections]`.

**Search:** Ollama native search disabled (`USE_NATIVE_SEARCH = False`) — requires paid subscription. DuckDuckGo (`ddgs`) is active.

## Configuration

All tunables in `config.py` (not git-ignored, no `.env`):
- `MODEL_NAME`, `OLLAMA_HOST`
- `USE_NATIVE_SEARCH`, `MAX_SEARCH_RESULTS`, `SEARCH_TIMEOUT`, `MAX_RETRIES`, `MAX_TOOL_STEPS`
- `VERBOSE_DEFAULT`
- `EMBED_MODEL` — `nomic-embed-text` (Ollama)
- `RERANK_MODEL` — `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers)
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_RETRIEVE_K`, `RAG_RERANK_TOP_K`, `RAG_SCORE_THRESHOLD`, `RAG_MAX_CHUNKS`

## Tests

`tests/test_queries.py` — all model and search calls mocked via `unittest.mock`. No Ollama instance needed.

Mock pattern: `patch.object(orchestrator, '_call_ollama', return_value=iter([chunk]))` where each chunk is a `MagicMock` with `.message.content`, `.message.tool_calls`, `.done`.

Tests cover: search trigger behaviour, search_done event payload, verbose toggle, conversation reset.

## Constraints

- Always use `uv` — never `pip` or `venv`.
- Keep all models and search local — no cloud APIs or API keys.
- Do not change `MODEL_NAME` from `gemma4:26b` unless explicitly requested.
- Do not set `USE_NATIVE_SEARCH = True` — requires paid Ollama subscription.
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

**Gemma4 quirks to keep in mind:**
- Emits `tool_calls` in an intermediate streaming chunk (`done=False`), not in the final `done=True` chunk — the `accumulated_tool_calls` pattern in `orchestrator.py` handles this.
- Occasionally emits LaTeX math notation (e.g. `$\rightarrow$`) — `preprocessLatex()` in `index.html` converts common commands to Unicode before rendering.
