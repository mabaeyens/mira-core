# CLAUDE.md

## Project

**Mira** — local AI assistant using Gemma 4:26b via Ollama with autonomous web search, file attachments (PDF/HTML/images/text), and RAG for large documents. CLI + local web interface.

## Reference docs — read when working on these subsystems

| File | When to read |
|------|-------------|
| `docs/architecture.md` | Event protocol, RAG internals, cancel mechanism, DDGS rationale, test mock patterns |
| `hardware-specs.md` | M5 hardware specs and Ollama env var rationale |
| `DEVELOPMENT.md` | Full decision log and build history |

## Commands

```bash
uv sync                                               # install dependencies
python main.py                                        # CLI
python server.py                                      # web interface → http://localhost:8000
uv add <package>                                      # add dependency
uv run pytest                                         # all tests (no Ollama needed)
uv run pytest tests/test_queries.py::test_name        # single test
```

Requires Ollama running (`http://localhost:11434`) with `gemma4:26b` and `nomic-embed-text` pulled. Override with `OLLAMA_HOST` env var.

## Architecture

```
main.py (CLI)     server.py (FastAPI + SSE)
        └─────────────────┘
                 │
      ChatOrchestrator (orchestrator.py)
            │  stream_chat(user_message, attachments=None) → yields events
            ├── _call_ollama() → ollama.chat(stream=True)
            ├── SearchEngine (search_engine.py) → ddgs.text()
            └── RagEngine (rag_engine.py)
                      ├── ollama.embed() → nomic-embed-text
                      ├── chromadb.EphemeralClient (in-memory)
                      └── CrossEncoder (sentence-transformers)

file_handler.py   — load_file() / load_file_bytes(): PDF→RAG, HTML→text, image→base64
config.py         — all tunables including RAG_* knobs
tools.py          — Ollama tool schema for web_search
prompts.py        — build_system_prompt() injects today's date + search rules
formatter.py      — Rich console helpers (CLI only)
static/index.html — single-page web UI (vanilla HTML/CSS/JS + marked.js)
```

Events yielded by `stream_chat()`: `thinking`, `token`, `search_start/done`, `fetch_start/done/context`, `rag_indexing/done/context`, `stats`, `warning`, `done`, `error`. Full payload schema in `docs/architecture.md`.

**RAG:** PDFs always go through RAG. HTML/text > 80k chars also go through RAG. When files are indexed in the current turn, score threshold is bypassed (`float('-inf')`) to prevent hallucination on meta-instructions like "summarize this" — do not remove this. Full explanation in `docs/architecture.md`.

**Cancel:** `POST /cancel` sets a `threading.Event`; `produce()` breaks out; `event_stream()` rolls back `conversation_history` to pre-turn length. Details in `docs/architecture.md`.

## Configuration

All tunables in `config.py` (not git-ignored, no `.env`):
- `MODEL_NAME`, `OLLAMA_HOST`
- `USE_NATIVE_SEARCH`, `MAX_SEARCH_RESULTS`, `SEARCH_TIMEOUT`, `MAX_RETRIES`, `MAX_TOOL_STEPS`
- `VERBOSE_DEFAULT`
- `EMBED_MODEL` — `nomic-embed-text` (Ollama)
- `RERANK_MODEL` — `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers)
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_RETRIEVE_K`, `RAG_RERANK_TOP_K`, `RAG_SCORE_THRESHOLD`, `RAG_MAX_CHUNKS`

## Constraints

- Always use `uv` — never `pip` or `venv`.
- Keep all models and search local — no cloud APIs or API keys.
- Do not change `MODEL_NAME` from `gemma4:26b` unless explicitly requested.
- Do not set `USE_NATIVE_SEARCH = True` — DDGS is a deliberate privacy choice (see `docs/architecture.md`).
- Do not change `EMBED_MODEL` from `nomic-embed-text` unless explicitly requested.
- `ChatOrchestrator` must remain display-agnostic — no print/console calls in `orchestrator.py`.
- Tests mock `_call_ollama`, not higher-level methods. Mock chunks need `.message.content`, `.message.tool_calls`, `.done`.
- ChromaDB `EphemeralClient` uses UUID collection names — required; fixed names cause collisions in tests.

## Working style

**Proceed without confirmation** for file edits and git operations on this repo.

**After any non-trivial fix, write a test before closing the session.**

**Bug triage:**
- Empty/wrong response → code bug (streaming, tool call handling, SSE)
- Poor answer despite having data → prompt engineering, not code

**Gemma4 quirks:**
- Emits `tool_calls` in an intermediate chunk (`done=False`) — handled by `accumulated_tool_calls` in `orchestrator.py`.
- Occasionally emits LaTeX (e.g. `$\rightarrow$`) — `preprocessLatex()` in `index.html` converts to Unicode.
