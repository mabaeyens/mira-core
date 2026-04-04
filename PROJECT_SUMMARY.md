# Project Summary: ollama Search Tool (Gemma 4 + Ollama)

## 1. Original Intent & Goal
To build a **local, private AI assistant** running on macOS that:
- Uses **Gemma 4:26b** (via Ollama) as the core reasoning engine.
- Possesses **autonomous web search capabilities** to answer questions about events after the model's training cutoff (April 2024).
- Decides **when** to search based on query context (ReAct pattern) rather than relying on manual triggers.
- Provides both a **CLI interface** and a **local web UI** with streaming markdown responses.
- Runs entirely locally with **zero-cost** search (DuckDuckGo) and no external API keys.

## 2. Technology Stack & Decisions

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| **LLM** | `gemma4:26b` | High reasoning capability, MoE architecture (4b active params), available via Ollama. |
| **Runtime** | Ollama (v0.20.2+) | Native tool calling support, local execution, easy model management. |
| **Language** | Python 3.12+ | 3.9 is EOL (April 2026); 3.12 offers better performance and syntax. |
| **Package Manager** | `uv` | Fast dependency resolution, reproducible lockfiles. |
| **Search Engine** | DuckDuckGo (`ddgs`) | Free, no API key. Ollama native search disabled (requires paid subscription). |
| **CLI UI** | Rich library | Spinners, formatted output, toggleable verbosity. |
| **Web UI** | FastAPI + SSE + vanilla HTML/JS | No build step, full control, streaming via SSE, markdown via marked.js. |
| **Storage** | Local FS + GitHub + Proton Drive | Secure, version-controlled, end-to-end encrypted backup. |

## 3. Architecture Overview

The system follows a **ReAct (Reasoning + Acting)** loop built on an event-based streaming architecture:

```
main.py (_render_stream)     server.py (/chat SSE endpoint)
          └────────────────────────────┘
                      │  consumes events
         ChatOrchestrator.stream_chat()
                      │  yields events
         ┌────────────┴────────────┐
    _call_ollama()          SearchEngine
  ollama.chat(stream=True)   ddgs.text()
```

**Events yielded by `stream_chat()`:**
1. `thinking` — model is processing; consumers show spinner
2. `token` — answer token; consumers append and render incrementally
3. `search_start` — search beginning; consumers show search indicator
4. `search_done` — results ready; consumers update indicator with count
5. `done` — turn complete with full content
6. `error` — error with message

**System prompt** is built dynamically (`build_system_prompt()`) and includes today's date, so the model correctly identifies past events as searchable.

## 4. Key Design Decisions

### A. Autonomous Search Triggers
- Model decides dynamically based on system prompt rules (date-aware) and its own judgment.
- No hardcoded heuristics — the model handles edge cases.

### B. Search Strategy
- **Active**: `ddgs` Python library (free, no API key).
- **Disabled**: Ollama native `web_search` requires a paid Ollama subscription — `USE_NATIVE_SEARCH = False`.
- **Fallback**: If search fails, model is instructed to inform the user and answer from internal knowledge.

### C. Transparency (Verbose Mode)
- CLI: `/verbose`, `/quiet`, `/toggle` commands; spinner always visible.
- Web: checkbox in header; search chip always visible (result count shown regardless).

### D. Event-Based Architecture
- `ChatOrchestrator` is display-agnostic: yields typed events, no I/O.
- CLI renderer (`_render_stream` in `main.py`) handles Rich spinners and stdout.
- Web renderer (`/chat` endpoint in `server.py`) forwards events as SSE via asyncio queue + background thread.
- Single mockable boundary: `_call_ollama()` — tests mock this with a list of MagicMock chunks.

### E. Environment Management
- **Tool**: `uv` (not `pip`/`venv`).
- **Workflow**: `uv venv --python 3.12` → `source .venv/bin/activate` → `uv sync`.

## 5. Current Project Status

### Completed ✅
- All core source files (`main.py`, `orchestrator.py`, `search_engine.py`, `tools.py`, `prompts.py`, `formatter.py`, `config.py`)
- Event-based streaming architecture (`stream_chat()` generator)
- CLI with streaming output, spinners, search status, verbose mode
- Web interface: FastAPI server (`server.py`) + single-page UI (`static/index.html`)
  - SSE streaming with live markdown rendering (marked.js)
  - Thinking animation, search chips, verbose toggle, reset
- 7 unit tests (all mocked, no Ollama needed)
- Full documentation (README, CLAUDE.md, PROJECT_SUMMARY.md)
- Git repository at `github.com/mabaeyens/ollama-web-search`

## 6. File Structure

```
ollama-web-search/
├── main.py                 # CLI entry point + _render_stream()
├── server.py               # FastAPI web server + SSE /chat endpoint
├── orchestrator.py         # ChatOrchestrator: stream_chat() generator
├── search_engine.py        # DuckDuckGo search (ddgs)
├── tools.py                # Ollama tool schema for web_search
├── prompts.py              # build_system_prompt() — date-aware
├── formatter.py            # Rich console helpers (CLI only)
├── config.py               # All tunables
├── pyproject.toml          # uv project config
├── uv.lock                 # Locked dependencies
├── .gitignore
├── README.md
├── CLAUDE.md
├── PROJECT_SUMMARY.md
├── static/
│   └── index.html          # Web UI (vanilla HTML/CSS/JS)
└── tests/
    ├── conftest.py          # sys.path injection
    └── test_queries.py      # 7 unit tests
```

## 7. Roadmap

### Phase 2 — File Attachments

Allow attaching files (PDF, HTML, images) to queries so the model can reason over local content.

#### Supported file types

| File type | Extraction | Ollama API |
|-----------|-----------|------------|
| Images (PNG, JPG, …) | Raw bytes → base64 | `images` field on the message dict |
| HTML | BeautifulSoup → readable text | Prepended to user message content |
| Text / code | Read as-is | Prepended to user message content |
| PDF (text-based) | `pdfplumber` → text | Prepended to user message content |
| PDF (scanned) | No text layer — warn user; optionally send pages as images | Image path or skip |

#### Non-obvious constraints to keep in mind

1. **Ollama multimodal format** — images can't be appended as text. The message must carry `"images": ["<base64>"]`. `stream_chat()` signature must change to accept attachments.
2. **HTML must be parsed** — injecting raw HTML is wasteful and confusing for the model. Strip tags with BeautifulSoup.
3. **Scanned PDFs have no extractable text** — detect the empty-text case and warn or fall back to the image path.
4. **Context window guard** — large documents can overflow the 128k context. Truncate with a warning above a threshold (~80k chars). RAG is the right path for very large documents (Phase 3).
5. **File persistence across turns** — decide: inject once and leave in `conversation_history` (stays for whole session), or inject per message (single-turn). Default: single-turn (appended to the user message, not stored separately).
6. **Web UI upload mechanism** — `/chat` currently accepts `{"message": str}`. Files require multipart form data or a separate `/upload` endpoint. Multipart is simpler.
7. **CLI attach UX** — use `/attach <path>` to stage a file for the next message; cleared after each turn. Show staged file in the prompt.

#### Implementation steps

| Step | What | Files touched |
|------|------|---------------|
| 1 | `file_handler.py` — extract text from PDF/HTML/text; return base64 for images; detect scanned PDFs | new file |
| 2 | `orchestrator.py` — `stream_chat(user_message, attachments=None)`; prepend text content; add `images` key for image attachments | `orchestrator.py` |
| 3 | CLI `/attach <path>` command; staged file cleared after each turn; show in prompt | `main.py` |
| 4 | Web UI file input button + chip display; change `/chat` to accept multipart form data | `server.py`, `static/index.html` |
| 5 | Context size guard — truncate + warn if extracted content exceeds threshold | `file_handler.py` or `orchestrator.py` |

#### New dependencies

```
pdfplumber      # PDF text extraction
beautifulsoup4  # HTML stripping
```

### Phase 3 — RAG for large PDFs (future)

For documents too large to inject in full: chunk → embed (`sentence-transformers`) → store (ChromaDB) → retrieve relevant sections per query. Separate effort, not part of Phase 2.

## 8. Backlog / Open Decisions

- **CLI markdown rendering** — Resolved: CLI now buffers tokens and renders the full answer with Rich Markdown (`03c7bf1`).

## 9. Notes for Claude Code Context

- Do not suggest `pip` or `venv`; always use `uv`.
- Do not suggest cloud APIs; keep search local/free.
- Do not change the model to anything other than `gemma4:26b` unless requested.
- Do not set `USE_NATIVE_SEARCH = True` — Ollama native requires a paid subscription.
- `ChatOrchestrator` must remain display-agnostic — no print/console calls in orchestrator.py.
- Tests mock `_call_ollama`, not higher-level methods. Mock chunks need `.message.content`, `.message.tool_calls`, `.done`.
