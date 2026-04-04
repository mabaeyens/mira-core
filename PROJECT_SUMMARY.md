# Project Summary: ollama Search Tool (Gemma 4 + Ollama)

## 1. Original Intent & Goal
To build a **local, private AI assistant** running on macOS that:
- Uses **Gemma 4:26b** (via Ollama) as the core reasoning engine.
- Possesses **autonomous web search capabilities** to answer questions about events after the model's training cutoff (April 2024).
- Decides **when** to search based on query context (ReAct pattern) rather than relying on manual triggers.
- Provides a **clean CLI interface** with toggleable verbosity for transparency.
- Runs entirely locally with **zero-cost** search (DuckDuckGo fallback) and no external API keys.

## 2. Technology Stack & Decisions

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| **LLM** | `gemma4:26b` | High reasoning capability, MoE architecture, available via Ollama. |
| **Runtime** | Ollama (v0.20.2+) | Native tool calling support, local execution, easy model management. |
| **Language** | Python 3.12+ | 3.9 is EOL (April 2026); 3.12 offers better performance and syntax. |
| **Package Manager** | `uv` | Extremely fast dependency resolution and virtual environment management. |
| **Search Engine** | Hybrid (Ollama Native + DuckDuckGo) | Ollama native for integration; DuckDuckGo (`ddgs`) as free, keyless fallback. |
| **UI** | CLI (Rich Library) | Lightweight, supports markdown formatting, toggleable debug info. |
| **Storage** | Local FS + GitHub + Proton Drive | Secure, version-controlled, end-to-end encrypted backup. |

## 3. Architecture Overview

The system follows a **ReAct (Reasoning + Acting)** loop:

1. **User Input** → CLI receives query.
2. **Orchestrator** → Sends query to Gemma 4 with `web_search` tool definition.
3. **Model Decision**:
   - *If search needed*: Model outputs tool call `web_search(query="...")`.
   - *If not needed*: Model outputs direct answer.
4. **Execution**:
   - Orchestrator intercepts tool call.
   - Executes search (Ollama native → DuckDuckGo fallback).
   - Formats results.
5. **Feedback Loop**: Search results injected back into conversation history.
6. **Final Answer**: Model synthesizes results and outputs final response.

## 4. Key Design Decisions

### A. Autonomous Search Triggers
- **Heuristic + Model Judgment**: The system relies on the **System Prompt** to instruct Gemma 4 to search for:
  - Events after April 2024.
  - Keywords: "latest", "news", "today", "current", "2025", "2026".
  - Uncertain facts.
- **No Hardcoded Rules**: The model decides dynamically, allowing for nuanced handling of edge cases.

### B. Search Fallback Strategy
- **Primary**: Ollama's native `web_search` API (if available in the specific Ollama version).
- **Secondary**: `ddgs` Python library (free, no API key, robust).
- **Error Handling**: If both fail, the model is instructed to inform the user and fall back to internal knowledge.

### C. Transparency (Verbose Mode)
- **Toggleable**: Users can switch between:
  - *Quiet*: Only final answer (clean for copy-pasting).
  - *Verbose*: Shows search queries, result tables, and intermediate steps (for debugging/verification).
- **Implementation**: Controlled via `/toggle`, `/verbose`, `/quiet` CLI commands.

### D. Environment Management
- **Tool**: `uv` (not `pip`/`venv`).
- **Workflow**: `uv python pin 3.12` → `uv venv` → `uv add ...` → `uv run ...`.
- **Rationale**: Faster installs, reproducible lockfiles (`uv.lock`), modern standard.

## 5. Current Project Status

### Completed
- **Architecture Design**: Defined and agreed upon.
- **Code Generation**: All source files created:
  - `main.py` (CLI Entry)
  - `orchestrator.py` (Logic Loop)
  - `search_engine.py` (Search Execution)
  - `tools.py` (Tool Definitions)
  - `prompts.py` (System Prompts)
  - `formatter.py` (Rich UI)
  - `config.py` (Settings)
  - `pyproject.toml` (Dependencies)
  - `.gitignore`
  - `README.md` (Setup Instructions)
  - `tests/test_queries.py` (Validation)
- **Documentation**: Full setup guide and troubleshooting steps written.

### Pending Actions (User Side)

#### 1. Environment Setup
Python 3.12 is installed. Create the virtual environment and install dependencies:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

#### 2. Model Pull
Pull the model:
```bash
ollama pull gemma4:26b
```

#### 3. File Creation
Ensure all generated files are saved in the `ollama_web_search` directory.

#### 4. Testing
Run `uv run python main.py` and test with queries like:
- `"Who wrote Hamlet?"` → Should **not** search
- `"What's the latest news on AI in 2026?"` → Should search

## 6. File Structure Reference

```
ollama_web_search/
├── main.py                 # CLI entry point
├── orchestrator.py         # Core ReAct loop
├── search_engine.py        # Search logic (Ollama + DDG)
├── tools.py                # Tool schema
├── prompts.py              # System instructions
├── formatter.py            # Rich text output
├── config.py               # Config constants
├── pyproject.toml          # Project config (uv)
├── .gitignore              # Git exclusions
├── README.md               # User guide
└── tests/
    └── test_queries.py     # Test cases
```

## 7. Next Steps for Development

### Phase 1 — Web Interface
Replace the CLI with a local browser UI delivering streaming responses with proper markdown rendering.

**Architecture:**
- `server.py` — FastAPI app with a `/chat` SSE endpoint that pipes the Ollama token stream to the browser, and `/static` serving for the frontend
- `static/index.html` — Single-page chat UI in vanilla HTML/JS consuming the SSE stream; `marked.js` for incremental markdown rendering
- `ChatOrchestrator` reused almost unchanged; streaming already implemented

**Why FastAPI + SSE over alternatives:**
- Gradio/Streamlit: faster to build but generic UI, harder to customise for file uploads later
- React frontend: better long-term but adds a build step and npm dependency
- FastAPI + plain HTML: no build step, full control, straightforward to extend

**Effort estimate:** ~1 day

### Phase 2 — File Attachments (requires web interface)
Allow attaching files to queries so the model can reason over local content.

| File type | Approach |
|-----------|----------|
| Images | Gemma 4 is natively multimodal — pass image bytes directly to Ollama |
| Text / code | Extract content, inject into conversation context |
| Large PDFs | RAG: chunk → embed (`sentence-transformers`) → store (ChromaDB) → retrieve relevant sections per query |

**Effort estimate:** images ~half day; text files ~half day; PDF RAG ~2 days

## 8. Backlog / Open Decisions

- **CLI: streaming vs markdown rendering** — Current CLI streams raw tokens (typewriter effect, no markdown). Once the web interface is live, revisit: either revert CLI to blocking + markdown, or implement a buffered approach that streams silently then renders with Rich Markdown. Deferred until web UI is done.

## 9. Notes for Claude Code Context

- Do not suggest `pip` or `venv`; always use `uv`.
- Do not suggest cloud APIs; keep search local/free.
- Do not change the model to anything other than `gemma4:26b` unless requested.
- Focus on robustness: ensure error handling covers network failures and empty search results.
- Security: remind user to keep `.env` (if added later) and `.venv` in `.gitignore`.
