# ollama Search Tool

A local AI assistant powered by **Gemma 4:26b** and **Ollama** with autonomous web search capabilities.

## Features

- **Autonomous Search**: Model decides when to search based on query type
- **Dual Search Engines**: Ollama native search with DuckDuckGo fallback
- **Clean CLI**: Rich-formatted output with toggleable verbosity
- **Private**: Runs entirely on your local machine

## Prerequisites

- **Python 3.12+**
- **Ollama** (v0.20.2+)
- **uv**
- **Gemma 4:26b** model pulled: `ollama pull gemma4:26b`

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

## Running

```bash
uv run python main.py
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/toggle` | Toggle verbose mode (show/hide search details) |
| `/verbose` | Enable verbose mode |
| `/quiet` | Disable verbose mode |
| `/reset` | Reset conversation history |
| `/quit` | Exit |

## Testing

All model and search calls are mocked — no Ollama instance needed to run tests.

```bash
uv run pytest                                              # all tests
uv run pytest tests/test_queries.py::test_toggle_verbose  # single test
```

## Roadmap

### Web Interface (planned)
Replace the CLI with a local browser UI that provides streaming responses with proper markdown rendering, matching the feel of claude.ai.

**Approach:** FastAPI backend exposing a Server-Sent Events (SSE) endpoint that pipes Ollama token stream to the browser. Vanilla HTML/JS frontend with `marked.js` for incremental markdown rendering. No build step, no npm.

Key pieces:
- `server.py` — FastAPI app with `/chat` SSE endpoint and `/static` file serving
- `static/index.html` — chat UI consuming the SSE stream
- Reuses existing `ChatOrchestrator` with minimal changes

### File Attachments (planned, after web interface)
Allow attaching files to queries so the model can reason over local documents.

- **Images**: Gemma 4 is natively multimodal — pass image bytes directly to Ollama
- **Text / code files**: Extract content and inject into the conversation context
- **Large documents (PDFs)**: Requires RAG — chunk, embed with `sentence-transformers`, store in ChromaDB, retrieve relevant sections per query

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `gemma4:26b` | Ollama model |
| `USE_NATIVE_SEARCH` | `True` | Try Ollama native search first |
| `MAX_SEARCH_RESULTS` | `5` | Results per search |
| `MAX_TOOL_STEPS` | `5` | Max tool calls per turn |
| `MAX_RETRIES` | `3` | API error retries per call |
| `SEARCH_TIMEOUT` | `30` | DuckDuckGo timeout in seconds |
| `VERBOSE_DEFAULT` | `False` | Start in verbose mode |
| `OLLAMA_HOST` | `http://localhost:11434` | Override via `OLLAMA_HOST` env var |
