# ollama Search Tool

A local AI assistant powered by **Gemma 4:26b** and **Ollama** with autonomous web search, file attachments, and RAG for large documents. Available as a CLI tool and a local web interface with streaming markdown responses.

## Features

- **Autonomous Search**: Model searches the web and fetches full page content when snippets aren't enough — sources are shown as clickable links
- **Streaming responses**: Tokens buffered and rendered as formatted markdown
- **Two interfaces**: Rich CLI and local web UI (FastAPI + SSE)
- **File attachments**: PDFs (RAG), HTML, images (multimodal), text/code files — tested with books up to 34 MB
- **RAG**: Large documents chunked, embedded with `nomic-embed-text`, reranked with CrossEncoder — retrieved automatically on every turn, with hallucination guard for meta-queries (summarize, translate)
- **Private**: Runs entirely on your local machine — no cloud APIs, no API keys

## Prerequisites

- **Python 3.12+**
- **Ollama** (v0.20.2+) running locally
- **uv** package manager
- **Gemma 4:26b** model: `ollama pull gemma4:26b`
- **nomic-embed-text** model (for RAG embeddings): `ollama pull nomic-embed-text`

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

> On first use, the CrossEncoder reranker model (~100MB) downloads automatically from HuggingFace and caches to `~/.cache/huggingface/`.

## Running

**1. Start the Ollama server** (if not already running as a background service):
```bash
ollama serve
```

**2. CLI:**
```bash
python main.py
```

**2. Web interface** — open `http://localhost:8000` in your browser:
```bash
python server.py
```

> `server.py` and `main.py` do **not** start Ollama automatically — they will fail with a connection error if `ollama serve` is not running first.

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/toggle` | Toggle verbose mode (show/hide search details) |
| `/verbose` | Enable verbose mode |
| `/quiet` | Disable verbose mode |
| `/reset` | Reset conversation history and RAG index |
| `/attach <path>` | Stage a file for the next message (PDF, HTML, image, text) |
| `/files` | List currently staged attachments |
| `/detach` | Clear all staged attachments |
| `/rag-list` | List documents currently in the RAG index |
| `/rag-remove <name>` | Remove a document from the RAG index |
| `/quit` | Exit |

## Web Interface

- Streaming responses with live markdown rendering
- 📎 upload button — attach files from your machine
- 📂 path button — load files directly from the server's disk by absolute path (same machine)
- Search chips expand to show clickable source links; fetch chips link directly to the fetched page
- Green Documents panel showing RAG-indexed files with per-doc remove
- Status bar showing current operation (Thinking / Searching / Reading / Indexing)
- **Status line** — header badges `↑Xk ↓Xk` (session tokens) and `ctx:N%` (context window fill, color-coded: grey → red → dark as it fills)
- **Stop button** — aborts the current response mid-stream; history rolls back as if the turn never happened
- Verbose toggle and conversation reset in the header
- Enter to send, Shift+Enter for newline

## File Attachments

| File type | Behaviour |
|-----------|-----------|
| PDF (any size) | Always indexed via RAG |
| HTML | Text extracted (BeautifulSoup); RAG if > 80k chars |
| Text / code | Injected directly; RAG if > 80k chars |
| Images | Passed via Ollama multimodal API (base64) |
| Scanned PDF | Warning emitted; no text extractable |

RAG documents persist in the session index across turns — no need to re-attach for follow-up questions. Use `/rag-remove` or Reset to clear.

## Testing

All model, search, and fetch calls are mocked — no Ollama instance needed to run tests.

```bash
uv run pytest                                              # all tests (23 tests)
uv run pytest tests/test_queries.py::test_toggle_verbose  # single test
uv run pytest tests/test_cancel.py                        # cancel/stop tests only
```

Tests cover: search trigger behaviour, `fetch_url` dispatch, Gemma4 intermediate-chunk tool calls (`accumulated_tool_calls`), RAG threshold bypass for same-turn attachments, verbose toggle, conversation reset, stop/cancel endpoint, event stream abort, history rollback on cancel, stats event emission, token count capture, context % bounds.

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `gemma4:26b` | Ollama chat model |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model for RAG |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker |
| `USE_NATIVE_SEARCH` | `False` | Disabled — Ollama native search is free-tier but requires a phone-verified account with no privacy guarantees; DuckDuckGo used instead |
| `MAX_SEARCH_RESULTS` | `5` | Results per web search |
| `MAX_TOOL_STEPS` | `5` | Max tool calls per turn |
| `MAX_RETRIES` | `3` | API error retries per call |
| `SEARCH_TIMEOUT` | `30` | DuckDuckGo timeout in seconds |
| `VERBOSE_DEFAULT` | `False` | Start in verbose mode |
| `OLLAMA_HOST` | `http://localhost:11434` | Override via `OLLAMA_HOST` env var |
| `RAG_CHUNK_SIZE` | `400` | Words per RAG chunk |
| `RAG_CHUNK_OVERLAP` | `40` | Word overlap between chunks |
| `RAG_RETRIEVE_K` | `10` | Candidates retrieved before reranking |
| `RAG_RERANK_TOP_K` | `4` | Chunks injected after reranking |
| `RAG_SCORE_THRESHOLD` | `0.0` | Minimum CrossEncoder score to inject |
| `RAG_MAX_CHUNKS` | `10000` | Warn when index exceeds this size |
