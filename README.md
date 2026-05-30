# Mira

A local AI assistant with autonomous web search, file attachments (PDF/HTML/images/text), and RAG for large documents. Available as a CLI tool and a local web interface with streaming markdown responses.

Runs on a local mlx-lm backend for inference — no cloud APIs, no API keys. Ollama is used only for RAG embeddings (nomic-embed-text) and will be phased out once sentence-transformers embeddings are wired in directly.

## Features

- **Autonomous Search**: Model searches the web and fetches full page content when snippets aren't enough — sources are shown as clickable links
- **Streaming responses**: Tokens buffered and rendered as formatted markdown
- **Two interfaces**: Rich CLI and local web UI (FastAPI + SSE)
- **File attachments**: PDFs (RAG), HTML, images (multimodal), text/code files — tested with books up to 34 MB
- **RAG**: Large documents chunked, embedded, reranked with CrossEncoder — retrieved automatically on every turn, with hallucination guard for meta-queries (summarize, translate)
- **Private**: Runs entirely on your local machine — no cloud APIs, no telemetry

## Prerequisites

- **Python 3.12+** and **uv**
- **mlx-lm 0.31.3+**: `uv tool install mlx-lm` — inference engine (port 8080)
- **Model**: `mlx-community/gemma-4-26b-a4b-it-4bit` (cached locally via mlx-lm on first run)
- **Ollama** v0.24.0+ — only needed for RAG embeddings: `ollama pull nomic-embed-text`

mlx-lm is started automatically by the server (`backend_manager.py`). Ollama must be running separately if RAG is used.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

> On first use, the CrossEncoder reranker model (~100 MB) downloads automatically from HuggingFace and caches to `~/.cache/huggingface/`.

## Running

**CLI:**
```bash
python main.py
```

**Web interface** — open `http://localhost:8000` in your browser:
```bash
python server.py
```

For remote access (iPad via Tailscale), the server also listens on HTTPS port **8443** — configure `SSL_CERTFILE` / `SSL_KEYFILE` in the plist (see [macOS LaunchAgent](#macos-launchagent-optional)) and connect to `https://<mac-hostname>:8443`.

mlx-lm is started and managed automatically by the server. See `docs/model-comparison-m5-macbook.md` for benchmarks and model alternatives.

## macOS LaunchAgent (optional)

To run the web server as a background service that starts at login:

```bash
cp com.mab.mira.plist.template com.mab.mira.plist
cp start-mira-server.sh.template start-mira-server.sh
```

Edit both files — replace `<MIRA_DIR>` and `<YOUR_HOME>` with your actual paths. If you are not using HTTPS/Tailscale, remove the `SSL_CERTFILE` / `SSL_KEYFILE` keys from the plist.

```bash
chmod +x start-mira-server.sh
cp com.mab.mira.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.mab.mira.plist
```

Both filled-in files are git-ignored (they contain local paths). Only the `*.template` originals are committed.

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
- 📂 folder browser — navigate the server's filesystem, filter by extension, select multiple files; files with wrong or missing extension are shown greyed out with a rejection warning
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
| Images | Passed via multimodal API (base64) |
| Scanned PDF | Warning emitted; no text extractable |

RAG documents persist in the session index across turns — no need to re-attach for follow-up questions. Use `/rag-remove` or Reset to clear.

## Configuration

Copy `mira.yaml.example` to `mira.yaml` and edit. All fields are optional — omit any to keep the built-in default.

```yaml
backend: mlx-lm
model: mlx-community/gemma-4-26b-a4b-it-4bit
host: http://localhost:8080

embed_backend: ollama
embed_model: nomic-embed-text
embed_host: http://localhost:11434

context_window: 65536
```

| Setting | Default | Description |
|---------|---------|-------------|
| `backend` | `mlx-lm` | Inference backend (`mlx-lm` or `ollama`) |
| `model` | `mlx-community/gemma-4-26b-a4b-it-4bit` | Model identifier |
| `host` | `http://localhost:8080` | Backend host URL |
| `embed_backend` | `ollama` | Embedding backend for RAG |
| `embed_model` | `nomic-embed-text` | Embedding model |
| `embed_host` | `http://localhost:11434` | Embedding host URL |
| `context_window` | `65536` | Token context window |

Additional settings (not user-configurable via `mira.yaml` — edit `core/config.py` only if needed):

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_SEARCH_RESULTS` | `5` | Results per web search |
| `MAX_TOOL_STEPS` | `10` | Max tool calls per turn |
| `MAX_RETRIES` | `3` | API error retries per call |
| `SEARCH_TIMEOUT` | `30` | DuckDuckGo timeout in seconds |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker |
| `RAG_CHUNK_SIZE` | `400` | Words per RAG chunk |
| `RAG_CHUNK_OVERLAP` | `40` | Word overlap between chunks |
| `RAG_RETRIEVE_K` | `10` | Candidates retrieved before reranking |
| `RAG_RERANK_TOP_K` | `4` | Chunks injected after reranking |
| `RAG_SCORE_THRESHOLD` | `0.0` | Minimum CrossEncoder score to inject |
| `RAG_MAX_CHUNKS` | `10000` | Warn when index exceeds this size |

## Testing

All model, search, and fetch calls are mocked — no inference server needed to run tests.

```bash
uv run pytest                                              # all tests
uv run pytest tests/test_queries.py::test_toggle_verbose  # single test
uv run pytest tests/test_cancel.py                        # cancel/stop tests only
```

Tests cover: search trigger behaviour, `fetch_url` dispatch, intermediate-chunk tool calls (`accumulated_tool_calls`), RAG threshold bypass for same-turn attachments, verbose toggle, conversation reset, stop/cancel endpoint, event stream abort, history rollback on cancel, stats event emission, token count capture, context % bounds.

## 🛠️ Development Workflow: Human-AI Collaboration

This project is the result of a strategic collaboration between human design and AI-assisted code generation.

- **Architecture & Logic:** Fully defined by the author. This includes system structure, business rules, data flow, and implementation strategy.
- **Code Generation:** The syntactic implementation and line-by-line code writing was performed by **Claude Code**, following precise and iterative instructions provided by the author.
- **Supervision & Refinement:** All code was manually reviewed, tested, and adjusted to ensure quality, consistency, and compliance with project standards.

This approach demonstrates the ability to direct advanced AI tools to accelerate development without sacrificing creative control or technical quality.

## 📄 License

This project is licensed under the **MIT License**. You can find the full text in the [`LICENSE`](./LICENSE) file.

> **Note on authorship:** Although much of the source code was generated by an AI, the creative direction, architecture, and final integration are human work. Usage rights are granted under the terms of the MIT License.

## 🚀 Contributing

Feel free to fork this project!
- If you find a bug, open an issue.
- If you have an improvement, submit a Pull Request.
- Feel free to use this code in your own projects!
