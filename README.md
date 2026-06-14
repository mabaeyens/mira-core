# Mira

A local AI assistant with autonomous web search, file attachments (PDF/HTML/images/text), and RAG for large documents. Available as a CLI tool and a local web interface with streaming markdown responses.

Runs on a local inference backend ([oMLX](https://omlx.ai)) — no cloud APIs, no API keys. RAG embeddings use `sentence-transformers` (`nomic-ai/nomic-embed-text-v1.5`) locally — no external services required.

See [CHANGELOG.md](CHANGELOG.md) for recent changes.

## Native apps

Mira is also available as native **macOS + iOS apps** — see [askmira.es](https://askmira.es)
for an overview and TestFlight access, or the
[mira-apps](https://github.com/mabaeyens/mira-apps) repo for the SwiftUI source.

## Features

- **Autonomous Search**: Model searches the web via Brave Search and fetches full page content when snippets aren't enough (Jina fallback for JS-rendered pages) — sources are shown as clickable links
- **Streaming responses**: Tokens buffered and rendered as formatted markdown
- **Two interfaces**: Rich CLI and local web UI (FastAPI + SSE)
- **File attachments**: PDFs (RAG), HTML, images (multimodal vision), text/code files — attach a screenshot and ask about it; tested with books up to 34 MB
- **RAG**: Large documents chunked, embedded, reranked with Qwen3-Reranker-0.6B-4bit (mlx, in-process) — retrieved automatically on every turn, with hallucination guard for meta-queries (summarize, translate)
- **Adaptive thinking**: Qwen3.6-35B uses extended reasoning on complex questions; suppressed automatically for trivial queries — zero overhead (≤14ms warm)
- **Conversation search**: Search past conversations by content — model can call `search_conversations()` or use the `/conversations/search` API endpoint
- **Scheduled reminders**: Set reminders in natural language; delivered via macOS Notification Center
- **Temp workspace**: Model can read and write files in a per-session temp directory when no project is open
- **Private**: Runs entirely on your local machine — no cloud APIs, no telemetry

## Requirements

macOS on Apple Silicon (the inference backends are MLX-based). Everything else —
`uv`, Python 3.12+, the virtualenv, and `mira.yaml` — is handled by the installer.

## Install

Pick whichever fits; each leaves you with a working backend in one command.

**Fresh machine** (clones to `~/mira-core`):
```bash
curl -LsSf https://raw.githubusercontent.com/mabaeyens/mira-core/main/install.sh | bash
```

**Already cloned the repo:**
```bash
make install
```

**As a package** (installs the `mira` command via uv):
```bash
uv tool install --editable .   # run from inside the checkout
mira setup
```

The installer checks/installs `uv`, runs `uv sync`, and creates `mira.yaml`. Opt into
the extras with flags (the curl one-liner takes them after `-s --`):

```bash
make install ARGS="--with-ollama --with-launchagent"
# or:  curl -LsSf .../install.sh | bash -s -- --with-ollama --with-launchagent
```

| Flag | Effect |
|------|--------|
| `--with-ollama` | `brew install ollama` + `ollama pull gemma4:26b` (optional Gemma4 backend) |
| `--with-launchagent` | install & load the macOS LaunchAgent (server runs at login) |
| `--with-tailscale <host>` | configure HTTPS/Tailscale cert paths in the LaunchAgent |

The one thing the installer **can't** do for you (it's GUI-gated): install the oMLX
app and load the model. Download from
[github.com/jundot/omlx/releases](https://github.com/jundot/omlx/releases), drag to
`/Applications`, open it once to accept the prompts, and load `Qwen3.6-35B-A3B` in its
model library. Run `mira doctor` (or `make doctor`) any time to see what's still missing.

> On first use, the `nomic-ai/nomic-embed-text-v1.5` embedding model and
> `Qwen3-Reranker-0.6B-4bit` reranker download automatically from HuggingFace and cache
> to `~/.cache/huggingface/`. For best ollama performance, add
> `export OLLAMA_CONTEXT_LENGTH=65536` and `export OLLAMA_FLASH_ATTENTION=1` to `~/.zprofile`.

## Running

```bash
mira serve     # web UI at http://localhost:8000   (or: make serve)
mira chat      # interactive CLI                    (or: make chat)
```

For remote access (iPad via Tailscale), the server also listens on HTTPS port **8443** —
install with `--with-tailscale <host>` and connect to `https://<mac-hostname>:8443`.

oMLX is started and managed automatically by the server. See
`docs/model-comparison-m5-macbook.md` for benchmarks and model alternatives.

## macOS LaunchAgent (optional)

`make install ARGS="--with-launchagent"` renders the plist + wrapper from their
`*.template` files (substituting your paths), installs to `~/Library/LaunchAgents/`, and
loads it — no manual editing. Add `--with-tailscale <host>` to keep the HTTPS cert keys.
The generated files are git-ignored; only the `*.template` originals are committed. Logs
go to `/tmp/com.mab.mira.log`.

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
- Upload button — attach files from your machine
- Folder browser — navigate the server's filesystem, filter by extension, select multiple files; files with wrong or missing extension are shown greyed out with a rejection warning
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
backend: omlx
model: Qwen3.6-35B-A3B
host: http://localhost:8080

embed_model: nomic-ai/nomic-embed-text-v1.5

context_window: 65536
```

| Setting | Default | Description |
|---------|---------|-------------|
| `backend` | `omlx` | Inference backend (`omlx`, `dflash`, `mlx-lm`, or `ollama`) |
| `model` | `Qwen3.6-35B-A3B` | Model identifier (omlx name; use `mlx-community/Qwen3.6-35B-A3B-4bit` for dflash/mlx-lm) |
| `host` | `http://localhost:8080` | Backend host URL |
| `embed_model` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model for RAG |
| `context_window` | `65536` | Token context window |

Additional settings (not user-configurable via `mira.yaml` — edit `core/config.py` only if needed):

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_SEARCH_RESULTS` | `5` | Results per web search |
| `MAX_TOOL_STEPS` | `10` | Max tool calls per turn |
| `MAX_RETRIES` | `3` | API error retries per call |
| `SEARCH_TIMEOUT` | `30` | DuckDuckGo timeout in seconds |
| `RERANK_MODEL` | `Qwen3-Reranker-0.6B-4bit` | RAG reranker (mlx, in-process) |
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

## Development Workflow

This project is the result of a strategic collaboration between human design and AI-assisted code generation.

- **Architecture & Logic:** Fully defined by the author. This includes system structure, business rules, data flow, and implementation strategy.
- **Code Generation:** The syntactic implementation and line-by-line code writing was performed by **Claude Code**, following precise and iterative instructions provided by the author.
- **Supervision & Refinement:** All code was manually reviewed, tested, and adjusted to ensure quality, consistency, and compliance with project standards.

This approach demonstrates the ability to direct advanced AI tools to accelerate development without sacrificing creative control or technical quality.

## License

This project is licensed under the **MIT License**. You can find the full text in the [`LICENSE`](./LICENSE) file.

> **Note on authorship:** Although much of the source code was generated by an AI, the creative direction, architecture, and final integration are human work. Usage rights are granted under the terms of the MIT License.

## Contributing

Feel free to fork this project!
- If you find a bug, open an issue.
- If you have an improvement, submit a Pull Request.
- Feel free to use this code in your own projects!
