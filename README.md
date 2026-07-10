# Mira

A local AI assistant with autonomous web search, file attachments (PDF/HTML/images/text), and RAG for large documents. Available as a CLI tool and a local web interface with streaming markdown responses.

Runs entirely on local inference — no cloud APIs, no API keys. The default backend is **mira-mlx**, Mira's own MLX-based inference server (bundled, no extra app to install); [oMLX](https://omlx.ai), dflash, mlx-lm, vllm-mlx, and Ollama are also supported as alternative backends. RAG embeddings use `sentence-transformers` (`nomic-ai/nomic-embed-text-v1.5`) locally — no external services required.

See [CHANGELOG.md](CHANGELOG.md) for recent changes.

## Native apps

Mira is also available as native **macOS + iOS apps** — see [askmira.es](https://askmira.es)
for an overview and TestFlight access, or the
[mira-apps](https://github.com/mabaeyens/mira-apps) repo for the SwiftUI source.

## Testing Mira

Mira is public and ready for testing. Two ways in: the apps (iPhone/iPad/Mac via
TestFlight at [askmira.es](https://askmira.es), easiest), or this repo (run the
backend yourself; Apple Silicon, and the models are big, so check the preflight below).

Two things I'd love to know: is it actually useful, and what broke. Feedback,
questions, and whether it's useful go in
[Discussions](https://github.com/mabaeyens/mira-core/discussions); bugs and crashes go
in [Issues](https://github.com/mabaeyens/mira-core/issues/new/choose). I read both.

## Features

- **Autonomous Search**: Model searches the web via Brave Search and fetches full page content when snippets aren't enough (Jina fallback for JS-rendered pages) — sources are shown as clickable links
- **Streaming responses**: Tokens buffered and rendered as formatted markdown
- **Two interfaces**: Rich CLI and local web UI (FastAPI + SSE)
- **File attachments**: PDFs (RAG), HTML, images (multimodal vision), text/code files — attach a screenshot and ask about it; tested with books up to 34 MB
- **RAG**: Large documents chunked, embedded, reranked with Qwen3-Reranker-0.6B-4bit (mlx, in-process) — retrieved automatically on every turn, with hallucination guard for meta-queries (summarize, translate)
- **Adaptive thinking**: Qwen3.6-35B uses extended reasoning on complex questions; suppressed automatically for trivial queries — zero overhead (≤14ms warm)
- **Multiple model families**: Qwen3.6 (MoE, the primary default) and Mistral-family models (Ministral 3 14B) both fully supported, including tool-calling — pick per-conversation from the model picker
- **Conversation search**: Search past conversations by content — model can call `search_conversations()` or use the `/conversations/search` API endpoint
- **Scheduled reminders**: Set reminders in natural language; delivered via macOS Notification Center
- **Temp workspace**: Model can read and write files in a per-session temp directory when no project is open
- **Private**: Runs entirely on your local machine — no cloud APIs, no telemetry

## Requirements

macOS on Apple Silicon (the inference backends are MLX-based). Everything else —
`uv`, Python 3.12+, the virtualenv, and `mira.yaml` — is handled by the installer.

## Install

**Most people want this** — on a fresh machine it clones to `~/mira-core` and sets up a
working backend in one command:

```bash
curl -LsSf https://raw.githubusercontent.com/mabaeyens/mira-core/main/install.sh | bash
```

<details><summary>Alternatives (already cloned, or installing as a package)</summary>

**Already cloned the repo:**
```bash
make install
```

**As a package** (installs the `mira` command via uv):
```bash
uv tool install --editable .   # run from inside the checkout
mira setup
```

</details>

The installer checks/installs `uv`, runs `uv sync`, and creates `mira.yaml`. Opt into
the extras with flags (the curl one-liner takes them after `-s --`):

```bash
make install ARGS="--with-ollama --with-launchagent"
# or:  curl -LsSf .../install.sh | bash -s -- --with-ollama --with-launchagent
```

| Flag | Effect |
|------|--------|
| `--with-ollama` | `brew install ollama` + `ollama pull gemma4:26b` (optional Gemma4 backend) |
| `--with-ocr` | `brew install tesseract` (OCR for scanned PDFs — see below) |
| `--with-launchagent` | install & load the macOS LaunchAgent (server runs at login) |
| `--with-tailscale <host>` | configure HTTPS/Tailscale cert paths in the LaunchAgent |
| `--skip-preflight` | skip the disk + memory check |
| `--force` | proceed even when free disk is below the recommended headroom |

### Disk & memory

Before doing any work the installer runs a **preflight** (`mira preflight`): it lets you
pick which models count toward the budget, estimates total disk (the default
`Qwen3.6-35B-A3B` alone is ~19 GB; a full multi-model install can top 70 GB), and checks
you have that **plus ~15 GB breathing room** free — otherwise it stops (override with
`--force`). It also warns when RAM is tight: on a 32 GB Mac the large models can't be
resident at once (Mira loads one at a time), and below 24 GB the default model may OOM at
large context. Run `mira preflight` standalone any time to see the budget.

### No extra app required

The default backend is **mira-mlx**, Mira's own inference server — it's a bundled
Python module (`core/inference/mira_mlx_server.py`) built on `mlx-lm`, started and
managed automatically by the server. `uv sync` pulls its dependencies as part of the
normal install; there's no separate GUI app to download or model library to configure.

If you'd rather use [oMLX](https://omlx.ai) instead (a separate GUI app with its own
model library, `~0ms` TTFT after warm-up), it's still fully supported — download it
from [github.com/jundot/omlx/releases](https://github.com/jundot/omlx/releases), load
a model in its library, and set `backend: omlx` in `mira.yaml`.

Run `mira doctor` (or `make doctor`) any time to confirm your configured backend is
detected and see what else is missing. (`--with-ollama` installs Ollama as another
alternative backend.)

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

Whichever backend is configured in `mira.yaml` is started and managed automatically by
the server. See `docs/model-comparison-m5-macbook.md` for benchmarks and model
alternatives.

### Access control

The server exposes tools that **run shell commands and read/write files**, so a sniffed
credential = full remote code execution. Network exposure is therefore locked down by
default and never sends anything sensitive in plaintext:

- **No token set (default):** the server binds **`127.0.0.1` only** and refuses to
  expose a non-loopback interface. Not reachable from other devices.
- **With a token** (`auth_token:` in `mira.yaml`, **`chmod 600` it**, or the `MIRA_TOKEN`
  env var): HTTP `:8000` stays **loopback-only**, and HTTPS `:8443` is served on the
  **Tailscale interface only** — so the off-host socket exists solely on your tailnet,
  and the token + payloads are always encrypted. Every route except `/health` and the
  static UI requires `Authorization: Bearer <token>` (the apps send it automatically),
  and a source-IP allowlist (loopback + tailnet) is enforced as defense-in-depth. If
  Tailscale is down at startup, `:8443` **fails closed to loopback** — start Tailscale
  then restart the server (`/mira-server restart`) to enable remote access.

See **[docs/remote-access.md](docs/remote-access.md)** for the full posture: travelling
with Tailscale (and why Proton VPN conflicts on iOS), and the opt-in plain-LAN mode.

## macOS LaunchAgent (optional)

`make install ARGS="--with-launchagent"` renders the plist + wrapper from their
`*.template` files (substituting your paths), installs to `~/Library/LaunchAgents/`, and
loads it — no manual editing. Add `--with-tailscale <host>` to keep the HTTPS cert keys.
The generated files are git-ignored; only the `*.template` originals are committed. Logs
go to `/tmp/com.mab.mira.log`.

## Uninstall

If you installed the `mira` command with `uv tool`, remove it by its **package** name
(`mira-core`), not the executable name:

```bash
uv tool uninstall mira-core   # NOT `uv tool uninstall mira`
```

If you installed the LaunchAgent, unload and remove it:

```bash
launchctl unload ~/Library/LaunchAgents/com.mab.mira.plist
rm ~/Library/LaunchAgents/com.mab.mira.plist
```

The rest is self-contained: delete the checkout (its `.venv`, `mira.yaml`, and local DBs go
with it). Shared caches under `~/.cache/uv` and `~/.cache/huggingface` are left for other tools;
remove the Mira model weights from `~/.cache/huggingface/hub` manually if you want the space back.

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
- PDFs are text-extracted and RAG-indexed automatically; **scanned PDFs** (no text layer) are OCR'd page-by-page when `tesseract` is installed (`--with-ocr`, or `brew install tesseract`), otherwise a clear warning is shown. OCR is capped (50 pages, per-page timeout) and only runs on pages with no extractable text
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
backend: mira-mlx
model: mlx-community/Qwen3.6-35B-A3B-4bit
host: http://localhost:8080

embed_model: nomic-ai/nomic-embed-text-v1.5

context_window: 65536
```

| Setting | Default | Description |
|---------|---------|-------------|
| `backend` | `mira-mlx` | Inference backend (`mira-mlx`, `omlx`, `dflash`, `mlx-lm`, `vllm-mlx`, or `ollama`) |
| `model` | `mlx-community/Qwen3.6-35B-A3B-4bit` | Model identifier — an mlx-community repo id for mira-mlx/dflash/mlx-lm/vllm-mlx (e.g. `mlx-community/Ministral-3-14B-Instruct-2512-4bit` for the Mistral family), or omlx's own model name (`Qwen3.6-35B-A3B`) when `backend: omlx` |
| `host` | `http://localhost:8080` | Backend host URL |
| `embed_model` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model for RAG |
| `context_window` | `65536` | Token context window |

Additional settings (not user-configurable via `mira.yaml` — edit `core/config.py` only if needed):

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_SEARCH_RESULTS` | `5` | Results per web search |
| `MAX_TOOL_STEPS` | `10` | Max tool calls per turn |
| `MAX_RETRIES` | `3` | API error retries per call |
| `SEARCH_TIMEOUT` | `30` | Web search timeout in seconds |
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

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, workflow, and
how to open a pull request. For security issues, see [SECURITY.md](SECURITY.md). Feel free to
fork it and use this code in your own projects.
