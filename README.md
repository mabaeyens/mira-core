# Mira

A local AI assistant with autonomous web search, file attachments (PDF/HTML/images/text), and RAG for large documents. Available as a CLI tool and a local web interface with streaming markdown responses.

Runs entirely on local inference — no cloud APIs, no API keys. The default backend is **mira-mlx**, Mira's own MLX-based inference server (bundled, no extra app to install); [oMLX](https://omlx.ai) is the supported alternative, and mlx-lm and vllm-mlx also work. RAG embeddings use `sentence-transformers` (`nomic-ai/nomic-embed-text-v1.5`) locally — no external services required.

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
- **File attachments**: PDFs (RAG), HTML, images, text/code files — attach a screenshot and ask about it; tested with books up to 34 MB. Images take one of two paths: OCR (default — the text is extracted with `tesseract` and folded into the prompt, which is cheaper and works well for error dialogs, menus and terminal output), or real vision. omlx always reads images directly; mira-mlx does too when you set `mira_mlx_vision: true`, which loads the Qwen3.6 checkpoint's own vision tower at a cost of about 1.1 GB
- **RAG**: Large documents chunked, embedded, reranked with Qwen3-Reranker-0.6B-4bit (mlx, in-process) — retrieved automatically on every turn, with hallucination guard for meta-queries (summarize, translate)
- **Adaptive thinking**: Qwen3.6-35B uses extended reasoning on complex questions; suppressed automatically for trivial queries — zero overhead (≤14ms warm)
- **Native speculative decoding**: mira-mlx can run a Qwen3.5/3.6 checkpoint's own multi-token-prediction (MTP) head as a self-speculator for faster generation (~1.2–1.3× on the MoE, lossless with the runaway guard) — no external app. Off by default; needs an MTP-equipped model (`mira_mlx_mtp_enabled`, see [`docs/multi-token-prediction.md`](docs/multi-token-prediction.md))
- **Multiple model families**: Qwen3.6 (MoE, the primary default) and Mistral-family models (Ministral 3 14B) both fully supported, including tool-calling — pick per-conversation from the model picker
- **Conversation search**: Search past conversations by content — model can call `search_conversations()` or use the `/conversations/search` API endpoint
- **Scheduled reminders**: Set reminders in natural language; delivered via macOS Notification Center
- **Temp workspace**: Model can read and write files in a per-session temp directory when no project is open
- **Private**: Runs entirely on your local machine — no cloud APIs, no telemetry

## Requirements

macOS on Apple Silicon (the inference backends are MLX-based). Everything else —
`uv`, Python (`>=3.12,<3.14`), the virtualenv, and `mira.yaml` — is handled by the installer.

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

**Homebrew** (macOS, via the tap):
```bash
brew tap mabaeyens/tap
brew trust mabaeyens/tap   # Homebrew 6+ only — it declines untrusted third-party taps
brew install mira
```
On Homebrew 5 and earlier there's no trust step. This installs the `mira` command; the
venv and the model are fetched on first run (`mira serve` or `mira fetch-model`), since
Homebrew's sandbox has no network at build time. Details in `packaging/homebrew/README.md`.

</details>

The installer checks/installs `uv`, runs `uv sync`, and creates `mira.yaml`. Opt into
the extras with flags (the curl one-liner takes them after `-s --`):

```bash
make install ARGS="--with-ocr --with-launchagent"
# or:  curl -LsSf .../install.sh | bash -s -- --with-ocr --with-launchagent
```

| Flag | Effect |
|------|--------|
| `--with-ocr` | `brew install tesseract` (OCR for scanned PDFs and image attachments — see below) |
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

The default backend pulls its ~19 GB model on the first `mira serve`, which otherwise
looks like a hang. Run **`mira fetch-model`** ahead of time to download it explicitly,
with progress — it's idempotent (a cached model returns at once) and takes an optional
repo id (`mira fetch-model mlx-community/<repo>`) to pull a model other than the
configured default.

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
detected and see what else is missing.

`mlx-lm` and `vllm-mlx` are supported too, if you already run them. The dflash and
Ollama backends were removed on 2026-08-01; everything they served runs on one of the
four above.

> On first use, the `nomic-ai/nomic-embed-text-v1.5` embedding model and
> `Qwen3-Reranker-0.6B-4bit` reranker download automatically from HuggingFace and cache
> to `~/.cache/huggingface/`. Nothing else needs configuring — mira-mlx sizes its
> context window and caches from the RAM it finds.

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
  Tailscale isn't up yet, `:8443` **fails closed to loopback** and the server retries the
  bind every 15s until Tailscale comes up — no manual restart needed. A monthly LaunchAgent
  auto-renews the 90-day Tailscale HTTPS cert.
- **Connecting remotely also needs `allowed_hosts`.** A `Host`-header check (anti-DNS-
  rebinding) runs before auth. Loopback and bare IPs in the source allowlist pass
  automatically, but a hostname must be listed — and since the Tailscale cert covers the
  MagicDNS name only, remote clients must use that name. Install with
  `--with-tailscale <host>` to have it written for you, or add it by hand:

  ```yaml
  allowed_hosts:
    - your-mac.tailXXXX.ts.net
  ```

  Without it the server returns **403 to every request**, which the apps surface as
  "could not reach server" even though the network and token are fine.

See **[docs/remote-access.md](docs/remote-access.md)** for the full posture: travelling
with Tailscale (and why Proton VPN conflicts on iOS), the opt-in plain-LAN mode, and a
troubleshooting table for "could not reach server".

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
| Scanned PDF (no text layer) | OCR'd page-by-page with `tesseract` when installed (capped at 50 pages, per-page timeout, only pages with no extractable text); a clear warning otherwise |
| HTML | Text extracted (BeautifulSoup); RAG if > 80k chars |
| Text / code | Injected directly; RAG if > 80k chars |
| Images | OCR by default — text extracted with `tesseract` and folded into the prompt. Read as real images on omlx, or on mira-mlx with `mira_mlx_vision: true` |

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
| `backend` | `mira-mlx` | Inference backend (`mira-mlx`, `omlx`, `mlx-lm`, or `vllm-mlx`) |
| `model` | `mlx-community/Qwen3.6-35B-A3B-4bit` | Model identifier — an mlx-community repo id for mira-mlx/mlx-lm/vllm-mlx (e.g. `mlx-community/Ministral-3-14B-Instruct-2512-4bit` for the Mistral family), or omlx's own model name (`Qwen3.6-35B-A3B`) when `backend: omlx` |
| `host` | `http://localhost:8080` | Backend host URL |
| `embed_model` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model for RAG |
| `context_window` | `65536` | Token context window |
| `auth_token` | unset | Required to reach Mira from another device — see *Access control* above |
| `brave_api_key` | unset | Makes Brave the primary search provider instead of the DuckDuckGo fallback |
| `mira_mlx_vision` | `false` | Read image attachments with the model's own vision tower instead of OCR (mira-mlx only) |

Those are the ones most people touch. There are **45 in total**, covering sampling, thinking
budget, the shell sandbox, MoE expert offload and the opt-in performance flags — all listed with
their defaults in **[docs/configuration.md](docs/configuration.md)**, and annotated with the
reasoning behind the awkward ones in `mira.yaml.example` itself.

A handful of RAG and search internals (`RAG_CHUNK_SIZE`, `RAG_RETRIEVE_K`, `MAX_TOOL_STEPS`,
`SEARCH_TIMEOUT` and friends) have no `mira.yaml` equivalent — edit `core/config.py` if you need
them.

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

## Acknowledgements

Mira is built on a lot of other people's work. This is what it leans on, what it copies outright, and under which licence.

**Code vendored into this repo.** `core/inference/qwen3_vl_vision.py` is the Qwen3-VL vision tower from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.6.8 (MIT, © 2025 Prince Canuma), copied rather than depended on because mlx-vlm pulls in opencv-python, mlx-audio and a torch stack Mira does not otherwise need. The file carries its own attribution header.

**The inference stack.** mira-mlx is not an engine of its own. It drives [mlx](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) (both MIT, Apple) directly: model definitions, weight loading, `BatchGenerator` for continuous batching, `LRUPromptCache` for cross-turn prefix reuse and `ToolCallFormatter` for tool-call parsing are all theirs. What Mira adds on top is scheduling, RAM-aware sizing, disk cache overflow and MoE expert offload. [oMLX](https://omlx.ai), vllm-mlx and mlx-lm's own server are supported as alternative backends, unmodified and unbundled.

**Native multi-token prediction.** Mira's own MTP self-speculative decoding (`core/inference/mtp/`) is written from scratch, but the mechanism was learned by reading three references: the MTP head design from the [DeepSeek-V3](https://arxiv.org/abs/2412.19437) paper and Qwen3's own MTP, the native Qwen3.5/3.6 MTP support proposed in [mlx-lm PR #990](https://github.com/ml-explore/mlx-lm/pull/990), and [oMLX](https://github.com/jundot/omlx)'s Apache-2.0 `patches/mlx_lm_mtp` layer (© jundot), which showed how the head attaches to mlx-lm and, in particular, the RMSNorm weight-shift convention that sanitize has to preserve. No code is copied; the implementation is Mira's, and the Apache-2.0 attribution for the ideas learned from oMLX is recorded in the [`NOTICE`](./NOTICE) file.

**Models downloaded at runtime.** `Qwen3.6-35B-A3B` (Apache-2.0, Alibaba), `nomic-embed-text-v1.5` (Apache-2.0, Nomic AI) for embeddings and `Qwen3-Reranker-0.6B` for reranking. The MLX conversions come from the [mlx-community](https://huggingface.co/mlx-community) org.

**Retrieval and serving.** ChromaDB, sentence-transformers, transformers and python-multipart (Apache-2.0), PyTorch (BSD-3-Clause), FastAPI (MIT), and Starlette, uvicorn and sse-starlette (BSD-3-Clause).

**Documents and the web.** PyMuPDF (AGPL-3.0 or Artifex commercial, see below), trafilatura (Apache-2.0), Markdown (BSD-3-Clause), and beautifulsoup4, markdownify, ddgs, curl_cffi, httpx and rich (MIT).

> **A note on PyMuPDF.** It is dual licensed, AGPL-3.0 or a commercial licence from Artifex, and it is the only strong copyleft in Mira's dependency tree. Mira does not redistribute it: the wheel declares it as a dependency and pip fetches it from Artifex directly, so hosting Mira yourself changes nothing. It is worth knowing about if you fork Mira into something closed or build a hosted product on top of it. `pypdfium2` (BSD) is the usual escape hatch.

## License

This project is licensed under the **MIT License**. You can find the full text in the [`LICENSE`](./LICENSE) file.

> **Note on authorship:** Although much of the source code was generated by an AI, the creative direction, architecture, and final integration are human work. Usage rights are granted under the terms of the MIT License.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, workflow, and
how to open a pull request. For security issues, see [SECURITY.md](SECURITY.md). Feel free to
fork it and use this code in your own projects.
