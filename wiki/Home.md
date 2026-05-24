# Mira

A local AI assistant with autonomous web search, file attachments, and RAG for large documents. Runs entirely on your machine — no cloud APIs, no API keys.

Two interfaces: a CLI and a local web UI.

---

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- **Ollama** (v0.24.0+) — with `gemma4:26b-mlx` and `nomic-embed-text` pulled

---

## Setup

```bash
git clone git@github.com:mabaeyens/mira-core.git
cd mira-core
uv sync
```

> On first use, the CrossEncoder reranker (~100 MB) downloads automatically from HuggingFace and caches to `~/.cache/huggingface/`.

### Backend configuration

Copy the example config and edit as needed:

```bash
cp mira.yaml.example mira.yaml
```

`mira.yaml` is git-ignored. Default backend is Ollama with `gemma4:26b-mlx`:

```yaml
backend: ollama
model: gemma4:26b-mlx
host: http://localhost:11434
embed_backend: ollama
embed_model: nomic-embed-text
context_window: 65536
```

---

## Running

**Start Ollama first** (`ollama serve`), then:

```bash
# CLI
python main.py

# Web interface → http://localhost:8000
python server.py
```

The web server does not start the inference backend automatically — it will fail with a connection error if the backend is not running.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/attach <path>` | Stage a file for the next message (PDF, HTML, image, text) |
| `/files` | List staged attachments |
| `/detach` | Clear staged attachments |
| `/rag-list` | List documents in the RAG index |
| `/rag-remove <name>` | Remove a document from the RAG index |
| `/toggle` | Toggle verbose mode (show/hide search and fetch details) |
| `/verbose` | Enable verbose mode |
| `/quiet` | Disable verbose mode |
| `/reset` | Reset conversation history and RAG index |
| `/quit` | Exit |

---

## Web Interface

- Streaming responses with live markdown rendering
- File attach button — upload PDFs, images, text, HTML from your machine
- Folder browser — navigate the server filesystem, filter by extension, select multiple files
- Search chips expand to show clickable source links; fetch chips link directly to the fetched page
- Documents panel — shows RAG-indexed files with per-doc remove
- Status bar — current operation (Thinking / Searching / Reading / Indexing)
- Token counter — session input/output tokens and context window fill percentage
- Stop button — aborts the current response mid-stream; history rolls back as if the turn never happened
- Enter to send, Shift+Enter for newline

---

## File Attachments

| File type | Behaviour |
|-----------|-----------|
| PDF (any size) | Always indexed via RAG |
| HTML | Text extracted; RAG if > 80k chars |
| Text / code | Injected directly; RAG if > 80k chars |
| Images | Passed via multimodal API (base64) |
| Scanned PDF | Warning shown; no text extractable |

RAG documents persist across turns in the same session. Use `/rag-remove` or Reset to clear.

---

## Run as a Background Service (macOS)

To have the web server start automatically at login:

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

Both filled-in files are git-ignored — only the `*.template` originals are committed.
