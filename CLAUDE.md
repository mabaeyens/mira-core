# Mira Core — Claude Code Reference

FastAPI backend for Mira.

## Project Stack

- **Framework:** FastAPI (Python 3.11+)
- **LLM Engine:** mira-mlx (Mira-owned MLX server, `core/inference/mira_mlx_server.py`; default backend as of 2026-07-09) — Qwen3.6-35B-A3B is the normal default model, 64k+ context; Mistral-family models (Ministral 3 14B) also fully supported, including tool-calling. omlx (~0ms TTFT after warm-up) is the backup backend; mlx-lm and vllm-mlx are also selectable via `mira.yaml`. dflash and ollama were retired 2026-08-01. Optional vision on mira-mlx via `mira_mlx_vision`, off by default
- **Embeddings:** sentence-transformers (`nomic-ai/nomic-embed-text-v1.5`, local, 768 dims)
- **Vector DB:** ChromaDB (ephemeral, for RAG)
- **Server:** Port 8000 (HTTP) / 8443 (HTTPS)

## Key Files

- `server.py` — FastAPI entry point, SSE streaming, backend/model switch endpoints
- `core/orchestrator.py` — `stream_chat()`, `_call_llm()`, tool dispatch loop
- `core/inference/mira_mlx_server.py` — mira-mlx: Mira-owned MLX inference server (continuous batching, RAM-aware sizing, disk-backed prompt cache, `/v1/stats`)
- `core/inference/disk_prompt_cache.py` — mira-mlx's disk-overflow prompt cache
- `core/hardware.py` — RAM-aware sizing (context window, prompt-cache/disk-cache budgets) shared by mira-mlx
- `core/backend_manager.py` — starts/stops/switches all inference backends (`KNOWN_BACKENDS`: mira-mlx, omlx, mlx-lm, vllm-mlx)
- `core/rag_engine.py` — embedding (sentence-transformers), ChromaDB, CrossEncoder reranker
- `core/config.py` — all tunables (`EMBED_MODEL`, `CONTEXT_WINDOW`, `RAG_*`, etc.)

## Constraints

- Always validate user input (command injection, path traversal)
- Connection resilience is critical — check mira-apps for UI patterns
- Shell operations must use `subprocess` with explicit args list, never shell=True

## Reference docs

| File | When to read it |
|---|---|
| `BACKLOG.md` | What's pending, known bugs |
| `CHANGELOG.md` | What shipped in each version |
| `mira.yaml` | Active backend/model and all runtime tunables |
| `SECURITY_AUDIT.md` | Findings from the monthly `/security-review`. **Gitignored, local only** — findings about Mira's own code are never published |
| `docs/` | Architecture and design notes |

## Commands

```bash
make serve      # start the web server (port 8000)
make chat       # interactive CLI
make doctor     # health check the install
make install    # deps + config (ARGS="--with-ocr --with-launchagent")
```

`/mira-server` manages the LaunchAgent (start | stop | install | reload | restart | logs | status).
It, `/core-release` and `/security-review` are local Claude Code skills on the maintainer's
machine, not part of this repo.

## Release

Run `/core-release` — updates CHANGELOG.md, commits, tags, and publishes a GitHub release.
SemVer. Run after a batch of features lands on main.

This is a Python backend: `/mira-validate` and `/mira-release` are **mira-apps** commands and do
not apply here. Spec format and release cadence live in the root `CLAUDE.md`.
