# Mira Core — Claude Code Reference

FastAPI backend for Mira.

## Project Stack

- **Framework:** FastAPI (Python 3.11+)
- **LLM Engine:** omlx (local inference, Qwen3.6-35B-A3B, 64k context; ~0ms TTFT after startup warm-up)
- **Embeddings:** sentence-transformers (`nomic-ai/nomic-embed-text-v1.5`, local, 768 dims)
- **Vector DB:** ChromaDB (ephemeral, for RAG)
- **Server:** Port 8000 (HTTP) / 8443 (HTTPS)

## Key Files

- `server.py` — FastAPI entry point, SSE streaming, backend/model switch endpoints
- `core/orchestrator.py` — `stream_chat()`, `_call_llm()`, tool dispatch loop
- `core/rag_engine.py` — embedding (sentence-transformers), ChromaDB, CrossEncoder reranker
- `core/config.py` — all tunables (`EMBED_MODEL`, `CONTEXT_WINDOW`, `RAG_*`, etc.)

## Constraints

- Always validate user input (command injection, path traversal)
- Connection resilience is critical — check mira-apps for UI patterns
- Shell operations must use `subprocess` with explicit args list, never shell=True

## Validation & Release

Before any release:
1. Run `/mira-validate` — builds simulator + sideloads to device
2. Manual smoke check (2 min): launch, open conversation, send message, check specific feature
3. Run `/mira-release` — bumps version, archives both platforms, uploads to TestFlight

**Release cadence:** One per week (Friday or Monday).
**Security audit:** Run `/security-review` last weekend of each month.

## Spec Format (5 bullets)

When a new bug or feature request arrives, write the spec to `specs/<slug>.md` before implementing. The `specs/` folder is gitignored (local only). Once implemented, the relevant detail moves to README or architecture docs — the spec file can then be deleted.

1. Problem: what is broken or missing
2. Files: which files to change and which functions to touch first
3. Constraint: a hard rule (don't show X if Y, match pattern Z)
4. Edge cases: (a) case 1, (b) case 2
5. Done: acceptance criteria (2–3 bullets)
