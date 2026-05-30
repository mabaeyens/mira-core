# Mira Core — Claude Code Reference

FastAPI backend for Mira. See `collaboration-notes.md` for session guidance and `../MIRA_WORKFLOW.md` for complete development workflow.

## Project Stack

- **Framework:** FastAPI (Python 3.11+)
- **LLM Engine:** mlx-lm (local inference, gemma-4-26b-a4b-it-4bit, 64k context)
- **Embeddings:** sentence-transformers (`nomic-ai/nomic-embed-text-v1.5`, local, 768 dims)
- **Vector DB:** ChromaDB (ephemeral, for RAG)
- **Server:** Port 8000 (HTTP) / 8443 (HTTPS)

## Key Files

- `server.py` — FastAPI entry point, SSE streaming, backend/model switch endpoints
- `core/orchestrator.py` — `stream_chat()`, `_call_llm()`, tool dispatch loop
- `core/rag_engine.py` — embedding (sentence-transformers), ChromaDB, CrossEncoder reranker
- `core/config.py` — all tunables (`EMBED_MODEL`, `CONTEXT_WINDOW`, `RAG_*`, etc.)
- See `collaboration-notes.md` for patterns and session guidance

## Constraints

- Always validate user input (command injection, path traversal)
- Connection resilience is critical — check mira-apps for UI patterns
- Shell operations must use `subprocess` with explicit args list, never shell=True

## Workflow Reference

See `../MIRA_WORKFLOW.md` for:
- Session checklist and 5-bullet spec format (section 2)
- Validation before releasing (section 5)
- Release cadence (1 per week, section 7)
- Monthly security audit (section 6)
- Token efficiency tips (sections 1 and 8)
