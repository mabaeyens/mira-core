# Mira Core — Claude Code Reference

FastAPI backend for Mira. See `collaboration-notes.md` for session guidance and `../MIRA_WORKFLOW.md` for complete development workflow.

## Project Stack

- **Framework:** FastAPI (Python 3.11+)
- **LLM Engine:** mlx-lm (local inference, gemma-4-26b-a4b-it-4bit, 64k context)
- **Vector DB:** ChromaDB (ephemeral, for RAG)
- **Server:** Port 8000 (HTTP) / 8443 (HTTPS)

## Key Files

- `APIClient.swift` — Connection resilience: probes, startup status, retry logic
- `OllamaSearchApp.swift` — Entry point, reconnect banner, auto-connect logic
- See `collaboration-notes.md` for full file reference and patterns

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
