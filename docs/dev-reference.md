# Dev reference

## Commands

```bash
uv sync                                                    # install dependencies
python main.py                                             # CLI
python server.py                                           # web server → http://localhost:8000 (https → :8443)
uv add <package>                                           # add dependency
uv run python -m pytest --tb=short -q                      # all tests (no LLM server needed)
uv run python -m pytest tests/test_queries.py::test_name   # single test
```

## Hardware

MacBook Pro M5 32GB unified memory — see `docs/model-comparison-m5-macbook.md` for the current model verdict and benchmark history.

## Ports

| Service | Port | Notes |
|---------|------|-------|
| Mira web server (HTTP) | 8000 | local browser / iOS on same network |
| Mira web server (HTTPS) | 8443 | Tailscale / remote iOS access |
| mlx-lm inference | 8080 | started automatically by `backend_manager.py` |
| Ollama (RAG embeddings) | 11434 | start manually when RAG is needed; transitional — see `docs/mlx-embeddings-spec.md` |
