# Dev reference

## Commands

```bash
uv sync                                                    # install dependencies
python main.py                                             # CLI
python server.py                                           # web server → http://localhost:8000 (https → :8443)
uv add <package>                                           # add dependency
uv run python -m pytest --tb=short -q                      # all tests (no LLM server needed)
uv run python -m pytest tests/test_queries.py::test_name   # single test
uv run python scripts/benchmark.py                         # latency benchmark (Ollama + mlx-lm; writes to /tmp/)
uv run python scripts/benchmark.py --help                  # see options: --skip-ollama, --skip-mlx, --reps N
uv run python scripts/bench_standard.py --base-url http://localhost:8080   # pp/tg throughput bench (any OpenAI-compatible backend on 8080, incl. mira-mlx)
uv run python scripts/bench_compare.py --model <tag> --project-name <proj> # quality/agentic bench; --model is a label only, switch backend first via /models/switch
```

## Hardware

MacBook Pro M5 32GB unified memory — see `docs/model-comparison-m5-macbook.md` for the current model verdict and benchmark history.

## Ports

| Service | Port | Notes |
|---------|------|-------|
| Mira web server (HTTP) | 8000 | local browser / iOS on same network |
| Mira web server (HTTPS) | 8443 | Tailscale / remote iOS access |
| LLM inference backend | 8080 | started automatically by `backend_manager.py`; default backend is mira-mlx (`core/inference/mira_mlx_server.py`), also used by omlx/dflash/mlx-lm/vllm-mlx |
| Ollama | 11434 | optional inference fallback; no longer required for RAG embeddings (sentence-transformers runs locally) |
