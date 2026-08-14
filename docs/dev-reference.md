# Dev reference

## Commands

```bash
uv sync                                                    # install dependencies
python main.py                                             # CLI
python server.py                                           # web server → http://localhost:8000 (https → :8443)
uv add <package>                                           # add dependency
uv run python -m pytest --tb=short -q                      # all tests (no LLM server needed)
uv run python -m pytest tests/test_queries.py::test_name   # single test
uv run python scripts/bench_standard.py --base-url http://localhost:8080   # pp/tg throughput bench (any OpenAI-compatible backend on 8080, incl. mira-mlx)
uv run python scripts/bench_compare.py --model <tag> --project-name <proj> # quality/agentic bench; --model is a label only, switch backend first via /models/switch
uv run python scripts/bench_eval.py --write-baseline       # score a completed run, update docs/quality-baseline.md
```

## Data directory

Everything Mira persists lives under one directory, `~/.local/share/mira` by
default: `conversations.db`, the Chroma RAG store, mira-mlx's prompt cache and
the expert-profile logs. Override it with **`MIRA_DATA_DIR`**:

```bash
MIRA_DATA_DIR=/tmp/mira-scratch uv run python server.py   # a throwaway instance
```

`tests/conftest.py` sets it to a fresh temp dir for every run. That is not
cosmetic: before it existed the suite wrote to the real database and left rows in
real conversation history. It has to be set before any project module is
imported, because `core/config.py` resolves the paths at import time and
`core/db.py` binds `DB_PATH` again — a fixture would run far too late.
`tests/test_data_isolation.py` is the tripwire if that ordering is ever broken.

## Hardware

MacBook Pro M5 32GB unified memory — see `docs/model-comparison-m5-macbook.md` for the current model verdict and benchmark history.

## Ports

| Service | Port | Notes |
|---------|------|-------|
| Mira web server (HTTP) | 8000 | local browser / iOS on same network |
| Mira web server (HTTPS) | 8443 | Tailscale / remote iOS access |
| LLM inference backend | 8080 | started automatically by `backend_manager.py`; default backend is mira-mlx (`core/inference/mira_mlx_server.py`). omlx, mlx-lm and vllm-mlx share the same port — only one runs at a time |

`scripts/benchmark.py` and `scripts/bench_standard.py` both predate the 2026-08-01 retirement
of the dflash and ollama backends. `benchmark.py`'s Ollama comparison arm (which shells out to
the `ollama` CLI on :11434, not through Mira) is now opt-in behind `--with-ollama` and off by
default, so a plain run benches only mlx-lm; `bench_standard.py` carries a harmless `ollama`
label branch but works fine against any OpenAI-compatible backend on 8080. Neither is the one to
reach for: `bench_compare.py` (quality/agentic) and `bench_eval.py` (scoring) are current, and
`bench_standard.py` remains useful for raw pp/tg throughput.

Past runs are indexed in `docs/bench-archive/README.md`; accepted scores are in
`docs/quality-baseline.md`.
