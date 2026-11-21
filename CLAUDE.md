# Mira Core

FastAPI backend for Mira (Python `>=3.12,<3.14`, `.python-version` pins 3.13, CI tests the 3.12 floor).

## Stack facts that are easy to get wrong

- The default backend is mira-mlx (`core/inference/mira_mlx_server.py`), a child of `server.py`. omlx is the backup; mlx-lm and vllm-mlx are selectable. dflash and ollama are retired. `KNOWN_BACKENDS` in `core/backend_manager.py` is the list
- The default model and every tunable live in `mira.yaml` (`docs/configuration.md` has defaults, `core/config.py`'s `_get()` calls are the source of truth). Switching models: edit `mira.yaml`, `--model` is a label only
- Mistral-family models (Ministral) are the small-Mac model and must keep working, including tool calls
- Ports 8000 (HTTP) and 8443 (HTTPS) are hard-coded

## Constraints

- Always validate user input (command injection, path traversal)
- Connection resilience is critical, check mira-apps for UI patterns
- Shell operations use `subprocess` with an explicit args list, never `shell=True`
- Add dependencies with `uv add`, never `uv sync`
- The production server on :8000 and its engine may be in use by another session (mira-apps device tests). Check before starting, stopping or loading anything
- Never put a security finding about Mira's own code in a tracked file. It goes in the gitignored `SECURITY_AUDIT.md` or `notes/`
- Run the full pytest suite in two halves with production stopped. All 60+ files collected at once exits with no output
- Tests that touch mlx start with `pytest.importorskip("mlx.core")`, imports below it
- Precision and numerics tests run with `MLX_ENABLE_TF32=0`

## Reference docs

| File | When to read it |
|---|---|
| `BACKLOG.md` | What's pending, known bugs |
| `CHANGELOG.md` | What shipped in each version |
| `mira.yaml` | Active backend/model and all runtime tunables |
| `docs/configuration.md` | Every setting with its default. `mira.yaml.example` carries the reasoning |
| `docs/ga-readiness.md` | The 0.3 / 1.0 go-public checklist and its proofs |
| `SECURITY_AUDIT.md` | Findings from the monthly `/security-review`. Gitignored, local only |

## Commands

```bash
make serve      # start the web server (port 8000)
make chat       # interactive CLI
make doctor     # health check the install
make install    # deps + config (ARGS="--with-ocr --with-launchagent")
```

`/mira-server` manages the LaunchAgent (start | stop | install | reload | restart | logs | status). It, `/core-release` and `/security-review` are local skills on the maintainer's machine, not part of this repo.

## Release

Run `/core-release`: updates CHANGELOG.md, commits, tags, publishes a GitHub release. SemVer, run after a batch of features lands on main. `/mira-validate` and `/mira-release` are mira-apps commands and do not apply here. Spec format and release cadence are in the root `CLAUDE.md`.
