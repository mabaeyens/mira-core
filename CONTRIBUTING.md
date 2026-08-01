# Contributing to mira-core

Thanks for your interest in mira-core — the local AI assistant backend (FastAPI + local
inference, RAG, and autonomous web search). This is a small, single-maintainer project, but
issues, ideas, and pull requests are welcome. Feel free to fork it and build something of
your own.

## Reporting bugs and requesting features

Open an [issue](https://github.com/mabaeyens/mira-core/issues). The issue template will
prompt you for your OS, Python version, the mira-core version, and the active backend +
model — please fill those in, they make bugs far easier to reproduce.

**Security issues do not go in public issues** — see [SECURITY.md](SECURITY.md).

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.12+ (`>=3.12,<3.14`) |
| [uv](https://docs.astral.sh/uv/) | latest |
| macOS | Apple Silicon (MLX-based inference) |
| Inference backend | mira-mlx (bundled) or oMLX 0.4.3+ |

The native macOS + iOS clients live in
**[mira-apps](https://github.com/mabaeyens/mira-apps)** — this repo is the server they talk
to.

## Setup

```bash
make install          # installs uv, syncs deps, writes mira.yaml
# or the one-liner installer:
#   curl -LsSf https://raw.githubusercontent.com/mabaeyens/mira-core/main/install.sh | bash
```

`scripts/setup.sh` is idempotent — re-run it any time. Then:

```bash
make serve            # web UI + API at http://localhost:8000
make chat             # interactive CLI
make doctor           # health-check the install
```

## Workflow

- **Spec first.** For anything non-trivial, jot a short spec (problem, files to touch, a hard
  constraint, edge cases, acceptance criteria) before writing code — see the 5-bullet spec
  format in `CLAUDE.md`.
- **One feature or fix per pull request.** Keep commits coherent — squash trial-and-error
  noise before opening the PR.
- **Match the surrounding code.** Always validate user input (command injection, path
  traversal). Shell operations use `subprocess` with an explicit args list, never
  `shell=True`. Tunables live in `core/config.py`; the orchestration loop is in
  `core/orchestrator.py`.

## Before you open a pull request

- Run the test suite: `uv run pytest`. All model, search, and fetch calls are mocked, so no
  inference server is needed to run tests.
- Don't commit secrets, API keys, or local config (`mira.yaml`, signing credentials, etc.).
- Update the README / CHANGELOG if the change is user-facing.

## Pull request

Fill in the PR template (summary, linked issue, how you tested). Then open it against `main`.
