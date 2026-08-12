# Contributing to mira-core

Thanks for your interest in mira-core — the local AI assistant backend (FastAPI + local
inference, RAG, and autonomous web search). This is a small, single-maintainer project, but
issues, ideas, and pull requests are welcome. Feel free to fork it and build something of
your own.

## Reporting bugs and requesting features

Open an [issue](https://github.com/mabaeyens/mira-core/issues). The bug template asks for your
macOS version, Mac model, Python version, the mira-core version, and the active backend + model.
Please fill those in — they make bugs far easier to reproduce. `make doctor` prints most of them
in one go, and the template has a spot to paste it.

Questions, ideas, and "is this actually useful" feedback are better in
[Discussions](https://github.com/mabaeyens/mira-core/discussions).

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

- **Spec first.** For anything non-trivial, write a five-bullet spec before writing code. The
  format is below. It takes about five minutes and it is the single biggest reason changes here
  land in one pass instead of three.
- **One feature or fix per pull request.** Keep commits coherent — squash trial-and-error
  noise before opening the PR.
- **Match the surrounding code.** Always validate user input (command injection, path
  traversal). Shell operations use `subprocess` with an explicit args list, never
  `shell=True`. Tunables live in `core/config.py`; the orchestration loop is in
  `core/orchestrator.py`.

## The five-bullet spec

Five headings, no more. The point is not ceremony — it is that bullets 3 and 4 are where the
work actually gets decided, and writing them first is much cheaper than discovering them in
review.

1. **Problem** — what is broken or missing. Observable behaviour, not a proposed solution.
2. **Files** — which files change, and which function you touch first.
3. **Constraint** — one hard rule the implementation must not violate.
4. **Edge cases** — the two or three cases that will break a naive version.
5. **Done** — two or three acceptance criteria you can actually check.

A worked example, from a fix that shipped on 2026-08-12:

> **1. Problem.** Retrying a failed reply saves the question twice. The client re-sends the
> same message, the server appends it again, and the conversation ends up with two identical
> user rows and one answer. Visible in the app as a duplicated question after any retry.
>
> **2. Files.** `server.py` — the `/chat` handler, inside `event_stream()` before the history
> snapshot is taken. `core/db.py` — needs a new `drop_last_turn()`.
>
> **3. Constraint.** The rollback must happen under the conversation lock and before the
> snapshot, or a concurrent request sees a half-rolled-back history. The client cannot fix this
> itself: by the time it knows the reply failed, it has already sent the message.
>
> **4. Edge cases.** (a) A retry on a brand-new conversation, where there is no previous turn
> to drop. (b) A turn that persisted more than one question and one answer — tool calls write
> rows too, so deleting "the last two rows" is wrong; delete from the last `user` row forward.
> (c) FTS rows carry no matchable row id, so the conversation's search index has to be rebuilt
> rather than patched.
>
> **5. Done.** Three sends against one conversation — first send, plain re-send, retry re-send
> — leave 2, 4 and 4 rows respectively. `pytest` green. The duplicate no longer appears in the
> app.

Note what bullet 4 bought: "delete the last two rows" is the obvious implementation and it is
wrong, and the spec caught that before any code existed.

## Before you open a pull request

- Run the test suite: `uv run pytest`. All model, search, and fetch calls are mocked, so no
  inference server is needed to run tests.
- Don't commit secrets, API keys, or local config (`mira.yaml`, signing credentials, etc.).
- Update the README / CHANGELOG if the change is user-facing.

## Pull request

Fill in the PR template (summary, linked issue, how you tested). Then open it against `main`.
