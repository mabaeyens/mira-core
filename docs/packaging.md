# Packaging Mira — wheels, options, and what we chose

A reference for how mira-core is packaged and released, the decisions behind it, and how
they compare to Python packaging best practice.

## 1. The two artifacts: sdist vs wheel

| | **sdist** (`.tar.gz`) | **wheel** (`.whl`) |
|---|---|---|
| What it is | Source distribution — raw files + `pyproject.toml`; the installer **builds** it on the user's machine | Built distribution — a zip of the already-laid-out files, installed by **copying**, no build step |
| Filename encodes | name + version | name + version + Python tag + ABI tag + platform tag |
| Speed/safety | Slower, can fail at install time (needs build tools) | Fast, deterministic |

Our wheel is `mira_core-0.8.2-py3-none-any.whl`:

- `mira_core` = package name, `0.8.2` = version (from the git tag)
- `py3` = any Python 3, `none` = no compiled C-ABI of our own, `any` = any OS

`py3-none-any` ("pure-Python, universal") matters: **our wheel is platform-agnostic, but Mira is
not.** Mira is macOS-Apple-Silicon-only because of its *dependencies* (`mlx`, `mlx-metal`), not
its own code. That constraint lives in `pyproject.toml` dependency markers
(`sys_platform == 'darwin'`), so the wheel installs anywhere but the deps won't resolve off
macOS. That is normal and fine.

## 2. The packaging decision stack

"Packaging" conflates four independent choices:

| Layer | Options | Mira's choice |
|---|---|---|
| **Build backend** | setuptools, **hatchling**, flit, pdm, maturin (Rust), scikit-build (C++) | **hatchling** — modern default, good for pure-Python |
| **Version source** | hardcoded in `pyproject.toml`; `__version__` in code; **VCS / git-tag** (hatch-vcs, setuptools-scm) | **hatch-vcs** (tag-driven) |
| **Distribution channel** | PyPI; private index; **git/GitHub**; Homebrew; npm; conda | **GitHub release asset + source repo** |
| **Install UX** | `pip install`, `uv add`, `uv tool install`, `uvx`, `pipx` | `uv tool install --editable .` + the `install.sh`/`make` bootstrap |

## 3. Best practice vs. what we chose

| Best practice | What Mira does | Verdict |
|---|---|---|
| `pyproject.toml` (PEP 621), no `setup.py` | Yes | Match |
| **Single source of truth for version** | Git tag via hatch-vcs — version cannot drift | **Match** (gold-standard option; many projects settle for "remember to bump the toml") |
| Build wheel **on a clean tree at the tag** | Release script enforces this (else you get a `.dev`/`.dirty` suffix) | Match |
| `src/` layout (prevents accidental imports of the working dir) | Flat layout (`main.py`, `server.py`, `core/` at root) | **Deliberate deviation** — see below |
| Publish to PyPI for `pip install <name>` | GitHub asset only | **Deliberate** — correct for Mira |
| Wheel is self-contained / runnable from `site-packages` | Partial — code installs, but Mira needs the **checkout** to run | **The one real caveat** |

## 4. The two deviations, and why they're right for Mira

**Flat layout instead of `src/`.** The textbook recommendation is `src/` because it forces you to
test the *installed* package, not the source dir. Mira is flat because the repo predates packaging
and the entry points (`server.py`, `main.py`) are run directly. Low cost here since we don't
`pip install mira-core` from PyPI and import it as a library — but it is the one thing to change if
we ever want a clean library-style package.

**Not on PyPI.** This is the correct call, and worth being explicit about *why*:

> Mira is an **application that runs from its checkout**, not a library. It needs `mira.yaml`,
> `static/`, the SQLite DBs, and the rendered LaunchAgent — all relative to a real directory. A
> PyPI wheel installed into `site-packages` has the code but none of that runtime context. That is
> why `mira doctor` and the `mira` CLI resolve a **repo root** (`MIRA_HOME` → source dir →
> `~/mira-core`) instead of assuming `site-packages`.

So the wheel's honest role is: **a convenient, versioned snapshot of the code attached to each
release** — not a "pip-installable product." The real install paths are the `install.sh` one-liner
and `make install`, with `uv tool install --editable .` as the packaged-command option (editable
keeps the code in the checkout, so paths resolve).

## 5. When each install path is the right one

- **`curl … install.sh | bash`** — fresh machine, nothing assumed. Clones + sets up.
- **`make install`** — already cloned; day-to-day.
- **`uv tool install --editable .`** — want a global `mira` command but still run from the
  checkout. This is the "package" answer.
- **The release wheel** — pinning an exact version, an air-gapped copy, or
  `uv tool install <url-to-wheel>` without cloning (works for the CLI/doctor; the full server still
  wants a checkout).

### Uninstalling the `mira` tool

`uv tool` keys on the **package** name, not the executable. The package is `mira-core` and the
executable is `mira`, so:

```bash
uv tool uninstall mira-core   # correct
uv tool uninstall mira        # error: `mira` is not installed
```

This bit us during the fresh-install test — a teardown that ran `uv tool uninstall mira` silently
left the tool behind.

## 6. How a release is cut

Versioning is tag-driven, so the version lives in exactly one place — the git tag. The
`/core-release` skill automates this; the mechanics:

1. Update `CHANGELOG.md`; commit `CHANGELOG.md` + any config changes on a clean tree.
2. `git tag vX.Y.Z` at that commit and push the tag (backfill tags for any untagged changelog
   versions first).
3. `uv build --wheel` — hatch-vcs reads the tag, so the wheel is named `mira_core-X.Y.Z`. Build
   only after tagging on a clean tree, or the version gets a `.devN`/`.dXXXXXXXX` suffix.
4. `gh release create vX.Y.Z` and `gh release upload vX.Y.Z dist/*.whl` (glob — never hardcode the
   version in the filename).

Relevant config in `pyproject.toml`:

```toml
[project]
name = "mira-core"
dynamic = ["version"]          # version is NOT stored here

[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[tool.hatch.version]
source = "vcs"                 # the git tag is the single source of truth

[tool.hatch.build.targets.wheel]
include = ["core", "mira_cli.py", "main.py", "server.py"]
```

## TL;DR

We picked the **most-correct version strategy that exists** (tag-driven, zero-drift) and the
**right distribution model for an app** (GitHub + source, not PyPI). The single honest caveat is
that the wheel is a code snapshot, not a run-anywhere product — inherent to Mira being a
directory-bound application, handled cleanly with `MIRA_HOME` resolution and the bootstrap scripts.
The only thing to revisit *if goals change* (e.g. wanting `pip install mira-core` to Just Work) is
the `src/` layout + bundling runtime data — not worth it today.
