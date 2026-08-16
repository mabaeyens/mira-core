# Homebrew distribution for Mira

`mira.rb` here is the **canonical** formula; a copy lives in the tap repo
`mabaeyens/homebrew-tap` under `Formula/mira.rb`. This directory is the source of truth.

The tap is **published and multi-project** — its short name is `mabaeyens/tap`, and it's
meant to hold formulae for several tools (Mira today, others later; see "Adding another
project" below).

## Install

```bash
brew tap mabaeyens/tap
brew trust mabaeyens/tap   # Homebrew 6+ only: it refuses to load untrusted third-party taps
brew install mira
```

On Homebrew 5 and earlier there is no trust step — `brew tap` then `brew install mira` is
enough. `brew install mabaeyens/tap/mira` is the fully-qualified form if the bare `mira`
name is ever ambiguous (it isn't today — nothing named `mira` is in homebrew-core).

## What `brew install` gives you (and what it doesn't)

`brew install mira` installs the `mira` command and the source tree into the Homebrew
Cellar. It does **not** download the ~19 GB model or build the Python venv at install
time — Homebrew's build sandbox has no network. Instead:

- `mira doctor`, `mira preflight`, `mira --help` run immediately (stdlib-only).
- The first command that needs dependencies (`mira serve`, `mira chat`,
  `mira fetch-model`) triggers a one-time `uv` sync of the venv (~1–2 min).
- `mira fetch-model` pulls the model on demand.

Mutable state stays out of the Cellar: config at `~/.config/mira/mira.yaml`
(via `MIRA_CONFIG`), data at `~/.local/share/mira` (the default). Both survive
`brew upgrade`; only the bundled venv is rebuilt after an upgrade.

## The tap repo

The tap is a public GitHub repo, `mabaeyens/homebrew-tap` (the `homebrew-` prefix is what
lets `brew tap mabaeyens/tap` resolve it). It's cloned locally at `~/Projects/homebrew-tap`.
Each tool is a single file under `Formula/`; `mira.rb` is the only one today.

## Adding another project (Vera, etc.)

The tap is deliberately generic so it can host every tool:

1. Author the project's formula (its own `packaging/homebrew/<name>.rb` in that repo is a
   good home for the canonical copy).
2. Copy it into `~/Projects/homebrew-tap/Formula/<name>.rb` and push.
3. Users then run `brew tap mabaeyens/tap` once and `brew install <name>`.

`/brew-release <project>` handles the per-release version bump for any project wired into
its `PROJECTS` map.

## Bumping on each release — use `/brew-release`

`url` + `sha256` pin one tagged source tarball, so after a project's release tag lands the
formula must be re-pinned. **Run `/brew-release mira`** (the project name; defaults to
`mira`) — it finds the latest tag for that project, computes the tarball sha256, updates
both the canonical copy and the tap's `Formula/<name>.rb`, pushes both, and confirms
`brew info` sees the new version. It's a mechanical job, run by a Haiku agent.

The manual equivalent, if you ever need it:

```bash
VER=1.4.0
URL="https://github.com/mabaeyens/mira-core/archive/refs/tags/v${VER}.tar.gz"
# curl/wget are intercepted here on redirect — use python for the sha:
SHA=$(python3 -c "import urllib.request,hashlib; print(hashlib.sha256(urllib.request.urlopen('$URL').read()).hexdigest())")
# edit url + sha256 in mira.rb, copy into ~/Projects/homebrew-tap/Formula/mira.rb, push both
```

## Validate

```bash
ruby -c packaging/homebrew/mira.rb                         # syntax
brew info mabaeyens/tap/mira                               # taps + parses, no install
brew install --build-from-source mabaeyens/tap/mira        # end-to-end, on a clean machine
```

## Design notes / tradeoffs (why it's shaped this way)

- **Formula, not cask.** Casks are for `.app`/DMG bundles. Mira is a CLI + server,
  so a formula is correct; the *oMLX* app is the cask-shaped piece, and it stays a
  manual/optional install.
- **uv-driven, not resource-enumerated.** The idiomatic Homebrew Python formula
  lists every wheel as a `resource` with its own sha256. For Mira's MLX/ML stack
  that list would be huge and break on every dependency bump. Delegating to `uv`
  trades Homebrew-hermetic builds for a maintainable formula — a deliberate call
  for a personal tap, not homebrew-core.
- **Lazy venv + lazy model.** Neither can happen in Homebrew's sandbox, and forcing
  a 19 GB download at install time would be hostile. First-run sync + `fetch-model`
  keep `brew install` fast and honest.
- **Personal tap, not homebrew-core.** homebrew-core requires hermetic offline builds
  (every dependency vendored with a pinned sha256) plus notability review. Mira's
  uv-delegated deps, 19 GB first-run model download, and Apple-Silicon-only runtime
  make it a poor fit and it would be rejected. The tap adds discoverability and
  `brew upgrade` at a per-release bump cost, which is the right trade here.
