# Homebrew distribution for Mira

`mira.rb` here is the **canonical** formula; the tap repo `mabaeyens/homebrew-mira`
holds a copy under `Formula/`. This directory is the source of truth. The tap is
**published** — install with:

```bash
brew tap mabaeyens/mira
brew install mira
```

## What `brew install` gives you (and what it doesn't)

`brew install mabaeyens/mira/mira` installs the `mira` command and the source tree
into the Homebrew Cellar. It does **not** download the ~19 GB model or build the
Python venv at install time — Homebrew's build sandbox has no network. Instead:

- `mira doctor`, `mira preflight`, `mira --help` run immediately (stdlib-only).
- The first command that needs dependencies (`mira serve`, `mira chat`,
  `mira fetch-model`) triggers a one-time `uv` sync of the venv (~1–2 min).
- `mira fetch-model` pulls the model on demand.

Mutable state stays out of the Cellar: config at `~/.config/mira/mira.yaml`
(via `MIRA_CONFIG`), data at `~/.local/share/mira` (the default). Both survive
`brew upgrade`; only the bundled venv is rebuilt after an upgrade.

## The tap repo

The tap is a separate public GitHub repo, `mabaeyens/homebrew-mira` (the `homebrew-`
prefix is what lets `brew tap mabaeyens/mira` resolve it). It's cloned locally at
`~/Projects/homebrew-mira`, and holds a single file: `Formula/mira.rb`, a copy of the
canonical formula in this directory.

It was created once with:

```bash
gh repo create mabaeyens/homebrew-mira --public --clone
```

You should not need to recreate it. Anyone installs with:

```bash
brew tap mabaeyens/mira && brew install mira
```

## Bumping on each release — use `/mira-brew-release`

`url` + `sha256` pin one tagged source tarball, so after `/core-release` tags a new
version the formula must be re-pinned. **Run `/mira-brew-release`** — it finds the
latest mira-core tag, computes the tarball sha256, updates both this canonical copy
and the tap's `Formula/mira.rb`, pushes both, and confirms `brew info` sees the new
version. It's a mechanical job, run by a Haiku agent.

The manual equivalent, if you ever need it:

```bash
VER=1.4.0
URL="https://github.com/mabaeyens/mira-core/archive/refs/tags/v${VER}.tar.gz"
# curl/wget are intercepted here on redirect — use python for the sha:
SHA=$(python3 -c "import urllib.request,hashlib; print(hashlib.sha256(urllib.request.urlopen('$URL').read()).hexdigest())")
# edit url + sha256 in mira.rb, copy into ~/Projects/homebrew-mira/Formula/mira.rb, push both
```

## Validate

```bash
ruby -c packaging/homebrew/mira.rb                          # syntax
brew info mabaeyens/mira/mira                               # taps + parses, no install
brew install --build-from-source mabaeyens/mira/mira        # end-to-end, on a clean machine
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
- **Honest verdict.** For a 19 GB-model, stateful, Apple-Silicon-only local server,
  the `curl | bash` installer remains the primary channel; Homebrew adds
  discoverability and `brew upgrade`, at the cost of the per-release bump.
