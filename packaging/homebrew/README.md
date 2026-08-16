# Homebrew distribution for Mira

`mira.rb` here is the **canonical** formula. It is not live until it's published in
a tap. This directory is the source of truth; the tap holds a copy.

## What `brew install` gives you (and what it doesn't)

`brew install mabaeyens/tap/mira` installs the `mira` command and the source tree
into the Homebrew Cellar. It does **not** download the ~19 GB model or build the
Python venv at install time — Homebrew's build sandbox has no network. Instead:

- `mira doctor`, `mira preflight`, `mira --help` run immediately (stdlib-only).
- The first command that needs dependencies (`mira serve`, `mira chat`,
  `mira fetch-model`) triggers a one-time `uv` sync of the venv (~1–2 min).
- `mira fetch-model` pulls the model on demand.

Mutable state stays out of the Cellar: config at `~/.config/mira/mira.yaml`
(via `MIRA_CONFIG`), data at `~/.local/share/mira` (the default). Both survive
`brew upgrade`; only the bundled venv is rebuilt after an upgrade.

## Publishing the tap (one-time)

The tap is a separate GitHub repo named `homebrew-tap`. Creating and pushing it is
a manual step (it needs your GitHub account — not automated from here):

```bash
gh repo create mabaeyens/homebrew-tap --public --clone
mkdir -p homebrew-tap/Formula
cp packaging/homebrew/mira.rb homebrew-tap/Formula/mira.rb
cd homebrew-tap && git add Formula/mira.rb && git commit -m "mira 1.3.0" && git push
```

Then anyone installs with:

```bash
brew install mabaeyens/tap/mira
# or:  brew tap mabaeyens/tap && brew install mira
```

## Bumping on each release

`url` + `sha256` pin one tagged source tarball. After tagging a new release
(`/core-release`), update both:

```bash
VER=1.4.0
URL="https://github.com/mabaeyens/mira-core/archive/refs/tags/v${VER}.tar.gz"
SHA=$(curl -LsS "$URL" | shasum -a 256 | awk '{print $1}')
# edit mira.rb: set url to $URL and sha256 to $SHA, then copy into the tap and push
```

(Worth folding into `/core-release` later so the formula bump is automatic.)

## Validate before publishing

```bash
ruby -c packaging/homebrew/mira.rb          # syntax
brew audit --formula --strict packaging/homebrew/mira.rb   # if you have the tap tapped
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
- **Honest verdict.** For a 19 GB-model, stateful, Apple-Silicon-only local server,
  the `curl | bash` installer remains the primary channel; Homebrew adds
  discoverability and `brew upgrade`, at the cost of this maintenance. Ship the tap
  if that discoverability is worth the per-release bump; otherwise the one-liner is
  already complete.
```
