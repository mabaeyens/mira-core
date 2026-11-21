# Going public: readiness checklist

Single source of truth for the move to 0.3 and then 1.0 with a public App Store release.
An item is done only when its **proof** has been run and passed, with the date. "Implemented"
is not done. Security findings about Mira's own code never go in this file (they stay in the
local, gitignored `SECURITY_AUDIT.md`); this file only records that the review was done.

Status: `open` / `in progress` / `done (date)` / `decided: <what>`.

## Goal

Download, install, and it works, like Ollama or oMLX. Settings are something you change
afterwards, not something you need in order to start.

## A. One-step install (mira-core)

| # | Item | Status | Proof |
|---|---|---|---|
| A1 | A first run needs no prior tools beyond what the installer fetches or clearly asks for (git/Xcode CLT, uv) | open | Fresh macOS user or VM: install command from the README reaches a healthy server |
| A2 | The model downloads as part of install by default, with progress and resume, not behind a default-No prompt | in progress | Built 2026-11-21 (`--recommended`); clean-account run still owed. | Fresh install with `-y`: model present in the HF cache, `mira doctor` green |
| A3 | A fresh install gets the same model and tuned config the owner runs (MTP head, kv4, vision, idle unload) | in progress | Built 2026-11-21: sidecar fetch is byte-identical to the owner's, and a throwaway server loaded the built dir with MTP armed. Clean-account run owed. | `mira doctor` reports MTP active; `/info` and the engine log match the owner's setup |
| A4 | The installer ends with a running server (LaunchAgent loaded, health checked) and prints the URL | done on this Mac (2026-11-21) | Simulated first install (isolated HOME, no uv, empty config): recommended install reached a running MTP engine and answered a chat. Only a second physical Mac is untested. | `curl localhost:8000/health` returns ok right after the installer exits |
| A5 | RAM-aware: a 16 or 24 GB Mac is steered to a model that fits instead of failing or thrashing | in progress | Profile picks the 14B below 31 GB. Open: confirm the 24 GB context-floor trap on a real 24 GB Mac. | `mira preflight` on a simulated 16 GB host picks the smaller model |
| A6 | Re-running the installer is safe and upgrades in place | open | Run twice; config and model untouched, server restarted cleanly |
| A7 | Uninstall is one command and leaves no LaunchAgent or stray files | open | `mira uninstall` then `launchctl list` and the install dir are clean |

## B. Packaging and distribution

| # | Item | Status | Proof |
|---|---|---|---|
| B1 | A packaging path is chosen for 1.0 (formula, notarized menu-bar app, or app-managed) | decided: notarized DMG with a menu-bar app (2026-11-21) | Spec in `specs/mira-menubar-app.md` (local) |
| B2 | Homebrew tap formula tracks the latest tag (`/brew-release`) | open | `brew install mabaeyens/tap/mira` on a clean machine, then A1 to A4 |
| B3 | Release artifacts are signed and notarized if a non-brew download is offered | in progress | Spike 2026-11-21: bundle notarized (Accepted), stapled, Gatekeeper accepts it as Notarized Developer ID with the quarantine flag set, and the full server boots from it. Open: the real app, DMG, update path. | `spctl -a -vv` accepts the artifact on a clean Mac |
| B4 | Update path exists and is tested | open | Old version to new version without losing config or data |

## C. Engine behaviour on a user's Mac

| # | Item | Status | Proof |
|---|---|---|---|
| C1 | Idle unload and wake on request, default on for installs | in progress | Server half shipped 2026-11-21 (`docs/engine-lifecycle.md`); app half in mira-apps, device test open |
| C2 | Proactive decompress off by default (the user's work comes first) | done (2026-11-21) | Code default is already false; the profile pins it. | Fresh `mira.yaml` has it false |
| C3 | The server never takes the machine: floor under memory pressure, no swap storms | open | Soak: other apps open, Mira idle and active, no swap growth |

## D. Apps (mira-apps)

| # | Item | Status | Proof |
|---|---|---|---|
| D1 | A first-time user can reach a server from the iPhone (pairing, LAN, Tailscale, QR) | open | Fresh device, no manual config files |
| D2 | App Review has a working path without the reviewer owning a Mac server | open | Decision and test notes recorded here |
| D3 | The engine-lifecycle UI is device-tested | open | TEST_PLAN.md run on a physical device |
| D4 | Release cadence respected (5-day minimum between TestFlight pushes) | open | Release notes and dates |

## E. Project hygiene before the repo or apps go public

| # | Item | Status | Proof |
|---|---|---|---|
| E1 | Monthly `/security-review` run on the release candidate | open | Date recorded; findings stay local |
| E2 | Licences and attribution complete (open: PyMuPDF AGPL-3.0) | open | `THIRD_PARTY` or equivalent matches installed dist-info |
| E3 | README quick start matches what the installer really does | open | A reader follows only the README on a clean machine |
| E4 | CHANGELOG current; version bumped (0.3 or 1.0) and tagged | open | `/core-release` run |
| E5 | No secrets or personal paths in tracked files or default config | open | Scan of tracked files and the generated `mira.yaml` |

## Found by the clean-account install test (2026-11-21)

Fixed: prompts swallowed under `curl | bash`, model silently skipped, README `fetch-model`
syntax, doctor showing red for optional items, final message naming a `mira` command that is
not on PATH. Still open:

- The embedding and reranker models (~1 GB) download silently on first use, and the
  embedder pulls unpinned remote code from `nomic-ai/nomic-bert-2048` (`core/rag_engine.py`).
- Ports 8000 and 8080 are hard-coded, so a second instance or a clean test run cannot move.
- `mira` is only at `.venv/bin/mira` unless installed with `uv tool install --editable .`.
- `--with-launchagent` ignores the `launchctl load` exit code and writes generated files
  into the repo.

## Signing spike result (2026-11-21)

Embedded runtime (python-build-standalone 3.13 plus the locked dependencies, no venv) in an
unsigned `Mira.app`: 1.2 GB on disk, 368 MB zipped, 322 native binaries. Signed inside-out
with the Developer ID Application certificate, hardened runtime, timestamp, **no
entitlements, no `--deep`**: `codesign --verify --strict` passes, and under the hardened
runtime mlx runs a GPU matmul and torch, sentence-transformers, chromadb, PyMuPDF,
onnxruntime and mlx-lm import. Signing 322 binaries takes 15 s. Notarization: Accepted in
5.5 min for a 368 MB zip. Stapled; `spctl --assess` accepts the quarantined copy ("Notarized
Developer ID"). The bundled server booted with config and data outside the bundle, MTP armed,
chat answered, zero files written inside the bundle, signature intact afterwards. Findings:
the launcher must forward SIGTERM to the Python child (killing it orphaned the server);
prune `test` directories from site-packages (10 joblib test archives drew warnings).

## Decisions

- Packaging path for 1.0 is undecided. Research recommends a notarized DMG with a menu-bar
  app that embeds the runtime and supervises the server (Mac App Store apps stay thin
  clients; a sandboxed Store app cannot run the server). Needs a signing spike first.
