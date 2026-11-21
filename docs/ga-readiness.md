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
| A2 | The model downloads as part of install by default, with progress and resume, not behind a default-No prompt | in progress | Built 2026-11-21 (`--recommended`) and run end to end on a simulated clean account, but with the 19 GB model symlinked from the existing cache. A from-scratch download is not yet run. Fresh install with `-y`: model present in the HF cache, `mira doctor` green |
| A3 | A fresh install gets the same model and tuned config the owner runs (MTP head, kv4, vision, idle unload) | done on this Mac (2026-11-21) | Sidecar fetch is byte-identical to the owner's; the simulated install ran with MTP armed, kv4 and a 15 minute idle unload, and answered a chat. | `mira doctor` reports MTP active; `/info` and the engine log match the owner's setup |
| A4 | The installer ends with a running server (LaunchAgent loaded, health checked) and prints the URL | done on this Mac (2026-11-21) | Simulated first install (isolated HOME, no uv, empty config): recommended install reached a running MTP engine and answered a chat. Only a second physical Mac is untested. | `curl localhost:8000/health` returns ok right after the installer exits |
| A5 | RAM-aware: a 16 or 24 GB Mac is steered to a model that fits instead of failing or thrashing | in progress | Profile picks the 14B below 31 GB. Open: confirm the 24 GB context-floor trap on a real 24 GB Mac. | `mira preflight` on a simulated 16 GB host picks the smaller model |
| A6 | Re-running the installer is safe and upgrades in place | open | Run twice; config and model untouched, server restarted cleanly |
| A7 | Uninstall is one command and leaves no LaunchAgent or stray files | open | `mira uninstall` then `launchctl list` and the install dir are clean |

## B. Packaging and distribution

| # | Item | Status | Proof |
|---|---|---|---|
| B1 | A packaging path is chosen for 1.0 (formula, notarized menu-bar app, or app-managed) | decided: notarized DMG with a menu-bar app (2026-11-21) | Spec in `specs/mira-menubar-app.md` (local) |
| B2 | Homebrew tap formula tracks the latest tag (`/brew-release`) | open | `brew install mabaeyens/tap/mira` on a clean machine, then A1 to A4 |
| B3 | Release artifacts are signed and notarized if a non-brew download is offered | done on this Mac (2026-11-21) | `mira-menubar/scripts/release.sh` built, signed, notarized (app and DMG both Accepted) and stapled `Mira-0.1.0.dmg` (609 MB); Gatekeeper accepts it as Notarized Developer ID. Open: a Mac that is not the build machine, and the update path (B4). | `spctl -a -vv` accepts the artifact on a clean Mac |
| B4 | Update path exists and is tested | open | Old version to new version without losing config or data |

## C. Engine behaviour on a user's Mac

| # | Item | Status | Proof |
|---|---|---|---|
| C1 | Idle unload and wake on request, default on for installs | done (2026-11-21) | Server half shipped (`docs/engine-lifecycle.md`), the recommended profile sets a 15 minute window, and the mira-apps device tests passed (D3). Wake takes 9 to 30 s. |
| C2 | Proactive decompress off by default (the user's work comes first) | done (2026-11-21) | Code default is already false; the profile pins it. | Fresh `mira.yaml` has it false |
| C3 | The server never takes the machine: floor under memory pressure, no swap storms | open | Soak: other apps open, Mira idle and active, no swap growth |

## D. Apps (mira-apps)

| # | Item | Status | Proof |
|---|---|---|---|
| D1 | A first-time user can reach a server from the iPhone (pairing, LAN, Tailscale, QR) | open | Fresh device, no manual config files |
| D2 | App Review has a working path without the reviewer owning a Mac server | open | Decision and test notes recorded here |
| D3 | The engine-lifecycle UI is device-tested | done (2026-11-21) | Physical iPhone against HTTPS 8443: asleep banner, wake on composer focus, first message after a sleep (29.8 s wake), Unload now, and Unload now while busy (409 and the Busy message). The first run exposed a client watchdog that cancelled any wake over 15 s; fixed in mira-apps and re-verified. |
| D4 | Release cadence respected (5-day minimum between TestFlight pushes) | open | Release notes and dates |

## E. Project hygiene before the repo or apps go public

| # | Item | Status | Proof |
|---|---|---|---|
| E1 | Monthly `/security-review` run on the release candidate | open | Date recorded; findings stay local |
| E2 | Licences and attribution complete (open: PyMuPDF AGPL-3.0) | open | `THIRD_PARTY` or equivalent matches installed dist-info |
| E3 | README quick start matches what the installer really does | open | A reader follows only the README on a clean machine |
| E4 | CHANGELOG current; version bumped (0.3 or 1.0) and tagged | open | `/core-release` run |
| E5 | No secrets or personal paths in tracked files or default config | open | Scan of tracked files and the generated `mira.yaml` |

## F. Menu-bar app (`mira-menubar`, private repo)

Supervises the embedded server, runs first-run setup, ships as the notarized DMG (B1).

| # | Item | Status | Proof |
|---|---|---|---|
| F1 | Supervisor starts and stops the server as one process group, restarts with backoff, gives up on a crash loop | done (2026-11-21) | 25 unit tests with a stub server, none load the model; stop kills the child process too |
| F2 | First-run flow: RAM and disk checks, model download with progress, then the server starts | in progress | Logic unit-tested; the window renders on the notarized build. Not run: the real download from the app |
| F3 | Quitting the app (menu, SIGTERM, logout) stops the server it started and leaves one it did not start | done (2026-11-21) | First build hung on SIGTERM (found by launching it); fixed, the rebuilt notarized app exits in 1 s and a stub server plus child both die |
| F4 | Build, sign, notarize, staple and DMG in one command | done (2026-11-21) | `scripts/release.sh`; both submissions Accepted twice |
| F5 | The menu itself works: status, Open, Wake, Free Memory, Start, Stop, Restart, Open at Login | open | Not looked at yet (a screenshot did not capture the menu bar) |
| F6 | The app adopts a server that is already running without restarting it | open | Unit-tested in the status logic only; run against a live server |
| F7 | Open at Login via `SMAppService` | open | Toggle on an installed copy, log out and in |
| F8 | Isolated-run support: the app ignores `HOME`, so a test run uses the real `~/Library/Application Support/Mira` | open | A path override, then a test run that touches nothing real |
| F9 | Pairing the iPhone app and the Mac client (QR or keychain with URL and token) | open | D1. The menu-bar app generates no token today (loopback needs none). The Mac client reads its token from the keychain (`com.mab.mira.apitoken`, sandboxed), so a token the menu-bar app creates reaches it only through a shared access group or a paste. Decide before adding token generation. |
| F10 | Updates (Sparkle) | open | B4 |

## G. Web UI parity (`static/index.html`)

Reviewed 2026-11-21 against the server routes and the mira-apps client. The web UI calls
`/chat /cancel /compact /reset /verbose /status /health /browse /conversations /projects
/rag/documents`. Streaming events and file attach, folder browse, projects, approvals, think
and verbose are all handled.

| # | Gap | Status | Notes |
|---|---|---|---|
| G1 | Engine asleep/waking state and click-to-wake | done (2026-11-21) | Header badge, polls `/health`. Click-to-wake verified against a stub; hiding after wake not seen on screen. |
| G2 | Model badge showed a filesystem path and a stale `gemma4:26b` placeholder | done (2026-11-21) | Shows the last path component |
| G3 | Memories: list, add, delete (`/memories`) | open | In the native apps, absent in the web UI |
| G4 | Conversation rename, search and move to project (`PATCH /conversations/{id}`, `/conversations/search`) | open | Only create, open and delete exist |
| G5 | Reminders list and delete (`/reminders`) | open | |
| G6 | Backend and model switching, model pull (`/backends /backend /models /models/switch /models/pull`) | open | CLI and mira.yaml only; decide whether the web UI should expose it |
| G7 | Free memory now (`/engine/unload`) | open | Needs the 409 busy case handled, as in the native apps |
| G8 | `divergence_guard` stream event is ignored | open | Harmless; no user-visible notice |
| G9 | Bundled copy: the web UI served by the menu-bar app is the mira-core inside the app, so web changes reach users only after a rebuild | noted | |

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
the launcher must forward SIGTERM to the Python child (killing it orphaned the server;
resolved: the app owns the process group and kills it whole, F1); prune `test` directories
from site-packages (10 joblib test archives drew warnings; still open). Two build bugs found
while turning the spike into `release.sh`: universal binaries must be de-duplicated before
parallel signing, and the Python runtime must be copied as real files, not a link into
`~/.local/share/uv`.

## Decisions

- Packaging path for 1.0: a notarized DMG with a menu-bar app that embeds the runtime and
  supervises the server (Mac App Store apps stay thin clients; a sandboxed Store app cannot
  run the server). The signing spike passed and the app is built (section F). The App Store
  position rests partly on forum excerpts: check Apple guidelines 2.4.5 and 2.5.2 before
  relying on it.
- Optional, not needed: a heartbeat or "waking" SSE event while `/chat` holds during a wake.
  The app handles it with a longer grace period.
