# Backlog

## Done
- [2026-07-06] Released v0.9.1 (docs-only: inference tuning results write-up); tag + GitHub release published.
- [2026-07-06] Stopped attaching a built wheel to GitHub releases — no install path consumes it (users install via `git clone` + `install.sh`/`setup.sh` + `uv`, not `pip install`). Removed the wheel asset from v0.9.1 and updated the `core-release` skill (`~/.claude/skills/core-release/SKILL.md`) to drop the `uv build --wheel` / `gh release upload` step.
- [2026-07-06] Designed and adversarially reviewed a "fully automated installer" concept (no manual GUI steps). Decision: keep status quo — the one manual step (installing `oMLX.app` and loading `Qwen3.6-35B-A3B` via its in-app model library) is an Apple Gatekeeper/TCC security boundary, not a scripting gap. Rejected scripting around it (quarantine-stripping, driving permission dialogs, reverse-engineering oMLX's undocumented model-store format) as a security-review-failing pattern. Full report: `~/.claude/plans/partitioned-tinkering-deer.md`.

## Pending
- Ease-of-install follow-ups considered but not started (all optional, low priority given "keep status quo" decision):
  - Retry/resume logic on `ollama pull` in `scripts/setup.sh` (reuse the 3×/10s pattern already in `scripts/prefetch_models.py`).
  - Auto-open the oMLX GitHub Releases page + `/Applications` in Finder during setup, to reduce the manual step to "open the .dmg, drag the icon."
  - A separate, explicitly opt-in **headless/non-interactive install mode** (e.g. `--headless`) using the `dflash-mlx` backend (fully HF-scriptable, no GUI) for automated/remote/CI provisioning only — NOT a change to the interactive default backend, since `dflash` has ~48s TTFT vs. `omlx`'s near-0ms warm TTFT. Would need: port-8080 conflict check between backends, `mira_cli.py` `COMPONENTS`/preflight disk-math update, `mira.yaml.example` default swap for that mode only.

## Notes
- `pyproject.toml` stays tag-driven via `hatch-vcs` — the git tag IS the version, never hand-edit. `mira.yaml` is gitignored runtime config, not tracked (confirmed 2026-07-06, no leaked secrets — a local `mira.yaml` with a real token exists only on disk, never committed).
- Installer already automates nearly everything (uv, Python deps incl. mlx stack, disk/RAM preflight, mira.yaml bootstrap, optional ollama/tesseract/LaunchAgent via brew, doctor health check). The oMLX GUI step is the sole exception and is expected to stay manual indefinitely absent an oMLX-side scriptable install/model API.
