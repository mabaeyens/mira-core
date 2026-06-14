# Backlog

See [CHANGELOG.md](CHANGELOG.md) for recent changes.

## Pending

- [ ] Unsloth UD-MLX-4bit bench — `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit` (15 GB) cached locally; bench vs uniform 4-bit
- [ ] Scanned PDF OCR — detect scanned PDFs (empty text layer), run OCR (e.g. `tesseract`) before indexing
- [ ] Server-side auth token check — add `verify_token` FastAPI dependency to `/chat`; reads `MIRA_TOKEN` env var; no-op if unset. Client already sends `Bearer` token. ~15 lines in `server.py`
- [ ] HTTPS on LAN — self-signed CA on startup, `.mobileconfig` endpoint, QR code sheet in mira-apps connection settings (Tailscale HTTPS already works; this covers direct LAN only)
- [ ] `server.py` startup `pkill -f "python.*server\.py"` (~line 763) is greedy — it kills *every* server.py process, so a second instance can't run and a fresh install on a machine already running Mira would kill production. Consider matching the venv path or using a PID file. (Found while testing fresh install — blocked validating test server boot.)
- [ ] **Test rot (found 2026-06-14 during arch-hardening):** `tests/test_fs_shell_tools.py` — 4 `run_shell` tests fail on result-shape drift (KeyError on `exit_code`/stderr keys, force/timeout assertions). Unrelated to the hardening phases; verify against current `core/shell_tools.run_shell` contract.
- [ ] **Test misfile (found 2026-06-14):** 5 `test_browse_*` tests live inside `tests/test_cancel.py` and fail because they browse `/tmp` / `tmp_path`, which the home-only `_safe_path` guard now (correctly) 403s. Move to their own file and point them at paths under `$HOME`, or stub `Path.home()`.

## Notes

- 2026-06-14 — **Versioning is tag-driven** (hatch-vcs): the git tag is the single source of truth; never hardcode a version in `pyproject.toml`. Default to conservative **patch (z) bumps**; minor/major only when explicitly requested; v1.0 is a deliberate milestone. **Tags mark releases, not commits** — most commits don't merit a bump or CHANGELOG entry. The `/core-release`, `/mira-release`, and `/vera-ship` skills are aligned to this. iOS apps differ: version lives in `project.pbxproj` and is tagged *after* shipping (not VCS-derived). See `docs/packaging.md`.
- 2026-06-14 — Fresh install validated end-to-end in an isolated `/tmp` env with production left intact. Uninstall the uv tool by **package** name (`uv tool uninstall mira-core`), not `mira`.
