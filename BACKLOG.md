# Backlog

## Done
- [2026-04-25] github_clone_repo tool — clones a GitHub repo via `gh repo clone`, auto-creates a Mira project in the DB (local_path + github_repo set), returns project_id; `_clone_and_register` helper in orchestrator handles the DB write post-clone

## Done
- [2026-04-25] Step 3 — app project picker: Projects sidebar section, project rows (tap→new scoped chat, context-menu delete), AddProjectSheet (name+path+repo), conversation rows show project badge, loadProjects on startup
- [2026-04-25] Step 2 — per-conversation workspace: workspace_root flows from project.local_path through orchestrator; fs/shell tools receive root per-call; tool list filtered when no local workspace; system prompt adapts to project context
- [2026-04-25] Step 1 — projects DB + API: projects table, project_id FK on conversations, /projects CRUD endpoints, POST /conversations accepts project_id
- [2026-04-25] edit_file(path, old_str, new_str) — targeted string-replace patch tool; rejects if old_str matches zero or >1 locations; prefer over write_file for existing files
- [2026-04-25] github_create_pr(repo, title, body, head, base) — opens a PR; base defaults to repo default branch
- [2026-04-25] github_merge_pr(repo, pr_number, merge_method) — merges PR with confirmation gate; supports merge/squash/rebase
- [2026-04-25] MAX_TOOL_STEPS 5 → 10 — end-to-end coding flows (read→edit→commit→PR→merge) no longer hit the ceiling
- [2026-04-25] /health returns 503 during Ollama warm-up — added _ollama_ready flag; 503+{"status":"starting"} while model loads, 200+{"status":"ok"} once ready; macOS native app splash stays visible until model is warmed
- [2026-04-25] shell sandbox: run_shell rejects absolute paths outside WORKSPACE_ROOT — closes escape where cwd was sandboxed but command args (ls /, cat /etc/passwd) were not
- [2026-04-25] prompt Rule 1: model answers capability questions from system prompt without calling tools
- [2026-04-25] Fixed Mac app "couldn't read data" error: LaunchAgent was still running from old `ollama-web-search/` path after refactor — reloaded with `mira-core` plist
- [2026-04-25] Cleaned up stale db files at repo root and backup after confirming all conversations intact
- [2026-04-25] Fixed missing conversations: after `core/` refactor, `DB_PATH` resolves to `core/conversations.db` — merged old root-level db (12 conversations) into new location
- [2026-04-25] Updated `mira-server` skill plist source path from `ollama-web-search` to `mira-core`
- [2026-04-25] Refactored all core modules into `core/` package (Option A architecture)
- [2026-04-25] Updated plist paths to `mira-core`; added `AssociatedBundleIdentifiers` to plist

## Pending

### Harness quality
- [ ] Parallel tool execution — orchestrator runs only the first tool call per step; if the model emits multiple tool calls in one response, the rest are dropped; execute all in parallel and merge results before the next turn
- [ ] Shell timeout 30s → configurable per-call — long builds and test suites time out; add optional `timeout` arg to `run_shell` (cap at e.g. 300s)


## Notes
- Projects have three modes: local-only (local_path only), GitHub-only (github_repo only), or both. Tool availability depends on local_path — no local path means fs/shell tools are hidden from the model entirely
- workspace_root flows per-call through all fs_tools and shell_tools via a `root` parameter; shell sandbox pattern rebuilt per-call from the active root (not cached at import time)
- `_LOCAL_TOOLS` set in tools.py drives filtering in orchestrator._active_tools — add new local-only tools there
- `DB_PATH = Path(__file__).parent / "conversations.db"` in `core/config.py` means the database lives at `mira-core/core/conversations.db`, not the repo root — be aware when backing up
- The `mira-server` skill copies plist from `~/Documents/Projects/mira-core/com.mab.mira.plist`; reload is required any time the plist or server paths change
- Mac app bundle ID remains `com.mab.OllamaSearch` (no rebuild needed for Python-side refactors)
- _ollama_ready is set True after the warm-up loop regardless of success/failure — app never hangs indefinitely if Ollama is permanently down; it just gets a working server with a broken first chat rather than a stuck splash
