# Changelog

## [Unreleased]

### Added
- **Coding tools** (19 new Ollama tools): filesystem CRUD (`read_file`, `write_file`, `list_files`, `search_files`, `move_file`, `delete_file`), shell execution (`run_shell`), and 13 GitHub tools covering repos, files, branches, issues, PRs, and code search.
- `workspace.py`: sandbox enforcement — all filesystem and shell paths are resolved against `WORKSPACE_ROOT`; traversal escapes are blocked at the path level.
- Destructive operation guards: `delete_file`, `github_delete_file`, `github_delete_branch` require `confirm=true`; `run_shell` requires `force=true` for `rm -rf`, `git push --force`, `git reset --hard`, `sudo`, and similar patterns.
- `tool_start` / `tool_done` SSE events with human-readable labels; rendered as status chips in the web UI.
- GitHub auth via `gh` CLI keyring (`gh auth token`) — no stored tokens in config.
- `httpx` added as dependency for GitHub REST calls.
- **Test suite**: 102 tests total across `test_workspace.py`, `test_fs_shell_tools.py`, `test_github_tools.py`, `test_coding_tools.py`. All GitHub and HTTP calls are mocked; no Ollama or network access required.

### Fixed
- `server.py`: bind to `0.0.0.0` instead of `127.0.0.1` so iOS/remote clients on the same network can reach the server.
