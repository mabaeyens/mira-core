# Mira Core — Collaboration Notes

Reference for Claude Code sessions working on mira-core. See `MIRA_WORKFLOW.md` (root) for the complete workflow guide and practices.

---

## Session Checklist

Before starting work on mira-core:

- [ ] Have you written a 5-bullet spec? (Do this outside Claude Code)
- [ ] Have you checked the sibling app (`mira-apps`) for patterns? (grep for NavigationSplitView, @Observable, etc.)
- [ ] Is this a follow-up to recent work? If yes, skip `/mira-status`.

---

## Key Patterns

### Entry point

`OllamaSearchApp.swift` — Contains `@main`, `startReconnect`, `autoConnect`.

### API client

`APIClient.swift` — Probe logic, startup status check, connection recovery. Check here for the latest timeout and retry patterns.

### Server communication

Connection resilience is critical. Key files:
- `APIClient.startupStatus()` — quick optimistic probe
- `APIClient.probe()` — full connectivity check
- `OllamaSearchApp.startReconnect()` — banner logic

**Pattern:** Quick probe first, then show banner only if probe fails. This prevents spurious reconnection attempts.

### Dependencies

- FastAPI backend (`uv run python server.py`)
- mlx-lm (local, port 8080) — `gemma-4-26b-a4b-it-4bit`
- sentence-transformers — `nomic-ai/nomic-embed-text-v1.5` (RAG embeddings, local, no server)
- ChromaDB (bundled with backend)

### Testing

No unit test framework. Validation is manual:
- Run `python server.py` locally
- Connect via mira-apps
- Test the specific feature you changed

---

## Validation Workflow

Before any release:

1. Run `/mira-validate` — builds for simulator and sideloads to device
2. Manual smoke check (2 minutes): app launches, send a message, verify the specific change works
3. Check icon assets (if any visual changes): no JPEG compression artifacts at crop boundaries

See `MIRA_WORKFLOW.md` section 5 for full details.

---

## Release Cadence

**Target:** One release per week (Friday or Monday).

**Why one per week?** Each TestFlight build notifies testers. Too many builds confuse the testing story.

**Process:**
1. Ensure all changes are committed and pushed
2. Run `/mira-validate` and manual smoke check
3. Run `/mira-release` to bump version, archive, and upload
4. Done. Next release is a week away.

See `MIRA_WORKFLOW.md` section 7 for full details.

---

## Bug Tracking

Open bugs are tracked in the root `BUGS.md` file (if it exists) or in the "Known bugs" section of `BACKLOG.md` (optional).

Completed work lives in `git log`. Don't archive completed bugs into a Done section — once fixed, they're deleted after 90 days.

---

## Monthly Security Audit

Last weekend of each month, run `/security-review` on this repo. Fix HIGH and MEDIUM issues before the next release.

**What to audit:**
- Input validation (command injection, path traversal)
- Shell sandbox escapes (subprocess calls must not be injectable)
- API authentication gaps
- Dependency updates (any CVEs in requirements.txt?)

---

## Token Efficiency Tips

- Compact at 50% context, not 95% (use `/compact`)
- Write a 5-bullet spec before opening Claude Code
- One deliverable per session — note scope creep for next time
- Use `grep` before `Read` to find the exact section

See `MIRA_WORKFLOW.md` section 1 for full details.

---

## Git and GitHub

- Always `git pull origin main` before any commit or push
- Commits should be coherent (one feature/fix per commit, not trial-and-error reversals)
- Use `git rebase -i` to squash commits before pushing if you have many small changes
- Release commits are tagged with version: `git tag v0.1.25 && git push origin v0.1.25`
