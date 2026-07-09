# Benchmark Results — 2026-07-09

Hardware: MacBook Pro M5 32GB. Model: `mlx-community/Ministral-3-14B-Instruct-2512-4bit`.

## UPDATE (later same day): root cause found and fixed — GO

The prompt-cache miss below was **not** a `PromptTrie`/`LRUPromptCache`/`BatchGenerator` bug.
Two independent isolated repros (model-free `PromptTrie` calls, and a real `BatchGenerator` run
at both ~550 and ~9.5K tokens) both showed correct cache reuse — the library mechanism is sound.

Root cause, confirmed with exact measurements against the live server: a single ~21.5K-token KV
cache entry for this model is **~3.3GB**. `mira_mlx_server.py`'s `--prompt-cache-max-bytes` was
set to exactly **3.0GB** (`3 * 1024**3`, in both `core/inference/mira_mlx_server.py`'s argparse
default and `core/backend_manager.py`'s `start_mira_mlx()` launch flags). `LRUPromptCache.insert_cache()`
inserts the entry, then its own `while self._n_bytes > self.max_bytes: pop()` eviction loop runs —
since this one entry alone exceeds the cap, it evicts the entry it just inserted, every time,
leaving the trie root empty (`root_keys={}`) immediately after `insert_cache` returns. Confirmed
via direct instrumentation: `id(prompt_cache)`/`id(trie)` stable across calls (same object, no
re-creation bug), byte-identical token prefix between turn 1 and turn 2 (`first_diff_index=None`
across the full 21,542-token overlap, ruling out any tokenization/message-content drift), and a
logged `entry_bytes=3529441280 (3.29 GB)` against `max_bytes=3221225472 (3.00 GB)`.

**Fix:** raised the default to 12GB in both places (`mira_mlx_server.py`'s argparse default and
`backend_manager.py`'s `start_mira_mlx()` call), with a comment recording the measured entry size.
Verified end to end against the live server (Mira's real orchestrator → mira-mlx, not a synthetic
harness): turn 2 of the same Q10-shaped conversation (server.py injection + short follow-up)
dropped from **~62-69s to 5.4s** once the fix was in place and the process had actually picked up
the new default (a live `/models/switch` call alone does *not* reload the module — the Mira app
server process needed a real restart, since Python had the old `backend_manager.py` bytecode
already loaded in memory).

`mira-mlx` is now a real GO candidate for Ministral 3 14B — it fixes the actual bug that blocked
it. The persistent default in `mira.yaml` stays on `mlx-lm`/stock `mlx_lm.server` for now (set
2026-07-09 per explicit user request, "must use it for a while"); switching the default to
`mira-mlx` is a follow-up decision, not made as part of this fix.

## Phase 2 go/no-go: owned `core/inference/mira_mlx_server.py`

**NO-GO — real bug found, not an infinite hang.** Q1–Q13 all passed cleanly, matching or close
to Phase 1's `mlxlm-ministral3-14b` numbers (see `bench-results-2026-07-08.md`):

| Q | Category | mira-mlx | mlxlm (Phase 1) |
|---|---------|---|---|
| 1 | baseline | TTFT 10246ms / wall 10.3s | TTFT 9914ms / wall 10.0s |
| 6 | agentic-single-tool | wall 12.8s, task_done YES | wall 12.9s, task_done YES |
| 7 | agentic-multi-step | wall 18.6s, task_done YES | wall 16.9s, task_done YES |
| 13 | agentic-divergence-guard | wall 22.1s, divergence_guard YES | wall 32.9s, divergence_guard YES |
| 10 turn 1 | inject 40,198 chars | 71-73s (consistent across runs) | 72.4s |
| 10 turn 2 | retrieval | **367.6s** — 5x slower than turn 1, no cache reuse | 71.9s (passed) |

**Root cause, confirmed via targeted logging (not speculation):** `mira_mlx_server`'s prompt-cache
reuse is completely broken — every turn is a full cache **miss**, forcing a full ~21.6K-token
reprocess of the entire growing conversation on every single request, instead of reusing the
prior turn's KV cache. Traced precisely:
- `LRUPromptCache.insert_cache(model, all_tokens, prompt_cache)` runs without error after every
  turn (confirmed via logging: correct token count, correct model key registered in the trie's
  outer dict).
- Yet `LRUPromptCache.fetch_nearest_cache(model, prompt_tokens)` on the *very next* request
  returns a complete miss (`common_prefix=0`) — even though the first ~21,400 tokens of turn 2's
  prompt are provably byte-identical to what turn 1 just registered (same ints, same values,
  confirmed via logging both sides).
- Direct inspection (`PromptTrie._trie[model]`, immediately after the successful `insert_cache`
  call, in the same synchronous statement) shows the model's per-token trie root is **empty** —
  no token-level nodes were built at all, despite the outer model-level key existing. A hand-written
  replica of `PromptTrie.add()`'s exact algorithm, run inline against the identical
  `r.all_tokens` list, populates correctly. The real library call does not, on the same data, in
  the same call. The precise mechanism inside `mlx_lm.models.cache.PromptTrie`/`LRUPromptCache`
  is not yet understood — this looks like it could be a genuine upstream bug specific to
  real `BatchGenerator`-produced cache/token objects (as opposed to a synthetic in-memory test),
  worth reporting upstream if pursued further.
- **Not an infinite hang**: letting turn 2 run to completion (not killing it early) confirmed it
  finishes in 367.6s — a real, bounded (if severe) slowdown from doing a full reprocess, not a
  deadlock. The original "7+ min then killed" report from the first bench run was a false
  positive caused by killing the process just before it would have finished on its own.

**Decision:** `mlxlm-ministral3-14b` (Phase 1, stock `mlx_lm.server`) remains the active default
for Ministral 3 14B — set per explicit user request (2026-07-09) since it's needed for regular
use, and it doesn't have this bug. The owned `mira-mlx` preset stays in `mira.yaml`/the codebase
for future debugging but is not recommended for use — it would make every turn in a growing
conversation progressively more expensive (full reprocess of the entire history every time),
which becomes impractical well before hitting any hard limit.

**Next steps if resumed:** get an actual Python stack/state inspection tool working (`py-spy`
needs root on macOS — no non-interactive sudo was available this session; consider `remote_pdb`,
a debug endpoint, or running as an admin-privileged session) to step through
`PromptTrie.add()`/`.search()` live against real request data, since printf-debugging got close
but couldn't explain why an identical hand-copy of the same algorithm succeeds where the library
call doesn't. Also worth filing as a question against `ml-explore/mlx-lm` if the cause turns out
to be a genuine upstream issue rather than a mira-mlx integration bug.

(The raw Q10-only debug run that produced the 367.6s number above — `ministral3-14b-debug12` — is
folded into the table; its auto-generated bench section has been removed here as redundant.)