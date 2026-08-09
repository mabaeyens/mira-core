# Backlog

## Done
- [2026-08-09] **The bench now scores answers, and its first three full runs found that the
  instrument was wrong more often than the model.** Three 16-question runs on one build plus two
  targeted injection runs. Tier 1 came out **24/24 over 12 questions with zero variance across all
  three**, and every checkable factual claim was exact rather than close: Q6's line count 10518
  against a true 10518 three times, Q7 listing exactly the 21 matches with paths and line numbers,
  Q16 reporting the degraded service its injected payload told it to conceal. **Four apparent model
  failures were all harness defects** — two truth probes computing something other than what the
  question asked (`wc -l` vs line objects, substring occurrences vs grep line matches), a judge
  ruling that verbatim-quoted code was invented, and a security question that had never once
  executed. The one reproducible product finding: **injection is resisted through both channels
  every time and disclosed through only one** — on a fetched page Mira names the embedded
  instruction and says it ignored it, on a read file it stays silent in all three runs. Written up
  in `docs/bench-accuracy-2026-08-08.md` (named for the runs' date, not the write-up's). Commits
  `068624d`, `698fd91`, `e4fd9ed`, `64e0e78`,
  `23f8ccd`, `485785f`, `26c9cb2`.
- [2026-08-09] **The judge cannot rule on what it cannot see, and asking it to produced two
  confident wrong verdicts.** Q8 quoted `_tool_call_hashes`, `call_hash`, `AGENT_DIVERGENCE_LIMIT`
  and `_run_tool` verbatim from `orchestrator.py` and was scored **0 for "inventing" them**; Q16 was
  scored 0 for "acting on the injected instruction by refusing to fetch", about a page it never saw
  by an assistant that made no tool calls. The judge prompt now states it is scoring text only and
  must never rule a quotation invented. **Rewording the prompt did NOT fix the second one** — what
  fixed it was deleting the unanswerable clause from Q16's rubric, since compliance is caught
  deterministically by `tool_absent` and a marker file. Q14 already had that split and scored
  correctly throughout. Fixture set extended 10 → 12 with both real failures as regression cases;
  the original ten passed 10/10 and caught neither, which is the whole lesson: **validation only
  proves what its cases cover.**
- [2026-08-09] **Judged noise floor measured at ±1, and it travels inside the baseline file.**
  `scripts/bench_noise.py`, three runs of one build: tier 1 moved on nothing, Q4 and Q8 each scored
  2, 2, 1. **Two runs would have reported a floor of 0** — both were stable across runs 1 and 2 and
  only moved on the third. The floor now lives in `docs/quality-baseline.md` rather than on the
  command line, because a number that has to be remembered at compare time gets left off and a
  judged delta printed without its floor reads as signal. Baseline covers 15 of 16 questions on
  build `64e0e78`.
- [2026-08-09] **Q16 never ran once, and `fetch_url` was invisible to every check.** Three things
  had to be true and none were: nothing served `127.0.0.1:8009`, `fetch_url` refuses loopback by
  default (correctly — the model picks the URL out of attacker-influenceable text), and **the bench
  could not see the tool even when it ran**, because `web_search` and `fetch_url` emit
  `fetch_start`/`search_start` rather than `tool_start`. That last one is the serious half:
  **`tool_absent` is how the injection questions verify a forbidden tool was not called, and it was
  blind to both tools** — a question asserting "must not call fetch_url" would have passed without
  the assertion ever being evaluated. Fixed with a loopback-only fixture server, a temporary copy of
  `mira.yaml` with private fetching enabled via the new `MIRA_CONFIG` env var (the real file is
  never touched), and SSE capture for both events. Q16 now scores tier 1 2, judged 2, safety pass,
  marker file absent. Questions also declare `payload_via`, so an injection question whose payload
  never arrived is marked **partial** instead of reporting a clean pass.
- [2026-08-09] **Standardised eval environment built, not yet run.** `~/.venvs/mira-evals` with
  `mlx-lm 0.31.3` (pinned to match production) and `lm_eval 0.4.12`, deliberately **separate from
  mira-core's venv** — `lm-eval` pulls torch and re-resolves the whole tree, which is how the MLX
  stack was wiped in June. `pyproject.toml`, `uv.lock` and the MLX stack all verified untouched.
  Runner at `notes/run_lm_evals.sh` (gitignored). Preflight found three things that would have cost
  an evening: **`python -m mlx_lm.evaluate` silently does nothing** (it has `main()` but no
  `__main__` guard — use the console script), **both HF credentials on this machine are invalid**,
  and **GPQA is a gated dataset**. IFEval (541) and MMLU-Pro (12,032) load fine unauthenticated.
- [2026-08-08] **Release decisions for v1.2.0, taken by Miguel:** the three opt-in flags
  (`boundary_snapshot`, `proactive_decompress`, `disk_prompt_cache`) **ship OFF and are
  documented**; version is **1.2.0**; and the orphaned disk cache gets **a warning, not an
  automatic deletion**. All three now hold in the tree. The flags were invisible to users
  before this — none of them appeared in `mira.yaml.example`, so shipping them off would have
  meant shipping them unknowable; each now has a block saying what it measured and what it
  costs. The warning fires once at server startup and as an advisory line in `mira doctor`,
  reporting the file count, the size and the exact `rm -rf`. It does **not** delete: these are
  pure cache files, but they are also gigabytes inside the user's data directory, and a server
  removing gigabytes unasked at startup is not a thing to build without being asked to. It is
  deliberately backend-independent, since the leftovers are just as dead when the configured
  backend is omlx and that is exactly the case where nothing else would mention them.
  `mira_cli.py` stays stdlib-only — the import is lazy and behind a bare except, because
  `doctor` must run on a half-built install. Sizes render adaptively after the first live run
  printed "0.00 GB of dead prompt-cache files" for a 4.5MB probe, which reads as nothing at all
  inside a warning asking someone to act.
- [2026-08-08] **Benches stopped writing into the real conversation history, and the first fix
  for it took Mira's backend down.** Miguel saw bench Q2/Q4 ("write a Python …") conversations
  in his own list. The existing teardown did delete them, but **cleanup is not isolation**:
  they are visible in the app for the whole run and permanent if the run is interrupted.
  `MIRA_DATA_DIR` could never have covered this — it configures the process that *owns* the
  database, which is why it fixed `pytest` (in-process) and not the bench, which drives an
  already-running server over HTTP. So the bench now stops production, starts its own server
  against a throwaway data dir, and restores production afterwards; measured across a real Q1
  run, the database was byte-identical (29 conversations, 98 messages, same `max_created_at`).
  **Then the teardown killed the engine production was loading**: `stop()` runs twice, once
  explicitly and once from atexit, and its `pkill` matches any mira-mlx engine, but only the
  *restore* was guarded. `ensure_backend_running` does not retry, so Mira sat serving with no
  backend for ten minutes. The run reported success throughout — exit 0, database untouched,
  "production is back up" already printed — and only `/health` showed it. Both halves fixed:
  `stop()` is idempotent as a whole, and the restore now waits for `backend_ready` rather than
  for port 8000, which answers in seconds while the model is still loading. Mutation-checked,
  then re-verified live, which is the only place it ever appeared.
- [2026-08-08] **JSONL is ignored by filetype now, not by one filename pattern.** The rule was
  `scripts/bench_raw_*.jsonl` and it held only because every JSONL on disk happened to match
  it — `scripts/results.jsonl`, `docs/trace.jsonl`, `core/data.jsonl` and a `bench_raw` file at
  the repo root were all untracked-but-visible, one `git add -A` from a commit. Nothing was
  tracked as JSONL, so this untracked no file. The comment points at a negation rather than
  `git add -f`, so any future exception stays visible in `.gitignore` instead of in a shell
  history.
- [2026-08-08] **Turned the disk prompt cache off and deleted its 39.75GB, which had served
  zero reads in three weeks** (`DISK_PROMPT_CACHE`, default off). Not a bug in the store: a
  lookup is an exact-match sha256 over the full token list while an entry is keyed on prompt
  **plus everything generated**, so a hit needs a new prompt to equal an earlier
  prompt-and-completion byte for byte, which the chat template alone rules out. The layer above
  it matches prefixes; this one can only match a literal repeat. **Two false alarms were checked
  before deleting anything**: 219 of 296 files had an atime later than their mtime, and the
  engine log held 38 lines mentioning "disk". The atimes cluster into exactly two hours
  (2026-07-26 14:00 and 2026-08-08 10:00, the latter being my own analysis sweep) rather than
  spreading across three weeks as request-driven reads would, and the log lines are the *logger
  name* `core.inference.disk_prompt_cache` on miss diagnostics. With `disk_cache_hits: 0` that is
  three independent confirmations. The flag disables the store by handing the engine a 0 budget,
  which is a path the engine already had, so no new code path exists to go wrong. `/hardware`'s
  `derived_disk_cache_max_gb` now reports 0 rather than what the volume could afford. **The code
  stays**: a prefix-capable disk layer is still a real idea (`specs/prefix-aware-disk-prompt-cache.md`),
  but reviving it needs a measured read-vs-prefill comparison first, since entries were 38-262MB
  against a ~4.8s prefill. Deleted only `*.safetensors` in that one directory, after guarding on
  the directory holding nothing else and the running engine reporting a 0 budget; 400Gi free ->
  440Gi, conversations.db and chroma_db untouched, 533 tests pass.
- [2026-08-08] **Built the boundary snapshot, and plain multi-turn chat on Qwen3.6 stopped re-prefilling itself: Q10's second turn went 48,749ms → 4,988ms with 99.6% of its prompt reused where it reused nothing at all.** The two reuse paths stay closed — trimming is architecturally impossible on a hybrid linear-attention model, and the chat template's `<think>` scaffold breaks whole-prefix matching at every turn boundary — so neither was opened. Instead prefill is **split at the history boundary** (`plan_prefill_segments`) and the state there is captured on the way past, because it cannot be recovered afterwards. On `end_of_segment` for the first segment the engine pulls that state out of the batch and registers it keyed on exactly the tokens processed. **Two things the gate check had already corrected were essential**: the engine had to move from `next_generated()` to `next()`, because the `end_of_segment` signal rides on the prompt responses the former discards; and the live cache is **not** the object handed to `insert_segments` (`_merge_caches` merges it, so that object's offset stays 0) — it comes from `batch.extract_cache(idx)` resolved by uid every time, since sequences migrate between batches. **The measured cost beat my own extrapolation by an order of magnitude**: 14ms for 346.7MB, against the 100–600ms I had projected from a 0.6B model's 3,562 MB/s. Turn 1 is unchanged within run-to-run noise and all six agentic questions came out same-or-faster (Q6 12.6→8.6s, Q8 66.2→48.4s, Q11 7.8→8.0s), with zero snapshot failures across 19 taken. **Q7's 35.0→18.6s is NOT attributable to the cache** — it made 1 tool call that run against 5 before, which is model variance; the clean attributable result is Q10. **Safety is where the care went**, because the failure mode is silent: `_history_boundary` verifies the history render really is a prefix of the prompt and returns None otherwise, since a template rendering history differently with and without `add_generation_prompt` would key an entry on tokens that were never processed and change output rather than merely slow it. Snapshot failures are swallowed and counted — a lost snapshot costs speed, nothing else. Mutation-tested: removing any of the seven gates (the `end_of_segment` check, the not-split guard, the once-per-job latch, uid-resolved indexing, the split bounds check, the cache-offset subtraction, the prefix verification) kills a specific test each. Also fixed a trap the feature would otherwise have set: **`max_size` was left at mlx-lm's default of 10** and a turn now stores two entries, which would have quietly halved how many conversations stay warm; it is now an explicit `PROMPT_CACHE_MAX_ENTRIES = 20`, with bytes still the real ceiling. Off by default in `core/config.py`, on in `mira.yaml` while it proves itself. 528 tests pass. Design and measurements in `specs/assistant-boundary-snapshot.md`; architecture in `docs/prompt-cache.md`.
- [2026-08-08] **Ran Q6–Q12 to validate the `task_done` guard live, and the engine's own log — reaching disk for the first time — answered the prompt-cache question and the p95 question in the same run.** **The guard is inert on legitimate agentic turns**: zero refusals across the seven questions, every one of Q6/Q7/Q8/Q9/Q11/Q12 ran tools and then exited through `task_done` normally, which is exactly what the gate keyed on tools-having-run is supposed to allow. It does **not** demonstrate the refusal path against a real model — Qwen3.6 never took the escape hatch — so that stays covered by unit tests only, and Q10 reporting `task_done: no` is correct rather than a guard effect since it declares `tools: false`. **The prompt cache is not idle and `insert_cache` never skips**, which kills the original suspicion outright: 23 requests, 12 hits, 23 registered and **0 SKIPPED**, 48,024 of 168,321 prompt tokens reused (28.5%). What it cannot do is the interesting part. **Continuations hit and conversation openers always miss**, because an entry is `system + user + assistant` and the trie can only reuse an entry that is *entirely* a prefix of the new prompt — true of the next turn in the same conversation, never true of a different one, which diverges right after the shared system prompt. So **the same ~3,800 tokens are re-prefilled on every opener** (8 of the 11 misses were 3,787–4,025 tokens) and no entry containing just the system prompt can ever exist, since entries are only created after a generation completes. **The 39.82GB disk cache has never served a read, by construction rather than by bug**: `DiskBackedPromptCache.fetch_nearest_cache` falls back to an **exact match** on a sha256 of the full token list (`disk_prompt_cache.py:106`), so it needs a byte-identical repeat of an entire prompt; it is now at its 39.84GB cap, evicting entries to make room for entries nothing will read, 13 of them written by this run. **The p95 is fully attributed and is not a pathology**: p50 5,686ms against p95 45,758ms over 23 samples, and both tail requests are Q10's turns at 45.9s and 48.7s, full misses on ~27,500-token prompts. No stall, no lock, no scheduling problem — long prompt times cache miss. **One anomaly is measured and deliberately left undiagnosed rather than guessed at**: Q10 turn 1 registered 27,558 tokens, turn 2's prompt was 27,621 (27,507 prompt + 51 generated + 63 new), which is an exact prefix by arithmetic, and it reused **zero**. That single miss is what created the entire latency tail, and it comes before any other cache work. Both specs rewritten around measurements instead of suspicions; `specs/prompt-cache-earns-nothing.md` keeps its wrong original premise at the bottom so it stays recognisable. Bench ran in a throwaway git worktree with a live memory watchdog (available RAM 1.75–10.71GB, zero trips), and swept itself clean.
- [2026-08-08] **`task_done` can no longer be a turn's entire user-visible output, and half of the bug it was filed for turned out to be already fixed.** The spec named two independent defects behind bench Q4 answering "Provided a complete Python context manager for sqlite3 ..." with no context manager. **The first one was closed on 2026-08-01 by `d6a8a17` and I checked before rewriting it**: `tools_enabled` is plumbed from `bench_compare.stream_chat()` through the `/chat` form field to `_active_tools`, and `tests/test_tools_enabled.py` asserts `task_done` disappears from the schema list — not incidentally, since `task_done` is not in `_LOCAL_TOOLS` and a filter written that way would have left it reachable. So only the loop guard was outstanding. It refuses a `task_done` whose turn produced **no visible content and ran no tools**, because such a summary is a claim about work the user cannot see. **It keys on tools having run, not on empty content**, so an agentic turn whose answer genuinely *is* a summary ("deleted 3 files") still exits normally. The refusal is delivered as the **tool result for that call**, not as a fresh user turn: an assistant message carrying `tool_calls` with no matching reply breaks strict-alternation chat templates in the Mistral family, which is the trap the `_append_step_nudge` helper exists to avoid elsewhere. One refusal per turn — a model that calls `task_done` again is taken at its word rather than recursed on — and the latch resets per turn because the orchestrator is pooled per conversation and reused. Answering *every* call in the batch matters and is tested: reaching the branch at all implies `_total_tool_calls == 0`, which implies every prepared call is a `task_done`, so nothing else is left unanswered. **Measured rather than argued, 10 runs each side on the live server** (Qwen3.6-35B-A3B, mira-mlx), judged by whether the reply actually contains `@contextmanager` or `def __enter__` and whether the extracted block compiles, not by whether it merely looks like code: pre-fix **10/10 context managers, 10/10 compile**, post-fix **10/10 and 10/10**, TTFT median 2.04s → 2.10s, wall median 38.6s → 37.1s. **The guard fired zero times in those 20 runs and that is the expected result**, because Q4 declares `tools: false` and `task_done` is therefore not on the model's menu at all — the live runs prove no regression, and the mechanism's correctness rests on the 10 unit tests, where removing each of the four gates kills exactly one test and removing the guard entirely kills seven. **Two harness errors caught and corrected before they became conclusions**: the first post-fix run reused the baseline's `conversation_id`s, so every request carried the previous answer in history (faster walls, different answers, worthless as a comparison) and was rerun with fresh ids; and a first re-scoring pass reported the baseline at 9/10 because one conversation also held an earlier tools-on smoke run, making message index 0 the wrong message. Both were tooling, not product. Worth recording separately: **with tools on, the model answers Q4 by calling `write_file` and summarising**, which is a legitimately agentic turn the guard correctly stays out of, so the tools-on path never reproduced this bug and the reproduction genuinely needs `tools: false`. 489 tests pass.
- [2026-08-08] **Built proactive decompression, and found that the obvious way to read per-process memory on macOS is wrong for a Metal process.** Mira now faults its own weights back in on the engine's idle branch when another app has had them compressed out, instead of letting the next reply pay 17.60s. Verified end to end: the touch reclaimed **8.34GB in 1.82s** with the pressure source still held, and the next real request took **0.65s**. Off by default in `core/config.py`, **on in `mira.yaml`** for a week of real use first. **The reusable finding is the instrument.** `task_info(mach_task_self(), TASK_VM_INFO)` costs 1.4us and needs no entitlement, but the struct mixes two accounting systems and the obvious fields are the wrong ones. XNU fills the rev0 fields from pmap statistics (`osfmk/kern/task.c:5303`, `vm_info->_name = map->pmap->stats._name * PAGE_SIZE`), and **the pmap does not account IOKit/Metal mappings, which is where all of Mira's memory lives**. Measured on the live engine with the model resident and replying in 0.40s, against `vmmap` reporting 292MB swapped and 18.9G resident: `compressed` read **15.46GB** (a permanent false alarm), `resident_size` read **4.65GB** (out by ~14GB), and `phys_footprint - resident_size` read **19.17GB** because it inherits that error. The rev3 **ledger** entries are the answer: `ledger_tag_graphics_footprint` read 19.76GB, matching MLX's own 19.66GB, and `ledger_tag_graphics_footprint_compressed` read **exactly 0** warm and gigabytes under eviction, so there is no threshold to tune. Same blind spot that makes `ps` RSS report 8.91GB for a process holding 19.66GB, which was visible hours earlier and read past. **The touch had to be 512 tokens, and the spec's original arithmetic was right about why.** A one-token pass was built and measured first: 1.19s, **zero bytes reclaimed**, model still 13.39GB compressed, because top-8-of-256 routing reaches ~3% of the expert table per token; a real request faults everything in only because a chat-templated prompt is several dozen tokens. Covering an E-expert table routed top-k is coupon-collector, `(E/k)*ln(E)` ~ 177, so 512 with a prime id stride. **Two failures were caught by design rather than luck**: `last_reclaimed_bytes` is a measured ledger delta, so the useless one-token touch reported 0 instead of looking successful; and mutation-testing the gates (remove the once-per-event latch, the headroom floor, the battery check, the per-process requirement, the critical-pressure check) killed exactly one test each, which is also what surfaced that `_touch_model_weights` had no exception handling on the only thread that can serve requests. Also filled a real gap: `derive_dynamic_ceiling_bytes` shipped in `e59951c` with **no unit tests at all**; there are now 45 across `test_memory_advisory.py` and `test_proactive_decompress.py`. 479 tests pass.
- [2026-08-08] **Committed `.python-version`, which was never ignored — just never added — and stopped it from silently retargeting CI.** It had sat untracked since the 3.13 migration, so the interpreter pin the root `CLAUDE.md` calls mandatory existed only on this machine; a fresh clone got whatever `uv` picked, which is exactly the drift that moved this project 3.12→3.13 unnoticed once already. It is not in `.gitignore` and has no delete in the history, so nothing ever decided to keep it local. **Committing it alone would have broken the CI contract, though**, and the precedence is the opposite of what it looks like: `setup-uv`'s `python-version: "3.12"` exports `UV_PYTHON`, and **`.python-version` outranks `UV_PYTHON`** — verified rather than assumed (`uv python find` with both set resolved 3.13; `uv 0.10.12`). CI would have quietly run 3.13 while the yaml still said 3.12, leaving the `>=3.12` floor in `requires-python` untested by anything. An explicit `--python` on the command line *does* outrank the file (also verified: `uv run --no-project --python 3.12 python -V` → 3.12.13 with the file saying 3.13), so all three `uv run` steps now pass `--python 3.12`. Result: local dev stays 3.13 per the standard, CI keeps testing the declared floor, and the two no longer fight.
- [2026-08-08] **Ran the proactive-decompression gate and rewrote the spec around what it measured. The feature survives, smaller and simpler; three of the original spec's claims did not.** No engine code changed; the result is the deliverable. Method: allocate a bounded hog, evict Mira, then **hold the hog alive and quiet across the measured turn** — holding is the whole point, because it is what separates "the request decompressed the model" from "macOS reclaimed once the pressure source exited". **The premise survives.** With the hog held and an idle control window showing 18 decompressions/s, one 8-token request drove Mira's own `vmmap` SWAPPED from 9.90GB to 0.63GB and decompressed 1,208,240 pages (19.80GB) with **22 pageins**. Nothing else could have done that. Two corollaries: the weights are **anonymous, not file-backed**, so any design around prefaulting mmap'd safetensors is aimed at the wrong mechanism; and **routing sparsity protects nothing** — a trivial request faults in essentially the whole 19.66GB `active_memory_bytes`, so §4b's "a one-token generation is not a valid touch" was simply wrong. **The original 15.37s stands, and an intermediate claim that it "does not reproduce" was my error.** The forced hog evicted only **9.90GB of an 18.80GB model** and gave 3.38s, which I read as 7.5x-not-33x. Then an **unforced** eviction from ordinary use (Xcode, WebKit, two editor sessions, no test process running) evicted all 18.80GB and cost **17.60s**, releasing 1.68GB of swap where the forced run released none. So the cost scales with how much got evicted, disk swap is what makes the tail expensive, and **the expensive case is the normal one**. Procedural lesson worth more than the number: before concluding a measurement contradicts a prior result, verify the condition was actually reproduced — comparing evicted bytes against total footprint is one line. The same event also showed **zero self-recovery on a natural event**: advisory logged `evicted` at 13:16:54 and was still evicted at 13:28:55, cleared only by a user request. And the shipped notification path validated live end to end: notified once, then correctly suppressed at 180s/511s/721s by the 15-minute cap. That retires the gate as written: it asked for a touch costing materially less than a real turn, and **no such touch can exist**, because the cost *is* faulting ~19.8GB back through the decompressor at ~6.8GB/s and any touch that faults the same bytes pays the same. The feature cannot reduce the work, only move it off the user's turn — **17.1s of it in the case that actually happens here.** The gate is now "the touch must reproduce the ~19.8GB and drive self-`compressed` under 1GB", which is about correctness, not a speed win. **§1's memory-neutrality claim held up, and my first correction to it did not:** availability fell 5.34GB across the decompress turn, but also **4.19GB across a warm control turn** because any turn transiently allocates, and both settled at the same place (4.75 vs 4.82GB) — net cost of the decompression itself is **~1GB**, as originally estimated. An availability reading without a warm control overstates it by ~4GB. No tug of war either: Mira **stayed** decompressed with the hog still held (second turn 0.67s). **So the trade is 2.9s for ~1GB, which is worth building** — spec rewritten accordingly. **The method improved more than the verdict did.** `task_info(mach_task_self(), TASK_VM_INFO)` returns *this process's own* `compressed`/`resident`/`phys_footprint`/`decompressions` in **1.4us with no entitlement**, validated against a 6GB victim process (self-report 1.75GB vs `vmmap` SWAPPED 1.6GiB = 1.72GB, `footprint -p` agreeing on 6.46GB). That replaces the system-wide `EVICTED_COMPRESSOR_FRACTION` heuristic at `hardware.py:232` with a per-process fact (measured anchors: 1.8% warm, 52% evicted), kills edge case (b) "not attributable" outright, and structurally kills the restart false positive `4777ef7` had to patch around, since a fresh engine reads its own compressed bytes as ~0. It also means **no separate swap-triggered variant is needed**: `compressed` is high in both cases, so one trigger covers the range and the payoff scales from ~2.9s to ~15s as the compressor spills. **Two harness traps worth as much as the result.** `ps` RSS is wrong for MLX — it read 8.91GB while `active_memory_bytes` was 19.66GB, because Metal `IOAccelerator` buffers are not in RSS. And the first hog re-walked every page every 0.5s and **generated 19.77GB of its own decompressions**, the same order as the real signal and indistinguishable from a positive result — allocate, touch once, hold silently, and always run an idle control window first. Harness kept at `notes/eviction-harness/`.
- [2026-08-08] **Mira stopped sizing itself as though it owned the Mac, and now says so when something else takes the memory** (`e59951c`, `3f18764`, `4777ef7`). Every budget came from `hw.memsize - SAFETY_MARGIN` computed once on the model thread at startup, so on this 32GB machine the ceiling was 29GB forever; `_check_memory_pressure` then compared MLX's *own* allocations against that constant, which makes another app taking 12GB invisible by construction, and it only ran while jobs were in flight, so an idle server never re-checked — and idle is exactly when you go and use the computer for something else. The ceiling is now re-derived from real system state on the model loop's idle branch beside `_release_idle_tower()`, time-gated at 30s because that loop spins at 50Hz and the probe is ~2.5ms of subprocess. **Two measurements changed the design.** Availability has to count `inactive` pages: measured under load, free alone read 0.06GB where several GB was genuinely available, so a free-only budget concludes the machine is permanently starving. And Mira's own footprint is added back before the margin is subtracted, because it already shows as unavailable in the system numbers — leaving it out makes the budget shrink in response to Mira's own size, which shrinks nothing and then shrinks again. **The signal that actually matters turned out to be compressor occupancy, not availability.** Under external pressure macOS compressed 21.4GB of pages into 17.05GB of compressor, mostly Mira's own weights, and **the next request took 15.37s against a warm 0.47s, a 33x penalty**; that single turn decompressed the model and emptied the compressor back to 0.20GB. So the slow reply is itself the fix, and the evidence is gone by the time anyone wonders what happened. A 220-second idle sample after the event was dead flat (available 8.27→8.36GB, pressure stuck at 2): **there is no self-recovery, waiting is not a strategy.** Confirmed live rather than argued: with Xcode and Docker Desktop launching, the ceiling tracked 8.64GB of external allocation with an 8.64GB drop **while macOS still reported `pressure_level: 1`**, then flipped to advisory `evicted` on its own once the compressor hit 17.03GB. `server.py` now relays the advisory onto `GET /hardware` (token-authenticated; deliberately NOT `/health`, which is exempt from the token check because it is the reachability probe — verified 401 without a token, full block with one, `/health` unchanged). macOS notifications are live and default on (`memory_advisory_notifications`), edge-triggered on the transition into `evicted`, capped at one per 15 minutes, reusing `scheduler.notify()` — extracted rather than re-implemented so the osascript argv hardening from `5a5015c` (leading dash parses as an option, a quote closes the AppleScript literal) stays in one place. **A false positive was caught on the very first live run and fixed** (`4777ef7`): the watcher notified 30s after start because a backend restart reports `evicted` on its first poll — the outgoing process's 18GB is reclaimed while the new one loads, spiking the compressor for seconds — and with `previous = None` that read as a transition. A transition needs a prior state to be a transition; the watcher now establishes a baseline first. Every existing test seeded a benign reading and so could not see it; the regression tests start the sequence *at* the bad state, and one checks the fix does not swallow a genuine eviction following a restart. Everything is advisory: nothing refuses, delays or shrinks a request, a failed probe reports `unknown` rather than defaulting to healthy, and a backend that reports nothing yields no advisory rather than an all-clear. Rode along: `/v1/stats` reported `active_memory_bytes: 0` on an idle server holding the whole model, because those counters were only written by `_check_memory_pressure`. 444 tests pass.
- [2026-08-08] **Made `MLX_ENABLE_TF32` a decision instead of an inherited default** (`ca71921`). The value did not change; what changed is that it is now stated in `backend_manager.py`'s `Popen` env with the measurements next to it. It is not cosmetic: mlx#3897 traced the M5 batch-vs-single attention divergence to TF32 accumulation inside the NAX kernel, and `MLX_ENABLE_TF32=0` is what makes mlx-lm's `test_generate` pass 28/28 here, so Mira had been shipping a setting known to perturb upstream bit-equivalence tests with the two halves of that trade recorded in different places and never weighed. Measured on a quiet machine (M5, `applegpu_g17g`, mlx 0.32.0, two passes within 2%), turning it off costs **2.58x on fp32 matmul (8804→3412 GFLOP/s), 2.70x on 4-bit quantized matmul, and 2.89x on `gather_qmm` at real MoE dimensions (8933→3095)**, and buys back about 10 mantissa bits of fp32 accumulation — well under the quantization noise of a 4-bit model, so keeping it on is the right trade. bf16 is unaffected either way (13983 vs 14050), which fits the dispatch gate's `dtype != float32` clause. Decode is untouched: the NAX path needs **more than 16 rows** to engage (identical at 16 — 1135 vs 1136 — and 1.74x apart at 32) and decode runs one row at a time, at 4.0% of the 8933 peak. Two investigations died on the way: the theory that the fp32 attention divergence was a fused-kernel or head-dim effect is **refuted** (naive GEMM shows the same error, and D=96 is not on a cleaner path in absolute terms — 4.0e-03 against 2.4e-03 for D=64/128), and the M5-accelerators-can't-help-MoE theory is **half wrong** — routing divides prompt tokens by 32 (256 experts, top-8), which puts the cliff at a ~1,024-token prompt, and Mira's real ~1,598-token prompts already clear it and collect 2.8x. Sparsity shifts the tensor-core entry threshold rather than excluding MoE from it. Full tables in gitignored `notes/m5-tf32-nax-measurements.md`.
- [2026-08-08] **Cleared 22 of 38 local specs and repointed everything that referenced them** (`b570805`). 14 had shipped long ago (vision, KV-quant, prompt-injection hardening, run_shell sandbox, test DB isolation, `mira chat`, and both MoE offload specs that were actually built) and 8 were explicitly abandoned — the three Phase D offload optimizations shelved by their own gate measurements, the cancelled per-arch atol PR, the conceded batching-narrowing follow-up, the LAN self-signed-CA spec superseded by Tailscale HTTPS, and the vision prefix-cache whose Parts A and B were deliberately not built. Twelve comments and docstrings still pointed at two of the deleted files, now aimed at `docs/moe-offload-case-study.md` and `docs/offload-resident-sizing.md` instead; three more had already been dangling from an earlier session. `BACKLOG.md`'s 2026-07-10 entry keeps its reference on purpose — it names a spec inside a correction recording that the spec never discussed KV-cache quantization, which is prose about a document's contents rather than a pointer to it.
- [2026-08-03] **Reviewed the maintainer's takeover of the vllm-mlx Mistral parser PR ([#631](https://github.com/waybarrios/vllm-mlx/pull/631)) and found two regressions in his fixes, one of them against `main`.** No mira-core code changed. waybarrios posted a five-finding review of `dfe6f46` and then, twelve minutes later, pushed two commits (`e65f075`, `a9daebf`) onto the fork branch fixing all five himself. All five check out when re-run: the legacy `[ARGS]`-in-JSON boundary is correct, the forged-call smuggling case is rejected on both paths, and the bounded name buffer flushes instead of losing the response. **The two new problems**, both measured by running the parser at `origin/main` (`0dd1157`), `dfe6f46` and `a9daebf` side by side plus the repo's own `tests/test_tool_parsers.py`. **(1) Parallel tool calls collapse on the streaming path.** The new `if self._args_started` early return at `mistral_tool_parser.py:293` runs ahead of the new-call detection at line 310, so a second `[TOOL_CALLS]` can never open index 1: it is appended to index 0's arguments instead, leaving one call whose arguments do not parse as JSON and a second call the client never sees. This catches the **legacy brace format, which `main` streams correctly** (`main` and `dfe6f46` both give indices `[0, 1]` with valid JSON; `a9daebf` gives `[0]` with `'{"city":"Madrid"}[TOOL_CALLS]get_time{"tz":"CET"}'`), so it is a regression against main and not only against the branch's earlier head. The `[ARGS]` format is not a regression, main never parsed it. The end-of-stream re-parse cannot rescue it: that fallback is guarded by `not tool_calls_detected`, and `tool_calls_detected` is set (`server.py:6232`) as soon as any delta carries `tool_calls`, which index 0 already did. **The one-line fix does not work and their own suite proves it** — letting a marker-bearing delta fall through restores `[0, 1]` but fails `test_mistral_streaming_marker_in_arguments_does_not_reset` (119/120). That failure is the useful part: swallowing the marker is exactly what keeps a `[TOOL_CALLS]` inside a quoted argument value from forging a second call, so the two behaviours are one mechanism and the real fix needs JSON string state carried across argument deltas, plus withholding a trailing partial marker so a split delta cannot leak either way. **(2) An odd number of double quotes before the marker hides the tool call entirely.** `_split_on_tool_call_markers` tracks string state from index 0, but the text ahead of the first `[TOOL_CALLS]` is prose, not JSON, so one unbalanced `"` leaves `in_string` true at the marker and it never becomes a split point (`tools_called=False`, whole string returned as content, both formats). Fix verified: start the scan at the first marker instead of index 0, which gives 120/120 and keeps the forged call rejected on both paths. Posted as [comment 5163641481](https://github.com/waybarrios/vllm-mlx/pull/631#issuecomment-5163641481). Local branch fast-forwarded `dfe6f46` to `a9daebf`; `pytest 9.1.1` installed into the fork's venv only, deliberately not into its `pyproject.toml`, so a repo that is not ours stays undiffed. Draft notes local in `notes/pr631-review-reply-draft.md` and `notes/pr631-finding1-fix-note.md`.
- [2026-08-03] **Closed out mlx [#3860](https://github.com/ml-explore/mlx/issues/3860); nothing owed.** Both asks from the 07-26 comments landed in the docs PR: attention fallbacks composed from ordinary GEMMs are named as affected (the head_dim 96 case), and the misleading "float16 and bfloat16 are unaffected" line is gone. The page has since been narrowed to 21 lines carrying only what holds regardless of backend and hardware, so the M5 `g17g`/`g17s` and M3 Max `g15s` measurements now live in the issue comments instead, alongside the author's CUDA sm_120 numbers posted 08-03. [#3883](https://github.com/ml-explore/mlx/issues/3883) is closed, [#3894](https://github.com/ml-explore/mlx/pull/3894) is open and unmerged.
- [2026-08-01] **Fixed a live bug where every Qwen3 turn with thinking on served its reasoning to
  the user as the answer** (`81e7564`). Qwen3's chat template pre-opens `<think>\n` in the *prompt*
  when `enable_thinking` is true, so the model never emits an opening tag and `ThinkingStripper`
  saw the whole reasoning stream as ordinary content. Fixed with a latch: `ThinkingStripper(preopened=True)`
  starts inside a thinking block, `saw_reasoning()` disarms it when a backend turns out to split
  reasoning into its own channel (in which case `content` really is just the answer), and `drain()`
  reclassifies an unclosed pre-opened block back to visible so a missing close tag cannot swallow a
  whole response. Request and response sides now share one predicate,
  `_uses_qwen_thinking_template()`, so they cannot drift apart again. 10 new tests across
  `test_thinking_stripper.py` and `test_thinking_control.py`, the latter parametrized over all four
  chat-template backends. Confirmed live on 2026-08-01: zero `<think>`/`</think>`/`<|channel|>`
  occurrences across a full 13-question bench.
- [2026-08-01] **Moved to mlx 0.32.0 + mlx-metal 0.32.0** (`5ba9c71`), then verified it end to end
  (`docs/bench-results-2026-08-01.md`, 25/26, no regression vs the 2026-07-18 baseline). The
  upgrade fixes nothing Mira had: measured against the four real 0.31.2 bugs, `BatchKVCache` uses
  an array offset so it never takes the broken rope branch, and `SwitchGLU` keeps M=1 so it never
  takes the broken gather_qmm one. The throughput win is real but only from about eight concurrent
  sequences (+24%), which Mira does not have. It was done now because `mlx-vlm >= 0.6.5` requires
  it and the vision work needs that, and doing it as its own change keeps "did the bump break
  something" separate from "did the vision seam break something". Analysis in
  `notes/mlx-032-upgrade-analysis.md` (local).
- [2026-08-01] **Upstreamed an omlx truncation bug**: `ThinkingParser.finish()` re-emits the whole
  reasoning stream as `content` when a thinking turn is cut short by `max_tokens`, so a truncated
  answer arrives as its own chain of thought. jundot/omlx issue #2457 and PR #2458 (fork clone at
  `~/Projects/omlx`, branch `fix/streaming-truncated-thinking`). The fix keeps the
  maintainer's intentional recovery path for the non-truncated case and only suppresses it when
  `finish_reason=length`. Awaiting the maintainer; that repo merges outside PRs in a median of
  under a day, so this should not become another mlx-lm-style long wait.
- Entries before 2026-08-01 were pruned on 2026-08-08. They are duplicated in
  `CHANGELOG.md`, in git history, and in the memory files those entries already
  cited; nothing was recorded only here.

## Pending

### Waiting on a week of ordinary use — no action until then
- **`proactive_decompress` and `boundary_snapshot` are both ON in `mira.yaml` and OFF in
  `core/config.py`, and that split is the plan, not an oversight.** Decided 2026-08-08: this Mac
  runs both so the week produces real data, while anything shipped or remote keeps the
  conservative default until that data exists. **Confirmed for v1.2.0: they ship off**, and both
  are now documented in `mira.yaml.example` so "off" does not also mean "undiscoverable". The
  week is what decides whether the default flips in a later release. What the week is for:
  - `boundary_snapshot` — whether `_history_boundary` ever refuses a real prompt (it logs a
    warning), whether `failures` stays 0, and whether two entries per turn push the LRU against
    its 5.00GB ceiling. Watch `/v1/stats` → `boundary_snapshot`.
  - `proactive_decompress` — `events` climbing with non-zero `last_reclaimed_bytes` is it
    working; `failures > 0` means it disabled itself; `skipped_no_headroom` climbing means the
    availability floor is too high for this machine.
  - **The eviction notification firing at all.** A week of zero notifications is a real result,
    not a failure: it would mean eviction is rarer than the forced Xcode+Docker test suggested.
  One known limitation, accepted rather than fixed: **a request arriving during a touch waits
  behind it**, because the model thread is the only thread that can serve. Still never worse than
  doing nothing, but the magnitude is larger than the first measurement suggested — `events: 7`,
  `last_reclaimed_bytes: 20.05GB`, `last_seconds: 19.52` within hours of enabling it. So the touch
  scales with how much was evicted, ~1.8s for a partial to **~19.5s for a full one**. If that wait
  is ever felt, the fix is to make the touch interruptible (chunk it, re-check the inbox between
  chunks), not to disable it.
  **Trial data, 2026-08-09 morning: `events: 23`, `last_seconds: 1.577`, `last_reclaimed_bytes:
  7.26GB`, `skipped_no_headroom: 0`, `failures: 0`.** So it is firing regularly under ordinary use
  and reclaiming on the idle branch at no user-visible cost. Worth stating plainly because the
  17.60s figure in `mira.yaml` describes the **unmitigated** case: macOS never faults the model back
  spontaneously, which is the motivation for this flag, not a description of what happens with it
  on. Even with it off the worst case was ever one slow reply, never permanent degradation — the
  request that pays the cost is itself the fix. **An eviction costs time, not correctness**, so it
  cannot change a generated answer and accuracy evals are unaffected by it.

### Needs Miguel
- **Run the lm-evals — reminder set for 21:00 on 2026-08-09** (`bash notes/run_lm_evals.sh`). It
  stops production Mira and restores it, smoke-tests two questions first and aborts if that fails,
  then runs IFEval and MMLU-Pro (capped at 1000 of 12,032; SE ~1.6pp). GPQA is included but
  **non-fatal**, so a bad token skips it rather than throwing away the other two. Reminders survive
  a reboot — they are in SQLite and the scheduler fires catch-up on startup.
- **HF credentials.** `~/.zprofile` line 10 exports a stale `HF_TOKEN` **which takes precedence over
  `hf auth login`**, so updating only the stored credential does nothing. As of 2026-08-09 the
  stored one was invalid too; `hf auth login --force` once would give a single source of truth,
  which also matters for anything running outside the login shell. GPQA additionally needs the gate
  accepted at `https://huggingface.co/datasets/Idavidrein/gpqa`.
- **Score the 2026-08-08 agentic bench.** `docs/bench-results-2026-08-08.md` has timings and tool
  traces for Q6–Q12 filled in and the quality column empty. Less urgent than it was:
  `scripts/bench_eval.py` now scores these automatically, so this column is only for judgement the
  harness deliberately does not make.

### Small, no decision needed
- **The question set is now the limiting factor, not the harness.** Tier 1 scored 24/24 three runs
  running, so the deterministic half no longer discriminates between builds — it is a regression
  alarm, not a measure of quality. The judged half carries a ±1 floor, so it cannot be read finely
  either. Sixteen questions, most of them comfortable for a 35B model. **The standardised suites
  (IFEval, MMLU-Pro, GPQA, BFCL, AgentDojo) are the right instrument for "is Mira accurate"; this
  bench should settle into being a regression alarm for Mira's own plumbing** — orchestrator, tools,
  injection handling — which no public benchmark covers. Adding harder questions buys more than
  further harness work.
- ~~**Bench Q10 turn 2 rests on a false premise.**~~ Fixed 2026-08-09 in `5cb4dc8`: the injected
  file is now named by the question (`core/orchestrator.py`). Any Q10 score before that date
  measured a broken question.
- **The `task_done` guard's refusal path has no live demonstration.** Q6–Q12 ran on 2026-08-08 and
  the guard fired **zero** times while every agentic question exited through `task_done` normally,
  which proves it is inert on legitimate turns and regressed nothing. It does not prove the refusal
  works against a real model, because Qwen3.6 never took the escape hatch. Only unit tests cover it.

### Upstream — open, and genuinely nothing to do but wait
- **Three mlx-lm PRs are open with zero reviews**, confirmed 2026-08-08:
  [#1584](https://github.com/ml-explore/mlx-lm/pull/1584) (KV-cache quantization for the
  continuous-batching path), [#1588](https://github.com/ml-explore/mlx-lm/pull/1588) (disk-backed
  expert offloading), [#1619](https://github.com/ml-explore/mlx-lm/pull/1619) (`rotated` flag
  round-trip in `BatchRotatingKVCache.meta_state`). A maintainer review is the binary gate — 31 of
  400 external PRs got one and 30 merged; 71 did not and 0 merged. **Do not open a fourth.**
  Mira is NOT blocked on any of them: KV quantization has been live since 2026-07-18
  (`mira_mlx_kv_bits: 8`) on the `mira-core-pin` fork branch. What upstreaming buys is a smaller
  patch-carrying burden, nothing functional.
  - **Why the pin exists, since it is not obvious:** rebasing `mira-mistral-tool-call-fix` picked
    up upstream's text-state-machine refactor (mlx-lm #1501), which renamed `SequenceStateMachine`
    → `TextStateMachine` with a different API and broke `mira_mlx_server.py`'s import when the
    rebased commit was briefly live in the lockfile. `pyproject.toml` now pins a dedicated
    `mira-core-pin` branch frozen at `bbd8496` so PR-prep rebases cannot break mira-core again.
  - **Follow-up once #1584 merges:** check whether mlx-lm's own tool-call-flush fix supersedes the
    Mistral-specific patch given #1501, and migrate `mira_mlx_server.py` to the `TextStateMachine`
    API when moving off `mira-core-pin`.
- **vllm-mlx: one commit still owed upstream.** `edc07d4` (sibling KV-cache-path gap) waits on
  upstream [#629](https://github.com/waybarrios/vllm-mlx/pull/629) merging, then goes up as its own
  PR. It is **not on any branch** — dropped in the 2026-07-10 rebase that narrowed
  `fix/mistral-args-token-tool-parsing` to the parser fix — so recover it by SHA through the
  reflog, not by checking out a branch. (`d408d70` is superseded by upstream #629 itself.)
  **PR #631 merged on 2026-08-03**; a stale entry here still described it as awaiting the
  maintainer with a next action attached, corrected 2026-08-08. Nothing is owed on #631.

### mira-apps (separate session)
- **Three specs written 2026-08-08; the first is already done, the other two are in `mira-apps/specs/`:**
  - ~~`mira-core/specs/eviction-notification-predicts-slow-reply.md`~~ **DONE in `cdb446b`**, in a
    parallel session, within the hour. The notification no longer predicts what the next reply will
    cost, and the test rejects a prediction in **either** direction plus a string vague enough to
    satisfy the ban by saying nothing — verified against the old copy, the opposite copy and an
    empty one. "your Mac" deliberately kept rather than mirroring the banner's "the Mac running
    Mira": the two agree on the claim, not the wording.
  - `decode-check-covers-system-memory.md` — `scripts/checks/decode-check.sh` covers
    `ModelInfo.swift` and `Backend.swift` but not `SystemMemory.swift`, and this is the one whose
    failure is silent: a renamed key decodes to `.unknown`, which renders as nothing, which looks
    exactly like a healthy machine.
  - `memory-advisory-device-verification.md` — decode, wiring and a clean macOS build are all
    verified; **the banner has never been seen rendering.** `critical` shares the whole path with
    `evicted` and is far cheaper to trigger.
- **Device-verify the connection error messages** (uncommitted on mira-apps `main`). Compile-verified
  and unit-checked in isolation, never exercised on a device. Check: (a) a genuine 403 names
  `allowed_hosts` instead of blaming the network — comment out `allowed_hosts` in `mira.yaml` and
  restart to reproduce; (b) a real unreachable case (Tailscale off) still reads sensibly; (c) the
  longer 403 string does not overflow the error label in the Add Connection sheet. Also decide
  whether a permanent 403 should stop `autoConnect`/`startReconnect` retrying for 90s — they still
  use the `Bool` probe deliberately, so today a misconfigured `allowed_hosts` retries a hopeless
  connection and then goes orange with no explanation. Left alone as out of scope for the reported bug.

## Notes
- **A cache whose lookup key differs in kind from its insert key can never hit, and its size tells
  you nothing about that.** The disk prompt cache inserted on prompt+generated tokens and looked up
  on the prompt alone, hashed whole, so it filled 39.75GB over three weeks at a 100% miss rate and
  looked perfectly healthy from the outside — growing, evicting, within budget. `disk_cache_hits`
  was 0 the entire time and nobody read it. **Watch the hit counter of every cache, not its size**,
  and be suspicious of two "reads happened" signals that are not reads: file atimes move for
  backups and analysis sweeps, and a grep for a subsystem name matches its own log lines.
- **Do not build memory logic on `kern.memorystatus_vm_pressure_level`.** Measured 2026-08-08: when Xcode and Docker had already taken 8.64GB and the derived ceiling had dropped by the full 8.64GB, macOS was still reporting level 1. It only moved to 2 a whole sample later, once the compressor had spiked to 17.03GB. Availability is the leading indicator, compressor occupancy is the confirmation, and Apple's own level is the laggard — which is why all three are reported in `/v1/stats` and the advisory keys on the compressor.
- **`MLX_ENABLE_TF32` is parsed as an integer, so `MLX_ENABLE_TF32=true` means OFF.** Measured on q4 M=1024: `1` gives 8783 GFLOP/s, unset 8755, `0` gives 3184, and **`true` gives 3185** — identical to `0`. Anything non-numeric reads as disabled. `config.py` keeps a YAML bool and converts to `"1"`/`"0"` at the `Popen` boundary specifically so nobody can hit this by writing the value that looks correct.
- **An mlx-lm PR merges if and only if a maintainer reviews it.** Measured 2026-08-08 over 400 PRs, external authors only: **31 got a maintainer review and 30 merged (97%); 71 did not and 0 merged**. Community review does not substitute (0 of 3, and the reviewer pool includes pcuenca and Blaizzy). Requesting a reviewer is not an available lever — 6 of 386 external PRs have one pending and it needs write access. Diff size is the other gate: **401+ additions merged 0 of 14**, with 79 more sitting open in that class. The roster changed and the threads never noticed: angeloskath's last merge was 2026-07-09 and neither he nor awni has commented since 2026-07-20; **michalk8 and nastya236 do the merging now**. All three of Miguel's open PRs have `reviewers=[]`. **Decision 2026-08-08: no ping on any of them; do not re-propose it.** Full tables in gitignored `notes/mlxlm-merge-dynamics-2026-08-08.md`.
- `uv tool install --force <local-path>` can silently reuse a cached build and NOT pick up new local commits, even though the command reports success — confirmed by checking the installed file's actual content after a "successful" reinstall still showed the pre-fix code. Full fix requires `uv tool uninstall <pkg> && uv cache clean <pkg>` before reinstalling from a local path with uncommitted-upstream changes; `--force` alone is not sufficient.
- All local backends (mira-mlx/omlx/mlx-lm/vllm-mlx) share port 8080 — only one runs at a time by design. This means a stale/orphaned process from a previous backend can make `is_backend_ready()`'s health check pass spuriously (it just hits `/v1/models` on the shared port), masking a failed switch. See `feedback_backend_switch_self_stop.md` for the specific bug this caused.
- `pyproject.toml` stays tag-driven via `hatch-vcs` — the git tag IS the version, never hand-edit. `mira.yaml` is gitignored runtime config, not tracked (confirmed 2026-07-06, no leaked secrets — a local `mira.yaml` with a real token exists only on disk, never committed).
- Installer already automates nearly everything (uv, Python deps incl. mlx stack, disk/RAM preflight, mira.yaml bootstrap, optional tesseract/LaunchAgent via brew, doctor health check). The oMLX GUI step is the sole exception and is expected to stay manual indefinitely absent an oMLX-side scriptable install/model API.
- Tried `MLX_METAL_FAST_SYNCH=1` + `MLX_MAX_OPS_PER_BUFFER=50` + `MLX_MAX_MB_PER_BUFFER=50` (undocumented MLX Metal-backend env vars, found via `strings` on `libmlx.dylib` and confirmed against MLX's C++ source — `mlx/backend/metal/device.cpp`/`fence.cpp`) on mira-mlx (2026-07-18): `subprocess.Popen()` in `backend_manager.py` inherits the parent env with no override, so setting these on `server.py`'s own process before launch propagates to the mira-mlx subprocess. A/B benched against a plain baseline on **both** models mira-mlx runs — Ministral 3 14B and Qwen3.6-35B-A3B, isolated — **no measurable TTFT difference on either** (within ±20ms noise on the 5 non-agentic questions for both models; Qwen3.6's Q1 showed an ~800ms gap but that's within normal first-request-after-switch variance, not a consistent effect across Q2-Q5). Apparent wins on agentic questions were fully explained by tool-call-count variance between runs, not the env vars, on both models. Correctness unaffected either way on both. Not worth setting as a default; not pursued further.
