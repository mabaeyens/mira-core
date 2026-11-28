# Backlog

## Pending

Prod Mira is DOWN by default (Safari needed); `/mira-server start` when wanted. The LaunchAgent was left unloaded after the 2026-10-10 bench.

### MTP verify-window overlap: loose ends
- The expert profiler logs the MTP head's MoE block as backbone layer 0 (`mtp.layers.0` matches the `layers\.(\d+)` regex in `expert_profiler.py`). Give the head its own id before the next profile. Until then, drop layer 0 when analysing.
- Greedy output with MTP on is not byte-stable run to run (4 of 5 prompts diverged between two identical runs). Likely the depth controller varying the verify batch shape; not isolated. Matters for any eval that assumes determinism.
- The `-MTP` model dir produces token noise when loaded with MTP off (the plain snapshot is fine). Not a prod issue, but `mira_mlx_mtp_enabled: false` on that dir is a trap.

### Inkling comparison: follow-ups
- After the next Mira release closes: decide whether to file an Issue or PR for the control-string guard on omlx, mlx-lm and vllm-mlx (they are unguarded, not our engines). Searched 2026-11-28: no open or closed Issue or PR on jundot/omlx, ml-explore/mlx-lm or waybarrios/vllm-mlx covers user text tokenizing as control tokens (closest, mlx-lm #875, is about gpt-oss output channels), so nothing to join; a new report would be first.
- Decide whether to delete the 15.4 GB Inkling q4l copy, once all benches and comparisons have run successfully (needs approval); update `MODEL_REGISTRY.md` either way.
- Optional: rerun the 44-item benchmark on current code (v4 was aborted at 7 items; v3 plus the Qwen code-01 correction stands meanwhile).
- "bench server never reported backend_ready": diagnosed for 2026-11-26 (both runs). Each launched in a ~35 s DarkWake window and the Mac slept 15-17 min mid-load; on wake the wall-clock deadline had jumped past 420 s, so `_wait_for` made one instant probe and gave up. Fixed: `_wait_for` uses `time.monotonic()` and the run holds `caffeinate -i`. Unexplained: 2026-11-24 23:56-00:03, a genuine 7 min timeout with no sleep (power log no longer covers it); the bench now keeps the server log, so a recurrence is diagnosable.
- Remaining hardening items are in local `notes/` (gitignored).
- Checked 2026-11-28: the wrap-up before a forced thinking close works live on the Qwen engine (sentence lands before `</think>`, the answer follows cleanly). History never stores think blocks on current code; the single stored leak is an Aug 11 corpus row from before pre-opened handling.

### Ship wave 2026-11-22: follow-ups
- Waiting on App Review: iOS 0.4.0 build 2 (Waiting for Review since 2026-11-22). TestFlight external beta link `join/crTWN4kN` is also in review; swap the old TestFlight link in mira-web (home, support) only when the iOS session says it is approved.
- Server: `/cancel` with an empty conversation id cancels every active turn (a Stop on a brand-new first turn). Fix: return the conversation id at turn start (SSE event or header) so the client can cancel its own turn. Details in `notes/ios-followups.md`.
- Check the home page capability card still matches the one cached model (it names Gemma4 and Llama).
- Mac App Store: not submitted (no demo mode, a Mac reviewer sees only a connect screen). Revisit when Mira Desktop has a demo or review path. Direct download covers Mac meanwhile.
- Mira Desktop store screenshots: one 1280x800 shot exists (`~/Desktop/mira-store-shots/macos/01-home.png`); automated capture failed while the Mac was in use.
- Optional: Spanish App Store localisation; a TestFlight beta group for the new app; Mira Desktop 'Pair' first-run text only mentions typing the Tailscale address.

### Engine on-demand lifecycle (daily-use footprint lever)
Built and pushed (`8232f9a`, default off, see `docs/engine-lifecycle.md`). Wake measured 9 to 14 s. Direction is daily usability and a 0.3 then 1.0 App Store release, not speed: the engine is a child of `server.py` and ~20GB stays wired around the clock.
- mira-apps: show sleeping/waking, call `/engine/wake` on composer focus, "Unload now" action, chat timeouts that allow a wake. Separate repo, needs its own spec and device check.
- Pick the daily default (proposed 15 min) and enable `engine_idle_unload_minutes` in the live `mira.yaml` after one real week.
- Retry the independent adversarial review of these levers on context-mode 2.x; it did not run because the 1.0.169 hook appended its routing block to every Agent prompt without an idempotency check.
- Not worth pursuing as headline levers: smaller default context (already derived to ~39k, 64k OOMs), smaller prefill step, kv2 (kv4 was judged keep).

### Gemma4-26B vs Qwen3.6: open follow-ups
GPQA proxy n=100, fresh engine each, `MLX_ENABLE_TF32=0`, kv4 (2026-10-10): gemma4-26b 48.0% vs Qwen3.6-35B-A3B-MTP 45.0%, a gap inside proxy noise. Earlier "gemma4 is worse" results were tool-calling and 24K-context OOM, which the proxy does not touch.
- Not measured: peak memory, tool-calling, long context, thinking-on gate. Next: a `bench_compare` tool-calling run and a peak-RAM reading before any default change.
- The proxy fixture printed `subset_sha` 35be7d860593; the eval-contract notes record af3777f46584. Qwen's 45.0% matches the old baseline, but find out why the hash differs.

### omlx 0.7.0 follow-ups
- PR jundot/omlx#4414 (non-streaming truncated-thinking fix) is open and unit-tested only. The stale bot closes after 60 days without commits: rebase onto `main` by early December if unreviewed. The non-streaming leak (jundot/omlx#3796) does not materially affect Mira.
- MTP speed and accept rate on 0.7.0 were not re-measured (only load plus a one-token reply).
- Cleanup left: `~/Downloads/omlx-0.7.0-dmg/` and the `~/Projects/omlx` clone, which goes after #4414 is merged or closed.

### `derive_resident_expert_fraction` is blind to gpt-oss-120b
- **`gpt-oss-120b-MXFP4-Q8` reports `num_experts=None` in its config, so `_classify_weight_bytes`
  finds 0GB of experts and `derive_resident_expert_fraction` returns the `floor_fraction` (0.3)
  unchanged** — the same early return a dense model gets. Found 2026-08-11 while modelling what a
  128GB Mac would do: at 32GB the model is over-DRAM either way so nothing is lost today, but on a
  larger machine gpt-oss-120b is the obvious thing to run and the MoE sizing logic cannot see it.
  The fix is a config-shape question, not a policy one: find what gpt-oss calls its expert count
  (the key is not `num_experts`) and teach `_classify_weight_bytes` that spelling. Related and
  already known: attention-sink models cannot use quantized KV, so gpt-oss needs the unquantized
  path regardless. **No action while this Mac has 32GB** — it changes nothing at this size.
- **The 128GB crossover, recorded so it is not re-derived:** the ceiling is
  `min(0.55 x total, 0.78 x total - 3GB, total - 3GB)` = **70.4GB on a 128GB Mac**, so the offload
  path only engages above that. Verified against the cached models: the 8-bit Qwen3.6 (35.1GB) and
  gpt-oss-120b (59.0GB) both go from over-DRAM at 32GB to **fully resident** at 128GB. A 4-bit
  100B-class model therefore never exercises offload on 128GB — that needs a 4-bit 235B-class
  (~132GB) or an 8-bit 120B.

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
    **2026-08-14: caught the opposite — a false positive.** The cause classifier keyed
    `external_pressure` on the OS pressure level alone, but compressing Mira's own ~20GB idle model
    is itself enough to raise pressure to warn, so the treadmill fired the notification seven times
    in one morning with other apps holding under 1GB. Fixed: the cause now attributes the compressor
    per-process (`compressor_bytes - self_compressed_bytes`) and stays `idle_reclaim` unless the
    non-Mira remainder clears `EXTERNAL_COMPRESSED_MIN_BYTES`. Lands on the next engine restart.
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

### Documentation leftovers from the 2026-08-12 sweep
- **mira-apps has no feature request template, and now nothing else to fall back on.** The deleted
  file was bug-and-feature both, opening with a Type selector; `bug_report.md` is bug-only by
  design. So a feature request or a question currently lands on a form asking for steps to reproduce
  and an expected-vs-actual. Either a second template or a `config.yml` with a discussions link
  closes it — small, and worth doing before anyone files the next one.
- **The three settings the sweep made configurable for the first time deserve a look now that
  someone can actually set them.** `shell_sandbox`, `shell_sandbox_allow_network` and
  `url_fetch_allow_private` were read by `config.py` and absent from `mira.yaml.example`, so their
  non-default paths have never been exercised by anyone deliberately choosing them. Documenting a
  security control is not the same as having reviewed what happens when it is switched off.
### Parked from the Unit 4 agent harness (`mabaeyens/hf-courses`)
- **Per-turn trail of queries and pages.** Only worth building if tool-using evals resume; they are cancelled for now. Archive fallback and the recovery rule shipped 2026-11-22.
### Small, no decision needed
- **The question set is now the limiting factor, not the harness.** Tier 1 scored 24/24 three runs
  running, so the deterministic half no longer discriminates between builds — it is a regression
  alarm, not a measure of quality. The judged half carries a ±1 floor, so it cannot be read finely
  either. Sixteen questions, most of them comfortable for a 35B model. **The standardised suites
  (IFEval, MMLU-Pro, GPQA, BFCL, AgentDojo) are the right instrument for "is Mira accurate"; this
  bench should settle into being a regression alarm for Mira's own plumbing** — orchestrator, tools,
  injection handling — which no public benchmark covers. Adding harder questions buys more than
  further harness work. **Six added 2026-08-12 (17-22), each one a bug that shipped** — see the Done
  entry. They are written and unit-tested; **they have not been run against a live server yet**, so
  whether they actually discriminate is unmeasured.
- **The `task_done` guard's refusal path has no live demonstration.** Q6–Q12 ran on 2026-08-08 and
  the guard fired **zero** times while every agentic question exited through `task_done` normally,
  which proves it is inert on legitimate turns and regressed nothing. It does not prove the refusal
  works against a real model, because Qwen3.6 never took the escape hatch. Only unit tests cover it.
  **Bench question 20 exists to bait it** as of 2026-08-12 — a task the model is likely to want to
  declare done early — but it has not been run, so this stays open until it has.

### Upstream — open, and genuinely nothing to do but wait
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
  - ~~`decode-check-covers-system-memory.md`~~ **DONE in `74b2824`** — the decode check now covers
    `SystemMemory.swift`, which was the one whose failure was silent: a renamed key decodes to
    `.unknown`, which renders as nothing, which looks exactly like a healthy machine.
  - ~~`memory-advisory-device-verification.md`~~ **DONE in `3765330`** — the banner has been seen
    rendering. Both specs are gone from `mira-apps/specs/`; this list was stale until 2026-08-12.
- **Two specs written 2026-08-12, both in `mira-apps/specs/`, neither built:**
  - `memory-notifications-only-when-actionable.md` — the selective rule. Note what the
    investigation turned up first: **the silence is mira-core's, not the app's.**
    `MEMORY_ADVISORY_NOTIFICATIONS` has defaulted to False since 2026-08-11 (`core/config.py:324`),
    `core/memory_watch.py` still records every transition, and the server log shows 12
    `model evicted by another process` warnings on 2026-08-12 alone. Only `evicted` ever notified —
    there was never a separate pressure notification to miss. The spec is about the distinction the
    old design lacked: an interrupt has to be actionable or explain something the user caused,
    everything else belongs in the ambient banner. **Open question recorded in the spec:** whether
    mira-core adds a `cause`/`user_actionable` field or the client infers it. The lean is mira-core —
    it is the side that knows which process took the memory.
  - `connection-errors-device-verification.md` — the three device checks: (a) a genuine 403 names
    `allowed_hosts` instead of blaming the network, (b) a real unreachable case (Tailscale off) still
    reads sensibly, (c) the longer 403 string does not overflow the error label in the Add Connection
    sheet. **Open question recorded in the spec:** whether a permanent 403 should stop the 90s
    retry — the lean is to keep retrying but surface the reason on the first refusal, since today a
    misconfigured `allowed_hosts` retries a hopeless connection and then goes orange with no
    explanation.

## Parked (revisit trigger in bold)
- **Smaller-footprint Qwen3.6 (REAP): dead end.** `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit-REAP-19B` scored 28.0% on the GPQA proxy vs 45.0% stock (2026-11-21, near chance). Other REAP checkpoints are unchecked; measure quality first, no `trust_remote_code`, and log any download in `MODEL_REGISTRY.md`. Context, not weights, is the limit on 32GB: wall is prefill transient plus the Metal ceiling, not KV.
- **Inkling 2 Mini: built, not adopted. Revisit when upstream Inkling support (PR #2086) merges (still open and unreviewed 2026-11-22), or when the variable-loop checkpoint (`TML-d24-varloop-research`, 1 or 2 passes instead of 3) runs in mlx-lm.** Ported on the fork (`mabaeyens/mlx-lm` branch `mira-inkling`) and wired into mira-mlx (`7592701`), but decode is 28.7 vs 57.5 t/s (gate 35 not met), prefill 0.36-0.69x, ~2.5-3x longer answers. Open gaps: kv quant on CacheList, `enable_thinking=false` still reasons, vision untested, Mira pin not moved. Checkpoints were deleted 2026-11-21; re-download to revisit. Untried: `thinkingmachines/TML-d24-varloop-research` (needs conversion). The 2026-11-20 HF launch post adds no Mac speed data; an independent M5 Pro run on the PR measured 28.7 t/s, same as ours. Fewer loop passes would cut layer visits 48 to 36 or 24, so decode could clear the gate, but quality is unevaluated and the PR config only carries an `is_varloop` flag.
- **Ternary-Bonsai-2-27B (`prism-ml`, Qwen3.8-27B ternary, MLX 2-bit, 8.6GB): tested, no-go. Revisit if prism-ml's MLX kernels reach mlx-lm / mira-mlx, or for a 16GB-Mac model.** On this M5/32GB via the pack's own mlx-vlm loader: decode 11.3 t/s at 2k and 10.2 at 8k (gate 35), prefill ~58 t/s, peak 14.4-16.5GB, so the 8.6GB footprint win mostly disappears. Output was coherent. The headline 47 t/s is an M5 Max; their M5 Pro llama.cpp figure is 28. It needs its bundled loader (stock loaders return wrong output silently), has no MTP, and bypasses mira-mlx. Script: `notes/bonsai_smoke.py`.
- **Fused quantized SDPA (mlx PR #3026): deferred. Recheck at the next mlx bump.** It is the lever against the 73 to 11 t/s decode collapse from 2k to 64k, but does not apply to v0.32.3 (7 of 15 files conflict) and the Metal toolchain is not installed. mlx is at 0.32.3 (831 tests pass, proxy unchanged at 45.0%).
- **Headroom options if the 32GB budget gets tight (nothing benched today):** Ministral-3-14B 4bit (7.9GB), Qwen3.8-27B 4bit (15.8GB, 8.3 t/s), gemma-4-26B-A4B REAP-14B (11.2GB, pruning risk unmeasured). Qwen3.8-Flash-Next and GLM-5.3-Flash do not fit.
- **Unmeasured idea:** stop an MTP draft chain early on low draft confidence. Expected gain looks small at accept ~0.77.
- **MLX Model Explorer bench** (`codelion/mlx-model-explorer`, `bench/mlx_explorer_bench.py`): not run; perplexity is a coarse signal.
- **justfit-llm 2.6.0 (Qwen3.8-27B with a DFlash2 draft tree, native Rust/Metal runtime): tested, deleted. Revisit if a Mac port of Mira's own draft work needs a comparison point, or if it ships notarized.** On this M5/32GB: 22.7 t/s decode at 1K, 19.8 at 49K, prefill ~310 t/s, against 8.3 t/s for the dense 27B in mlx-lm. That is 2.7x and holds at depth, but the README claims 59.5 decode on an M5 Pro and the 16K point was never measured here (swap hit 16 of 16.4 GB, so I stopped). Engine held 18.6 GB resident; it does not coexist with Mira on 32GB. Output quality unchecked. Binaries are not notarized. The Oceanus/Reddit post is CUDA-only and does not apply to a Mac.


## Done
- [2026-11-28] Read Cohere's 2309.05444 (training-only, no use for Mira) and 2610.38981 (verify cost on sparse MoE follows unique experts per window). Measured it on the live MoE with the expert profiler: a 4-token verify window touches ~21 unique experts per layer against 8 for one token, for ~2.5 tokens per cycle, which caps MTP near 1.2x. Measured 1.25x at max depth 3 and 1.21x at depth 2 against the plain 4-bit snapshot (3 interleaved repeats each, spread ~1%). No new lever. Write-up in local `specs/mtp-verify-expert-overlap.md`, data in `notes/mtp_overlap_*`.
- [2026-11-28] Inkling 2 Mini q4l vs Qwen3.6-35B-A3B-4bit-MTP harness comparison finished; results in local `notes/live_with_2026-11-25/FINAL_ASSESSMENT.md`. Harness fixes shipped from it: `read_file` window cap, leading blank lines, forced-close wrap-up, history snapshot kept on non-trimmable models, quoted think tags, boundary one token short. Message text containing tokenizer control strings now stays plain text on every backend (1d71695). Suite and CI green.
- [2026-11-22] Shipped the Wayback fallback in `fetch_url` and the RULE 9 recovery rule from the Unit 4 harness (e3fd314). Smoke-tested Ternary-Bonsai-2-27B (no-go, see Parked) and re-checked Inkling 2 Mini against the HF launch post (no change). Bonsai files deleted with approval (`~/.bonsai-test`, `~/.venvs/bonsai-smoke`).
- [2026-11-22] Tested justfit-llm 2.6.0 on the M5/32GB (decode 2.7x the dense mlx-lm baseline, 16K point not reached), then deleted `~/.justfit-test` and its venv with approval and updated `MODEL_REGISTRY.md`. Read the Unit 4 agent harness for findings Mira could borrow; they are under Pending. Prod Mira is still down.

- [2026-11-22] Docs consolidation: 43 to 28 tracked `.md` files. Removed `docs/bench-archive/` and the orphan 2026-08-17 bench; merged the MoE design and prior-art docs into `moe-offload-case-study.md` and the decode-wall and custom-kernels essays into `multi-token-prediction.md`; folded `dev-reference.md` into `architecture.md`; re-verified `architecture.md`, `omlx-ctl.md`, `diagram.md` and `SECURITY.md` against the code; pruned this file from 715 to ~216 lines. Removed text is in git history.
- [2026-11-22] iOS 0.4.0 build 2 submitted for App Review (iOS only) with demo mode; listing entered via the ASC API from `notes/asc-listing-0.4.0.md`. Mira Desktop build 2 DMG notarized and attached to mira-apps v0.4.0. Keychain token and on-device dictation verified on a device; privacy page, home and support copy rewritten for the Mac-server model (mira-web `885b910`, `498e60c`, `8f50805`, `4bcef38`).
- [2026-11-21] mira-core v1.5.1 released (PyMuPDF replaced by pypdfium2, Origin host+port check, proxy headers off, loopback needs no token); menu-bar 0.1.2 notarized with Sparkle update verified; model cache trimmed to the live model (95 GB to 21 GB); two security reviews done, findings in gitignored `SECURITY_AUDIT.md`.

## Notes
- **MoE MTP is capped by expert reads per verify pass, not by the head or the draft depth (2026-11-28).** A verify window reads ~2.1x one token's experts for ~2.5 emitted tokens, so a bandwidth-bound step gains about 1.2x; measured 1.21-1.25x. Depth 3 beats depth 2 by 3% (real, outside the spread); the head's own MoE reads are ~5% of a step. The dense model has no such growth, hence 1.70x. A cheaper verify (fewer unique experts) is the only lever; the sort-threshold gather fix is a separate cost.
- Docs layout: client-side flows (iOS connection, macOS startup) belong in mira-apps, not `docs/diagram.md`. The appendices in `multi-token-prediction.md` and `moe-offload-case-study.md` are long by design and stay as is.
- **Mixed-precision quant is NOT a speed-free quality lever on this MoE — NULL 2026-08-17.** Tried
  `Qwen3.6-35B-A3B-4bit-DWQ` as the one remaining speed-free quality lever (uniform 4-bit baseline
  = GPQA-diamond 70.7% flex). Despite the "4bit" name it is mixed 4/8-bit (`config.json` top-level
  `bits:8` + 512 per-layer overrides), and decoded **~28% slower resident on the M5** (58.6/56.7 →
  42.0/40.4 t/s), prefill unchanged — the same decode-bandwidth ceiling the 2026-05-30 OptiQ audit
  found. A speed gate killed it before any quality run. Every quality-recovery quant on mlx-community
  raises average bits, so none is speed-free; uniform 4-bit stays. Reopen only for a genuinely
  ≤4-bit-average build (sub-4-bit / block-sparse). Build deleted. Don't re-download.
- **A cache whose lookup key differs in kind from its insert key can never hit, and its size tells
  you nothing about that.** The disk prompt cache inserted on prompt+generated tokens and looked up
  on the prompt alone, hashed whole, so it filled 39.75GB over three weeks at a 100% miss rate and
  looked perfectly healthy from the outside — growing, evicting, within budget. `disk_cache_hits`
  was 0 the entire time and nobody read it. **Watch the hit counter of every cache, not its size**,
  and be suspicious of two "reads happened" signals that are not reads: file atimes move for
  backups and analysis sweeps, and a grep for a subsystem name matches its own log lines.
- **Do not build memory logic on `kern.memorystatus_vm_pressure_level`.** Measured 2026-08-08: when Xcode and Docker had already taken 8.64GB and the derived ceiling had dropped by the full 8.64GB, macOS was still reporting level 1. It only moved to 2 a whole sample later, once the compressor had spiked to 17.03GB. Availability is the leading indicator, compressor occupancy is the confirmation, and Apple's own level is the laggard — which is why all three are reported in `/v1/stats` and the advisory keys on the compressor.
- **`MLX_ENABLE_TF32` is parsed as an integer, so `MLX_ENABLE_TF32=true` means OFF.** Measured on q4 M=1024: `1` gives 8783 GFLOP/s, unset 8755, `0` gives 3184, and **`true` gives 3185** — identical to `0`. Anything non-numeric reads as disabled. `config.py` keeps a YAML bool and converts to `"1"`/`"0"` at the `Popen` boundary specifically so nobody can hit this by writing the value that looks correct.
- **An mlx-lm PR merges if and only if a maintainer reviews it.** Measured 2026-08-08 over 400 PRs, external authors only: **31 got a maintainer review and 30 merged (97%); 71 did not and 0 merged**. Community review does not substitute (0 of 3, and the reviewer pool includes pcuenca and Blaizzy). Requesting a reviewer is not an available lever — 6 of 386 external PRs have one pending and it needs write access. Diff size is the other gate: **401+ additions merged 0 of 14**, with 79 more sitting open in that class. The roster changed and the threads never noticed: angeloskath's last merge was 2026-07-09 and neither he nor awni has commented since 2026-07-20; **michalk8 and nastya236 do the merging now**. All three of my open PRs have `reviewers=[]`. **Decision 2026-08-08: no ping on any of them; do not re-propose it.** Full tables in gitignored `notes/mlxlm-merge-dynamics-2026-08-08.md`.
- `uv tool install --force <local-path>` can silently reuse a cached build and NOT pick up new local commits, even though the command reports success — confirmed by checking the installed file's actual content after a "successful" reinstall still showed the pre-fix code. Full fix requires `uv tool uninstall <pkg> && uv cache clean <pkg>` before reinstalling from a local path with uncommitted-upstream changes; `--force` alone is not sufficient.
- All local backends (mira-mlx/omlx/mlx-lm/vllm-mlx) share port 8080 — only one runs at a time by design. This means a stale/orphaned process from a previous backend can make `is_backend_ready()`'s health check pass spuriously (it just hits `/v1/models` on the shared port), masking a failed switch. See `feedback_backend_switch_self_stop.md` for the specific bug this caused.
- `pyproject.toml` stays tag-driven via `hatch-vcs` — the git tag IS the version, never hand-edit. `mira.yaml` is gitignored runtime config, not tracked (confirmed 2026-07-06, no leaked secrets — a local `mira.yaml` with a real token exists only on disk, never committed).
- Installer already automates nearly everything (uv, Python deps incl. mlx stack, disk/RAM preflight, mira.yaml bootstrap, optional tesseract/LaunchAgent via brew, doctor health check). The oMLX GUI step is the sole exception and is expected to stay manual indefinitely absent an oMLX-side scriptable install/model API.
- Tried `MLX_METAL_FAST_SYNCH=1` + `MLX_MAX_OPS_PER_BUFFER=50` + `MLX_MAX_MB_PER_BUFFER=50` (undocumented MLX Metal-backend env vars, found via `strings` on `libmlx.dylib` and confirmed against MLX's C++ source — `mlx/backend/metal/device.cpp`/`fence.cpp`) on mira-mlx (2026-07-18): `subprocess.Popen()` in `backend_manager.py` inherits the parent env with no override, so setting these on `server.py`'s own process before launch propagates to the mira-mlx subprocess. A/B benched against a plain baseline on **both** models mira-mlx runs — Ministral 3 14B and Qwen3.6-35B-A3B, isolated — **no measurable TTFT difference on either** (within ±20ms noise on the 5 non-agentic questions for both models; Qwen3.6's Q1 showed an ~800ms gap but that's within normal first-request-after-switch variance, not a consistent effect across Q2-Q5). Apparent wins on agentic questions were fully explained by tool-call-count variance between runs, not the env vars, on both models. Correctness unaffected either way on both. Not worth setting as a default; not pursued further.
- **A fact that must stay current belongs in exactly one file; everywhere else links to it.** The
  mlx-lm pin SHA was written out in four places and went stale in three of them within hours of
  moving — `docs/packaging.md` was still showing `branch = "mira-mistral-tool-call-fix"`, a branch
  that no longer exists, in precisely the form `docs/mlx-lm-pin.md` was written to forbid. That
  document had predicted the failure in its own closing paragraph and then suffered it, which is the
  useful part: a warning is not a mechanism. **The SHA now appears in `pyproject.toml` and
  `docs/mlx-lm-pin.md` and nowhere else**, stated as a rule inside the doc so the next copy is a
  visible violation rather than an oversight. Same rule now applies to settings: `docs/configuration.md`
  is the one reference, README and `docs/architecture.md` point at it instead of keeping short lists
  that silently fall behind `config.py`.
- **`_get()` in `core/config.py` is the definition of what a setting is.** Anything read through it
  is configurable and must appear in `docs/configuration.md`; anything not read through it is a
  constant, however much it looks like a knob. Written down because the drift it caused was invisible
  from either end — seven keys the code honoured had never been offered to a user, three of them
  security controls, while `docs/architecture.md` advertised `reranker_model` as source-only weeks
  after it became a config field.
- **GitHub silently ignores `.github/ISSUE_TEMPLATE.md` once a `.github/ISSUE_TEMPLATE/` directory
  exists.** No warning, no precedence note in the repo, nothing in the UI. This is how a
  well-written template in mira-apps went unseen for its entire life while GitHub's unedited default
  — browser, iPhone 6, "Version [e.g. 22]" — collected every bug report for a macOS and iOS app.
  If a template ever looks like it is not being used, check for the directory before rewriting it.
- 2026-11-22: GUI capture of Mira Desktop fails when another app has focus or the user is active (window not capturable, AX tree empty). Needs the Mac idle, or the user takes the shots.
- 2026-11-22: a fresh MIRA_DATA_DIR is seeded from the stale `core/conversations.db`; delete migrated rows before any screenshot session.
