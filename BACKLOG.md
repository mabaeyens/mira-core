# Backlog

## Done
- [2026-07-26] **Repaired all 9 deselected tests and removed the deselect list from CI; suite is 330 passed / 0 failed with nothing skipped.** The premise in the old Pending entry was wrong: these did **not** "fail on the Linux runner for their own reasons", they failed on macOS too, which made them locally reproducible and properly fixable rather than a blind CI guess. Three independent causes, all test rot behind security hardening that landed later. **(1) The five `/browse` tests** pointed at `/tmp` and pytest's `tmp_path`, both outside `Path.home()`, so `server._safe_path` 403'd them before any assertion ran. Added a `home_dir` fixture that mkdtemps inside `$HOME` and tears down after; `test_browse_nonexistent_path_returns_error` now uses a missing path *inside* home so it exercises the not-a-directory branch instead of being short-circuited by the guard. Added `test_browse_outside_home_is_forbidden` to cover the 403 the old tests were tripping over by accident. **(2) The four `run_shell` tests were running against the real `~/workspace`, not the fixture** — including `test_run_shell_force_bypasses_guard`, which executes `rm`. The `ws` fixture patched only `core.workspace.WORKSPACE_ROOT`, but `core/shell_tools.py:26` does its own `from .config import WORKSPACE_ROOT` and so holds a separate module-level binding that `run_shell` reads. Fixture now patches both. The fixture docstring calling `core.workspace` "the single source of truth" was the thing that hid this. **(3) `test_run_shell_timeout`** patched `core.shell_tools.SHELL_TIMEOUT`, which can never work: `run_shell(timeout: int = SHELL_TIMEOUT)` binds that default once at function-definition time, so `sleep 10` ran to completion under the real 30s and returned `exit_code: 0`. Now passes `timeout=1` explicitly. Also fixed `test_run_shell_captures_stderr`, which used an absolute path the absolute-path guard correctly refuses, so it got an error dict and died on `KeyError: 'stdout'`. Rode along: **`test_queries.py::test_stats_token_capture_with_real_counts`**, which was failing but was never on the deselect list, so CI was either red or passing by coincidence of runner RAM. It asserted `context_pct == round(1024 / 65536 * 100)` against a hardcoded window while `CONTEXT_WINDOW` is derived from available RAM, so the literal only held on the machine it was written on. Now asserts against `orchestrator.context_window`.
- [2026-07-26] **Bumped the mlx-lm pin to `65fcb4c` to carry the `insert_segments` kv_bits fix. Defensive, NOT a live bug fix here — the first read of this was wrong and is corrected below.** `mira_mlx_server.py:457` does call `insert_segments(..., caches=[cache])` with a cache restored from the disk prompt cache, and externally supplied caches do bypass `_make_new_cache()`, so on paper `mira_mlx_kv_bits: 8` looked like it was being ignored on every cache hit. **Measured against the real cache directory, it is not.** Scanned all 269 entries in `~/.local/share/mira/mira_mlx_cache` (safetensors headers only): 238 are `RotatingQuantizedKVCache` with `group_size=64, bits=8` baked in, and every entry written since 2026-07-18 17:23 through the newest (07-21 18:48) is quantized. The caches Mira hands to `insert_segments` were created quantized by `_make_new_cache()`, saved quantized, and restored quantized, and the fix explicitly skips entries that are already quantized. So KV-quant has been working correctly in Mira since it shipped. The pin bump is still worth keeping (it matches upstream, costs nothing, and covers any future path that supplies an unquantized cache), but the justification was wrong and the "silently a no-op" framing was mine, not a real defect. Fix is the one upstreamed as `3ebafab` on [mlx-lm#1584](https://github.com/ml-explore/mlx-lm/pull/1584), cherry-picked onto the frozen `mira-core-pin` branch rather than bumping to the PR head, deliberately: `kv-cache-quant-batching` sits on recent upstream and would drag in the #1501 `TextStateMachine` rename that broke this import once already. Suites green on the pin branch (`test_generate.py` + `test_prompt_cache.py`, 56 passed, `MLX_ENABLE_TF32=0`). Venv diffed before/after `uv sync` given the Jun 2026 MLX-stack wipe; only `mlx-lm` and `mira-core` moved. LaunchAgent reloaded (PID 965 to 15262), `backend_ready: true`. Two side findings: the installed rev was `5378ffb`, **one behind what `pyproject.toml` declared** (`e2b26fd`), so the venv had drifted from the lock; and the remaining 31 unquantized entries are all from an 8-minute window on 07-18 (17:15 to 17:23, the Phase C validation session), **5.72 GB that can never be hit** because `_key()` folds `kv_bits` into the hash, so a server running `kv_bits: 8` will never look them up. Deleting them needs a decision, not taken. The `self.rotated = bool(v[3])` bug ([mlx-lm#1250](https://github.com/ml-explore/mlx-lm/issues/1250), upstreamed as PR #1619) is genuinely dormant here for a reason I also first got wrong: not because `max_kv_size` is unset (it is set, the saved caches are rotating with `max_size=128000`) but because these are single-sequence `RotatingQuantizedKVCache`, which carries `keep/max_size/offset/idx/group_size/bits` and no `rotated` field at all. `rotated` belongs to the Batch variants, which this path never saves.
- [2026-07-24] **Cleared the review blocker on the KV-quant PR ([mlx-lm#1584](https://github.com/ml-explore/mlx-lm/pull/1584)) — root-caused the 8 batched-attention test failures to an M5 kernel quirk, not the PR.** The reviewer (`katlun-lgtm`) asked for a matched 32-seed sweep on the M5 before the divergence issue ([ml-explore/mlx#3897](https://github.com/ml-explore/mlx/issues/3897)) went to a maintainer. Ran it (repro in `/tmp/claude_m5_sweep.py`; not committed): D=64/96/128 × fp16/bf16 + an **fp32 control**, on both mlx 0.31.2 and a throwaway 0.32.0 venv (byte-identical cell-for-cell, so version-invariance on M5 is now *proven*, not inferred from the test suite). Findings: (1) the silicon split survives a matched seed sweep — M5 stays ~2.6 bits above the M3 Max at D=128 bf16, does not converge; (2) the **fp32 control is the clincher** — M5's masked path diverges at ~3e-4 (median 2⁻¹¹·⁷) where the M3 Max sits at the ~2e-7 floor, and being fp32 that pins it to the M5 fused-attention *kernel arithmetic*, not storage precision; (3) **D=96 is the mechanism tell** — katlun's follow-up M3 Max run showed D=96 is *identical on both machines* (both at floor), so the divergence is specifically the fused vectorized path that only engages on 64-aligned head dims (64, 128), while 96 falls back to a shared clean path both silicons run the same. Net: the 8 failures are pre-existing, M5-specific, and benign (argmax unchanged, ~1/32 logprob), so they don't gate the KV-quant work; the standing recommendation (an atol floor / per-arch tolerance in mlx-lm) holds regardless of whether Apple touches the kernel. Both replies posted humanized (no em-dashes) — [#3897 comment](https://github.com/ml-explore/mlx/issues/3897#issuecomment-5066681556), [#1584 comment](https://github.com/ml-explore/mlx-lm/pull/1584#issuecomment-5066681954). Investigation is closed and consistent from both silicons; now awaiting a maintainer. Full detail in the `project_m5_batched_attention_divergence` memory.
- [2026-07-21] **Prompt-injection hardening (step 8 — the last item of the 2026-07 security audit)**, shipped in v0.9.5. Added a per-process nonce trust boundary (`wrap_untrusted()` in `core/prompts.py`) + `RULE 10` at all six untrusted-content entry points in `core/orchestrator.py` — the `_wrap_observation` file/GitHub funnel, `fetch_url`, RAG chunks, search results, text attachments, OCR — so the model treats tool output as data, not instructions. Errors and approval-gate confirmations deliberately stay unwrapped (control metadata, not attacker content). Rode along: corrected `RULE 4`'s stale wording (destructive-action approval is out of band, not a `force` flag the model sets) and newline-escaped search result titles/URLs against forged `[N]` result blocks. New corpus (`scripts/bench_questions.yaml` Q14-16, `scripts/bench_fixtures/`, `scripts/serve_bench_fixtures.py`) + unit coverage (`tests/test_prompt_injection.py`: all six points asserted on real message dicts, reflected-nonce defeat, cache-stability). Full suite green apart from the pre-existing browse/fs_shell/stats baseline. Pre/post benches (3 runs each + a stashed pre-vs-post A/B) are LOCAL in `notes/bench_injection_{baseline,post}_2026-07-21.md`; shipped as defence-in-depth per Miguel's release decision, with the out-of-band approval gate (`core/approvals.py`) remaining the load-bearing control. Bench-corpus gotcha banked: `fetch_url` runs readability extraction that strips HTML comments, so an injection fixture must sit in visible page text (see the Q16 note in `bench_questions.yaml`).
- [2026-07-21] **Released mira-core v0.9.5** (CHANGELOG + tag + wheel + GitHub release) covering the three post-v0.9.4 security commits: run_shell OS sandbox (`5993dec`), listener model-identity check (`e61323f`), and the trust boundary (`c57fc6f`). v0.9.4 had already shipped earlier the same day without step 8, so the new work went out as the conservative patch bump v0.9.5 rather than a re-tag.
- [2026-07-21] **Fixed red CI** (`edcbf00`). The run_shell OS sandbox uses macOS `sandbox-exec`, absent on ubuntu runners, so `SHELL_SANDBOX` failed closed and broke four `test_run_shell_*` execution tests on every push from `5993dec` onward (they pass on macOS, which is why the original deselect list missed them). CI provision now writes `shell_sandbox: false` so those tests run unsandboxed and keep their coverage; the sandbox path itself stays covered by `test_shell_sandbox.py` (skips off macOS). Details + the deliberate do-not-deselect note under Pending.
- [2026-07-19] MoE offload throughput follow-up — **analysis complete, banked (no build this round)**.
  Diagnostic + 6 adversarial reviewers + fresh-server/standalone gates at fraction 0.3 on
  Qwen3.6-35B-A3B-4bit. Findings (`specs/moe-expert-offload-06-throughput-gate-findings.md`):
  (1) The "~18.2GB prefill peak" that framed the memory axis **no longer exists** — it was a
  pre-lazy-fix artifact; current diverse-prefill peak is 7.25GB. Owl-plan Approaches 1-3 closed.
  (2) **Disk is not the throughput bottleneck**: reads are 0.207ms each, 8-way parallel → ~14% of wall;
  `open`/`seek` <2%. A miss's real cost is engine-side (`mx.array` build + dequant + G-fold gather);
  the lever is miss count (hit_rate), not read speed. Specs 03/04/05 shelved (03 gated→moot, 04 NO-GO,
  05 deferred — see each spec's adversarial section).
  (3) **Hot-pinning the resident set** raises hit_rate but modestly: in-sample +10 pts was overfit;
  out-of-sample a hybrid (pin ~64 hottest + LRU) gives **+2.4 pts (~14% fewer misses)**, pure pinning
  (no LRU) hurts (-2 pts), adaptive LFU fails (+0.27). Left as scoped next-round work alongside the
  bigger-but-riskier G-fold gather compaction (owl Approach 2) for the ~8-12x prefill tax.
  Three optimization specs written + adversarially reviewed: `specs/moe-expert-offload-03/04/05-*.md`.
  No production code changed.
- [2026-07-18] MoE expert-offloading Phase 0 (profiling) — GO signal on both flagship models. Built
  `core/inference/expert_profiler.py` (opt-in, `--profile-experts`/`mira_mlx_profile_experts`) — patches
  `mlx_lm.models.switch_layers.SwitchGLU`/`SwitchMLP.__call__` at the class level (architecture-agnostic:
  Qwen3.6's `Qwen3NextSparseMoeBlock` and Gemma4's differently-named router aren't attribute-compatible,
  but both route through these shared primitives) and `scripts/analyze_expert_profile.py` (stdlib-only,
  computes concentration/adjacent-token-overlap against the true uniform-random baseline for the model's
  actual num_experts/top_k, not an arbitrary fixed threshold). Ran the full 13-question bench suite against
  both real production models on mira-mlx:
  - **Qwen3.6-35B-A3B** (256 experts, top-8): mean top-20% concentration 49.8% (2.5x uniform-random
    baseline of 19.9%), adjacent-token overlap 17.1% (10.8x baseline of 1.6%). 243,960 records, 40 layers.
  - **Gemma4-26B-A4B** (128 experts, top-8): mean concentration 72.5% (3.6x baseline of 20.3%), overlap
    25.4% (7.9x baseline of 3.2%) — an even stronger signal than Qwen3.6. 201,600 records, 30 layers.
  - **Conclusion**: both metrics show real, substantial skew/correlation well above the uniform-random null
    hypothesis on both models — GO for `specs/moe-expert-offload-02-runtime-cache.md` (fork-level
    SwitchGLU/SwitchMLP gather interception + disk-backed cold-expert store + `hardware.py` active-weight
    budget rework). The adjacent-token overlap result in particular (7.9-10.8x baseline) is a strong
    positive signal for "LLM in a Flash"-style windowing specifically, not just Eliseev & Mazur-style
    frequency-based caching. Raw logs (235MB/222MB JSONL, not committed) and bench results:
    `docs/bench-results-2026-07-18.md`, `scripts/bench_raw_2026-07-18_{qwen3.6,gemma4}-mira-mlx-expert-profile.jsonl`.
  - Profiling disabled again in `mira.yaml` after this run (`mira_mlx_profile_experts: false`) — this was a
    deliberate bench-driven window, not the multi-day live-traffic window the spec originally envisioned;
    re-enable for a longer real-usage window if a more representative sample is wanted before starting spec 02.
  Specs: `specs/moe-expert-offload-01-profiling.md` (done), `specs/moe-expert-offload-02-runtime-cache.md`
  (next — no longer blocked).
- [2026-07-19] MoE expert disk offloading (`specs/moe-expert-offload-02-runtime-cache.md`) — Phase A + B
  built and correctness-verified; **Phase C blocked on a real GPU-OOM crash found during live validation,
  not yet fixed.**
  - **Spike (before any implementation)**: byte-range `seek()+read()` against the live Qwen3.6-35B-A3B-4bit
    safetensors shards measured 0.3-0.6ms per expert slice — far under the tens-of-ms Eliseev & Mazur assume
    for their (network-attached-storage) setup, so Phase A skipped repacking entirely: the disk store reads
    straight from the model's existing shards.
  - **Phase A** (`~/Documents/Projects/mlx-lm`, `mira-core-pin` branch, commit `f0c66a4`, **not pushed**):
    added `enable_offload(resident_slots, fetch_fn)` to `SwitchLinear`/`QuantizedSwitchLinear`
    (`switch_layers.py`). First design used a fixed-size slot array with in-place row overwrites — found (via
    `tests/test_expert_offload.py`) a real bug: an eviction *later in the same forward call* could silently
    overwrite a slot an *earlier* expert in that same call had already resolved to, since the actual gather
    only runs after every index in the call resolves — corrupting output for that call. Rebuilt as a
    dict-keyed `expert_id -> weight` cache with a fresh temporary stacked tensor built per call — no shared
    slots, so this bug class can't occur, verified including the case where a single call's unique-expert
    count exceeds `resident_slots`. Verified bit-identical output vs. the unmodified eager path: synthetic
    `SwitchLinear`/`QuantizedSwitchLinear` unit tests, and a live end-to-end check against the real
    Qwen3.6-35B-A3B-4bit model (logits `mx.array_equal` on both a cold and warm cache pass).
  - **Phase B** (mira-core, uncommitted): `core/inference/disk_expert_cache.py` (byte-range reader, resolves
    each `(module_path, attr)` to its safetensors shard/offset once), `core/inference/expert_offload.py`
    (install hook, mirrors `expert_profiler.py`'s pattern; excludes `shared_expert`, which isn't
    per-expert-stacked so is never matched), `core/hardware.py` (`estimate_active_weight_bytes()` classifies
    on-disk bytes into per-expert-stacked vs. everything-else by safetensors header shape, threaded through
    `derive_prompt_cache_max_bytes`/`derive_context_window`/`fits_in_memory` — default/unset behavior
    unchanged, unit-tested), CLI/config/`backend_manager.py` wiring (`--resident-expert-fraction` /
    `mira_mlx_resident_expert_fraction`, same opt-in pattern as KV-quant), `/v1/stats` expert-cache hit-rate.
  - **Phase C attempt — found a real crash, not yet resolved**: enabled `mira_mlx_resident_expert_fraction:
    0.3` and restarted the live `com.mab.mira` service to bench it. The live backend crashed during its own
    startup warmup request (`Remote end closed connection without response`) — reproduced safely in an
    isolated standalone process (separate port, same payload) rather than continuing to poke at the live
    service: `libc++abi: terminating due to uncaught exception ... [METAL] Command buffer execution failed:
    Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` on a 1458-token prefill (the real
    system prompt + tool descriptions, not a toy prompt). **Root cause**: Qwen3.6 routes top-8-of-256 experts
    per token; a ~1458-token prefill makes ~11,664 expert selections in one batched forward call, so the
    *unique* experts touched per layer approaches all 256, not a small hot subset — spec 01's skew finding
    holds for steady-state/decode traffic but does not bound prefill-time diversity. At
    `resident_expert_fraction=0.3` (76 resident), a single large prefill needs to fetch and stack close to
    all 256 experts across all 40 layers, all queued into one lazy MLX graph before evaluation — that
    transient memory spike blew through the GPU's command-buffer budget. This is a correctness-preserving
    design (Phase A's guarantee held — no wrong output, a hard crash instead) but provides **no memory
    benefit during prefill**, since prefill's own expert diversity defeats the small-resident-set premise.
    Live service recovered immediately (`mira_mlx_resident_expert_fraction` unset, restarted, confirmed
    healthy) — outage window was contained to the debugging session.
  - **Decision (confirmed with user)**: stopped to document rather than rush a fix that session, then designed
    and implemented one next session (`~/Documents/Projects/mlx-lm` `mira-core-pin` commit `0458f97`). Traced
    through the "offload only during decode" idea the user proposed first and found it doesn't hold up on its
    own: for it to actually save RAM, the model can't stay fully resident during prefill, which means prefill
    needs the *same* bounded mechanism anyway — a phase toggle would just relocate the hard problem, not
    avoid it. Landed on **chunked/bounded resolve** instead: partition a call's unique experts into groups of
    at most `max_stack_size` (new `enable_offload()` param, defaults to `resident_slots`), one
    `gather_mm`/`gather_qmm` per group against the *entire* index tensor (out-of-group positions get a dummy
    index, discarded via `mx.where`), with a forced `mx.eval()` between groups so the previous group's
    temporary stack is actually released before the next is built — without that, MLX's laziness re-batches
    every group into one graph anyway and the bound is fiction. Fetches/evicts lazily per-group rather than
    resolving the whole call's expert set up front, which bounds the *cache dict itself* too, not just the
    final stacked tensor (the original design's dict cache ballooning to near-full-table size was as much a
    contributor to the crash as the stack). Costs more matmul work than a single unchunked call whenever a
    call needs more than one group (every group's call touches every position, not just its own) — that's the
    trade for a bound that holds regardless of call shape, with no prefill/decode signal needed from
    `BatchGenerator`/mira-core at all: decode's usual small unique-expert count still resolves in one group,
    unchanged cost from before.
  - **Found and fixed a real bug while implementing this**: `mx.gather_mm`/`gather_qmm`'s output has
    `indices.ndim + 2` dimensions, not `+ 1` as first assumed — true only for the `top_k=1` shape every prior
    test happened to use. The per-group `mx.where` mask needs padding to the output's *actual* rank (checked
    at runtime), not a fixed offset; got this wrong on the first pass and a same-shape-family test would not
    have caught it. Added `tests/test_expert_offload.py::test_offload_matches_baseline_with_realistic_top_k_routing`
    (`top_k=8` over 20 tokens, mirroring `SwitchGLU`'s real `(N, top_k)` indices shape) specifically because it
    caught this — confirmed the bug reproduces and the fix resolves it before moving on.
  - **Verified**: mira-core's test suite, including a new bounded-cache-size invariant test
    (`test_offload_bounds_cache_size_during_large_diverse_call` — instruments `fetch_fn` to assert the cache
    dict never exceeds `capacity + max_stack_size` entries during a worst-case call touching all 64 of a
    64-expert test model). Then re-ran the *exact* crashing scenario standalone (separate port, not the live
    service): same real system-prompt payload (~1458 tokens), `resident_expert_fraction=0.3` — **200 OK, no
    crash, no orphaned process** (previously: hard Metal OOM abort). Resident RSS was ~2.3GB (vs. Qwen3.6's
    normal ~20GB) confirming the memory saving is real at rest; TTFT on that first large/diverse prefill was
    ~42s (chunking's matmul-overhead cost, expected — this prompt forces ~76 chunks per module across 120
    modules), a short decode-heavy follow-up completed in ~13s. Live `com.mab.mira` service was never touched
    during this verification pass.
  - **Not yet done (at the time)**: an actual throughput/TTFT tradeoff table across a range of
    `resident_expert_fraction` values, and Eliseev & Mazur's gate-score-priority prefetch idea. Fork commit
    (`0458f97`, on `f0c66a4`) local/unpushed; mira-core Phase A/B changes uncommitted.
  Spec: `specs/moe-expert-offload-02-runtime-cache.md`.

- [2026-07-20] MoE expert disk offloading — continued Phase C validation. Ran the RAM/TTFT tradeoff table
  (standalone servers, not the live `com.mab.mira` service — user later gave blanket permission to disturb
  production for this work too, noted for future sessions) across `resident_expert_fraction` ∈
  {none, 0.15, 0.3, 0.5} against Qwen3.6-35B-A3B, and found two more real bugs plus one still-open design gap.
  - **`ps` RSS doesn't measure MLX's Metal-backed unified memory on macOS** — a live process hosting the fully
    loaded 20GB model showed 11MB RSS. Switched to `/v1/stats`'s `active_memory_bytes`/`peak_memory_bytes`
    (from `mx.get_active_memory()`/`mx.get_peak_memory()`), which do reflect it correctly (baseline: 19.25GB
    peak, matching the model's real size).
  - **Traced "offload only during decode" (the idea considered as an alternative fix the prior session)
    through fully and confirmed it doesn't work standalone**: `BatchGenerator._next()`
    (`mlx_lm/generate.py:1803-1879`) already separates prefill (`self._prompt_batch.prompt(prompts)`) and
    decode (`self._generation_batch.next()`) into distinct calls, so there's no per-token phase-mixing risk —
    but for the idea to save any RAM, the model can't stay fully resident during prefill either, which means
    prefill needs the *same* bounded mechanism anyway. A phase toggle relocates the hard problem rather than
    avoiding it; chunked/bounded resolve (already built) subsumes it.
  - **Bug 1 — sequential fetch was likely the dominant cost, not chunking's extra matmul work**: the crashing
    ~1458-token prefill from the prior session had 67,110 cache misses in *one call*; at ~0.3-0.6ms per
    sequential disk read that alone is 20-40s, matching the measured TTFT almost exactly. Parallelized the
    fetch loop in `_offload_chunked_gather` across a shared 8-worker `ThreadPoolExecutor`
    (`mlx-lm` `mira-core-pin` commit `dce6935`, paired with the fix below).
  - **Bug 2 (found by that change) — thread-affinity crash**: the parallelized fetch constructed `mx.array`
    objects on `ThreadPoolExecutor` worker threads. MLX streams are thread-local, and mira-mlx's engine pins
    model execution to one dedicated thread (a fix from earlier project history, `feedback_launchagent_reload`-
    adjacent), so the array crashed the instant it was used there: `RuntimeError: There is no Stream(gpu, N)
    in current thread`. **mira-core's pytest suite (48/48 passing across 8 repeated stress runs) did not catch
    this** — it only exercises the offload code in a plain script, never mira-mlx's pinned-thread architecture;
    only a real standalone server process reproduced it (same lesson as the original OOM: pytest correctness
    tests and real-server behavior tests catch genuinely different bug classes here). Fixed by splitting disk
    I/O + numpy parsing (thread-safe, done in parallel) from `mx.array` construction (must run on the calling
    thread, done after the parallel fetch via new `_offload_to_mx()`/`_offload_fetched_to_data()`) —
    `core/inference/disk_expert_cache.py`'s `fetch_fn` now returns raw `(np.ndarray, dtype_str)` instead of
    `mx.array`, since the fork must not import mira-core (wrong dependency direction) so the conversion logic
    now lives in both places by necessity (small, ~5-line duplication).
  - **Bug 3 — the chunk-size bound scaled *up* with the very setting meant to be safer**: `max_stack_size`
    defaulted to `resident_slots`, so at `resident_expert_fraction=0.5` (`resident_slots=128` on 256 experts),
    chunks of ~128 reproduced the *original* Metal OOM the chunking fix was supposed to prevent — a bound that
    grows with the setting isn't a bound. Fixed with a fixed `_OFFLOAD_DEFAULT_MAX_STACK = 64`, independent of
    `resident_slots` (still overridable per-call).
  - **Verified against a real standalone server (not just pytest) after both fixes**: 0.15 and 0.3 now
    complete the exact ~1458-token crashing prompt correctly; 0.3 dropped from 38.6s → 27.5s (confirms the
    concurrency fix is doing real work, not just correctness). 0.5 still crashed *at this point* — both that
    and the peak-memory gap below were resolved by the 2026-07-19 seed-view fix (see the RESOLVED bullet).
  - **RESOLVED (2026-07-19 follow-up) — peak memory now below baseline at every fraction**: the gap had a
    single root cause, and it was not "three overlapping copies" as first theorized — it was one wrong
    assumption. `module.weight[:resident_slots]` is an mx **view** that pins the *entire* parent buffer, so
    `module.weight = seed_weight` never freed the other experts; the whole table stayed resident and offload
    only piled cache + temp-stack memory on top of a still-full model. The "~0GB active right after load"
    reading that had suggested the table WAS being freed was a lazy-mmap artifact (safetensors weights don't
    fault into active memory until first touched — the short/long requests are what materialized them). Caught
    it by measuring MLX slice semantics directly *before* editing (applying the session's own "verify, don't
    assume" lesson): allocate (256,1024,1024) f32, slice `[:32]`, drop the parent → `mx.get_active_memory()`
    freed 0 bytes until the slice itself was also dropped. Fix (`mira-core-pin` commit `5378ffb`): seed the
    resident set from disk (same path a cold miss uses → genuinely independent buffers), replace
    weight/scales/biases with a 1-row stand-in (view of one now-resident expert, so `input_dims`/`output_dims`
    still resolve and the param tree stays intact) which drops the last reference to the full tensors, and
    stash the true expert count on the module since `num_experts` can no longer read it from `weight.shape[0]`.
    Compute path unchanged (offload `__call__` reads only the cache), so the unit suite stays bit-identical —
    7/7, added `test_offload_frees_full_weight_and_preserves_num_experts`, 5x no flakiness.

    Re-measured on a fresh real-model server (fork synced into the venv, restored after — live `com.mab.mira`
    disturbed freely per the standing permission):

    | fraction | peak GB (worst-case prefill) | was | steady-state resident GB | cold ~1458-tok prefill |
    |---|---|---|---|---|
    | none (baseline) | 19.25 | 19.25 | 18.31 | 1.8s |
    | 0.15 | 18.19 | 21.39 | 4.01 | 28.8s |
    | 0.3 | 18.21 | 23.99 | 6.58 | 21.8s |
    | 0.5 | 18.24 | crashed | 9.94 | 22.3s |

    Peak is now fraction-independent and *below* baseline (incl. 0.5, which no longer OOMs — this also closes
    the "re-verify 0.5 against the fully-fixed code" loose end). The worst-case diverse prefill is
    activation-bound, not expert-bound, so the resident fraction sets the *steady-state* floor (the real RAM
    win — now genuinely delivered: 4GB at 0.15 vs 18.3GB full) but not the transient peak. Output coherent and
    identical across all configs. Cost: cold-prefill latency (disk-bound), decode unaffected once warm.
  - **Scope note (2026-07-19) — CORRECTED by the step-0 probe below.** This bullet originally claimed the
    18.2GB peak was *activation*-bound (so only token-block prefill / compacted gather could lower it) and
    that `mlx_lm.load` is lazy with "active ~0 right after load." **Both wrong.** The peak is 100% load-time
    eager materialization of the full expert table (`lazy=False` → `mx.eval(model.parameters())`), and it was
    fixed outright by `load(..., lazy=True)` gated on offload (peak 18.21→7.25GB, table never wires). The
    token-block/compacted-gather "next steps" are moot. See the step-0 bullet below for the real mechanism.
  - **Now ON by default (2026-07-19)** at fraction 0.3, via a new `mira_mlx_expert_offload` flag (config
    default `true`; set `false` to fall back to the simpler fully-resident lazy-mmap path — the pre-offload
    behavior). Flag + fraction resolve in `core/config.py::_resolve_resident_fraction` into the single
    effective `MIRA_MLX_RESIDENT_EXPERT_FRACTION` the rest of the stack already gated on, so no downstream
    wiring changed. Live-confirmed on production after restart: `--resident-expert-fraction 0.3` on the
    subprocess, `/v1/stats` expert_cache hit_rate ~0.37, **active memory 6.58GB (down from 18.3GB — ~12GB
    freed)**, coherent output. Chosen for the steady-state RAM freeing even though the model fits at baseline;
    accepted cost is slower cold-prefill TTFT (short first message ~2.6s→~8s, decode + peak unchanged). The
    earlier "keep ≤0.2" restriction is moot — any fraction is safe; 0.15–0.3 is the sensible range.
  - **Prefill-peak diagnostic (2026-07-19) — SUPERSEDED by the step-0 probe below; kept for the trail.**
    Asked to evaluate 3 fixes for the peak; adversarial review (3 agents) refuted all three as targeting <1%
    of it (token-block prefill 0.3-0.5%, compacted gather 0.13% + a silent sort-order corruption trap below
    fraction 0.25, full weight-streaming rejected — non-expert is only 2.1GB and dense 100%/token → ~5 tok/s
    ceiling). Measured on a fresh server (offload 0.3): peak 18.21GB, FLAT across prefill_step_size and
    kv_bits. From those *server* readings I concluded the peak was a one-time FIRST-FORWARD transient (MLX
    faulting the full table in on first run). **That conclusion was wrong** — see the step-0 correction.
  - **Step-0 load-path probe (2026-07-19) — the peak is LOAD-TIME EAGER MATERIALIZATION, and FIXED.**
    A standalone probe with per-phase `reset_peak()` (which the live server never does — it never resets
    `mx.get_peak_memory()`, so the load high-water mark was still showing at first-request time and *looked*
    request-caused) settled the mechanism decisively. Trajectory at fraction 0.3, lazy=False (current):
    after `load()` **active/peak=18.17GB** (not ~0 — the earlier "active~0 after load" was read *after*
    install's stand-in swap had already freed it); after `install()` active 6.37 / peak 18.21, freed bytes to
    MLX's buffer *pool* (`cache`=16.88GB, released by `clear_cache`); every forward bounded (tiny 6.51,
    diverse ~1458 prefill 7.48). **No first-forward fault exists.** `mlx_lm.load` with `lazy=False` calls
    `mx.eval(model.parameters())` (utils.py:418) which wires the full stacked `(num_experts,…)` table at load.
    **Fix (shipped):** `load(..., lazy=True)` in `mira_mlx_server._run`, gated on
    `resident_expert_fraction is not None`. Under lazy=True `sanitize` still builds the stacked tensors but
    they stay unevaluated (0 wired bytes); `install()`'s stand-in swap drops them before any eval → the full
    table never materializes at any point. Measured: standalone whole-run peak **18.21→7.48GB**; real server
    `/v1/stats` peak **18.21→7.25GB** (pre-request 18.21→**0.0**), output bit-coherent, tests 12/12. The
    originally-scoped "filter expert keys" and "offload-native `__init__`" components proved unnecessary.
    Peak now scales with resident fraction, not table size → a model with an over-DRAM expert table can open
    (synthetic-checkpoint demonstration still TODO — the only claim resting on inference not measurement).
    Docs: `docs/moe-offload-lazy-load-design.md` (fix), `docs/moe-offload-case-study.md` §6b (the correction).
    Model is Qwen3.5 hybrid (`qwen3_5.py`: GatedDeltaNet linear-attn + full-attn every 4th layer +
    `qwen3_next` SparseMoeBlock), not `qwen3_moe`. Diagnostic scripts in the job scratch dir; earlier plan at
    `~/.claude/plans/resilient-sprouting-owl.md` (its 3 peak-fix approaches are all moot now).
  - **Over-DRAM demonstrated + throughput benched + offload made per-model AUTO (2026-07-19).**
    Downloaded `Qwen3.6-35B-A3B-8bit` (35GB, ~33.8GB expert table > 32GB RAM). Ran it via
    `load(lazy=True)`+offload(0.3) at **peak 12.71GB**, coherent output — a model that cannot load
    eagerly on this hardware (`scripts/moe_overdram_demo.py`, case study §9). Then benched throughput
    (`scripts/moe_throughput_bench.py`, fresh backend per config, 3022-tok prefill / 200-tok decode):
    | config | prefill cold/warm t/s | decode t/s | peak GB | hit |
    |---|---|---|---|---|
    | 4bit offload-OFF | 616 / 957 | **57.1** | 19.31 | — |
    | 4bit offload-ON 0.3 | 77 / 76 | **10.8** | 7.31 | 0.51 |
    | 8bit offload-ON 0.3 | 59 / 56 | **6.6** | 13.05 | 0.51 |
    **Finding: offload has a large throughput cost** (~5x decode, ~8-12x prefill on 4bit), NOT "decode
    unchanged" as the earlier note claimed — a diverse prefill touches ~all 256 experts but 0.3
    resident can't hold that set, so warm ~= cold and decode pays a per-token miss tax (hit_rate 0.51).
    **So offload is now per-model AUTO** (`mira_mlx_expert_offload: auto`, `config.resolve_offload_fraction`):
    enabled only when `hardware.fits_in_memory(resident_expert_fraction=None)` says the fully-resident
    model won't fit. The 4bit (fits at 19.3GB) now runs offload-OFF at full speed (live-verified: tiny
    prefill 0.61s vs ~40s offload-on, expert_cache None, active 18.25GB); the 8bit (over-DRAM) auto-
    enables offload. `on`/`off` still force it. Reverses the 2026-07-19 "default-on" decision on the
    strength of the throughput data (per [[feedback_confirm_before_reversing_decisions]], surfaced to
    and chosen by the user). Config `core/config.py` (tri-state mode + resolver, 11 config tests),
    launch decision `core/backend_manager.start_mira_mlx`. NOTE (separate gap found): the `/chat`
    orchestrator can't report tok/s because the mira-mlx backend doesn't emit
    `prompt_eval_count`/`eval_count` on its chunks, so `orchestrator` token counters + `context_pct`
    stay 0 — throughput must be timed against the backend (hence `scripts/moe_throughput_bench.py`,
    not the 13-question `/chat` bench). Worth fixing so the app's context% works.
  - Fork commits (all local, unpushed, on `mira-core-pin`): `0458f97` (chunked/bounded resolve), `dce6935`
    (concurrent fetch + thread-affinity fix + chunk-cap-independent-of-fraction fix), `5378ffb` (seed-view
    fix — frees the full table). mira-core changes (`disk_expert_cache.py`'s raw-data contract change, plus all
    of Phase A/B) remain uncommitted. **To make it live**: push `mira-core-pin`, bump the pin in
    `pyproject.toml` (or reinstall mlx-lm) — the offload code lives only in these unpushed commits, so
    `resident_expert_fraction` in `mira.yaml` would currently call an `enable_offload` the installed mlx-lm
    lacks.
  Spec: `specs/moe-expert-offload-02-runtime-cache.md` updated with full current status.
- [2026-07-18] Fixed two `core/backend_manager.py`/`server.py` bugs found while switching mira-mlx between
  models during the expert-offloading profiling run above: (1) `ensure_backend_running("mira-mlx")` always
  called `start_mira_mlx()` with no model argument, silently defaulting to its own hardcoded `MIRA_MLX_MODEL`
  constant (Ministral) instead of `core.config.MODEL_NAME` (mira.yaml's configured Qwen3.6) — every
  `server.py` cold start was silently ignoring the configured model. Fixed: `ensure_backend_running` now
  accepts an explicit `model` param; `server.py`'s startup thread passes `MODEL_NAME`. (2) `start_mira_mlx()`'s
  internal `_wait_for_ready` timeout (120s) was shorter than `GenerationEngine.start()`'s own internal
  readiness budget (180s) — a legitimately-slow-but-successful cold load (large model, memory pressure right
  after killing a previous backend) could raise `TimeoutError` in `switch_to_model()` while the process kept
  loading in the background and became healthy moments later, leaving `server.py`'s `_rt` state permanently
  stale (reported the old model while the new one was actually live and serving). Fixed: bumped the timeout
  to 200s, and added `_reconcile_stale_switch_failure()` in `server.py` — on any switch failure, briefly poll
  whether the target backend actually came up before giving up, so a switch that succeeds late is never
  permanently misreported as failed. Reproduced live (Qwen3.6→Gemma4 switch under memory pressure), fixed,
  and verified end-to-end (clean restart correctly booted Qwen3.6 per `mira.yaml`, `_rt`/`/backend` correctly
  reflected reality, no more `--profile-experts` on the process args after disabling it).
- [2026-07-18] KV-cache quantization, isolated timing re-bench (resolves the Phase C timing caveat above): re-ran the 13-question suite four times with no other apps/processes running (Safari helper processes only, idle CPU) — Ministral 3 14B and Qwen3.6-35B-A3B, each once with `kv_bits=8` and once fully unquantized (temporarily unset `mira_mlx_kv_bits` in `mira.yaml`, confirmed via the running subprocess's args each time, restored to `kv_bits=8` afterward). Correctness held again in all four runs (same tool-call/`task_done` pattern as every prior run).
  - **TTFT (first-token latency) is the cleanest metric** — unaffected by output length or agentic tool-call-round-trip count. On the five non-agentic questions (Q1–Q5, most directly comparable across runs), the quantized-vs-unquantized delta was within ±30ms for both models — indistinguishable from run-to-run noise. **Conclusion: KV-cache quantization has no measurable prefill/TTFT cost.**
  - **Wall-clock totals are not a clean signal for this bench harness**: agentic questions (Q6–Q13) have a variable number of tool-call round trips run-to-run even with identical prompts (e.g. Ministral Q6 was 1 tool call in one run, 2 in another) — each extra round trip adds a full decode+prefill cycle to wall time, swamping any quantization-specific effect. The multi-turn long-context question (Q10, the one exercising the largest KV cache in the suite) showed the biggest swings, but in **opposite directions per model** (Ministral: quantized 222s vs unquantized 119s, ~1.9x slower; Qwen3.6: quantized 67s vs unquantized 98s, ~1.5x *faster*) — a real per-token compute cost from quantization would show the same direction on both models, so this is more consistent with cache-hit/reuse variance across the four separate server restarts (fresh in-memory cache each time) than with quantization overhead.
  - **Net takeaway**: no evidence of a real speed cost from enabling `kv_bits=8`. `mira_mlx_kv_bits: 8` stays the local default. Raw results: `scripts/bench_raw_2026-07-18_*-isolated.jsonl`; markdown tables appended to `docs/bench-results-2026-07-18.md`.

- [2026-07-18] KV-cache quantization, Phase A + B + C (all phases complete — `mira_mlx_kv_bits: 8` now the local default in `mira.yaml`): Phase A discovered the real blocker was one level deeper than first thought — mira-core's `start_mira_mlx()` always passes `--max-kv-size`, so `BatchGenerator` always wraps caches in `RotatingKVCache`, and on merge that becomes `BatchRotatingKVCache` — the actual class `update_and_fetch()` runs on during real batched decode steps. Both `RotatingKVCache.to_quantized()` and `BatchRotatingKVCache.to_quantized()` were `NotImplementedError` stubs upstream. Implemented `RotatingQuantizedKVCache` (per-job) and `BatchRotatingQuantizedKVCache` (batch-merged, ~350 new lines mirroring `BatchRotatingKVCache`'s full surface — `_update_in_place`/`_update_concat`/`filter`/`extend`/`extract`/`merge`/`state`/`meta_state`) in the pinned `mabaeyens/mlx-lm` fork (`mira-mistral-tool-call-fix` branch, commit `bbd8496`, **pushed** 2026-07-18). `BatchGenerator` gained `kv_bits`/`kv_group_size`/`quantized_kv_start` params — quantization happens once at cache creation (offset=0, no precision lost); `quantized_kv_start > 0` is explicitly rejected for the batching path. Fork validated: full test suite unaffected (same 10 pre-existing, unrelated failures before/after); 6 new tests (rotation+quantization accuracy, mid-stream `to_quantized()`, save/load round-trip, full merge/filter/extend/extract, end-to-end `BatchGenerator` run forcing real rotation on `mlx-community/Qwen1.5-0.5B-Chat-4bit`). Phase B wired mira-core's lockfile to the pushed fork commit (`uv lock --upgrade-package mlx-lm && uv sync`) and threaded `kv_bits`/`kv_group_size`/`quantized_kv_start` end-to-end: `core/inference/mira_mlx_server.py` (`--kv-bits`/`--kv-group-size`/`--quantized-kv-start` CLI args → `GenerationEngine` → `BatchGenerator`), `core/config.py` (`MIRA_MLX_KV_BITS`/`MIRA_MLX_KV_GROUP_SIZE`, both `None`/`64` default = today's behavior), `core/backend_manager.py` (`start_mira_mlx()` only appends `--kv-bits`/`--kv-group-size` when `MIRA_MLX_KV_BITS` is set; also threaded into `fits_in_memory`/`derive_context_window`/the model-switch preset reporting), `core/hardware.py` (`estimate_kv_bytes_per_token` gained optional `kv_bits`/`kv_group_size`, formula `bits/8 + (2*KV_DTYPE_BYTES)/group_size` bytes/element, mirroring `QuantizedKVCache`'s own `mx.quantize` packing), `core/inference/disk_prompt_cache.py` (`_key()` folds `kv_bits`/`kv_group_size` into the hash so a config change or restart can't load a quantized cache entry as unquantized or vice versa), `mira.yaml.example` (documented `mira_mlx_kv_bits`/`mira_mlx_kv_group_size`). 12 new unit tests (hardware.py kv_bits math incl. default-unchanged regression, disk-cache key-collision regression, CLI/config threading). Live end-to-end smoke test: real server run with `--kv-bits 8 --kv-group-size 32 --max-kv-size 256`, sent a 400-token completion that pushed the cache past 256 tokens (forcing real rotation+quantization eviction), got a clean `finish_reason=length` response — not just unit-tested in isolation.

  Phase C (2026-07-18) validated all of the above against real production models with `mira_mlx_kv_bits: 8`/`mira_mlx_kv_group_size: 64` enabled via `mira.yaml` (confirmed the running mira-mlx subprocess picked up `--kv-bits 8 --kv-group-size 64` on both a fresh start and a live `/models/switch`). Ran the full 13-question bench suite (`scripts/bench_compare.py`) on both **Ministral 3 14B** and **Qwen3.6-35B-A3B**, both on mira-mlx:
  - **No regression**: every agentic question's tool-call sequence and `task_done` result matched (or, for Qwen3.6 Q13, slightly *improved* on) the 2026-07-10 unquantized baseline in `docs/bench-results-2026-07-10.md` — same `run_shell`/`write_file`/`read_file`/`edit_file` call patterns, same divergence-guard firing behavior. Spot-checked every raw response (`scripts/bench_raw_2026-07-18_*.jsonl`) for coherence: no NaN, no garbled/corrupted tokens, no truncation artifacts on either model — code generation, file read/write/edit tasks, and long-form reasoning all read as clean, correct output. Full results: `docs/bench-results-2026-07-18.md`.
  - **Timing numbers from this run are not reliable for a TTFT/throughput comparison** — the user was concurrently running MLX model conversions on the same machine during both bench sweeps, so wall-clock/TTFT figures in `docs/bench-results-2026-07-18.md` reflect contention, not the isolated cost of quantization. Correctness (not speed) is what Phase C set out to confirm.
  - **RAM/context-window win measured directly** via `hardware.derive_context_window()`: on this 32GB machine, at the current `mira.yaml` `requested_context: 65536` the RAM ceiling doesn't bind for either model even unquantized, so no visible change at today's default. Raising `requested_context` past the point where it binds shows the real effect — confirmed 1.88x KV-cache compression for both models (Ministral 3 14B: 163,840 → 87,040 bytes/token; Qwen3.6-35B-A3B: 81,920 → 43,520 bytes/token), which lifts the RAM-derived context ceiling from ~131K unquantized to ~247K-261K tokens quantized on this hardware. The win is real but only shows up once someone raises `requested_context` beyond ~131-138K tokens, or on a lower-RAM machine where the ceiling already binds at the default.
  - **Real production-model rotation test** (not just the Phase A/B smoke test's tiny `Qwen1.5-0.5B-Chat-4bit`): ran `mira_mlx_server.py` standalone against **Ministral 3 14B** with `--max-kv-size 800 --kv-bits 8 --kv-group-size 64`, sent a 567-token prompt + asked for 1200 generated tokens (1767 total, well past the 800-token rotation boundary) — clean `finish_reason=length`, no NaN/corruption, coherent output throughout.
  - Decision (confirmed with user): `mira_mlx_kv_bits: 8` stays enabled as the local `mira.yaml` default going forward, rather than reverting to unset after validation.
  - Not covered by this pass: an isolated (no concurrent load) timing re-bench, and a rotation test on the full-size Qwen3.6-35B-A3B specifically (the rotation test used Ministral 3 14B for faster iteration — the underlying cache code path is identical for both models, and Qwen3.6 was already exercised at the default 65536 `max_kv_size` during its own bench run without issue).

  Spec: `specs/mira-mlx-kv-quant.md` (all three phases marked done).
- [2026-07-18] mira-mlx has no real vision support (mlx-lm's `BatchGenerator` has no image-tensor seam — VLM code only reachable via the non-batched `generate`/`stream_generate` path), so instead of just rejecting images, `orchestrator.py` now runs them through OCR (`file_handler.ocr_image_from_base64()`, reusing the same optional `tesseract` binary already used for scanned-PDF OCR) and folds recovered text into the prompt as regular text — handles the common case (screenshots of error dialogs, menus, terminal output) without real vision. `mira_mlx_server.py::_prepare_messages()` still raises a `ValueError` (→ 400) as a backstop if a raw image somehow reaches it. When OCR is unavailable or finds no text (photos, diagrams), the user gets a clear inline error pointing at omlx or `brew install tesseract`. `backend_manager.PRESETS[backend]["vision"]` is the single source of truth for which backends need this fallback (only `omlx` is `True`). Full write-up: `docs/architecture.md` "Model quirks". Tests: `tests/test_file_handler.py` (`ocr_image_from_base64`), `tests/test_queries.py` (OCR-success and no-text-found paths).
- [2026-07-10] Fixed Qwen3.6 tool calls firing 0/7 on mira-mlx (Ministral got 6/7, omlx's Qwen3.6 got 7/7 — so this was mira-mlx-specific, not a Qwen3.6 capability gap). Three stacked bugs: `core/orchestrator.py`'s Qwen3-thinking backend allow-list omitted `mira-mlx`; mira-mlx's HTTP handler silently dropped `chat_template_kwargs`; the tool-text buffering fallback (needed for Mistral's one-sided marker) also captured Qwen's closing `</tool_call>` marker, corrupting the parser. Fixed all three; re-benched 7/7 on Qwen3.6, no regression on Ministral. Full writeup: `docs/architecture.md` "Model quirks", `docs/bench-results-2026-07-10.md`.
- [2026-07-10] mira-mlx Apple Silicon tuning: live MLX memory visibility (`active_memory_bytes`/`cache_memory_bytes`/`peak_memory_bytes`/`wired_limit_bytes` in `/v1/stats`), proactive Metal cache-limit tuning (`core/hardware.py:derive_cache_limit_bytes`), M5/NAX startup smoke-check logging, `--max-tokens` reconciled against the RAM-derived context window. Confirmed live: `mlx-lm`'s `BatchGenerator` already sets a proactive wired-memory limit internally — this part is correct. **Correction (2026-07-18, adversarial review):** the accompanying claim that "KV-cache quantization isn't reachable ... ruled out ... after adversarial review" is wrong — `specs/mira-mlx-apple-silicon-tuning.md` (the cited spec) never actually discusses KV-cache quantization, so no such review happened here. `maybe_quantize_kv_cache`/`QuantizedKVCache` are complete, tested primitives in the pinned `mabaeyens/mlx-lm` fork, already used by its non-batched `generate`/`stream_generate` path — just not threaded into `BatchGenerator`. See the Pending entry below, which is the accurate status.
- [2026-07-09] mira-mlx (`core/inference/mira_mlx_server.py`) promoted to the default backend in `mira.yaml`, replacing omlx as default — RAM-aware sizing, oversized-prompt handling, disk-backed prompt cache, and `/v1/stats` were implemented and live-verified first. omlx remains fully supported as an alternative backend. Full 32GB-M5 bench matrix (Qwen3.6-35B-A3B + Ministral-3-14B, throughput + 13-question quality/agentic suite) run 2026-07-10 — see `docs/model-comparison-m5-macbook.md` "Current Verdict".
- [2026-07-07] Fixed vllm-mlx + Ministral 3 14B end-to-end in Mira — full 13-question bench now passes (previously blocked on every final-turn/agentic-tool-call response). Three stacked bugs found and fixed:
  1. vllm-mlx issue #628 (natural-EOS `finish_reason` never stamped in `_stream_generate_impl`) — applied the upstream reporter's verified fix as a local commit on the existing fork branch.
  2. An untested sibling gap in the same file's system-KV-cache path (drives `mlx_lm.stream_generate` directly, same missing `finish_reason` on natural stop) — fixed locally, not yet upstream.
  3. Mira's own bug: `core/orchestrator.py`'s agent-loop step-budget nudge was appended as a bare `user`-role message, which violates Mistral's strict template rule requiring user/assistant alternation around tool calls. New `_append_step_nudge()` helper folds the nudge into the trailing `tool` message's content instead. Verified no regression on omlx/Qwen3.6 (7 nudge-triggering questions re-benched, divergence guard still fires).
  Also found and fixed a real latent bug in `core/backend_manager.py`'s `switch_to_model()`: each backend branch stopped every *other* Popen-managed backend but never its own prior process first, so switching to the same backend with a different model could orphan the running process and silently misreport which model was actually active. Fixed by adding each backend's own stop call before restarting it.
  Full details: memory `project_vllm_mlx_integration.md`, `feedback_backend_switch_self_stop.md`.
- [2026-07-06] Released v0.9.1 (docs-only: inference tuning results write-up); tag + GitHub release published.
- [2026-07-06] Stopped attaching a built wheel to GitHub releases; **reverted 2026-07-08** — the wheel-on-release decision stands (see `docs/packaging.md` §6). Restored `uv build --wheel` / `gh release upload` in the `core-release` skill (`~/.claude/skills/core-release/SKILL.md`). Backfilled `mira_core-0.9.1-py3-none-any.whl` onto the v0.9.1 GitHub release (built at the tag in an isolated worktree); v0.9.2 onward will have it again via the skill.
- [2026-07-06] Designed and adversarially reviewed a "fully automated installer" concept (no manual GUI steps). Decision: keep status quo — the one manual step (installing `oMLX.app` and loading `Qwen3.6-35B-A3B` via its in-app model library) is an Apple Gatekeeper/TCC security boundary, not a scripting gap. Rejected scripting around it (quarantine-stripping, driving permission dialogs, reverse-engineering oMLX's undocumented model-store format) as a security-review-failing pattern. Full report: `~/.claude/plans/partitioned-tinkering-deer.md`.

## Pending
- The `~/Documents/Projects/vllm-mlx` fork (branch `fix/mistral-args-token-tool-parsing`) still has two commits not pushed/PR'd upstream: `d408d70` (finish_reason=stop on natural EOS, mirrors upstream PR #629 for issue #628) and `edc07d4` (sibling KV-cache-path gap, flagged as untested by the #629 author — worth submitting as its own PR once #629 merges, or bundled with the existing #631 parser-fix PR). (The rest of the 2026-07-07 vllm-mlx pending list — orchestrator/backend_manager/config/server.py changes, bench-results docs — has since landed; see "Done" above.)
- KV-cache quantization for mira-mlx: **shipped and live since 2026-07-18** (`mira_mlx_kv_bits: 8`), carried on the `mira-core-pin` fork branch. Mira is NOT blocked on the upstream merge; what is pending here is only the upstreaming itself, which reduces the patch-carrying burden. (This entry previously read "awaiting review/merge", which wrongly implied the feature was not available. Corrected 2026-07-26, along with the no-op bug that correction surfaced, see Done.) Awaiting review/merge of upstream PR [ml-explore/mlx-lm#1584](https://github.com/ml-explore/mlx-lm/pull/1584) (issue [#1583](https://github.com/ml-explore/mlx-lm/issues/1583)), opened 2026-07-18 from a clean rebase onto `upstream/main` — no conflicts, same pre-existing test failures on both sides, formatted with black/isort per CONTRIBUTING.md. **Update 2026-07-24: the "same pre-existing test failures" are now root-caused and cleared as a review blocker** — the 8 batched-attention failures are an M5 fused-kernel numerics quirk (see the 2026-07-24 Done entry and [mlx#3897](https://github.com/ml-explore/mlx/issues/3897)), not introduced by this PR and benign (argmax unchanged). PR now blocked only on maintainer attention. **Important side-effect discovered during the rebase**: rebasing `mira-mistral-tool-call-fix` picked up upstream's "Text-based state machine for tool/reasoning parsing" refactor (mlx-lm PR #1501), which renamed `SequenceStateMachine` → `TextStateMachine` with a different API — this broke `mira_mlx_server.py`'s import when the rebased commit was briefly live in mira-core's lockfile. Fixed by decoupling: mira-core's `pyproject.toml` now pins to a dedicated `mira-core-pin` branch (frozen at the pre-rebase commit `bbd8496`) instead of `mira-mistral-tool-call-fix` directly, so future PR-prep rebases on the latter can't silently break mira-core again — see the comment above `[tool.uv.sources]` in `pyproject.toml`. **Follow-up once #1584 merges**: check whether mlx-lm's own tool-call-flush fix is still needed given the #1501 refactor (our Mistral-specific patch may now be superseded upstream), and decide whether to migrate `mira_mlx_server.py`'s `SequenceStateMachine` usage to the new `TextStateMachine` API when finally moving off the `mira-core-pin` branch. The isolated timing re-bench (below) is done.
- Ease-of-install follow-ups considered but not started (all optional, low priority given "keep status quo" decision):
  - Retry/resume logic on `ollama pull` in `scripts/setup.sh` (reuse the 3×/10s pattern already in `scripts/prefetch_models.py`).
  - Auto-open the oMLX GitHub Releases page + `/Applications` in Finder during setup, to reduce the manual step to "open the .dmg, drag the icon." (Lower priority now that mira-mlx is the default backend and doesn't require this step at all — only relevant to users who opt into `backend: omlx`.)
  - A separate, explicitly opt-in **headless/non-interactive install mode** (e.g. `--headless`) using the `dflash-mlx` backend (fully HF-scriptable, no GUI) for automated/remote/CI provisioning only — NOT a change to the interactive default backend. Note: the original rationale (dflash's ~48s TTFT vs. omlx's near-0ms warm TTFT) predates mira-mlx's promotion to default; mira-mlx itself is also fully HF-scriptable (no GUI), so this may be moot — revisit whether a separate headless mode is even still needed. Would need: port-8080 conflict check between backends, `mira_cli.py` `COMPONENTS`/preflight disk-math update, `mira.yaml.example` default swap for that mode only.

- ~~**CI test deselects.**~~ **DONE 2026-07-26, all 9 repaired and the deselect list removed entirely.** See the Done entry at the top.
  - **Separate, and already fixed — do NOT add these to the deselect list.** The `run_shell` OS sandbox (`5993dec`) uses macOS `sandbox-exec`, which is absent on ubuntu runners, so with `SHELL_SANDBOX` defaulting on it failed closed and broke four *more* execution tests — `test_run_shell_basic_command`, `test_run_shell_non_zero_exit_code`, `test_run_shell_allows_glob_exclusion_pattern`, `test_model_cannot_self_approve_destructive_command` (red on every push from `5993dec` until `edcbf00`; they pass locally on macOS where the sandbox exists, which is why the original deselect list — built from macOS-local failures — missed them). Fixed by making the CI provision step write `shell_sandbox: false` (`edcbf00`) so they exercise real command execution unsandboxed and keep their coverage; the sandbox path itself stays covered by `test_shell_sandbox.py`, which `skipif`s off macOS. Deselecting them would have silently dropped `run_shell` execution coverage on CI — the config approach is deliberate.

## Notes
- `uv tool install --force <local-path>` can silently reuse a cached build and NOT pick up new local commits, even though the command reports success — confirmed by checking the installed file's actual content after a "successful" reinstall still showed the pre-fix code. Full fix requires `uv tool uninstall <pkg> && uv cache clean <pkg>` before reinstalling from a local path with uncommitted-upstream changes; `--force` alone is not sufficient.
- All local backends (mira-mlx/omlx/dflash/mlx-lm/vllm-mlx) share port 8080 — only one runs at a time by design. This means a stale/orphaned process from a previous backend can make `is_backend_ready()`'s health check pass spuriously (it just hits `/v1/models` on the shared port), masking a failed switch. See `feedback_backend_switch_self_stop.md` for the specific bug this caused.
- `pyproject.toml` stays tag-driven via `hatch-vcs` — the git tag IS the version, never hand-edit. `mira.yaml` is gitignored runtime config, not tracked (confirmed 2026-07-06, no leaked secrets — a local `mira.yaml` with a real token exists only on disk, never committed).
- Installer already automates nearly everything (uv, Python deps incl. mlx stack, disk/RAM preflight, mira.yaml bootstrap, optional ollama/tesseract/LaunchAgent via brew, doctor health check). The oMLX GUI step is the sole exception and is expected to stay manual indefinitely absent an oMLX-side scriptable install/model API.
- Tried `MLX_METAL_FAST_SYNCH=1` + `MLX_MAX_OPS_PER_BUFFER=50` + `MLX_MAX_MB_PER_BUFFER=50` (undocumented MLX Metal-backend env vars, found via `strings` on `libmlx.dylib` and confirmed against MLX's C++ source — `mlx/backend/metal/device.cpp`/`fence.cpp`) on mira-mlx (2026-07-18): `subprocess.Popen()` in `backend_manager.py` inherits the parent env with no override, so setting these on `server.py`'s own process before launch propagates to the mira-mlx subprocess. A/B benched against a plain baseline on **both** models mira-mlx runs — Ministral 3 14B and Qwen3.6-35B-A3B, isolated — **no measurable TTFT difference on either** (within ±20ms noise on the 5 non-agentic questions for both models; Qwen3.6's Q1 showed an ~800ms gap but that's within normal first-request-after-switch variance, not a consistent effect across Q2-Q5). Apparent wins on agentic questions were fully explained by tool-call-count variance between runs, not the env vars, on both models. Correctness unaffected either way on both. Not worth setting as a default; not pursued further.
