# Running a 35B MoE on a 32GB Mac: MoE expert disk offloading

This is my write-up of building expert offloading for Mira, my local assistant. It runs on one
32GB Apple-silicon Mac (about 25GB Metal wired limit). The model is `Qwen3.6-35B-A3B-4bit` (256
experts, top-8 routing, roughly 16.9GB of expert weights on a 2.1GB dense backbone). The engine is
mira-mlx, my own MLX inference server, running on a pinned `mabaeyens/mlx-lm` fork.

I am writing it down partly because it is a clean example of a lesson I keep relearning: measure the
thing, do not reason about it. Twice in this project I was confidently wrong about where the memory
was going, and both times a twenty-line experiment settled it in a minute.

## 1. What I wanted

The 4-bit weights are about 19GB. They fit on 32GB, but with almost no headroom for KV cache,
context, or anything else on the machine. MoE models have a property I wanted to exploit: only
`top_k` of `num_experts` fire per token (here 8 of 256, about 3%), so most expert weights sit idle
most of the time. The idea was to keep only a fraction of experts resident in unified memory and
fetch the rest on demand, straight from the model's own safetensors shards by byte offset, with no
repacking and no separate cache format. I measured a per-expert read against the real checkpoint at
0.3 to 0.6ms, fast enough to make a disk-backed LRU cache practical.

## 2. The feature

I added `enable_offload()` to the shared `SwitchLinear` and `QuantizedSwitchLinear` primitives in the
fork: a dict-keyed LRU cache (`expert_id` to weight), seeded from the resident set, fetching cold
experts on a miss and evicting to a capacity bound. mira-core wires it up with a
`DiskExpertCacheStore` that resolves each expert's shard and offset once and serves byte-range reads,
RAM-aware sizing that budgets context and prompt-cache against the reduced footprint, and a
`/v1/stats` hit-rate. It is opt-in through one `mira.yaml` knob.

Simple in outline. The interesting part was everything that broke.

## 3. Five bugs, five lessons

**(a) Metal OOM on a big prefill.** A 1458-token prompt crashed with
`kIOGPUCommandBufferCallbackErrorOutOfMemory`. The cause is that a diverse prompt routes to nearly
every expert in one forward call (coupon-collector: 1024 selections over 256 experts covers about
98%), so the gather tried to stack almost the whole table at once. The fix was a chunked, bounded
resolve: partition a call's experts into groups of at most `max_stack`, one gather per group, with a
forced `mx.eval()` between groups so MLX actually releases each group before building the next. That
bound holds for any call shape, with no prefill/decode signal needed.

**(b) `ps` lies about Metal memory.** A process hosting the fully-loaded 20GB model showed 11MB RSS.
macOS `ps -o rss` does not account for MLX's Metal-backed unified memory. I switched to measuring
with `mx.get_active_memory()` and `mx.get_peak_memory()`, exposed through `/v1/stats`, and stopped
trusting RSS entirely.

**(c) The crash pytest could not see.** Parallelizing the disk fetches across a thread pool made cold
prefill much faster, and introduced `RuntimeError: There is no Stream(gpu, N) in current thread`.
MLX arrays are thread-affine, and mira-mlx pins model execution to one engine thread, so an
`mx.array` built on a worker thread crashes the first time the engine thread touches it. My pytest
suite, run eight times over to catch flakiness, never saw it, because pytest does not exercise the
pinned-thread architecture. Only a real server process did. The fix is to split the disk I/O
(thread-safe, parallel) from the `mx.array` construction (which has to happen on the calling thread).
The lesson is that unit tests and real-server tests catch different classes of bug here, and I need
both.

**(d) A bound that grew with the setting meant to make things safer.** `max_stack` defaulted to
`resident_slots`, so raising the resident fraction to 0.5 made the chunks bigger and reproduced the
original OOM. A bound has to be independent of how generous the setting is. I replaced it with a
constant cap.

**(e) The seed that never freed anything.** This is the crux of the whole story. Offload seeded its
cache by slicing the resident experts out of the loaded weight, `module.weight[:resident_slots]`, and
then set `module.weight` to that slice to "drop the rest." Peak memory came out higher than the
fully-resident baseline at every fraction. Offload was making things worse. A direct MLX experiment
settled it:

```
allocate (256,1024,1024);  seed = big[:32];  drop big
=> 0 bytes freed.  A prefix slice is a view that pins the entire parent buffer.
```

So `module.weight = module.weight[:k]` never released the other experts. It pinned the whole table
and offload piled cache on top. The "about 0GB active after load" reading that had made me think the
table was freed was a lazy-mmap artifact. The fix is to seed from disk (independent buffers) and
replace the weight with a 1-row stand-in, which genuinely drops the full tensor. Verified on the real
model:

| fraction | peak GB (was) | steady-state resident GB (was ~full) |
|---|---|---|
| none (baseline) | 19.25 | 18.31 |
| 0.15 | 18.19 (21.39) | 4.01 |
| 0.3  | 18.21 (23.99) | 6.58 |
| 0.5  | 18.24 (crashed) | 9.94 |

Steady-state resident now scales with the fraction, which is the RAM win the view-pin had masked
completely.

## 4. Shipping it honestly

I made offloading the default at fraction 0.3, with a flag to fall back to the simpler fully-resident
path. I stated the tradeoff I believed at the time: on this machine the model already fits, so this
trades slower cold-prefill latency for about 12GB of freed steady-state RAM. That framing turned out
to be too generous about the cost (see section 10), but the RAM number was real and live-verified.

One thing the table above made obvious: peak never dropped. It stayed at 18.2GB regardless of
fraction. That is what raised the real question.

## 5. The larger-than-DRAM question

I put it to myself plainly: this does not actually let me load models larger than DRAM. Prefill uses
almost all the memory for the model, so a larger model that would run fine in steady state still
cannot open in the first place. Experts cannot be sparsely loaded when the model opens.

Steady state was solved. The 18.2GB peak was the wall. So I lined up three candidate implementations
to bound the peak and put each through an adversarial review against the real checkpoint:

- Token-block prefill (smaller `prefill_step_size`): refuted. The N-scaled activation is about 0.3 to
  0.5% of the peak, and it is 1.5 to 3x slower.
- Compacted chunked gather (compute only each expert-group's own token positions): refuted for peak.
  The gather output is 0.13% of the peak, and worse, the "multi-group implies sorted" invariant it
  relies on is false below fraction 0.25, which silently corrupts output in exactly the
  larger-than-DRAM regime. Salvageable only as a safely-gated latency win.
- Full per-layer weight streaming (the LLM-in-a-Flash approach): rejected. The non-expert weights are
  only 2.1GB and dense (100% read per token), so streaming them means about 1GB of dense disk reads
  per decode token, a hard ceiling around 5 tok/s. It inverts the sparsity that makes offload work at
  all.

All three targeted terms under 1% of the peak. That was the signal. The dominant transient, about
11.6GB, was something none of them addressed, and static analysis could not even attribute it (the
three reviewers disagreed on the mechanism). So I stopped reasoning and measured.

## 6. The diagnostic, and being wrong before being right

First I ran a knob sweep on fresh servers (fixed offload 0.3, the same diverse 1458-token prefill):

| config | prefill(s) | peak GB |
|---|---|---|
| step 1024, kv8 | 24.1 | 18.21 |
| step 128,  kv8 | 46.9 | 18.21 |
| step 1024, kvFP16 | 22.1 | 18.21 |
| step 128,  kvFP16 | 46.0 | 18.21 |

Peak is exactly flat across block size and KV-quant. Not activations, not the KV score matrix.

Then I probed the live server and concluded, wrongly, that the peak was a one-time first-forward
transient: MLX faulting the full expert mmap in when the model first runs, then freeing it. It looked
airtight. Active read as about 0GB after load, and the first forward, even a 10-token "Hi", appeared
to spike to 18.21GB and settle to 6.5GB, and the freed amount (18.21 minus 6.49, about 11.78GB) was
exactly the offloaded 70% of the experts.

It was an artifact. The server never resets `mx.get_peak_memory()`, so the high-water mark set during
`load()` was still being reported when the first request arrived, which made a load-time cost look
request-caused. The "0GB active after load" that seemed to confirm it was read after `install()`'s
stand-in swap had already freed the table.

A standalone probe with a per-phase `reset_peak()` between load, install, and each forward settled it
in one run:

| phase | active | peak |
|---|---|---|
| after `load()` (lazy=False) | 18.17 | 18.17 |
| after `install()` (stand-in swap) | 6.37 | 18.21 |
| after `install` + `clear_cache()` | 6.37 | 0.00* |
| tiny forward / diverse 1458 prefill | 6.44 | 6.51 / 7.48 |

*(peak reset before the phase)*

The peak is entirely load-time eager materialization. `mlx_lm.load` with `lazy=False` calls
`mx.eval(model.parameters())`, which wires the full stacked expert table the instant `load()`
returns. No forward ever exceeds about 7.5GB. There is no first-forward fault, no mmap re-fault, and
no residual-reference mystery. The stand-in swap already frees the table (18.17 to 6.37 active, with
the 11.8GB going to MLX's buffer pool and released by `clear_cache`). My prediction from section 5
was right, the wall is the model opening, but the mechanism is plain eager load, one `mx.eval`.

I want to keep the wrong version on the record because it is the lesson turned on itself. My own first
diagnosis asserted a mechanism I had not isolated, and a twenty-line probe overturned it. Assume
nothing about MLX memory, including my own earlier conclusions.

## 7. Lessons

1. Verify MLX memory semantics, never assume them. The view-pin in 3(e) and the load-time
   materialization in 6 were both invisible to reasoning and obvious to a short experiment.
2. `ps` and RSS are blind to Metal unified memory. Use MLX's own counters.
3. pytest and real-server tests catch different bugs. The thread-affinity crash passed pytest eight
   times.
4. Adversarial review before implementing pays for itself. Three plausible approaches, all refuted
   against real numbers, saved three wasted builds, and the pattern of the refutations (all under 1%)
   is what pointed at the real cause.
5. Measure before building. Two cheap sweeps did what a week of static analysis could not: attribute
   the peak and reframe the whole problem.

## 8. The fix that shipped

Since the peak is a load-open cost, the fix is a load-path change, and a one-keyword one:
`load(..., lazy=True)`, gated on offload being enabled. Whole-run peak went from 18.21GB to 7.25GB on a
real server, output bit-coherent, and peak now scales with the resident fraction rather than the
expert-table size. The mechanism — why a deferred `sanitize` node costs nothing once the stand-in swap
drops it, and why this is safe only with offload — is in Appendix A below.

## 9. The over-DRAM demonstration, observed rather than inferred

The one remaining claim, that a model whose expert table exceeds DRAM now runs, was still resting on
the memory trajectory rather than on a direct observation. So I ran it for real:
`Qwen3.6-35B-A3B-8bit` (35GB on disk, about 33.8GB stacked expert table, larger than both this
machine's 32GB RAM and its ~25GB Metal wired limit), on the same 32GB Mac.

- Eager load is impossible, measured and safely aborted. Under `lazy=True` the table is 0.00GB.
  Evaluating expert modules one at a time up to a 7GB ceiling, 28 of 120 modules already reached
  7.22GB (264MB per module), which extrapolates to about 30.9GB for the expert table alone, past the
  wired limit before the dense 4.2GB is even counted. `lazy=False`, which evaluates all of it at once,
  cannot complete on this machine.
- Lazy plus offload runs it. `load(lazy=True)` gives 0.00GB, `install(offload 0.3)` seeds 9.59GB
  resident from disk, and a 64-token generation completes at peak 12.71GB with coherent output,
  59,346 cache hits against 30,615 cold-miss disk fetches, 33.3s cold. A model that cannot be loaded
  eagerly on this hardware loaded and generated at about 12.7GB resident.

That closes the loop. Not just sparse at steady state, but a genuinely larger-than-DRAM model opening
and running. Model size is now a free parameter: at fraction 0.3 the wall moved from "expert table
fits in RAM" to "0.3 times expert table plus dense weights fits in RAM," so this 32GB Mac can open MoE
checkpoints whose expert tables run to roughly 50 to 70GB. Reproduce with
`scripts/moe_overdram_demo.py`; the model is logged in `MODEL_REGISTRY.md`.

## 10. The throughput cost, and making offload pay for itself

Runnable is not free. I benched throughput on fresh backends (a 3022-token prefill, a 200-token
decode; `scripts/moe_throughput_bench.py`):

| config | prefill cold / warm | decode | peak | hit-rate |
|---|---|---|---|---|
| 4-bit offload-off | 616 / 957 t/s | 57.1 t/s | 19.31 GB | n/a |
| 4-bit offload-on 0.3 | 77 / 76 t/s | 10.8 t/s | 7.31 GB | 0.51 |
| 8-bit offload-on 0.3 (over-DRAM) | 59 / 56 t/s | 6.6 t/s | 13.05 GB | 0.51 |

Two things stand out. First, offload is expensive: about 5x slower decode and 8 to 12x slower prefill
on the 4-bit model. Second, warm prefill is no better than cold, because a diverse 3022-token prefill
routes to nearly all 256 experts while the cache holds only 30%, so it re-fetches most of the table on
every prefill. The LRU simply cannot hold a prefill's working set. Decode pays the same tax per token
(hit-rate 0.51, so roughly half of the top-8 selections across 40 layers fault to disk). My earlier
"decode unchanged" note was measured under kinder conditions and was wrong at scale.

This overturned my own "offload on by default" decision. On the 4-bit model, which fits at 19.3GB,
default-on was paying 5x decode for RAM the machine did not need. So offload is now per-model auto
(`mira_mlx_expert_offload: auto`): it turns on only when the fully-resident model would not fit, using
`hardware.fits_in_memory`. The 4-bit runs resident at full speed (live: 0.61s prefill against about
40s with offload on), and the 8-bit auto-offloads because it is the only way it runs at all. That is
the honest framing of the whole feature. Offload buys the ability to run a model I otherwise could
not, at a real speed cost, so I pay it only when I have to. From here two separate levers remained:
cut the per-token compute the tax rides on, or cut the miss count itself. Section 11 is what happened
when I pushed on both.

## 11. Clawing back the tax: two gather optimizations, and one wall

Both wins were already foreshadowed. Section 5 flagged the compacted chunked gather as "salvageable
only as a safely-gated latency win," and section 3(a) introduced the per-group `mx.eval()` barrier.
Each became a real, bit-identical change on the fork.

**Prefill compaction.** The mask-path chunked gather from 3(a) runs each of the G expert-groups over
*all* token positions and discards the out-of-group ones with `mx.where`, so it pays G times the
matmul on a diverse prefill (G is 4 at fraction 0.3). But `SwitchGLU`/`SwitchMLP` already sort routing
before dispatch whenever `indices.size >= 64`, which is exactly the multi-group prefill case, and pass
`sorted_indices=True`. When that holds and the flat index run is verified monotonic, each group's
positions are one contiguous slice, so I slice `x` to that segment, gather once over just those
positions, and concatenate, with no `mx.where`. It is guarded by an actual monotonicity check rather
than trusting the flag, so a mislabelled `sorted_indices` falls back to the mask path. That guard is
the answer to the silent-corruption risk section 5 raised: the invariant is verified per call, not
assumed. Measured on 4-bit, offload 0.3, a 3022-token prefill: cold prefill 75.7 to 136.2 t/s (1.8x),
bit-identical to the mask path, hit-rate and peak unchanged. The isolated gather micro-bench is 3.24x;
end-to-end is 1.8x because the gather is about half of offload-on prefill wall time and the rest is
attention plus the disk misses.

**Decode eval-skip.** The per-group `mx.eval()` from 3(a) exists only to bound memory *across* groups:
it forces each group's stack to release before the next one allocates. Decode always routes to a
single group, because a token's top-8 experts fit inside one `max_stack` of 64, so there is never a
next group to bound against. On that path the eval only forces a per-layer GPU sync, roughly 3 per
layer over 40 layers per token, where a fully-resident model evals the whole token once. Skipping it
when `len(groups) == 1` lets MLX defer the token's gather graph to the caller's next eval boundary,
which during generation is the sampler evaling each token anyway. The measurement that mattered was
peak, since deferring the graph is exactly what could have raised it. On the real over-DRAM case,
8-bit, offload 0.3, greedy so both runs select identical experts (hits and misses identical to the
token):

| variant | decode | peak | prefill |
|---|---|---|---|
| barrier every group | 7.15 t/s | 12.73 GB | 117.3 t/s |
| skip lone-group barrier | 8.09 t/s | 12.73 GB | 120.0 t/s |

That is +13% decode at byte-identical peak, output bit-identical. Peak not moving is the whole point:
on the real offload path only the resident fraction is ever held, so deferring one token's graph adds
a negligible transient instead of materializing the table. A synthetic that keeps every expert warm
shows the opposite, deferral raises peak and loses about 12%, but that warm-everything state is
precisely what offload exists to avoid, so it never occurs on the real path. This is why it had to be
settled on the real 8-bit model rather than in a microbench: the microbench's memory regime is the one
regime offload guarantees you are not in.

**The miss-count wall.** The other lever, cutting the miss count by raising hit-rate, I pushed on and
it did not give. Cross-layer prefetch is defeated by the load-balancing aux loss, which decorrelates
per-layer usage, so there is no cheap predictor. Co-activation-aware on-disk reordering is already
spent by LRU at decode: the first activation caches the expert and the adjacent-token reuse is then a
hit, not a read, so it would only help cold-prefill sequential reads. A warm-then-pin hot set beats
plain LRU by about 2.4 points within a topic but does not generalize across topics, and pure pinning
without LRU is worse than plain LRU. The realistic online ceiling (Belady, in-sample) is about 0.79,
and warm decode already sits at 0.71 to 0.83, so there is no large win hiding there. The only dial
that moves miss count freely is the resident fraction, which spends back the RAM offload exists to
save. So the tax stays where the measurements put it, and the wins that shipped are the compute-side
ones above: the prefill redundancy and the decode sync. Both live on the `mabaeyens/mlx-lm` fork and
are noted on the upstream offload PR.

---

The full technical timeline and every measurement live in `BACKLOG.md`, how to size the resident
set in `docs/offload-resident-sizing.md`, and the adversarial evaluation in
`~/.claude/plans/resilient-sprouting-owl.md`.

---

## Appendix A. Lazy expert load design

Status: implemented 2026-07-19. One gated keyword, `load(..., lazy=True)` when offload is enabled
(`core/inference/mira_mlx_server.py`). Whole-run peak on Qwen3.6-35B-A3B at fraction 0.3 dropped from
18.21GB to 7.48GB, and the full expert table no longer becomes a live buffer at any point in the
lifecycle, so a model whose expert table exceeds DRAM can open. The sections above have
the full story of how I got here, including the wrong turn.

I originally scoped this as a three-component change and wrote it up assuming the 18.21GB peak was a
first-forward transient. Both were wrong. A standalone probe with per-phase `reset_peak()` showed the
peak is purely load-time eager materialization (the live server never resets `mx.get_peak_memory()`,
so the load high-water mark was still showing at request time and looked request-caused), and the fix
collapsed to a single keyword. The rest of this doc reflects the corrected, measured mechanism.

### Problem

Expert disk offloading bounds steady-state and per-forward memory to about 6.5GB on Qwen3.6-35B-A3B at
resident fraction 0.3, but `mlx_lm.load` with the default `lazy=False` eagerly materializes the full
stacked `(num_experts, ...)` expert table into unified memory at load: 18.17GB active the instant
`load()` returns, before a single token. That one load-time allocation is the largest memory event in
the whole lifecycle and the wall that stops a model whose expert table exceeds DRAM from opening at
all, even when its offloaded steady state (6.5GB) would fit with room to spare.

### Root cause (load flow), as measured

`core/inference/mira_mlx_server.py` calls `mlx_lm.utils.load(model_path)` with the default
`lazy=False`:

1. `load_model` (`utils.py:282`) reads the raw safetensors and calls `model.sanitize(weights)`
   (`qwen3_5_moe.py:23-52`), which constructs the full stacked `(num_experts, ...)` expert tensors
   from the raw checkpoint (`experts.gate_up_proj` into `switch_mlp.gate_proj/up_proj/down_proj.weight`).
2. `model.load_weights(...)` (`utils.py:415`) binds them into the module tree, still unevaluated.
3. `mx.eval(model.parameters())` (`utils.py:418`) evaluates them, and active jumps to 18.17GB
   immediately (standalone probe: active and peak both 18.17 right after `load()`, before install; a
   redundant `mx.eval(model.parameters())` afterward adds nothing).
4. mira then runs `expert_offload.install()`, which seeds the resident set from disk and replaces each
   module's full weight with a 1-row stand-in, dropping the full-tensor references. Active falls from
   18.17 to 6.37GB, and the freed bytes go to MLX's buffer-reuse pool (`get_cache_memory` reads
   16.88GB) until `mx.clear_cache()` returns them to the OS. Every forward after that is bounded
   (tiny 6.51GB, diverse 1458-token prefill 7.48GB). No forward re-materializes the table.

So the full tensors are built and evaluated at load for nothing. The disk expert cache reads expert
bytes straight from the raw checkpoint by offset (`core/inference/disk_expert_cache.py`), so the model
never needs the stacked expert tensors to exist as live arrays. The only reason they cost 18GB is
step 3's eager eval.

### Goal (met)

The full `(num_experts, ...)` expert tensors must never become live MLX arrays. Then peak is about
equal to steady state (6.5GB), independent of expert-table size, and models with over-DRAM expert
tables run.

### The fix

The step-0 probe showed the peak is purely step 3's eager `mx.eval(model.parameters())`. It is not a
first-forward fault, and it is not something that survives `install()`'s stand-in swap. That collapses
the original three-component scope to one keyword:

`load(self.model_path, ..., lazy=True)`, gated on `self.resident_expert_fraction is not None`
(`core/inference/mira_mlx_server.py`). With `lazy=True`, `sanitize` still constructs the stacked
expert tensors, but they stay unevaluated graph nodes with zero wired bytes, and `install()` then
seeds the resident set from disk and its stand-in swap drops those nodes before anything forces their
eval, so the full table is never materialized at any point. Measured on a standalone probe at fraction
0.3:

| phase | `lazy=False` (before) | `lazy=True` (fix) |
|---|---|---|
| after `load()` | active/peak 18.17GB | active/peak 0.00GB |
| after `install()` | active 6.37 / peak 18.21 | active 5.08 / peak 5.08 |
| after diverse 1458-token prefill | peak 18.21 (set at load) | peak 7.48 |

End-to-end on a real server (`/v1/stats` peak): pre-request 18.21GB to 0.0GB, after a diverse prefill
18.21GB to 7.25GB. Output stays coherent.

The two other components I had originally scoped both turned out to be unnecessary:

- Filtering expert keys before `load_weights`: not needed, because unevaluated nodes cost nothing and
  the stand-in swap discards them. Skipping it avoids fork- and model-version-sensitive `sanitize`
  surgery.
- An offload-native `SwitchLinear.__init__`: not needed, because `load()` overwrites the `__init__`
  placeholder weights before any eval, and under `lazy=True` neither the placeholder nor the loaded
  table is ever materialized.

### Why `lazy=True` is safe only with offload

Without offload there is no `install()`, so there is no stand-in swap to drop a deferred table, and
`lazy=True` would merely postpone the same 18GB materialization to the first forward. The default
non-offload path keeps `lazy=False`, unchanged. A dense model (no `SwitchLinear`) launched with the
offload flag on is a benign misconfiguration: `install()` finds nothing to swap, and the dense weights
materialize on the first forward instead of at load, deferred but not broken.

### Open questions, resolved by the probe

- Why does the first forward fault the full table? It does not. The 18.21GB was load-time eager eval,
  and the "first-forward" reading was the server's un-reset `peak_memory` high-water mark from load
  still showing at request time. Per-phase `reset_peak()` in a standalone probe settled it.
- `sanitize` builds the tensors. True, but under `lazy=True` they are never evaluated, so their
  construction is free. No need to bypass `sanitize` or filter keys.
- The quantized path (`_maybe_qq`, `utils.py:410`) does not re-materialize experts under `lazy=True`:
  post-fix diverse-prefill output is coherent and peak stays in the 7.25 to 7.48GB range.

### Verification

- Peak: standalone probe whole-run peak 18.21 to 7.48GB; server `/v1/stats` peak 18.21 to 7.25GB.
- Correctness: `tests/test_expert_offload.py` plus `tests/test_config.py` all pass; live diverse
  prefill produces coherent output; offload hit/miss counters behave as before.
- The actual goal, demonstrated 2026-07-19: `Qwen3.6-35B-A3B-8bit` (about 33.8GB expert table, larger
  than 32GB RAM and the ~25GB wired limit) loaded via `load(lazy=True)` plus offload at 0.3 and
  generated coherent text at peak 12.71GB resident (seed 9.59GB, 59346 hits against 30615 cold-miss
  disk fetches). Eager load is impossible: the measured materialization trajectory extrapolates to
  about 30.9GB for the expert table alone. See case study section 9; reproduce with
  `scripts/moe_overdram_demo.py`.

### Risks and residual notes

- The change is one gated keyword in mira-core (no fork change), guarded behind the offload flag, with
  the offload test suite as the gate. Low blast radius.
- The MLX lazy and mmap lifecycle is the same "verify, do not assume" class that produced the earlier
  crashes, so every step here landed behind a probe measurement rather than reasoning.
- Over-DRAM is now directly observed rather than inferred, via the 8-bit run above. That 8-bit model
  is cached but is a demo artifact, not a production model (`MODEL_REGISTRY.md`).

---

## Appendix B. Prior art

An honest, mechanism-level map of how Mira's expert offloading relates to the two papers closest to it,
so I can cite them accurately without overclaiming or reimplementing. Companion to the build story above and Appendix A (the fix).

One thing to read this doc with in mind: the offloading lives on Apple-silicon **unified memory** (no
host/device copy) and reads experts straight from the model's own safetensors shards by byte offset (no
repack, no separate cache format). That single hardware fact flips most of the tradeoffs below, because
every paper here was written for a PCIe/flash tier boundary where the transfer is the enemy.

A note on numbers: an earlier version of this comparison concluded "disk is ~14% of wall, the miss cost
is engine-side." That was measured on the **4-bit model during prefill**, and I over-generalized it. A
2026-07-19 resident-fraction sweep on the **8-bit over-DRAM model during decode** showed the opposite:
there the cold read *is* on the critical path (~254us, ~66% of the per-miss wall), because decode faults
only ~1.3 experts per module call across 120 sequential modules, too little concurrency to fill the
8-way read pool, and the 33.8GB expert table exceeds the page cache left after the resident set. Where
that changes a conclusion below, it is flagged inline.

---

### 1. Eliseev & Mazur, "Fast Inference of Mixture-of-Experts Language Models with Offloading" (arXiv 2312.17238)

The Mixtral-offloading paper. Same family as what I built (a resident LRU expert cache with on-demand
fetch), which I arrived at independently for Apple Silicon. The two techniques it leans on most,
speculative expert prefetch and mixed 2-3 bit expert quantization, I do not do. Shared lineage and prior
art, not a reimplementation.

**Their setting vs mine.** They target Mixtral-8x7B on a consumer GPU with a strict tier boundary:
experts live in host RAM, get copied to VRAM over PCIe on demand, tens of milliseconds per expert. I
target Qwen3.6-35B-A3B on a 32GB Apple-silicon Mac with unified memory: a resident expert sits in the
same memory the GPU computes in, and a cold expert is an SSD byte-range slice read (measured 0.207ms for
a 4-bit slice, `core/inference/disk_expert_cache.py`), two orders of magnitude under their PCIe transfer.

| Paper technique | Mira's equivalent | Same / Different / Skipped |
|---|---|---|
| Resident LRU expert cache | `_offload_cache` + `_offload_lru`, capacity `round(n_experts·fraction)` | **Same** structure, in unified memory rather than a host-RAM tier feeding VRAM |
| On-demand fetch on a cold miss | `_offload_chunked_gather` fetches missing experts just-in-time, 8-way parallel | **Same** idea; the read is an SSD byte-range slice, not a PCIe copy |
| Speculative expert prefetch | none | **Skipped**: measured NO-GO (below) |
| Mixed 2-3 bit expert quant (HQQ) | read the model's own uniform-4-bit shards by offset, no repack | **Skipped / different**: no separate quant or cache format |
| Host-RAM cache + PCIe transfer to VRAM | SSD-direct read into the same unified memory the GPU uses | **Different hardware**: inverts the prefetch/transfer tradeoffs |
| Frequency-skewed expert reuse as why caching works | measured: top-20% concentration 2.5x uniform, adjacent-token overlap 10.8x uniform | **Same premise, independently measured** for this model |

**Where the hardware difference changes the tradeoff.**
- *Speculative prefetch pays for them, not for me.* Their prefetch hides a tens-of-ms PCIe transfer.
  Even where my read is on the critical path (8-bit decode), the blocker is not latency-to-hide but
  *predictability*: each decoder layer has its own independently-trained router over disjoint experts,
  and the load-balancing aux loss decorrelates usage, so "prefetch the next layer's experts from this
  layer's choice" has no signal (measured cross-layer top-K Jaccard 0.017, indistinguishable from
  uniform). A 2026 survey plus "Speculating Experts" (arXiv 2603.19289), which benchmarks the Qwen3-A3B
  family directly and finds speculation degrades task accuracy there, confirm it. Skipped, correctly.
- *Mixed low-bit expert quant is a memory lever, and memory is not the binding constraint here.* My
  quantization audit found uniform 4-bit optimal and lower-bit hurt quality; after the lazy-load fix the
  peak is already bounded. Reading the checkpoint's own 4-bit rows by offset also means no repack and
  byte-identical cold experts. Skipped.
- *My miss cost is split, and which half binds depends on quant and phase.* On 4-bit prefill the read is
  ~14% of wall (parallelized) and the engine-side `mx.array` build/dequant dominates. On 8-bit decode
  the cold read dominates instead (~66% of the per-miss wall) because concurrency is too low to hide it.
  Either way, "faster storage" is not the lever, the read that binds at decode is a *cold* read whose
  fix is fewer of them (residency) or fewer/larger ops (layout), not a faster device.

**What of theirs transferred (post-analysis).** Their frequency-skew insight → keep the hot experts
resident. Two ways to cash it: pin a learned hot set, or simply hold more of them. I measured hot-pinning
at ~+2.4 pts hit-rate out-of-sample but it failed to generalize cross-topic (the gate went negative), so
it is closed. The residency route shipped instead as RAM-aware sizing (`hardware.derive_resident_expert_
fraction`): an over-DRAM model sizes its resident fraction to available RAM rather than a flat 0.3,
measured +12% decode and +9% prefill on the 8-bit at f=0.45. Speculative prefetch and low-bit quant were
not adopted.

---

### 2. Alizadeh et al., "LLM in a Flash" (arXiv 2312.11514, Apple ML Research)

Written for exactly my hardware class (limited DRAM, weights on flash), so it is the most tempting to
borrow from. But it is about running a **dense** model whose FFN weights do not fit in DRAM by streaming
them from flash, exploiting ReLU-induced *neuron* sparsity. I run a **sparse-MoE** model, keep the small
dense backbone fully resident, and offload only the routed experts. Its windowing and predictor ideas map
onto things I already measured; its dense-streaming premise is the one I explicitly reject.

**The core difference.** Their sparsity is *activation* sparsity (with ReLU FFNs most neurons output zero,
so load only the active rows/columns). Mine is *routing* sparsity (a gate picks top-8 of 256 experts). The
16.9GB expert table is offloaded while the **2.1GB dense backbone stays resident** because it is read 100%
per token. That split is why I do not stream everything.

| "LLM in a Flash" technique | What it maps to for me | Transfers? |
|---|---|---|
| **Windowing**: reuse neurons active in a recent-token window, load only the delta | measured adjacent-token expert overlap, 10.8x uniform | **Already captured by LRU** (below) |
| **Row-column bundling**: store a neuron's up/down weights contiguously, read as one chunk | disk-read coalescing | **Measured, not adopted**: +14.3% at the shipped fraction, costs the no-repack property (below) |
| **Sparsity predictor**: predict which neurons fire, prefetch only those | cross-layer expert prediction | **No signal** for this MoE's routing (Jaccard 0.017) |
| **Optimized flash layout + large sequential reads** | byte-range slice reads | **Partly**: see the coalescing note below |
| **Stream dense FFN weights from flash** (the premise) | would mean streaming my dense backbone | **Rejected**: bandwidth ceiling |

**Windowing is already monetized by the LRU cache.** The 10.8x adjacent-token overlap is the windowing
premise: an expert active at token t is likely active at t+1. Under my resident LRU, that expert's first
activation pulls it resident, so its second is a cache hit, not a reload. The cache already spends the
windowing signal; what is left in the miss stream is the residency complement (the least-reused experts),
which is why an explicit windowing pass on top of LRU buys little.

**Row-column bundling / layout: measured, and not adopted.** The earliest version dismissed this as
non-paying because "reads are 0.2ms and parallel." That held for 4-bit prefill. On 8-bit decode the cold
read is on the critical path, and the reader is more fragmented than assumed: up/gate/down are three
separate modules and each reads weight+scales+biases as separate ops, so one cold expert is up to nine
`open`+`seek`+`read` calls. Fixing the reopen-per-read alone (one long-lived fd per shard via `os.pread`)
was measured at +4.5% decode and shipped.

A middle draft then framed the bundling win proper as a **prefill/TTFT** lever only. That was wrong too,
and the decode sweep is what corrected it: decode faults about 1.3 experts per module call across 120
sequential modules, so the 8-way reader has almost nothing to parallelize and the reads effectively
serialize on the critical path. Cold reads are 26.1% of the decode token at resident fraction 0.3 and
21.8% at 0.45. Decode is a genuine target, not just prefill.

Measured on our own reader rather than transferred from the upstream model (`scripts/moe_coalesce_ceiling.py`,
both arms reading never-touched regions at real slice sizes):

| lever | I/O speedup | decode @0.3 | decode @0.45 |
|---|---|---|---|
| A1, 3 slices to 1 read | 1.61x | +11.0% | +9.0% |
| A2, 9 slices to 1 read | 2.34x | +17.6% | +14.3% |

Both come in below the upstream model's 2.25x / 3.0x, and the gap looks like the `open()` per slice that
model assumes and our shared-fd reader no longer pays. **Not adopted.** Either lever needs a coalesced
side-file, which breaks reading straight from the model's own shards: no repack step, no second artifact
per model, nothing to invalidate on re-quant. At the fraction we actually ship that is +14.3%, which does
not buy the property back. A1 at +9.0% is not a candidate at all.

**Revisit trigger tested and retired (2026-07-20).** The stated trigger was "a model far enough over DRAM
to force a much smaller resident fraction, pushing the read share back up". We ran it on
`gpt-oss-120b-MXFP4-Q8` (56.3GB expert table, 1.8x RAM, mxfp4, a different architecture). The read share
did rise as predicted, 21.8% to **39.3%** — and A2 still got **worse**, +9.7% against +14.3%:

| | read share | A1 I/O | A2 I/O | A2 end-to-end |
|---|---|---|---|---|
| Qwen 8bit @0.45 (shipped) | 21.8% | 1.61x | 2.34x | +14.3% |
| gpt-oss @0.30 | 39.3% | 1.08x | 1.29x | +9.7% |

Coalescing is an IOPS/latency lever and gpt-oss reads are already bandwidth-bound: its module is
4050K + 253K (one read is 94% of the bytes, scattered already runs 5.8 GB/s) where Qwen's is
1024K + 32K + 32K (two tiny reads that are pure per-op latency floor, scattered only 2.5 GB/s).
**The two effects are anti-correlated** — a bigger model forces a smaller fraction (raising the read
share) *and* has larger slices (making reads bandwidth-bound) — and the second wins. So there is no
"bigger model" that rescues coalescing, and the trigger was based on a wrong model of where the win
comes from. Discussion: ml-explore/mlx-lm#1438.

**The sparsity predictor has no target here.** Their predictor guesses active neurons within a layer from
that layer's own input, available before the FFN runs. My analogue would be predicting the *next* layer's
experts (the only prefetch with headroom), and that fails on independent per-layer routers, arbitrary
index numbering, and aux-loss decorrelation.

**The rejection this paper forces me to state.** The most direct "LLM in a Flash" reading (stream all
weights, keep only a working set resident) is wrong for this model. My non-expert weights are only 2.1GB
(11%) but dense: read 100% per token. Streaming ~1GB of dense weights per decode token gives a hard
bandwidth ceiling around ~5 tok/s regardless of prefetch (prefetch hides latency, not throughput). So I
keep the dense backbone resident and offload only the sparse experts (read ~3% per token), the opposite
of streaming everything, and the right split because the paper's technique is tuned for
dense-with-activation-sparsity, not MoE-with-routing-sparsity.

---

### Citation sentences (paste-ready)

> My MoE expert offloading is in the same family as Eliseev and Mazur's Mixtral-offloading work (arXiv
> 2312.17238): a resident LRU expert cache with on-demand fetch on a cold miss. I built that structure
> independently for Apple Silicon, so I cite their paper as prior art rather than something I
> reimplemented. The two techniques it leans on most, speculative expert prefetch and mixed 2-3 bit
> expert quantization, I deliberately do not do: on unified memory a cold expert is an SSD slice read,
> not a tens-of-ms PCIe transfer, and I measured that cross-layer expert choice is not predictable for
> this model anyway.

> "LLM in a Flash" (Alizadeh et al., arXiv 2312.11514) is the closest prior work to my hardware, and my
> model shows the reuse signal it relies on (10.8x-uniform expert overlap between adjacent tokens). But my
> resident LRU cache already turns that reuse into cache hits. I also deliberately do not stream all
> weights from disk the way that paper does: its target is a dense model with ReLU activation sparsity,
> mine is a mixture-of-experts model, so I keep the small dense backbone (2.1GB, read every token)
> resident and offload only the routed experts. Streaming the dense part would cap throughput near 5
> tokens per second by bandwidth alone.
