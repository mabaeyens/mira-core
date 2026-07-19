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
`load(..., lazy=True)`, gated on offload being enabled. `mlx_lm.load` no longer materializes the full
`(num_experts, ...)` expert tensors. `sanitize` still constructs them, but as unevaluated graph nodes
with zero wired bytes, and the stand-in swap drops those nodes before anything evaluates them, so the
full table never becomes a live buffer at any point. Whole-run peak went from 18.21GB to 7.25GB on a
real server, output bit-coherent. Peak now scales with the resident fraction, not the expert-table
size. Details are in `docs/moe-offload-lazy-load-design.md`.

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

The full technical timeline and every measurement live in `BACKLOG.md`, the offload internals in
`specs/moe-expert-offload-02-runtime-cache.md`, and the adversarial evaluation in
`~/.claude/plans/resilient-sprouting-owl.md`.
