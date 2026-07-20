# Mira's MoE offloading vs the prior art

An honest, mechanism-level map of how Mira's expert offloading relates to the two papers closest to it,
so I can cite them accurately without overclaiming or reimplementing. Companion to
`docs/moe-offload-case-study.md` (the build story) and `docs/moe-offload-lazy-load-design.md` (the fix).

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

## 1. Eliseev & Mazur, "Fast Inference of Mixture-of-Experts Language Models with Offloading" (arXiv 2312.17238)

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

## 2. Alizadeh et al., "LLM in a Flash" (arXiv 2312.11514, Apple ML Research)

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

## Citation sentences (paste-ready)

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
