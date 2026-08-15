# Custom kernels on Apple Silicon: what they buy, and whether Mira should write one

Written 2026-08-11, prompted by finding real Metal prefill kernels inside oMLX
and asking why mira-mlx has none. It is an assessment, not a plan. Everything
below is marked as verified, estimated, or not checked, because the useful part
of this question is knowing which is which.

## 1. What a kernel is

A GPU kernel is a small program executed by many threads in parallel over the
elements of a tensor. A framework ships one per primitive: matmul, softmax,
RMSNorm, elementwise add. `y = softmax(x @ w)` dispatches a sequence of
pre-written kernels. A *custom* kernel is one written by hand, in Metal Shading
Language here, instead of composing the framework's built-ins.

## 2. What writing one actually buys

**Fusion, which is the main one.** Every framework op reads its inputs from
device memory and writes its output back. Five chained elementwise ops means
five round trips. A fused kernel does all five while the data sits in registers,
so one round trip. FlashAttention is the canonical case: fusing QK^T, softmax
and ·V means the N×N attention matrix is never written to memory at all, turning
attention from O(N²) memory traffic into O(N). The win is bandwidth, not
arithmetic.

**Exploiting structure the framework cannot see.** A generic matmul does not
know the weights are 4-bit packed into int32, that only k of N experts are
active for this token, or that the recurrence is a Gated DeltaNet. A custom
kernel dequantises in-register and skips the inactive path.

**Cutting dispatch overhead.** Each launch has a fixed cost. At batch 1 with
short sequences, decode can be dispatch-bound rather than compute-bound.

## 3. Stability is a separate axis and it cuts both ways

Kernels change numerics. A fused kernel can accumulate in fp32 while its inputs
are fp16, which improves conditioning. A hand-written kernel is also a new place
for bugs, and a different reduction order gives different results.

The sharpest case for Mira is one this repo already measured: **batch
invariance**. A kernel whose reduction order depends on batch size produces
different tokens at different batch sizes: 23 of 24 prompts here produced
different text at greedy between batch 1 and batch 4, which is why eval scores
carry a batch-composition term.
Custom kernels are the *fix* for that, not the cause. It is the one motivation
here that is about correctness rather than speed, and it is the one that
survives §6 intact.

## 4. The case for, on Apple Silicon

**MLX exposes a custom-kernel API.** `mx.fast.metal_kernel` — **verified
present** in the installed MLX 0.32.0. Metal source goes in as a Python string
and is JIT compiled. No C++ build, no separate toolchain.

**The ecosystem is thinner, so more is unpicked.** A novel architecture gets
CUDA kernels from the community within weeks. MLX often waits months, or never.
oMLX's `custom_kernels/qwen35_prefill/` exists because mlx-lm's Gated DeltaNet
path is generic Python.

**Unified memory removes the host-device copy** that makes some fusion patterns
pointless on discrete GPUs.

## 5. The case against

**The obvious fusions are already done.** `mx.fast` ships `rms_norm`, `rope`,
`layer_norm` and `scaled_dot_product_attention` as custom kernels inside MLX —
**verified** by inspecting the module. The canonical FlashAttention-style wins
are not available; Apple took them. What is left is architecture-specific, which
is a far narrower target and is exactly where oMLX went.

**Decode is bandwidth-bound and no kernel moves that wall.** Generating a token
requires reading the active weights from memory. Fusion helps when making many
passes over data; it does not help when the single unavoidable pass *is* the
cost. This is why the 2026-06-27 rejection of DFlash was right (MTP does not belong in this argument — see §6.5). **See
§6.2 — this argument is weaker than it looks.**

**Maintenance is the real cost.** A custom kernel pins the project to one model
architecture and one MLX version. The standing position across this project is
that carrying fork patches is the thing worth the most effort to avoid, and a
Metal kernel is the most expensive possible version of one.

**Tooling and hardware churn.** No CUTLASS or cuDNN equivalent, thinner
profiling, and on a laptop the GPU also drives the display. This M5 already has
an fp16/bf16 NAX gap with no opt-out.

## 6. Adversarial pass

Five things did not survive review of §1-5.

**6.1 "oMLX's kernels explain its prefill speed" is unsupported.** Inferred from
filenames and `.metallib` presence. No benchmark of mlx-lm's GDN path against
oMLX's kernel has been run. Their existence proves someone believed it was worth
writing, not that it pays. **Not checked.**

**6.2 "Decode is bandwidth-bound" only holds at the roofline — and now it's
measured to.** The argument requires that measured decode throughput sits at the
achievable memory bandwidth. That other half is now established. An MLX
microbench (2026-08-14, prod down, same runtime as decode: `mx.sum` read-stream,
copy, and a decode-representative GEMV that streams a big fp16 weight matrix
once, random inputs so nothing constant-folds) puts this M5's **achievable
ceiling at ≈ 128 GB/s** — read-stream flat at 128 GB/s from 0.5–4 GB, GEMV
plateauing at 125 GB/s. Non-MTP 27B decode demand is ≈ 15 GB/token × 8.12 tok/s
≈ **122 GB/s, or 95% of the ceiling.** Decode really does sit at the roofline, so
the bandwidth-wall argument holds and the reasoning is *not* backwards: there is
no idle bandwidth for a better single-token kernel to capture. **Checked.** (This
is also why MTP pays — §6.5 — it attacks the roofline from the other side, by
reading the weights fewer times, not faster.)

**6.3 "The barrier is low" conflates two things.** Calling
`mx.fast.metal_kernel` is easy. Writing a correct *and faster* tiled kernel with
the right threadgroup memory layout is not, and the API does not help with that
part. Low barrier to entry, unchanged barrier to competence.

**6.4 The strongest objection is opportunity cost, not difficulty.** The
question is not whether a kernel can be written but whether it is the best use
of the time. On 2026-08-11 a cache change took one turn from 48.7s to 5.0s and a
second one took conversation openers from 10% to 78.7% reuse. No kernel is going
to approach either. Step 4 of the roadmap already gates kernels behind
profiling, and that gate is doing real work.

**6.5 MTP does not belong in the bandwidth-wall argument (§5).** The 2026-06-27
rejection of MTP was practical, not physical: the 4-bit checkpoint had its
`mtp.*` head stripped and re-converting a ~20 GB checkpoint was not worth it
then. Baked back and run on omlx (2026-08-15), MTP gives the dense 27B **8.1 →
19.1 tok/s (2.36×)** and the 35B-A3B MoE 1.34×, with no kernel. It wins
*because* decode is bandwidth-bound: a verify step amortises one weight-read
across several speculated tokens, using the ALU that single-token decode leaves
idle. So a non-kernel technique already moves the effective decode wall — which
cuts the same way as §6.4, the wins that matter here are not kernels. It does
**not** settle §6.2 (achievable bandwidth is still unmeasured) and it is not an
argument for writing one.

## 7. Position

Not now, and the reason is §6.4 rather than difficulty. The order that survives
scrutiny is: cache work, then measure, then consider a kernel only if profiling
names the GDN prefill chunk as the hot spot.

Two things would change this:

- **§6.2 resolves against the current position.** It didn't — measured: decode
  sits at ≈ 95% of M5's ≈ 128 GB/s achievable bandwidth, so there is no headroom
  a single-token kernel could reclaim and the "bandwidth wall" holds. This bullet
  is now closed; it would only reopen if a faster resident-set or weight-format
  change lifted the ceiling itself.
- **Reproducible evals need batch invariance.** That is a correctness
  requirement, independent of speed, and no amount of cache work addresses it.
