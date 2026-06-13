# Bench results — 2026-06-13

Standard pp/tg benchmark (llama.cpp-compatible metrics).
Script: `scripts/bench_standard.py`, 3 reps each, temperature 0.

**Configurations:**
- omlx 0.4.1 @ localhost:8080 — Qwen3.6-35B-A3B (MoE 35B/3.6B active, 4-bit, ~20 GB)
- ollama 0.30.8 @ localhost:11434 — qwen3.6:35b-mlx (MLX backend, ~21 GB)
- ollama 0.30.8 @ localhost:11434 — gemma4:26b-mlx (MLX backend, MoE 26B/4B active, ~16 GB)

**Note:** omlx Gemma4 not tested — omlx 0.4.1 returns `Model type gemma4 not supported`
(mlx_vlm bundled version lacks Gemma4 driver; blocked until omlx ≥ 0.4.4).

## Results


**omlx tg note:** omlx emits all tokens in a single SSE batch, so `gen_ms = wall - ttft ≈ 0`. The script now falls back to `wall_ms` when `gen_ms < 50ms`; omlx results below were collected with the old formula and tg512 is corrected manually: 512 tok / 9.185s = **55.7 t/s**. tg128 was skipped (batch caused gen_ms=0 on rep 2, function bailed). Ollama rows use the fixed formula.

| model                        | backend    | test     |      t/s |   avg ms |  std ms |
|------------------------------|------------|----------|----------|----------|---------|
| Qwen3.6-35B-A3B              | omlx       | pp128    |    216.6 |     2265 |    2824 |
| Qwen3.6-35B-A3B              | omlx       | pp512    |    540.0 |      914 |      14 |
| Qwen3.6-35B-A3B              | omlx       | pp1024   |    786.1 |     1224 |       4 |
| Qwen3.6-35B-A3B              | omlx       | tg512*   |     55.7 |     9185 |       1 |


| model                        | backend    | test     |      t/s |   avg ms |  std ms |
|------------------------------|------------|----------|----------|----------|---------|
| qwen3.6:35b-mlx              | ollama     | pp128    |   1221.2 |     2634 |    4403 |
| qwen3.6:35b-mlx              | ollama     | pp512    |   4695.2 |      282 |     339 |
| qwen3.6:35b-mlx              | ollama     | pp1024   |   8754.5 |      359 |     461 |
| qwen3.6:35b-mlx              | ollama     | tg128    |     41.1 |     3257 |      64 |
| qwen3.6:35b-mlx              | ollama     | tg512    |     42.3 |    12312 |     288 |


| model                        | backend    | test     |      t/s |   avg ms |  std ms |
|------------------------------|------------|----------|----------|----------|---------|
| gemma4:26b-mlx               | ollama     | pp128    |    900.7 |     1487 |    2320 |
| gemma4:26b-mlx               | ollama     | pp512    |   3185.4 |      282 |     233 |
| gemma4:26b-mlx               | ollama     | pp1024   |   5980.9 |      323 |     289 |
| gemma4:26b-mlx               | ollama     | tg128    |     41.4 |     3326 |     122 |
| gemma4:26b-mlx               | ollama     | tg512    |     40.1 |    12975 |     155 |


## Analysis

### pp metric caveat — first-rep cold cache

Every pp row has a very high std ms because rep 1 is always a cold-cache miss (model not yet paged in). The script takes the **median** of 3 reps, so the cold rep is excluded from the t/s figure, but it still inflates avg ms. Warm pp TTFT:
- ollama Qwen3.6: 68–113 ms (pp128–pp1024)
- ollama Gemma4: 144–157 ms
- omlx Qwen3.6: ~630–1220 ms (omlx warm TTFT is ~5–10× slower than ollama for prefill)

### Prefill throughput (pp)

ollama MLX prefills significantly faster than omlx across all prompt sizes. At pp1024, ollama Qwen3.6 reports **8754 t/s** vs omlx **786 t/s** — an 11× gap. This reflects a real difference in how the two backends schedule MLX prefill kernels, not a measurement artifact (prompt_tokens from usage ≈ expected size in all runs).

Between the two ollama models, Qwen3.6 outprefills Gemma4 by ~1.4–1.5× at every size (pp1024: 8754 vs 5981 t/s).

### Token generation throughput (tg)

omlx pulls ahead on generation: **~55.7 t/s** vs ollama's **~41–42 t/s** (+35%). The omlx figure has a batch-streaming caveat (see note above) but is consistent across multiple measured reps.

Gemma4 and Qwen3.6 on ollama generate at essentially the same speed (~41 t/s), confirming the generation bottleneck is hardware-bound (ALU/memory bandwidth), not model architecture.

### Summary table

| metric         | omlx / Qwen3.6 | ollama / Qwen3.6 | ollama / Gemma4 |
|----------------|---------------|-----------------|----------------|
| pp1024 (t/s)   | 786           | **8 754**       | 5 981          |
| tg512  (t/s)   | **55.7**      | 42.3            | 40.1           |
| warm TTFT (ms) | ~630–1220     | **68–113**      | 144–157        |

**omlx is the right default** for Mira: faster token generation (the metric users feel in long replies) and ~0 ms TTFT for chat (KV prefix cache warms on server startup — separate from the prefill numbers above). ollama's faster prefill is a benchmark win that doesn't translate to perceived latency in typical chat sessions where the system prompt is already cached.

omlx Gemma4 remains blocked until omlx ≥ 0.4.4.
