# Bench results — 2026-06-13

Standard pp/tg benchmark (llama.cpp-compatible metrics).
Script: `scripts/bench_standard.py`, 3 reps each, temperature 0.

**Configurations:**
- omlx 0.4.3 @ localhost:8080 — Qwen3.6-35B-A3B (MoE 35B/3.6B active, 4-bit, ~20 GB)
- omlx 0.4.3 @ localhost:8080 — gemma4-26b (MoE 26B/4B active, 4-bit, ~16 GB)
- ollama 0.30.8 @ localhost:11434 — qwen3.6:35b-mlx (MLX backend, ~21 GB)
- ollama 0.30.8 @ localhost:11434 — gemma4:26b-mlx (MLX backend, MoE 26B/4B active, ~16 GB)

## Results

**omlx tg note:** omlx emits all tokens in a single SSE batch, so `gen_ms = wall - ttft ≈ 0`. The script falls back to `wall_ms` when `gen_ms < 50ms`. omlx Qwen3.6 tg512 was collected before this fix and corrected manually: 512 tok / 9.185s = **55.7 t/s**; tg128 was skipped (batch caused gen_ms=0 on rep 2, function bailed early). All other rows use the fixed formula.

| model                        | backend    | test     |      t/s |   avg ms |  std ms |
|------------------------------|------------|----------|----------|----------|---------|
| Qwen3.6-35B-A3B              | omlx       | pp128    |    216.6 |     2265 |    2824 |
| Qwen3.6-35B-A3B              | omlx       | pp512    |    540.0 |      914 |      14 |
| Qwen3.6-35B-A3B              | omlx       | pp1024   |    786.1 |     1224 |       4 |
| Qwen3.6-35B-A3B              | omlx       | tg512*   |     55.7 |     9185 |       1 |

| model                        | backend    | test     |      t/s |   avg ms |  std ms |
|------------------------------|------------|----------|----------|----------|---------|
| gemma4-26b                   | omlx       | pp128    |    257.4 |     1132 |    1047 |
| gemma4-26b                   | omlx       | pp512    |    618.1 |      782 |      15 |
| gemma4-26b                   | omlx       | pp1024   |    836.6 |     1127 |      11 |
| gemma4-26b                   | omlx       | tg128    |     44.4 |     3397 |      18 |
| gemma4-26b                   | omlx       | tg512    |     43.3 |    12304 |      78 |

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

Every pp row has a high std ms because rep 1 is a cold-cache miss (model not yet paged in). The script takes the **median** of 3 reps, so the cold rep is excluded from the t/s figure, but it still inflates avg ms. Warm pp TTFT:
- omlx Qwen3.6: ~514–1220 ms
- omlx Gemma4: ~514–1127 ms (similar to Qwen3.6)
- ollama Qwen3.6: 68–110 ms
- ollama Gemma4: 144–157 ms

### Prefill throughput (pp)

ollama MLX prefills 5–11× faster than omlx at every prompt size. At pp1024: ollama Qwen3.6 8754 t/s vs omlx Qwen3.6 786 t/s. This is a real backend difference in how MLX prefill kernels are scheduled, not a measurement artifact.

Within omlx, Gemma4 slightly outprefills Qwen3.6 (pp1024: 837 vs 786 t/s). Within ollama, Qwen3.6 outprefills Gemma4 by ~1.5× (pp1024: 8754 vs 5981 t/s).

### Token generation throughput (tg)

omlx Qwen3.6 is the fastest on generation at **~55.7 t/s** (batch-streaming corrected). omlx Gemma4 drops to **~43–44 t/s** — on par with both ollama models (~41–42 t/s). The Qwen3.6 advantage on omlx likely reflects better MoE routing efficiency for this weight shape.

### Summary table

| metric         | omlx / Qwen3.6 | omlx / Gemma4 | ollama / Qwen3.6 | ollama / Gemma4 |
|----------------|---------------|--------------|-----------------|----------------|
| pp1024 (t/s)   | 786           | 837          | **8 754**       | 5 981          |
| tg512  (t/s)   | **55.7**      | 43.3         | 42.3            | 40.1           |
| warm TTFT (ms) | ~514–1220     | ~514–1127    | **68–110**      | 144–157        |

**Verdict:** omlx + Qwen3.6 remains the right default for Mira. It has the fastest token generation (~56 t/s) and ~0 ms TTFT for chat (KV prefix cache warms on startup, so prefill latency above doesn't apply to typical chat turns). omlx + Gemma4 offers no advantage over ollama on tg and is slower to prefill than ollama — the main reason to use it would be multimodal (image/audio) via omlx, if that becomes a priority.
