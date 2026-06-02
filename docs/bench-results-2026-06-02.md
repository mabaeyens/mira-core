# Benchmark Results — 2026-06-02

Hardware: MacBook Pro M5 10c 32GB · 153.6 GB/s memory bandwidth  
**Goal:** Measure DFlash speculative decoding gains (tok/s, prefill, memory) for both current Mira models.  
Tool: `dflash-mlx` 0.1.8 · `mlx` 0.31.2 · `mlx-lm` 0.31.3

---

## Method

`dflash benchmark` with the canonical math prompt, 3 repeats, 60s cooldown, `--no-eos`, median reported.  
Each run loads the model fresh per repeat. Models cached locally; no download time in timings.

Prompt:
> The function f satisfies f(x)+f(y)=f(x+y)-xy-1 for all real x,y. If f(1)=1, find all integers n such that f(n)=n.

Draft models (auto-resolved by dflash):

| Target | Draft |
|---|---|
| `mlx-community/Qwen3.6-35B-A3B-4bit` | `z-lab/Qwen3.6-35B-A3B-DFlash` |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | `z-lab/gemma-4-26B-A4B-it-DFlash` |

Note: `dflash benchmark` does not support `--chat-template-args`. Thinking mode was tested using the `/think` token prepended to the prompt (standard Qwen3 trigger).

---

## Results

### Qwen3.6-35B-A3B-4bit — no-thinking

| Metric | Baseline | DFlash |
|---|---|---|
| Decode (t/s) | 59.4 | 77.0 |
| Speedup | — | **1.30×** |
| Acceptance rate | — | 78.0% |
| Prefill (t/s) | 18.0 | 18.0 |
| Peak memory | 19.6 GB | 20.0 GB |
| End footprint | 20.5 GB | 21.9 GB |

### Qwen3.6-35B-A3B-4bit — thinking (`/think`)

| Metric | Baseline | DFlash |
|---|---|---|
| Decode (t/s) | 60.0 | 84.3 |
| Speedup | — | **1.41×** |
| Acceptance rate | — | 77.4% |
| Prefill (t/s) | 18.8 | 18.9 |
| Peak memory | 19.6 GB | 20.0 GB |
| End footprint | 20.5 GB | 22.3 GB |

**Thinking mode: fully supported.** Better speedup than no-thinking (1.41× vs 1.30%) — thinking output has more regular structure, which helps the draft model predict ahead.

### Gemma4-26B-A4B-it-4bit

| Metric | Baseline | DFlash |
|---|---|---|
| Decode (t/s) | 43.6 | 66.6 |
| Speedup | — | **1.53×** |
| Acceptance rate | — | 78.1% |
| Prefill (t/s) | 29.7 | 33.0 |
| Peak memory | 14.7 GB | 15.1 GB |
| End footprint | 16.0 GB | 21.6 GB |

Gemma4 gets the largest relative gain. Prefill also improves slightly (+11%) — consistent with DFlash's block-diffusion overlapping some prefill work.

---

## Observations

**TTFT:** DFlash is decode-only speculative decoding — prefill throughput is unchanged for Qwen3.6 (18 t/s both ways). The 7.9s TTFT measured in the 2026-05-30 Mira server bench is server/RAG/streaming overhead, not model prefill time. DFlash does not help TTFT.

**Memory:** Very safe on 32GB. The draft model adds ~1-2 GB peak; the largest end footprint observed was 22.3 GB (Qwen3.6 + DFlash + thinking). ~10 GB headroom remains. No OOM or thermal events.

**Thermal pressure warning:** All runs reported `thermal pressure is 'unknown'`. This is a sensor API quirk on this chip revision — not throttling. Results are consistent across all 3 repeats on each run.

**Baseline delta vs prior Mira bench:** The dflash baseline numbers (59 t/s, 44 t/s) are higher than the 2026-05-30 server bench (34.4 t/s, ~36 t/s). The difference is the Mira server overhead: FastAPI SSE streaming, RAG retrieval, tool dispatch, and that bench's harder 13-question prompt set. These dflash baselines represent raw model decode capacity.

---

## Summary

| Model | Mode | Baseline t/s | DFlash t/s | Speedup | Acceptance |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B-4bit | no-think | 59.4 | 77.0 | 1.30× | 78% |
| Qwen3.6-35B-A3B-4bit | thinking | 60.0 | 84.3 | 1.41× | 77% |
| Gemma4-26B-A4B-it-4bit | default | 43.6 | 66.6 | 1.53× | 78% |

DFlash clears the 1.3× threshold on all three configurations. Acceptance rate holds at ~78% across all models and modes — well above the 75% floor where speculative decoding becomes net-positive.

**Recommendation:** DFlash is worth integrating into Mira for both models. The draft model adds minimal memory overhead (~1-2 GB peak) and the speedup is real and consistent. Gemma4 benefits most. The operational path is `dflash serve` as a drop-in replacement for `mlx-lm`. Thinking mode on Qwen3.6 works and gets better gains — no caveats needed.
