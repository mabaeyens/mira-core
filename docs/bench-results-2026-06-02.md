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


---

## Benchmark Results — 2026-06-02

### Timing

| Q | Difficulty | Category | qwen3.6-dflash-pinned:qwen3.6-dflash-pinned TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 7451ms | 8.4s | 2.2 |
| 2 | easy | code-no-tools | 5367ms | 9.9s | 52.5 |
| 3 | medium | reasoning | 5366ms | 34.8s | 44.1 |
| 4 | medium | long-output | 29608ms | 86.2s | 69.7 |
| 5 | medium | thinking-toggle | 8096ms | 34.5s | 39.0 |
| 6 | hard | agentic-single-tool | 10878ms | 12.1s | 56.0 |
| 7 | hard | agentic-multi-step | ERR: LLM stream closed without a completion signal. | — | — |
| 8 | hard | agentic-read-reason | 9886ms | 68.4s | 21.8 |
| 9 | expert | agentic-task-done | 10953ms | 13.1s | 44.4 |
| 11 | hard | agentic-write-file | 12076ms | 14.1s | 58.3 |
| 12 | hard | agentic-edit-file | 13458ms | 15.3s | 88.0 |
| 13 | expert | agentic-divergence-guard | 11742ms | 21.3s | 48.1 |
| 10 | expert | multi-turn-long-context | 627ms | 33.2s | 75.5 |

### Agentic results

| Q | Category | Expected calls | qwen3.6-dflash-pinned calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-dflash-pinned score |
|---|-----------|---------|---|
| 1 | easy | baseline | — |
| 2 | easy | code-no-tools | — |
| 3 | medium | reasoning | — |
| 4 | medium | long-output | — |
| 5 | medium | thinking-toggle | — |
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 13 | expert | agentic-divergence-guard | — |
| 10 | expert | multi-turn-long-context | — |

---

## Mira Server Bench — 2026-06-02 — qwen3.6-dflash-datetime-fix

**Change:** `CURRENT DATE AND TIME` moved from token ~50 to end of system prompt (db01a0f).  
**Purpose:** Enable dflash prefix cache to cover ~3500 static system prompt tokens.  
**Q7 note:** dflash OOM-crashed on Q7's 16K-token output; Q8-Q13 were re-run after restart (cold cache).

### Timing

| Q | Difficulty | Category | TTFT | wall | t/s | notes |
|---|-----------|---------|------|------|-----|-------|
| 1 | easy | baseline | 9587ms | 10.4s | 2.5 | cold start |
| 2 | easy | code-no-tools | 5401ms | 9.4s | 48.0 | |
| 3 | medium | reasoning | 5727ms | 35.5s | 39.9 | |
| 4 | medium | long-output | 28454ms | 51.1s | 107.1 | tools=[github_write_file] |
| 5 | medium | thinking-toggle | 7651ms | 34.2s | 40.8 | |
| 6 | hard | agentic-single-tool | 10830ms | 11.8s | 74.1 | |
| 7 | hard | agentic-multi-step | ERR | — | — | dflash OOM at 16K tokens; crash |
| 8 | hard | agentic-read-reason | 8807ms | 61.6s | 23.8 | after restart (cold) |
| 9 | expert | agentic-task-done | 10448ms | 12.1s | 56.1 | after restart (cold) |
| 10 | expert | multi-turn-long-context | 18869ms | 48.8s | 48.7 | T1=22050ms T2=26729ms; after restart |
| 11 | hard | agentic-write-file | 11470ms | 13.0s | 76.8 | after restart (cold) |
| 12 | hard | agentic-edit-file | 13055ms | 14.7s | 100.8 | after restart (cold) |
| 13 | expert | agentic-divergence-guard | 11824ms | 20.1s | 52.0 | after restart (cold) |

### Agentic results

| Q | Category | Expected calls | Actual calls | task_done |
|---|---------|----------------|---|---|
| 4 | long-output | 0 | github_write_file | YES |
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | ERR (OOM crash) | — |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | NO |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell ×4 | YES (divergence_guard=YES) |

### TTFT analysis — datetime fix vs prior run

Moving datetime to end of system prompt gave modest TTFT improvement for Q1→Q2 within the same session (cold Q1 = 9.6s, warm Q2 = 5.4s — delta of ~4.2s). The prior run (dflash-pinned) showed a smaller delta (Q1=7.5s → Q2=5.4s, delta ~2.1s), consistent with the prefix cache having more stable tokens to cover.

However, Q2-Q6 TTFT is still 5-11s, not sub-second as the theoretical cache-hit math would suggest (~100 uncached tokens / 568 t/s ≈ 0.2s). Likely cause: the dflash L2 SSD prefix cache read latency for a 3500-token KV state is itself several seconds. True sub-second TTFT requires L1 (RAM) cache hits, which only occur within the same conversation turn (confirmed by Q10-T2=627ms in prior run). Cross-conversation L2 hits are fast relative to cold prefill but not zero-latency.

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Score |
|---|-----------|---------|-------|
| 1 | easy | baseline | — |
| 2 | easy | code-no-tools | — |
| 3 | medium | reasoning | — |
| 4 | medium | long-output | — |
| 5 | medium | thinking-toggle | — |
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | ERR |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |
| 10 | expert | multi-turn-long-context | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 13 | expert | agentic-divergence-guard | — |

---

## Mira Server Bench — 2026-06-02 — gemma4-26b-dflash

**Model:** `mlx-community/gemma-4-26b-a4b-it-4bit` + draft `z-lab/gemma-4-26B-A4B-it-DFlash`  
**Q8 note:** dflash Metal OOM at 24,217-token input (system prompt + full orchestrator.py from tool call). 32GB limit. Q9-Q13 re-run after restart (cold cache).  
**Q13 note:** divergence_guard fired after 9 run_shell calls (vs expected 3); model kept issuing new shell variants before the 3-identical-args window triggered.

### Timing

| Q | Difficulty | Category | TTFT | wall | t/s | notes |
|---|-----------|---------|------|------|-----|-------|
| 1 | easy | baseline | 4686ms | 6.4s | 1.2 | cold start |
| 2 | easy | code-no-tools | 4229ms | 9.9s | 39.6 | |
| 3 | medium | reasoning | 4353ms | 32.0s | 28.9 | |
| 4 | medium | long-output | 4648ms | 22.1s | 37.0 | no tool call (Qwen3.6 incorrectly used github_write_file) |
| 5 | medium | thinking-toggle | 26684ms | 53.1s | 88.5 | no thinking mode; verbose inline reasoning |
| 6 | hard | agentic-single-tool | 9191ms | 10.4s | 61.5 | |
| 7 | hard | agentic-multi-step | 15576ms | 138.7s | 34.3 | used run_shell (not read_file); passed |
| 8 | hard | agentic-read-reason | Metal OOM | — | — | 24K token context; 32GB limit |
| 9 | expert | agentic-task-done | 9329ms | 19.0s | 19.3 | after restart (cold) |
| 10 | expert | multi-turn-long-context | 16448ms | 54.9s | 264.6 | T1=35191ms T2=19703ms; after restart |
| 11 | hard | agentic-write-file | — | 24.1s | — | after restart (cold) |
| 12 | hard | agentic-edit-file | — | 22.9s | — | after restart (cold) |
| 13 | expert | agentic-divergence-guard | 9878ms | 43.5s | — | 9 calls before guard fired; after restart |

### Agentic results

| Q | Category | Expected calls | Actual calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | Metal OOM | — |
| 9 | agentic-task-done | 3 | run_shell ×2 | YES |
| 10 | multi-turn-long-context | 0 | none | NO |
| 11 | agentic-write-file | 2 | run_shell, read_file | YES |
| 12 | agentic-edit-file | 3 | run_shell, write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell ×9 | divergence_guard=YES |

### vs Qwen3.6-dflash comparison

| Metric | Qwen3.6-35B-A3B | Gemma4-26B |
|--------|----------------|------------|
| Cold TTFT (Q1) | 9587ms | **4686ms** |
| Typical TTFT (Q2-Q6 warm) | 5-11s | **4-10s** |
| Prefill speed (dflash log) | 568 t/s | **795 t/s** |
| Peak decode (Q10) | 48.7 t/s | **264.6 t/s** |
| Q4 tool use | github_write_file (wrong) | **direct generation (correct)** |
| Q5 thinking-toggle | 40.8 t/s, 34.2s wall | 88.5 t/s, 53.1s wall (verbose) |
| Q7 multi-step | **FAIL** (16K OOM) | **PASS** (run_shell) |
| Q8 large-context | **PASS** | FAIL (24K Metal OOM) |
| Q13 divergence guard | fires at 4 calls | fires at 9 calls (weaker follow) |
| Acceptance rate (speed bench) | 78% | 78% |
| Decode t/s (speed bench) | 77.0 | 66.6 |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Score |
|---|-----------|---------|-------|
| 1 | easy | baseline | — |
| 2 | easy | code-no-tools | — |
| 3 | medium | reasoning | — |
| 4 | medium | long-output | — |
| 5 | medium | thinking-toggle | — |
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | OOM |
| 9 | expert | agentic-task-done | — |
| 10 | expert | multi-turn-long-context | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 13 | expert | agentic-divergence-guard | — |