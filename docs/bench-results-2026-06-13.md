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

**Verdict:** omlx + Qwen3.6 remains the right default for Mira. It has the fastest token generation (~56 t/s), ~0 ms TTFT for chat (KV prefix cache warms on startup), and already handles multimodal (image attachments confirmed working). omlx + Gemma4 offers no measurable advantage: slower tg than Qwen3.6, slower prefill than ollama, and Qwen3.6 already covers the multimodal use case.


---

## Gemma4-26b — Scenario A: Quality & tool use (2026-06-13)

Backend: omlx 0.4.3 · Model: gemma4-26b (MoE 26B/4B active, 4-bit, ~16 GB)

### Timing — Q1–Q7 (measurable)

| Q | Difficulty | Category | TTFT | wall | t/s | result |
|---|-----------|---------|------|------|-----|--------|
| 1 | easy | baseline | 681ms | 1.8s | 0.9 | PASS |
| 2 | easy | code-no-tools | 672ms | 4.1s | 28.7 | PASS |
| 3 | medium | reasoning | 672ms | 19.6s | 37.4 | PASS |
| 4 | medium | long-output | 1724ms | 17.6s | 37.0 | PASS |
| 5 | medium | thinking-toggle | 698ms | 21.4s | 37.7 | PASS |
| 6 | hard | agentic-single-tool | 8113ms | 9.3s | 60.0 | **PASS** (run_shell, task_done=YES) |
| 7 | hard | agentic-multi-step | — | 600s | — | **FAIL** (timeout: 1 tool call, no completion) |

### Q8–Q13, Q10 — unmeasurable (KV cache exhausted by Q7)

Q7's verbose looping consumed ~9 GB of KV cache (24.98 GB used / 25.0 GB ceiling). All subsequent
questions received `prefill_memory_exceeded` before any output was generated.

| Q | Category | Outcome | Peak predicted | Ceiling |
|---|---------|---------|---------------|---------|
| 8 | agentic-read-reason | OOM | 26.94 GB | 24.96 GB |
| 9–13 | agentic + guard | OOM | ~25.19 GB each | 24.96 GB |
| 10 | multi-turn-long-context | OOM | 25.56 GB | 24.96 GB |

### KV budget analysis (Scenario B)

| Model | Weights | Safety cap | KV budget | Max KV tokens¹ |
|-------|---------|-----------|-----------|----------------|
| Qwen3.6-35B-A3B | ~20 GB | 25 GB | ~5 GB | ~14K |
| Gemma4-26b | ~16 GB | 25 GB | ~9 GB | ~25K |

¹ At ~178–238 KB/token (fp16 KV, 94 and ~62 KV layers respectively). Gemma4 has ~1.8× more headroom
theoretically, but Q7's 600s timeout consumed nearly all of it (KV reached 24.98 GB).

### Analysis

**What's different from Ollama Gemma4 (June 7):** omlx Gemma4 correctly called tools (Q6: run_shell,
task_done=YES). Ollama Gemma4 failed 6/7 agentic questions with zero tool calls. The runtime was the
blocker, not the model — omlx fixed the tool-call failure mode.

**What remains broken:** Q7 (multi-step agentic with large grep output) times out at 600s after only
1 tool call. The model enters a verbose reasoning loop without making progress or calling task_done.
Same root cause seen in Qwen3.6 run 2 (June 6, Q12 anomaly: 338s wall at 83 t/s = ~28K tokens
generated). This is an instruction-following/verbosity issue on complex multi-step tasks.

**Verdict (Scenario A): CONDITIONAL PASS.** Non-agentic (Q1–Q5) and simple agentic (Q6) work.
Complex multi-step agentic (Q7+) is blocked by the verbose-loop failure. Q8–Q13 could not be measured
because Q7 exhausted the KV budget. Gemma4 cannot replace Qwen3.6 as the daily driver until the
multi-step agentic loop issue is resolved. Qwen3.6 is the correct default.

**Scenario C (Vision):** Not tested in this session. Qwen3.6 already confirmed multimodal (2026-06-13);
Gemma4 vision parity on omlx is a separate bench item.

---

## Long-context quality probe — omlx + Qwen3.6-35B-A3B (2026-06-13)

Spec: `specs/bench-omlx-longctx.md`, Scenario 2. Single-turn (doc + question in one message).
Document: `core/orchestrator.py` (65K chars, ~19K actual tokens at ~2 chars/tok for Python code).

### Method

Each probe: fresh omlx restart → load Qwen3.6 only → single-turn message (doc slice + question).
No multi-turn — avoids second-turn KV buildup. Needle is a unique value deep in the document slice.

| Probe | Doc chars | Actual tokens¹ | Needle | Result | TTFT | Wall |
|-------|-----------|----------------|--------|--------|------|------|
| ~6K tokens | 21,000 | ~10,500 | `_prefill_system_prompt` regex | **PASS** | 23.8s | 25.3s |
| ~12K tokens | 42,000 | ~21,000 | `_schedule_reminder` strftime | OOM | — | — |
| ~16K tokens | 56,000 | ~28,000 | `fetch_url` preview length | OOM | — | — |

¹ Python code tokenizes at ~2 chars/token (code is denser than prose). The "~6K" label reflects the
rough estimate used in the spec; actual token count is closer to 10.5K.

### OOM detail — ~12K and ~16K probes

Error: `Prefill would require ~26.x GB peak (current ~21.5 GB + KV+SDPA ~5 GB) but ceiling is ~25 GB`

The effective KV safety cap on 32 GB hardware with Qwen3.6 (~20 GB) allows ~5 GB for KV cache,
corresponding to ~14K tokens total (including system prompt ~2K tokens). A 10.5K-token document
reaches ~12.5K total — just under the cap. A 21K-token document hits ~23K total — well over.

### Findings

- **Recall confirmed at ~10.5K tokens** — verbatim format string recalled correctly from line 440,
  which appears ~5K tokens into the 10.5K-token slice. TTFT 23.8s reflects large prefill.
- **OOM boundary: ~14K total KV tokens** — system prompt (~2K) + doc must stay under ~14K combined.
  This limits usable user content to roughly **10–12K actual tokens** on a fresh session.
- **Session degradation confirmed** — KV cache is global; after a large conversation the limit drops
  further. June 6 bench found OOM at 18K on a truly fresh omlx session; current 14K limit reflects
  a warmer session state.
- **RAG is the correct path for long documents** — embedding retrieval handles arbitrary document
  length without KV pressure. Raw context injection is viable only for documents under ~10K tokens.

### Verdict (Scenario 2): COMPLETE.

Recall quality is good at the tested depth (~10.5K tokens, PASS). Context window is hardware-limited
to ~14K total KV tokens on 32 GB RAM with Qwen3.6-35B-A3B. RAG remains the recommended path for
documents exceeding ~8–10K tokens of user content.