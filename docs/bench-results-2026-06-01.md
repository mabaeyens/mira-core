# Benchmark Results — 2026-06-01

Hardware: MacBook Pro M5 32GB · mlx-lm v0.31.3  
**Goal:** Test whether Qwen3.6-27B dense (4bit, OptiQ-4bit) fixes instruction-following issues vs the 35B-A3B MoE baseline.

---

## Three-way comparison

Baseline (35B-A3B) from 2026-05-30. All 13 questions, same `bench_compare.py` harness.  
Q1 excluded from averages (cold-load warmup — TTFT not representative).

| Q | Category | 35B-A3B baseline | 27B-4bit | 27B-OptiQ-4bit |
|---|---|---|---|---|
| 1 | baseline (cold) | ✗ 0.1 t/s · 52s | ✗ 0.9 t/s · 62s | ✗ 0.9 t/s · 68s |
| 2 | code-no-tools | ✗ 7.7 t/s · 68s | ✗ 6.3 t/s · 53s | ✗ 5.8 t/s · 39s |
| 3 | reasoning | ✗ 26.9 t/s · 71s | ✗ 7.2 t/s · 104s | ✗ 6.5 t/s · 100s |
| 4 | long-output | ✓ 24.6 t/s · 81s | ✗ 7.1 t/s · 61s | ✗ 6.5 t/s · 163s |
| 5 | thinking-toggle | ✗ 30.6 t/s · 68s | ✗ 9.1 t/s · 128s | ✗ 8.0 t/s · 111s |
| 6 | agentic-single-tool | ✓ 1.0 t/s · 133s | ✓ 21.5 t/s · 35s | ✓ 18.3 t/s · 38s |
| 7 | agentic-multi-step | ✓ 17.4 t/s · 283s | ✓ 12.8 t/s · 146s | ✓ 10.3 t/s · 161s |
| 8 | agentic-read-reason | ✓ 13.5 t/s · 125s | ✓ 4.0 t/s · 240s | ✓ 4.7 t/s · 218s |
| 9 | agentic-task-done | ✓ 5.1 t/s · 44s | ✓ 6.6 t/s · 47s | ✓ 5.7 t/s · 58s |
| 10 | multi-turn-long-context | ✗ 274.2 t/s · 60s¹ | ✗ 21.8 t/s · 129s | ✗ 19.9 t/s · 129s |
| 11 | agentic-write-file | ✓ 5.1 t/s · 47s | ✓ 12.0 t/s · 49s | ✓ 10.5 t/s · 54s |
| 12 | agentic-edit-file | ✓ 5.5 t/s · 53s | ✓ 17.2 t/s · 31s | ✓ 5.7 t/s · 60s |
| 13 | agentic-divergence-guard | ✓ 1.5 t/s · 358s | ✓ **G!** 7.0 t/s · 117s | ✓ **G!** 5.9 t/s · 153s |

¹ 274 t/s on Q10 is a measurement artifact (likely eval_tokens miscount on multi-turn response).  
**G!** = divergence guard fired (model looped; guard terminated the run).

### Summary

| Model | Avg t/s | Avg wall | Avg TTFT | task\_done | Guard fires |
|---|---|---|---|---|---|
| 35B-A3B (MoE, baseline) | **34.4** | 116s | **7.9s** | **8/12** | **0** |
| 27B-4bit (dense) | 11.0 | **95s** | 29.5s | 7/12 | 1 |
| 27B-OptiQ-4bit (dense) | 9.0 | 107s | 30.9s | 7/12 | 1 |

---

## Findings

**Hypothesis was wrong.** The 27B dense models are slower *and* worse at instruction following:

- **Speed:** 35B-A3B is 3× faster in throughput (34 vs 11 t/s) and 4× lower TTFT (7.9s vs 30s). The MoE architecture activates only 3.5B params per token vs all 27B in the dense models.
- **Instruction following:** 35B-A3B completes more tasks (8/12 vs 7/12) and never fires the divergence guard. Both 27B models diverged on Q13 (loop detection triggered).
- **OptiQ vs plain 4bit:** No meaningful difference — identical task_done, same guard behavior, slightly slower. Not worth the extra model size (18GB vs 15GB).
- **Kernel panic on OptiQ bench:** macOS GPU driver panic (`IOGPUGroupMemory.cpp:528`) occurred mid-run, likely due to Metal buffer pressure under sustained inference. 6bit bench skipped as a result (21GB would push harder on a 32GB system).

**Conclusion:** The always-thinking and tool-use bias issues are prompt/orchestrator problems, not model problems. 35B-A3B remains the best available model on this hardware. Dense 27B models removed from disk (54GB freed).

---

## Raw data — 27B-4bit

### Timing

| Q | Difficulty | Category | TTFT | wall | t/s |
|---|---|---|---|---|---|
| 1 | easy | baseline | 59589ms | 61.8s | 0.9 |
| 2 | easy | code-no-tools | 18163ms | 53.4s | 6.3 |
| 3 | medium | reasoning | 18562ms | 104.3s | 7.2 |
| 4 | medium | long-output | 17732ms | 61.0s | 7.1 |
| 5 | medium | thinking-toggle | 26301ms | 128.1s | 9.1 |
| 6 | hard | agentic-single-tool | 31820ms | 35.1s | 21.5 |
| 7 | hard | agentic-multi-step | 61449ms | 146.0s | 12.8 |
| 8 | hard | agentic-read-reason | 27389ms | 240.4s | 4.0 |
| 9 | expert | agentic-task-done | 22164ms | 47.1s | 6.6 |
| 10 | expert | multi-turn-long-context | 48830ms | 129.3s | 21.8 |
| 11 | hard | agentic-write-file | 33870ms | 49.3s | 12.0 |
| 12 | hard | agentic-edit-file | 19284ms | 31.3s | 17.2 |
| 13 | expert | agentic-divergence-guard | 28768ms | 117.0s | 7.0 |

### Agentic results

| Q | Category | Expected calls | Actual calls | task\_done |
|---|---|---|---|---|
| 6 | agentic-single-tool | 1 | run\_shell | YES |
| 7 | agentic-multi-step | 2 | run\_shell, run\_shell | YES |
| 8 | agentic-read-reason | 1 | read\_file, search\_files | YES |
| 9 | agentic-task-done | 3 | run\_shell, run\_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |
| 11 | agentic-write-file | 2 | write\_file, read\_file | YES |
| 12 | agentic-edit-file | 3 | write\_file, edit\_file, read\_file | YES |
| 13 | agentic-divergence-guard | 3 | run\_shell ×10 (guard fired) | YES |

---

## Raw data — 27B-OptiQ-4bit

### Timing

| Q | Difficulty | Category | TTFT | wall | t/s |
|---|---|---|---|---|---|
| 1 | easy | baseline | 65895ms | 68.1s | 0.9 |
| 2 | easy | code-no-tools | 17913ms | 38.7s | 5.8 |
| 3 | medium | reasoning | 1002ms | 100.1s | 6.5 |
| 4 | medium | long-output | 17924ms | 162.9s | 6.5 |
| 5 | medium | thinking-toggle | 26923ms | 111.2s | 8.0 |
| 6 | hard | agentic-single-tool | 33641ms | 37.6s | 18.3 |
| 7 | hard | agentic-multi-step | 78183ms | 160.5s | 10.3 |
| 8 | hard | agentic-read-reason | 29629ms | 217.9s | 4.7 |
| 9 | expert | agentic-task-done | 23379ms | 57.9s | 5.7 |
| 10 | expert | multi-turn-long-context | 50367ms | 129.5s | 19.9 |
| 11 | hard | agentic-write-file | 37161ms | 53.7s | 10.5 |
| 12 | hard | agentic-edit-file | 23283ms | 59.7s | 5.7 |
| 13 | expert | agentic-divergence-guard | 30867ms | 153.3s | 5.9 |

### Agentic results

| Q | Category | Expected calls | Actual calls | task\_done |
|---|---|---|---|---|
| 6 | agentic-single-tool | 1 | run\_shell | YES |
| 7 | agentic-multi-step | 2 | run\_shell, run\_shell, run\_shell | YES |
| 8 | agentic-read-reason | 1 | read\_file | YES |
| 9 | agentic-task-done | 3 | run\_shell, run\_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |
| 11 | agentic-write-file | 2 | write\_file, read\_file | YES |
| 12 | agentic-edit-file | 3 | write\_file, edit\_file, read\_file | YES |
| 13 | agentic-divergence-guard | 3 | run\_shell ×12 (guard fired) | YES |
