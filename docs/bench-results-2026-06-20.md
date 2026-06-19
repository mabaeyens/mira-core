# Benchmark Results — 2026-06-20

Hardware: MacBook Pro M5 32GB · Backend: omlx 0.4.4 · Model: Qwen3.6-35B-A3B

## Benchmark Results — 2026-06-20

### Timing

| Q | Difficulty | Category | Qwen3.6-35B-A3B:Qwen3.6-35B-A3B TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 16017ms | 16.6s | 113.6 |
| 2 | easy | code-no-tools | 1918ms | 5.6s | 80.9 |
| 3 | medium | reasoning | 1901ms | 30.0s | 57.2 |
| 4 | medium | long-output | 2177ms | 14.6s | 62.3 |
| 5 | medium | thinking-toggle | 2038ms | 28.8s | 58.2 |
| 6 | hard | agentic-single-tool | 5184ms | 9.6s | 49.8 |
| 7 | hard | agentic-multi-step | ERR: Prefill context too large for available memory (pre-chunk guard at 2048 tokens, kv_len=6144): predicted peak would exceed prefill safety cap 21.8GB (90% of effective ceiling 24.2GB) | — | — |
| 8 | hard | agentic-read-reason | ERR: Prefill context too large for available memory (pre-chunk guard at 9688 tokens, kv_len=11736): predicted peak would exceed prefill safety cap 21.8GB (90% of effective ceiling 24.2GB) | — | — |
| 9 | expert | agentic-task-done | 3101ms | 10.9s | 61.8 |
| 11 | hard | agentic-write-file | 2278ms | 12.3s | 34.6 |
| 12 | hard | agentic-edit-file | 2950ms | 19.1s | 32.4 |
| 13 | expert | agentic-divergence-guard | 2550ms | 30.4s | 65.2 |
| 10 | expert | multi-turn-long-context | ERR: Prefill context too large for available memory (pre-chunk guard at 4096 tokens, kv_len=6144): predicted peak would exceed prefill safety cap 21.8GB (90% of effective ceiling 24.2GB) | — | — |

### Agentic results

| Q | Category | Expected calls | Qwen3.6-35B-A3B calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Qwen3.6-35B-A3B score |
|---|-----------|---------|---|
| 1 | easy | baseline | 2 |
| 2 | easy | code-no-tools | 2 |
| 3 | medium | reasoning | 2 |
| 4 | medium | long-output | 2 |
| 5 | medium | thinking-toggle | 2 |
| 6 | hard | agentic-single-tool | 2 |
| 7 | hard | agentic-multi-step | 0 (omlx OOM) |
| 8 | hard | agentic-read-reason | 0 (omlx OOM) |
| 9 | expert | agentic-task-done | 2 |
| 11 | hard | agentic-write-file | 2 |
| 12 | hard | agentic-edit-file | 1 (correct tool sequence, no confirmation text) |
| 13 | expert | agentic-divergence-guard | 2 |
| 10 | expert | multi-turn-long-context | 0 (omlx OOM) |

Suggested total: **19/26** — but the three 0s are omlx's 32GB prefill memory-guard (OOM), not model capability; on a memory-unconstrained backend Qwen3.6 would likely score ~25/26.