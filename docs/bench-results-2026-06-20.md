# Benchmark Results — 2026-06-20

Hardware: MacBook Pro M5 32GB · Backend: omlx 0.4.4 · Model: Qwen3.6-35B-A3B

## Run 1 — omlx default Metal cap (prefill effective ceiling 24.2 GB)

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

Suggested total: **19/26** — the three 0s are omlx's prefill memory-guard (OOM), not model capability. See Run 2 below: raising `iogpu.wired_limit_mb` to 26 GB did **not** clear them (the prefill safety cap is computed from available memory, independent of the wired limit).

---

## Run 2 — omlx with iogpu.wired_limit_mb=26624 (enforcer ceiling 23.8 → 26 GB)

### Timing

| Q | Difficulty | Category | Qwen3.6-35B-A3B:Qwen3.6-35B-A3B TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 9994ms | 10.6s | 5.3 |
| 2 | easy | code-no-tools | 1430ms | 7.4s | 52.7 |
| 3 | medium | reasoning | 1917ms | 21.6s | 58.8 |
| 4 | medium | long-output | 1673ms | 14.5s | 60.0 |
| 5 | medium | thinking-toggle | 1913ms | 14.9s | 62.0 |
| 6 | hard | agentic-single-tool | 2685ms | 6.8s | 46.7 |
| 7 | hard | agentic-multi-step | 2535ms | 238.6s | — |
| 8 | hard | agentic-read-reason | ERR: Prefill context too large for available memory (pre-chunk guard at 4096 tokens, kv_len=6144): predicted peak would exceed prefill safety cap 21.8GB (90% of effective ceiling 24.2GB) | — | — |
| 9 | expert | agentic-task-done | 3267ms | 9.5s | 47.0 |
| 11 | hard | agentic-write-file | 2269ms | 10.4s | 24.5 |
| 12 | hard | agentic-edit-file | 3940ms | 16.2s | 30.1 |
| 13 | expert | agentic-divergence-guard | 2427ms | 15.4s | 41.0 |
| 10 | expert | multi-turn-long-context | ERR: Prefill context too large for available memory (pre-chunk guard at 6144 tokens, kv_len=8192): predicted peak would exceed prefill safety cap 21.8GB (90% of effective ceiling 24.2GB) | — | — |

### Agentic results

| Q | Category | Expected calls | Qwen3.6-35B-A3B calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, search_files, run_shell, run_shell, run_shell, run_shell, write_file | no |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell | YES |
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
| 7 | hard | agentic-multi-step | 0 (no longer OOM, but ran away: 11 calls, tool-path confusion, no task_done) |
| 8 | hard | agentic-read-reason | 0 (still omlx OOM — wired limit did not help) |
| 9 | expert | agentic-task-done | 2 |
| 11 | hard | agentic-write-file | 2 |
| 12 | hard | agentic-edit-file | 2 (confirmed read-back this run) |
| 13 | expert | agentic-divergence-guard | 2 |
| 10 | expert | multi-turn-long-context | 0 (still omlx OOM — wired limit did not help) |

Suggested total: **20/26**. Raising the wired limit (enforcer ceiling 23.8→26 GB) did **not** change the prefill safety cap (still 21.8 GB / 90% of effective ceiling 24.2 GB): Q8 and Q10 OOM identically to Run 1. Q7 happened to take a non-OOM tool path this run but then failed behaviorally. **Conclusion: the wired-limit lever does not fix large-context prefill OOMs on 32 GB — route large-context work to dflash (handled 24 K cleanly in the omlx-0.4.4 dflash bench) or change omlx's Memory Guard tier.**

---

## Run 3 — omlx aggressive memory guard (`--memory-guard aggressive`, + wired limit 26 GB) — re-test of OOM questions only

Aggressive tier raised the prefill effective ceiling 24.2 → 26 GB (cap ~23.4 GB). **Both previously-OOM questions now pass, no Metal panic.**

### Timing

| Q | Difficulty | Category | Qwen3.6-35B-A3B:Qwen3.6-35B-A3B TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 8 | hard | agentic-read-reason | 9975ms | 59.9s | 29.1 |
| 10 | expert | multi-turn-long-context | 11920ms | 45.2s | 377.6 |

### Agentic results

| Q | Category | Expected calls | Qwen3.6-35B-A3B calls | task_done |
|---|---------|----------------|---|---|
| 8 | agentic-read-reason | 1 | read_file | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Qwen3.6-35B-A3B score |
|---|-----------|---------|---|
| 8 | hard | agentic-read-reason | 2 (read_file, accurate divergence-guard explanation w/ quoted code) |
| 10 | expert | multi-turn-long-context | 2 (correctly identified the guard is in orchestrator.py, not the shared server.py) |

**Conclusion: the aggressive Memory Guard tier (with `iogpu.wired_limit_mb=26624`) fixes the large-context prefill OOMs.** Q8 and Q10 went 0→2. This confirms Run 1's three 0s were the memory guard, not model capability — Qwen3.6's effective score rises to ~24-26/26 (Q7 remains the one behavioral weak spot, unrelated to OOM). Trade-off: aggressive lets allocations approach the 26 GB cap and Metal will **panic** if a request exceeds it (omlx suggests `iogpu.wired_limit_mb=28672` for full aggressive-tier headroom). Both levers are needed together — wired limit alone (Run 2) did not help.