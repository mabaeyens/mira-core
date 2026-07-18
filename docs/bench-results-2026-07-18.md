# Benchmark Results — 2026-07-18

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Benchmark Results — 2026-07-18

### Timing

| Q | Difficulty | Category | ministral3-14b-mira-mlx-kvq8:ministral3-14b-mira-mlx-kvq8 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 8062ms | 8.1s | — |
| 2 | easy | code-no-tools | 398ms | 26.4s | — |
| 3 | medium | reasoning | 400ms | 77.7s | — |
| 4 | medium | long-output | 397ms | 43.0s | — |
| 5 | medium | thinking-toggle | 285ms | 94.1s | — |
| 6 | hard | agentic-single-tool | 11662ms | 13.5s | — |
| 7 | hard | agentic-multi-step | 11047ms | 14.2s | — |
| 8 | hard | agentic-read-reason | 561ms | 2.7s | — |
| 9 | expert | agentic-task-done | 5094ms | 7.1s | — |
| 11 | hard | agentic-write-file | 4354ms | 6.5s | — |
| 12 | hard | agentic-edit-file | 6022ms | 7.1s | — |
| 13 | expert | agentic-divergence-guard | 16620ms | 20.1s | — |
| 10 | expert | multi-turn-long-context | 49134ms | 192.0s | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-mira-mlx-kvq8 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | none | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-mira-mlx-kvq8 score |
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

## Benchmark Results — 2026-07-18

### Timing

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-kvq8:qwen3.6-35b-mira-mlx-kvq8 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 4514ms | 4.5s | — |
| 2 | easy | code-no-tools | 2847ms | 6.2s | — |
| 3 | medium | reasoning | 2836ms | 27.9s | — |
| 4 | medium | long-output | 2903ms | 15.1s | — |
| 5 | medium | thinking-toggle | 2900ms | 18.6s | — |
| 6 | hard | agentic-single-tool | 8035ms | 8.1s | — |
| 7 | hard | agentic-multi-step | 7261ms | 18.2s | — |
| 8 | hard | agentic-read-reason | 23802ms | 41.1s | — |
| 9 | expert | agentic-task-done | 6942ms | 7.9s | — |
| 11 | hard | agentic-write-file | 6404ms | 7.0s | — |
| 12 | hard | agentic-edit-file | 4186ms | 12.5s | — |
| 13 | expert | agentic-divergence-guard | 5386ms | 17.1s | — |
| 10 | expert | multi-turn-long-context | 32986ms | 68.8s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-35b-mira-mlx-kvq8 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-kvq8 score |
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