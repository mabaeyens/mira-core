# Benchmark Results — 2026-07-07

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Benchmark Results — 2026-07-07

### Timing

| Q | Difficulty | Category | qwen36-omlx-regression:qwen36-omlx-regression TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Error code: 404 - {'detail': 'The model `Qwen3.6-35B-A3B` does not exist. Available model: `mlx-community/Ministral-3-14B-Instruct-2512-4bit`'} | — | — |
| 7 | hard | agentic-multi-step | ERR: Error code: 404 - {'detail': 'The model `Qwen3.6-35B-A3B` does not exist. Available model: `mlx-community/Ministral-3-14B-Instruct-2512-4bit`'} | — | — |
| 8 | hard | agentic-read-reason | ERR: Error code: 404 - {'detail': 'The model `Qwen3.6-35B-A3B` does not exist. Available model: `mlx-community/Ministral-3-14B-Instruct-2512-4bit`'} | — | — |
| 9 | expert | agentic-task-done | ERR: Error code: 404 - {'detail': 'The model `Qwen3.6-35B-A3B` does not exist. Available model: `mlx-community/Ministral-3-14B-Instruct-2512-4bit`'} | — | — |
| 11 | hard | agentic-write-file | ERR: Error code: 404 - {'detail': 'The model `Qwen3.6-35B-A3B` does not exist. Available model: `mlx-community/Ministral-3-14B-Instruct-2512-4bit`'} | — | — |
| 12 | hard | agentic-edit-file | ERR: Error code: 404 - {'detail': 'The model `Qwen3.6-35B-A3B` does not exist. Available model: `mlx-community/Ministral-3-14B-Instruct-2512-4bit`'} | — | — |
| 13 | expert | agentic-divergence-guard | ERR: Error code: 404 - {'detail': 'The model `Qwen3.6-35B-A3B` does not exist. Available model: `mlx-community/Ministral-3-14B-Instruct-2512-4bit`'} | — | — |

### Agentic results

| Q | Category | Expected calls | qwen36-omlx-regression calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | ERR | — |
| 11 | agentic-write-file | 2 | ERR | — |
| 12 | agentic-edit-file | 3 | ERR | — |
| 13 | agentic-divergence-guard | 3 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen36-omlx-regression score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 13 | expert | agentic-divergence-guard | — |

---

## Benchmark Results — 2026-07-07

### Timing

| Q | Difficulty | Category | qwen36-omlx-regression:qwen36-omlx-regression TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | 25763ms | 25.8s | 16083.0 |
| 7 | hard | agentic-multi-step | 13727ms | 25.8s | 67.8 |
| 8 | hard | agentic-read-reason | 28562ms | 47.0s | 34.9 |
| 9 | expert | agentic-task-done | 6683ms | 7.8s | 113.6 |
| 11 | hard | agentic-write-file | 8865ms | 9.3s | 245.2 |
| 12 | hard | agentic-edit-file | 1926ms | 12.4s | 15.7 |
| 13 | expert | agentic-divergence-guard | 2856ms | 43.3s | — |

### Agentic results

| Q | Category | Expected calls | qwen36-omlx-regression calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen36-omlx-regression score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 13 | expert | agentic-divergence-guard | — |

---

## Benchmark Results — 2026-07-07

### Timing

| Q | Difficulty | Category | ministral3-14b-vllmmlx-v5:ministral3-14b-vllmmlx-v5 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 5073ms | 5.2s | 23.2 |
| 2 | easy | code-no-tools | 4722ms | 17.7s | 15.9 |
| 3 | medium | reasoning | 377ms | 94.8s | 15.6 |
| 4 | medium | long-output | 4750ms | 48.7s | 15.7 |
| 5 | medium | thinking-toggle | 4793ms | 79.3s | 15.6 |
| 6 | hard | agentic-single-tool | 17443ms | 20.4s | 28.8 |
| 7 | hard | agentic-multi-step | 200263ms | 206.1s | 453.5 |
| 8 | hard | agentic-read-reason | 9394ms | 135.6s | 6.5 |
| 9 | expert | agentic-task-done | 28151ms | 30.8s | 38.5 |
| 11 | hard | agentic-write-file | 25685ms | 27.9s | 37.6 |
| 12 | hard | agentic-edit-file | 27620ms | 28.6s | 83.6 |
| 13 | expert | agentic-divergence-guard | — | 124.3s | — |
| 10 | expert | multi-turn-long-context | 174205ms | 259.0s | 13.9 |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-vllmmlx-v5 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell | YES |
| 8 | agentic-read-reason | 1 | list_files, read_file | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-vllmmlx-v5 score |
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