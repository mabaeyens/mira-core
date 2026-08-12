# Benchmark Results — 2026-07-10

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Benchmark Results — 2026-07-10

### Timing

| Q | Difficulty | Category | ministral3-14b:mira-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 9325ms | 9.4s | — |
| 2 | easy | code-no-tools | 439ms | 26.5s | — |
| 3 | medium | reasoning | 424ms | 87.9s | — |
| 4 | medium | long-output | 462ms | 40.8s | — |
| 5 | medium | thinking-toggle | 323ms | 88.4s | — |
| 6 | hard | agentic-single-tool | 10956ms | 12.9s | — |
| 7 | hard | agentic-multi-step | 13855ms | 17.8s | — |
| 8 | hard | agentic-read-reason | 598ms | 3.9s | — |
| 9 | expert | agentic-task-done | 5621ms | 8.0s | — |
| 11 | hard | agentic-write-file | 4643ms | 7.2s | — |
| 12 | hard | agentic-edit-file | 6621ms | 7.8s | — |
| 13 | expert | agentic-divergence-guard | 17996ms | 21.8s | — |
| 10 | expert | multi-turn-long-context | 42990ms | 156.0s | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b:mira-mlx calls | task_done |
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

| Q | Difficulty | Category | ministral3-14b:mira-mlx score |
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

## Benchmark Results — 2026-07-10

### Timing

| Q | Difficulty | Category | qwen3.6:mira-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 3102ms | 3.9s | — |
| 2 | easy | code-no-tools | 2821ms | 6.1s | — |
| 3 | medium | reasoning | 2801ms | 22.9s | — |
| 4 | medium | long-output | 3010ms | 20.4s | — |
| 5 | medium | thinking-toggle | 2915ms | 24.9s | — |
| 6 | hard | agentic-single-tool | 4149ms | 6.5s | — |
| 7 | hard | agentic-multi-step | 4288ms | 11.2s | — |
| 8 | hard | agentic-read-reason | 4109ms | 5.2s | — |
| 9 | expert | agentic-task-done | 4118ms | 6.0s | — |
| 11 | hard | agentic-write-file | 4052ms | 6.0s | — |
| 12 | hard | agentic-edit-file | 4076ms | 5.9s | — |
| 13 | expert | agentic-divergence-guard | 4115ms | 5.8s | — |
| 10 | expert | multi-turn-long-context | 28961ms | 72.3s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6:mira-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | none | no |
| 7 | agentic-multi-step | 2 | none | no |
| 8 | agentic-read-reason | 1 | none | no |
| 9 | agentic-task-done | 3 | none | no |
| 11 | agentic-write-file | 2 | none | no |
| 12 | agentic-edit-file | 3 | none | no |
| 13 | agentic-divergence-guard | 3 | none | no |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:mira-mlx score |
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

## Benchmark Results — 2026-07-10

### Timing

| Q | Difficulty | Category | qwen3.6:mira-mlx-fixed TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 10482ms | 10.5s | — |
| 2 | easy | code-no-tools | 2909ms | 6.3s | — |
| 3 | medium | reasoning | 2861ms | 34.0s | — |
| 4 | medium | long-output | 22356ms | 41.3s | — |
| 5 | medium | thinking-toggle | 2937ms | 21.3s | — |
| 6 | hard | agentic-single-tool | 8121ms | 8.2s | — |
| 7 | hard | agentic-multi-step | 10548ms | 20.6s | — |
| 8 | hard | agentic-read-reason | 21473ms | 32.1s | — |
| 9 | expert | agentic-task-done | 7253ms | 8.3s | — |
| 11 | hard | agentic-write-file | 6588ms | 7.2s | — |
| 12 | hard | agentic-edit-file | 4184ms | 12.6s | — |
| 13 | expert | agentic-divergence-guard | 4422ms | 27.4s | — |
| 10 | expert | multi-turn-long-context | 29128ms | 60.4s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6:mira-mlx-fixed calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:mira-mlx-fixed score |
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