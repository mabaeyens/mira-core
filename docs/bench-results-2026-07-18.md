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

---

## Benchmark Results — 2026-07-18

### Timing

| Q | Difficulty | Category | ministral3-14b-mira-mlx-kvq8-isolated:ministral3-14b-mira-mlx-kvq8-isolated TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 7967ms | 8.1s | — |
| 2 | easy | code-no-tools | 414ms | 25.5s | — |
| 3 | medium | reasoning | 386ms | 87.9s | — |
| 4 | medium | long-output | 410ms | 36.6s | — |
| 5 | medium | thinking-toggle | 276ms | 65.2s | — |
| 6 | hard | agentic-single-tool | 11638ms | 13.5s | — |
| 7 | hard | agentic-multi-step | 11155ms | 14.4s | — |
| 8 | hard | agentic-read-reason | 556ms | 4.1s | — |
| 9 | expert | agentic-task-done | 5068ms | 7.6s | — |
| 11 | hard | agentic-write-file | 4308ms | 6.4s | — |
| 12 | hard | agentic-edit-file | 6005ms | 7.4s | — |
| 13 | expert | agentic-divergence-guard | 16558ms | 20.2s | — |
| 10 | expert | multi-turn-long-context | 78270ms | 222.0s | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-mira-mlx-kvq8-isolated calls | task_done |
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

| Q | Difficulty | Category | ministral3-14b-mira-mlx-kvq8-isolated score |
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

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-kvq8-isolated:qwen3.6-35b-mira-mlx-kvq8-isolated TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 4336ms | 4.4s | — |
| 2 | easy | code-no-tools | 2862ms | 6.2s | — |
| 3 | medium | reasoning | 2841ms | 23.4s | — |
| 4 | medium | long-output | 2893ms | 14.5s | — |
| 5 | medium | thinking-toggle | 2931ms | 18.5s | — |
| 6 | hard | agentic-single-tool | 7998ms | 8.1s | — |
| 7 | hard | agentic-multi-step | 16455ms | 30.5s | — |
| 8 | hard | agentic-read-reason | 23711ms | 58.5s | — |
| 9 | expert | agentic-task-done | 5869ms | 6.6s | — |
| 11 | hard | agentic-write-file | 6396ms | 7.0s | — |
| 12 | hard | agentic-edit-file | 4203ms | 12.6s | — |
| 13 | expert | agentic-divergence-guard | 4225ms | 46.8s | — |
| 10 | expert | multi-turn-long-context | 32551ms | 67.4s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-35b-mira-mlx-kvq8-isolated calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, search_files | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-kvq8-isolated score |
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

| Q | Difficulty | Category | ministral3-14b-mira-mlx-unquant-isolated:ministral3-14b-mira-mlx-unquant-isolated TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 7998ms | 8.1s | — |
| 2 | easy | code-no-tools | 406ms | 26.3s | — |
| 3 | medium | reasoning | 388ms | 71.0s | — |
| 4 | medium | long-output | 403ms | 41.6s | — |
| 5 | medium | thinking-toggle | 280ms | 82.9s | — |
| 6 | hard | agentic-single-tool | 17099ms | 17.4s | — |
| 7 | hard | agentic-multi-step | 12048ms | 16.1s | — |
| 8 | hard | agentic-read-reason | 548ms | 2.7s | — |
| 9 | expert | agentic-task-done | 5228ms | 7.6s | — |
| 11 | hard | agentic-write-file | 4481ms | 6.7s | — |
| 12 | hard | agentic-edit-file | 6228ms | 7.4s | — |
| 13 | expert | agentic-divergence-guard | 17143ms | 20.9s | — |
| 10 | expert | multi-turn-long-context | 8079ms | 118.6s | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-mira-mlx-unquant-isolated calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | none | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-mira-mlx-unquant-isolated score |
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

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-unquant-isolated:qwen3.6-35b-mira-mlx-unquant-isolated TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 3893ms | 3.9s | — |
| 2 | easy | code-no-tools | 2787ms | 6.0s | — |
| 3 | medium | reasoning | 2779ms | 20.8s | — |
| 4 | medium | long-output | 7946ms | 18.1s | — |
| 5 | medium | thinking-toggle | 2827ms | 20.3s | — |
| 6 | hard | agentic-single-tool | 7757ms | 7.8s | — |
| 7 | hard | agentic-multi-step | 7075ms | 16.7s | — |
| 8 | hard | agentic-read-reason | 21813ms | 34.3s | — |
| 9 | expert | agentic-task-done | 4136ms | 10.6s | — |
| 11 | hard | agentic-write-file | 6202ms | 6.8s | — |
| 12 | hard | agentic-edit-file | 4071ms | 12.1s | — |
| 13 | expert | agentic-divergence-guard | 4073ms | 26.8s | — |
| 10 | expert | multi-turn-long-context | 30024ms | 98.3s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-35b-mira-mlx-unquant-isolated calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-unquant-isolated score |
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

| Q | Difficulty | Category | ministral3-14b-mira-mlx-mlxtune:ministral3-14b-mira-mlx-mlxtune TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 8414ms | 8.5s | — |
| 2 | easy | code-no-tools | 402ms | 41.2s | — |
| 3 | medium | reasoning | 387ms | 86.7s | — |
| 4 | medium | long-output | 413ms | 39.6s | — |
| 5 | medium | thinking-toggle | 260ms | 87.3s | — |
| 6 | hard | agentic-single-tool | 11671ms | 13.5s | — |
| 7 | hard | agentic-multi-step | 11171ms | 14.4s | — |
| 8 | hard | agentic-read-reason | 555ms | 4.1s | — |
| 9 | expert | agentic-task-done | 5080ms | 7.6s | — |
| 11 | hard | agentic-write-file | 4306ms | 6.4s | — |
| 12 | hard | agentic-edit-file | 5986ms | 7.4s | — |
| 13 | expert | agentic-divergence-guard | 16574ms | 20.2s | — |
| 10 | expert | multi-turn-long-context | 37376ms | 183.5s | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-mira-mlx-mlxtune calls | task_done |
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

| Q | Difficulty | Category | ministral3-14b-mira-mlx-mlxtune score |
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

| Q | Difficulty | Category | ministral3-14b-mira-mlx-mlxtune-baseline:ministral3-14b-mira-mlx-mlxtune-baseline TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 8105ms | 8.2s | — |
| 2 | easy | code-no-tools | 410ms | 25.5s | — |
| 3 | medium | reasoning | 404ms | 102.4s | — |
| 4 | medium | long-output | 403ms | 44.7s | — |
| 5 | medium | thinking-toggle | 274ms | 67.4s | — |
| 6 | hard | agentic-single-tool | 18167ms | 19.9s | — |
| 7 | hard | agentic-multi-step | 23007ms | 57.6s | — |
| 8 | hard | agentic-read-reason | 575ms | 3.9s | — |
| 9 | expert | agentic-task-done | 5108ms | 7.7s | — |
| 11 | hard | agentic-write-file | 4358ms | 6.5s | — |
| 12 | hard | agentic-edit-file | 6097ms | 7.5s | — |
| 13 | expert | agentic-divergence-guard | 16653ms | 20.9s | — |
| 10 | expert | multi-turn-long-context | 45451ms | 192.6s | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-mira-mlx-mlxtune-baseline calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, list_files, run_shell | YES |
| 8 | agentic-read-reason | 1 | none | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-mira-mlx-mlxtune-baseline score |
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

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-mlxtune:qwen3.6-35b-mira-mlx-mlxtune TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 10959ms | 11.0s | — |
| 2 | easy | code-no-tools | 2876ms | 6.2s | — |
| 3 | medium | reasoning | 2842ms | 23.3s | — |
| 4 | medium | long-output | 2890ms | 18.3s | — |
| 5 | medium | thinking-toggle | 2869ms | 22.6s | — |
| 6 | hard | agentic-single-tool | 7924ms | 8.0s | — |
| 7 | hard | agentic-multi-step | 10856ms | 24.0s | — |
| 8 | hard | agentic-read-reason | 23469ms | 64.8s | — |
| 9 | expert | agentic-task-done | 6945ms | 7.7s | — |
| 11 | hard | agentic-write-file | 6442ms | 7.3s | — |
| 12 | hard | agentic-edit-file | 7665ms | 8.3s | — |
| 13 | expert | agentic-divergence-guard | 6718ms | 22.5s | — |
| 10 | expert | multi-turn-long-context | 33036ms | 69.1s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-35b-mira-mlx-mlxtune calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, search_files | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-mlxtune score |
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

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-mlxtune-baseline:qwen3.6-35b-mira-mlx-mlxtune-baseline TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 10143ms | 10.2s | — |
| 2 | easy | code-no-tools | 2866ms | 7.1s | — |
| 3 | medium | reasoning | 2836ms | 22.6s | — |
| 4 | medium | long-output | 2878ms | 17.2s | — |
| 5 | medium | thinking-toggle | 2879ms | 24.1s | — |
| 6 | hard | agentic-single-tool | 7947ms | 8.0s | — |
| 7 | hard | agentic-multi-step | 18176ms | 46.0s | — |
| 8 | hard | agentic-read-reason | 23654ms | 41.2s | — |
| 9 | expert | agentic-task-done | 4221ms | 10.6s | — |
| 11 | hard | agentic-write-file | 6390ms | 7.0s | — |
| 12 | hard | agentic-edit-file | 7564ms | 8.2s | — |
| 13 | expert | agentic-divergence-guard | 6381ms | 23.0s | — |
| 10 | expert | multi-turn-long-context | 32789ms | 182.9s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-35b-mira-mlx-mlxtune-baseline calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, list_files, search_files, read_file | YES |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-mira-mlx-mlxtune-baseline score |
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