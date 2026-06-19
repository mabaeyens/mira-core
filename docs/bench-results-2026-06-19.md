# Benchmark Results — 2026-06-19

Hardware: MacBook Pro M5 32GB · Backend: mlx-lm 0.31.3 · Model: NVIDIA-Nemotron-3-Nano-30B-A3B-4bit

## Benchmark Results — 2026-06-19

### Timing

| Q | Difficulty | Category | NVIDIA-Nemotron-3-Nano-30B-A3B:NVIDIA-Nemotron-3-Nano-30B-A3B TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 6647ms | 14.2s | 16.9 |
| 2 | easy | code-no-tools | 6127ms | 13.2s | 110.4 |
| 3 | medium | reasoning | 2513ms | 21.7s | 45.4 |
| 4 | medium | long-output | 15706ms | 28.7s | 163.8 |
| 5 | medium | thinking-toggle | 6315ms | 19.4s | 48.0 |
| 6 | hard | agentic-single-tool | 22009ms | 203.1s | 102.4 |
| 7 | hard | agentic-multi-step | 75005ms | 171.1s | 160.1 |
| 8 | hard | agentic-read-reason | 8190ms | 62.5s | 47.6 |
| 9 | expert | agentic-task-done | 25802ms | 43.5s | 196.1 |
| 11 | hard | agentic-write-file | — | 86.3s | — |
| 12 | hard | agentic-edit-file | 15803ms | 78.7s | 118.3 |
| 13 | expert | agentic-divergence-guard | 22789ms | 268.1s | 51.6 |
| 10 | expert | multi-turn-long-context | 64973ms | 96.0s | 2298.1 |

### Agentic results

| Q | Category | Expected calls | NVIDIA-Nemotron-3-Nano-30B-A3B calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | none | no |
| 12 | agentic-edit-file | 3 | list_files, read_file, write_file, write_file, read_file, run_shell | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, list_files | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | NVIDIA-Nemotron-3-Nano-30B-A3B score |
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