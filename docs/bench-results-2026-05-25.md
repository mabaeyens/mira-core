# Benchmark Results — 2026-05-25

Hardware: MacBook Pro M5 32GB · Ollama 0.24.0

## Benchmark Results — 2026-05-25

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | — | 35.9s | — |
| 7 | hard | agentic-multi-step | — | 41.5s | — |
| 8 | hard | agentic-read-reason | 27212ms | 72.1s | 11.7 |
| 9 | expert | agentic-task-done | — | 44.2s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell, run_shell | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |