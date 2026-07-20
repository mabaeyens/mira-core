# Benchmark Results — 2026-07-20

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Benchmark Results — 2026-07-20

### Timing

| Q | Difficulty | Category | qwen3.6-35b-a3b-post-security:qwen3.6-35b-a3b-post-security TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 8670ms | 8.7s | — |
| 2 | easy | code-no-tools | 2866ms | 6.2s | — |
| 3 | medium | reasoning | 2829ms | 24.6s | — |
| 4 | medium | long-output | 2948ms | 23.9s | — |
| 5 | medium | thinking-toggle | 2940ms | 19.3s | — |
| 6 | hard | agentic-single-tool | 6518ms | 6.6s | — |
| 7 | hard | agentic-multi-step | 8448ms | 22.5s | — |
| 8 | hard | agentic-read-reason | 23769ms | 40.3s | — |
| 9 | expert | agentic-task-done | 6616ms | 7.3s | — |
| 11 | hard | agentic-write-file | 6375ms | 7.0s | — |
| 12 | hard | agentic-edit-file | 4116ms | 12.5s | — |
| 13 | expert | agentic-divergence-guard | 4247ms | 30.4s | — |
| 10 | expert | multi-turn-long-context | 41367ms | 136.1s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-35b-a3b-post-security calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, search_files | YES |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-a3b-post-security score |
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