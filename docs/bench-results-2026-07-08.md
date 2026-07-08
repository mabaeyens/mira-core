# Benchmark Results — 2026-07-08

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Phase 1 go/no-go: mlx-lm vs vllm-mlx for Mistral (Ministral 3 14B)

**GO.** The `mlx-lm` backend passes the one gate that killed vllm-mlx: Q10
(multi-turn, long-context) completes cleanly with **no ballooning and no hang**
— turn 1 (inject 40,150-char file) took 72.4s, turn 2 (retrieval against that
context) took 71.9s, essentially flat. The prior vllm-mlx run (`ministral3-14b-
vllmmlx-postfix`, same-day, same fixed timestamp/cache-invalidation bug already
patched) got through Q1–Q13 fine but then **hung indefinitely on Q10 turn 2** —
18+ minutes with an idle-but-open TCP connection and near-zero CPU growth,
consistent with the open `waybarrios/vllm-mlx#468` stream/thread-ownership bug
class. That run was killed manually; its raw results were never written to
disk (only captured in the terminal log), so there's no `vllmmlx-postfix`
section below — the partial numbers (Q1–Q13, all passing) matched or slightly
beat mlx-lm's, so nothing was lost by abandoning it except that it can't
finish a real multi-turn conversation.

Everything else on `mlx-lm` is solid: correct tool-calling across Q6/7/9/11/
12/13 (all `task_done=YES`), baseline decode ~15 t/s (physics-bound — dense
14B vs. Qwen3.6's ~3B-active MoE, expected, not a regression), TTFT dropping to
~400-500ms on later non-tool questions once the prompt-prefix cache warms up.

**Decision:** proceed to Phase 2 (owned `core/inference/mira_mlx_server.py`).
Phase 1's stock `mlx_lm.server` subprocess is the safe fallback if Phase 2's
glue code underperforms.

## Benchmark Results — 2026-07-08

### Timing

| Q | Difficulty | Category | qwen36-omlx-postfix:qwen36-omlx-postfix TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 46939ms | 47.0s | 40.5 |
| 2 | easy | code-no-tools | 1116ms | 4.6s | 52.6 |
| 3 | medium | reasoning | 1036ms | 22.6s | 53.5 |
| 4 | medium | long-output | 11699ms | 28.3s | 56.8 |
| 5 | medium | thinking-toggle | 1916ms | 29.2s | 56.8 |
| 6 | hard | agentic-single-tool | 20822ms | 20.8s | 89469.9 |
| 7 | hard | agentic-multi-step | 25631ms | 97.1s | 31.3 |
| 8 | hard | agentic-read-reason | 72787ms | 91.1s | 32.2 |
| 9 | expert | agentic-task-done | 9790ms | 10.8s | 115.4 |
| 11 | hard | agentic-write-file | 13719ms | 14.2s | 256.7 |
| 12 | hard | agentic-edit-file | 22830ms | 23.3s | 343.5 |
| 13 | expert | agentic-divergence-guard | 11162ms | 45.6s | 16.7 |
| 10 | expert | multi-turn-long-context | ERR: 'str' object has no attribute 'get' | — | — |

### Agentic results

| Q | Category | Expected calls | qwen36-omlx-postfix calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, search_files | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen36-omlx-postfix score |
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

## Benchmark Results — 2026-07-08

### Timing

| Q | Difficulty | Category | ministral3-14b-mlxlm:ministral3-14b-mlxlm TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 9914ms | 10.0s | 26.4 |
| 2 | easy | code-no-tools | 421ms | 28.1s | 15.1 |
| 3 | medium | reasoning | 424ms | 84.5s | 15.2 |
| 4 | medium | long-output | 429ms | 37.8s | 15.3 |
| 5 | medium | thinking-toggle | 324ms | 88.4s | 15.1 |
| 6 | hard | agentic-single-tool | 10935ms | 12.9s | 38.7 |
| 7 | hard | agentic-multi-step | 13566ms | 16.9s | 68.6 |
| 8 | hard | agentic-read-reason | 565ms | 2.8s | 15.3 |
| 9 | expert | agentic-task-done | 4698ms | 8.6s | 27.3 |
| 11 | hard | agentic-write-file | 4495ms | 6.7s | 36.6 |
| 12 | hard | agentic-edit-file | 6336ms | 7.5s | 72.0 |
| 13 | expert | agentic-divergence-guard | 28612ms | 32.9s | 98.9 |
| 10 | expert | multi-turn-long-context | 31176ms | 144.2s | 16.8 |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-mlxlm calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | none | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-mlxlm score |
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