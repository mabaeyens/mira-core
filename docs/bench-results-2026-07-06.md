# Benchmark Results — 2026-07-06

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Benchmark Results — 2026-07-06

### Timing

| Q | Difficulty | Category | ministral3-14b-ollama:ministral3-14b-ollama TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 10946ms | 11.2s | 8.0 |
| 2 | easy | code-no-tools | 443ms | 29.4s | 14.1 |
| 3 | medium | reasoning | 2954ms | 76.2s | 14.0 |
| 4 | medium | long-output | 3044ms | 33.9s | 14.1 |
| 5 | medium | thinking-toggle | ERR: "ministral-3:14b" does not support thinking (status code: 400) | — | — |
| 6 | hard | agentic-single-tool | 11113ms | 20.5s | 14.1 |
| 7 | hard | agentic-multi-step | 614ms | 8.6s | 14.2 |
| 8 | hard | agentic-read-reason | 549ms | 1.6s | 15.0 |
| 9 | expert | agentic-task-done | 544ms | 3.0s | 14.5 |
| 11 | hard | agentic-write-file | 499ms | 2.8s | 14.5 |
| 12 | hard | agentic-edit-file | 542ms | 1.4s | 15.2 |
| 13 | expert | agentic-divergence-guard | 566ms | 4.1s | 14.3 |
| 10 | expert | multi-turn-long-context | 140445ms | 328.4s | 14.9 |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-ollama calls | task_done |
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

| Q | Difficulty | Category | ministral3-14b-ollama score |
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

## Benchmark Results — 2026-07-06

### Timing

| Q | Difficulty | Category | ministral3-14b-vllmmlx:ministral3-14b-vllmmlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | ERR: LLM stream closed without a completion signal. | — | — |
| 2 | easy | code-no-tools | 4777ms | 19.3s | 14.7 |
| 3 | medium | reasoning | 392ms | 174.1s | 14.9 |
| 4 | medium | long-output | 4750ms | 60.0s | 15.3 |
| 5 | medium | thinking-toggle | 4810ms | 59.0s | 15.3 |
| 6 | hard | agentic-single-tool | ERR: LLM stream closed without a completion signal. | — | — |
| 7 | hard | agentic-multi-step | ERR: LLM stream closed without a completion signal. | — | — |
| 8 | hard | agentic-read-reason | ERR: LLM stream closed without a completion signal. | — | — |
| 9 | expert | agentic-task-done | ERR: LLM stream closed without a completion signal. | — | — |
| 11 | hard | agentic-write-file | ERR: LLM stream closed without a completion signal. | — | — |
| 12 | hard | agentic-edit-file | ERR: LLM stream closed without a completion signal. | — | — |
| 13 | expert | agentic-divergence-guard | ERR: LLM stream closed without a completion signal. | — | — |
| 10 | expert | multi-turn-long-context | ERR: LLM stream closed without a completion signal. | — | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-vllmmlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | ERR | — |
| 11 | agentic-write-file | 2 | ERR | — |
| 12 | agentic-edit-file | 3 | ERR | — |
| 13 | agentic-divergence-guard | 3 | ERR | — |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-vllmmlx score |
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

## Benchmark Results — 2026-07-06

### Timing

| Q | Difficulty | Category | ministral3-14b-vllmmlx:ministral3-14b-vllmmlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | ERR: LLM stream closed without a completion signal. | — | — |
| 2 | easy | code-no-tools | 4728ms | 15.7s | 15.7 |
| 3 | medium | reasoning | 370ms | 98.3s | 15.5 |
| 4 | medium | long-output | 4734ms | 57.4s | 15.6 |
| 5 | medium | thinking-toggle | 4757ms | 81.9s | 15.2 |
| 6 | hard | agentic-single-tool | ERR: LLM stream closed without a completion signal. | — | — |
| 7 | hard | agentic-multi-step | ERR: LLM stream closed without a completion signal. | — | — |
| 8 | hard | agentic-read-reason | ERR: LLM stream closed without a completion signal. | — | — |
| 9 | expert | agentic-task-done | ERR: LLM stream closed without a completion signal. | — | — |
| 11 | hard | agentic-write-file | ERR: LLM stream closed without a completion signal. | — | — |
| 12 | hard | agentic-edit-file | ERR: LLM stream closed without a completion signal. | — | — |
| 13 | expert | agentic-divergence-guard | ERR: LLM stream closed without a completion signal. | — | — |
| 10 | expert | multi-turn-long-context | ERR: LLM stream closed without a completion signal. | — | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-vllmmlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | ERR | — |
| 11 | agentic-write-file | 2 | ERR | — |
| 12 | agentic-edit-file | 3 | ERR | — |
| 13 | agentic-divergence-guard | 3 | ERR | — |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-vllmmlx score |
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

## Benchmark Results — 2026-07-06

### Timing

| Q | Difficulty | Category | ministral3-14b-vllmmlx-v2:ministral3-14b-vllmmlx-v2 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | ERR: LLM stream closed without a completion signal. | — | — |
| 2 | easy | code-no-tools | 4725ms | 19.5s | 15.6 |
| 3 | medium | reasoning | 4761ms | 105.0s | 15.4 |
| 4 | medium | long-output | 4738ms | 67.4s | 15.5 |
| 5 | medium | thinking-toggle | 4775ms | 130.7s | 15.3 |
| 6 | hard | agentic-single-tool | ERR: LLM stream closed without a completion signal. | — | — |
| 7 | hard | agentic-multi-step | ERR: LLM stream closed without a completion signal. | — | — |
| 8 | hard | agentic-read-reason | 7197ms | 11.0s | 15.4 |
| 9 | expert | agentic-task-done | ERR: LLM stream closed without a completion signal. | — | — |
| 11 | hard | agentic-write-file | ERR: LLM stream closed without a completion signal. | — | — |
| 12 | hard | agentic-edit-file | ERR: LLM stream closed without a completion signal. | — | — |
| 13 | expert | agentic-divergence-guard | ERR: LLM stream closed without a completion signal. | — | — |
| 10 | expert | multi-turn-long-context | ERR: LLM stream closed without a completion signal. | — | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-vllmmlx-v2 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | none | no |
| 9 | agentic-task-done | 3 | ERR | — |
| 11 | agentic-write-file | 2 | ERR | — |
| 12 | agentic-edit-file | 3 | ERR | — |
| 13 | agentic-divergence-guard | 3 | ERR | — |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-vllmmlx-v2 score |
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

## Benchmark Results — 2026-07-06

### Timing

| Q | Difficulty | Category | ministral3-14b-vllmmlx-v3:ministral3-14b-vllmmlx-v3 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | ERR: LLM stream closed without a completion signal. | — | — |
| 2 | easy | code-no-tools | 4720ms | 15.9s | 15.8 |
| 3 | medium | reasoning | 378ms | 62.1s | 15.6 |
| 4 | medium | long-output | 4721ms | 49.0s | 15.7 |
| 5 | medium | thinking-toggle | 4747ms | 91.7s | 15.6 |
| 6 | hard | agentic-single-tool | ERR: LLM stream closed without a completion signal. | — | — |
| 7 | hard | agentic-multi-step | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |
| 8 | hard | agentic-read-reason | ERR: LLM stream closed without a completion signal. | — | — |
| 9 | expert | agentic-task-done | ERR: LLM stream closed without a completion signal. | — | — |
| 11 | hard | agentic-write-file | ERR: LLM stream closed without a completion signal. | — | — |
| 12 | hard | agentic-edit-file | ERR: LLM stream closed without a completion signal. | — | — |
| 13 | expert | agentic-divergence-guard | ERR: LLM stream closed without a completion signal. | — | — |
| 10 | expert | multi-turn-long-context | ERR: LLM stream closed without a completion signal. | — | — |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-vllmmlx-v3 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | ERR | — |
| 11 | agentic-write-file | 2 | ERR | — |
| 12 | agentic-edit-file | 3 | ERR | — |
| 13 | agentic-divergence-guard | 3 | ERR | — |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-vllmmlx-v3 score |
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

## Benchmark Results — 2026-07-06

### Timing

| Q | Difficulty | Category | ministral3-14b-vllmmlx-v4:ministral3-14b-vllmmlx-v4 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 4843ms | 4.9s | 21.4 |
| 2 | easy | code-no-tools | 4760ms | 21.5s | 15.6 |
| 3 | medium | reasoning | 414ms | 108.2s | 15.3 |
| 4 | medium | long-output | 4774ms | 51.5s | 15.5 |
| 5 | medium | thinking-toggle | 4810ms | 99.4s | 15.3 |
| 6 | hard | agentic-single-tool | ERR: LLM stream closed without a completion signal. | — | — |
| 7 | hard | agentic-multi-step | ERR: LLM stream closed without a completion signal. | — | — |
| 8 | hard | agentic-read-reason | 7432ms | 14.6s | 15.2 |
| 9 | expert | agentic-task-done | 19913ms | 22.6s | 39.2 |
| 11 | hard | agentic-write-file | 25312ms | 26.5s | 55.0 |
| 12 | hard | agentic-edit-file | 27188ms | 28.3s | 74.4 |
| 13 | expert | agentic-divergence-guard | 67514ms | 71.7s | 86.2 |
| 10 | expert | multi-turn-long-context | 112316ms | 337.8s | 3.6 |

### Agentic results

| Q | Category | Expected calls | ministral3-14b-vllmmlx-v4 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | none | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | ministral3-14b-vllmmlx-v4 score |
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