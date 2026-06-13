# Benchmark Results — 2026-06-07

Hardware: MacBook Pro M5 32GB · Ollama 0.24.0

## Benchmark Results — 2026-06-07

### Timing

| Q | Difficulty | Category | gemma4-12b-omlx-qat4:gemma4-12b-omlx-qat4 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 2 | easy | code-no-tools | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 3 | medium | reasoning | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 4 | medium | long-output | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 5 | medium | thinking-toggle | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 6 | hard | agentic-single-tool | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 7 | hard | agentic-multi-step | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 8 | hard | agentic-read-reason | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 9 | expert | agentic-task-done | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 11 | hard | agentic-write-file | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 12 | hard | agentic-edit-file | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 13 | expert | agentic-divergence-guard | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |
| 10 | expert | multi-turn-long-context | ERR: Error code: 404 - {'error': {'message': "Model 'mlx-community/gemma-4-12B-it-qat-4bit' not found. Available models: Qwen3.6-35B-A3B, gemma4-26b, mlx-community--Qwen3-Reranker-0.6B-4bit, mlx-community--Qwen3.6-35B-A3B-4bit, mlx-community--gemma-4-12B-it-qat-4bit, mlx-community--gemma-4-26b-a4b-it-4bit", 'type': 'not_found_error', 'param': None, 'code': None}} | — | — |

### Agentic results

| Q | Category | Expected calls | gemma4-12b-omlx-qat4 calls | task_done |
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

| Q | Difficulty | Category | gemma4-12b-omlx-qat4 score |
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

## Benchmark Results — 2026-06-07

### Timing

| Q | Difficulty | Category | gemma4-12b-omlx-qat4:gemma4-12b-omlx-qat4 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 2 | easy | code-no-tools | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 3 | medium | reasoning | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 4 | medium | long-output | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 5 | medium | thinking-toggle | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 6 | hard | agentic-single-tool | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 7 | hard | agentic-multi-step | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 8 | hard | agentic-read-reason | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 9 | expert | agentic-task-done | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 11 | hard | agentic-write-file | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 12 | hard | agentic-edit-file | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 13 | expert | agentic-divergence-guard | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |
| 10 | expert | multi-turn-long-context | ERR: Error code: 500 - {'error': {'message': 'Internal server error', 'type': 'server_error', 'param': None, 'code': None}} | — | — |

### Agentic results

| Q | Category | Expected calls | gemma4-12b-omlx-qat4 calls | task_done |
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

| Q | Difficulty | Category | gemma4-12b-omlx-qat4 score |
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

## Benchmark Results — 2026-06-07

### Timing

| Q | Difficulty | Category | gemma4-12b-ollama-mlx:gemma4-12b-ollama-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 7446ms | 211.6s | — |
| 2 | easy | code-no-tools | 2836ms | 72.6s | 3.0 |
| 3 | medium | reasoning | 4271ms | 116.4s | 6.3 |
| 4 | medium | long-output | 2912ms | 89.3s | 4.0 |
| 5 | medium | thinking-toggle | 42952ms | 153.3s | 11.7 |
| 6 | hard | agentic-single-tool | 15766ms | 75.8s | 0.8 |
| 7 | hard | agentic-multi-step | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |
| 8 | hard | agentic-read-reason | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |
| 9 | expert | agentic-task-done | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |
| 11 | hard | agentic-write-file | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |
| 12 | hard | agentic-edit-file | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |
| 13 | expert | agentic-divergence-guard | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |
| 10 | expert | multi-turn-long-context | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |

### Agentic results

| Q | Category | Expected calls | gemma4-12b-ollama-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | ERR | — |
| 11 | agentic-write-file | 2 | ERR | — |
| 12 | agentic-edit-file | 3 | ERR | — |
| 13 | agentic-divergence-guard | 3 | ERR | — |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4-12b-ollama-mlx score |
|---|-----------|---------|---|
| 1 | easy | baseline | 2 |
| 2 | easy | code-no-tools | 2 |
| 3 | medium | reasoning | 2 |
| 4 | medium | long-output | 2 |
| 5 | medium | thinking-toggle | 1 (quality degraded vs Q3 despite thinking on; 43s TTFT) |
| 6 | hard | agentic-single-tool | 2 |
| 7 | hard | agentic-multi-step | 0 (timeout, 0 tool calls) |
| 8 | hard | agentic-read-reason | 0 (timeout, 0 tool calls) |
| 9 | expert | agentic-task-done | 0 (timeout, 0 tool calls) |
| 11 | hard | agentic-write-file | 0 (timeout, 0 tool calls) |
| 12 | hard | agentic-edit-file | 0 (timeout, 0 tool calls) |
| 13 | expert | agentic-divergence-guard | 0 (timeout, 0 tool calls) |
| 10 | expert | multi-turn-long-context | 0 (timeout, 0 tool calls) |

**Total: 11/26** — REJECTED. 6/7 agentic tasks failed with 0 tool calls + timeout. Throughput 3–12 t/s (vs ~100 t/s on omlx). Same root cause as June-4 Ollama test: Ollama MLX integration does not reliably sustain multi-step tool use for gemma4-12B.