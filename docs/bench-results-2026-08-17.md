# Benchmark Results — 2026-08-17

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Benchmark Results — 2026-08-17

### Timing

| Q | Difficulty | Category | qwen38-27b-mtp:qwen38-27b-mtp TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 7502ms | 7.5s | 245.4 |
| 2 | easy | code-no-tools | 1537ms | 6.9s | 22.3 |

### Agentic results


### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen38-27b-mtp score |
|---|-----------|---------|---|
| 1 | easy | baseline | — |
| 2 | easy | code-no-tools | — |

---

## Benchmark Results — 2026-08-17

### Timing

| Q | Difficulty | Category | qwen38-27b-mtp:qwen38-27b-mtp TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 1390ms | 1.4s | 1361.9 |
| 2 | easy | code-no-tools | 1549ms | 6.9s | 22.3 |
| 3 | medium | reasoning | 1589ms | 51.1s | 13.2 |
| 4 | medium | long-output | 1751ms | 135.1s | 17.2 |
| 5 | medium | thinking-toggle | 102271ms | 139.0s | 84.9 |
| 6 | hard | agentic-single-tool | 35453ms | 35.6s | 616.0 |
| 7 | hard | agentic-multi-step | 186183ms | 229.5s | — |
| 8 | hard | agentic-read-reason | ERR: Request aborted: process memory limit exceeded (usage 24.9 GB, ceiling 19.9 GB). Reduce context size or lower memory_guard_tier. | — | — |
| 9 | expert | agentic-task-done | — | 39.9s | — |
| 11 | hard | agentic-write-file | — | 16.6s | — |
| 12 | hard | agentic-edit-file | — | 23.3s | — |
| 13 | expert | agentic-divergence-guard | 113233ms | 118.4s | — |
| 10 | expert | multi-turn-long-context | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=33384, min_chunk=32): predicted peak would require ~21.14 GB (current 18.61 GB + KV 2.04 GB + min-chunk transient 503.59 MB) but prefill safety cap is 20.81 GB (90% of effective ceiling 23.12 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 22700716932, 'limit_bytes': 22340630937}, 'type': 'error'} | — | — |
| 14 | hard | injection-resistance-readfile | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=4423, min_chunk=32): predicted peak would require ~20.26 GB (current 19.49 GB + KV 276.50 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21748892316, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |
| 15 | hard | injection-over-caution | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=4413, min_chunk=32): predicted peak would require ~19.99 GB (current 19.23 GB + KV 275.88 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21461058012, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |
| 16 | hard | injection-resistance-fetchurl | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=4306, min_chunk=32): predicted peak would require ~19.98 GB (current 19.23 GB + KV 269.19 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21454078428, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |
| 17 | hard | plumbing-string-returning-tool | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=4319, min_chunk=32): predicted peak would require ~19.98 GB (current 19.23 GB + KV 270.00 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21454946780, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |
| 18 | expert | plumbing-reply-integrity | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=2247, min_chunk=32): predicted peak would require ~19.86 GB (current 19.23 GB + KV 140.50 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21319156188, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |
| 19 | expert | plumbing-legitimate-repetition | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=2212, min_chunk=32): predicted peak would require ~19.85 GB (current 19.23 GB + KV 138.31 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21316862428, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |
| 20 | expert | plumbing-task-done-refusal | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=4306, min_chunk=32): predicted peak would require ~19.98 GB (current 19.23 GB + KV 269.19 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21454176732, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |
| 21 | expert | plumbing-multi-turn-tool-memory | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=4349, min_chunk=32): predicted peak would require ~19.98 GB (current 19.23 GB + KV 271.88 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21456994780, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |
| 22 | expert | plumbing-injection-across-turns | ERR: Error code: 400 - {'error': {'message': 'oMLX prefill memory guard rejected this prompt: Prefill context too large for available memory (preflight safety guard, kv_len=4337, min_chunk=32): predicted peak would require ~19.98 GB (current 19.23 GB + KV 271.12 MB + min-chunk transient 503.59 MB) but prefill safety cap is 18.87 GB (90% of effective ceiling 20.97 GB). Reduce context length, free system memory, or loosen memory_guard_tier (safe → balanced → aggressive). To continue, set Memory Guard to aggressive, raise the custom memory guard ceiling, free system memory, or compact/reduce context.', 'type': 'invalid_request_error', 'param': None, 'code': 'prefill_memory_exceeded', 'omlx_code': 'prefill_memory_exceeded', 'estimated_bytes': 21456257500, 'limit_bytes': 20265866035}, 'type': 'error'} | — | — |

### Agentic results

| Q | Category | Expected calls | qwen38-27b-mtp calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | ERR | — |
| 14 | injection-resistance-readfile | 1 | ERR | — |
| 15 | injection-over-caution | 1 | ERR | — |
| 16 | injection-resistance-fetchurl | 1 | ERR | — |
| 17 | plumbing-string-returning-tool | 1 | ERR | — |
| 20 | plumbing-task-done-refusal | 1 | ERR | — |
| 21 | plumbing-multi-turn-tool-memory | 2 | ERR | — |
| 22 | plumbing-injection-across-turns | 2 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen38-27b-mtp score |
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
| 14 | hard | injection-resistance-readfile | — |
| 15 | hard | injection-over-caution | — |
| 16 | hard | injection-resistance-fetchurl | — |
| 17 | hard | plumbing-string-returning-tool | — |
| 18 | expert | plumbing-reply-integrity | — |
| 19 | expert | plumbing-legitimate-repetition | — |
| 20 | expert | plumbing-task-done-refusal | — |
| 21 | expert | plumbing-multi-turn-tool-memory | — |
| 22 | expert | plumbing-injection-across-turns | — |

---

## Benchmark Results — 2026-08-17

### Timing

| Q | Difficulty | Category | qwen38-27b-native-mtp:qwen38-27b-native-mtp TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | ERR: list index out of range | — | — |
| 3 | medium | reasoning | ERR: Error code: 503 - {'detail': 'mira-mlx engine is not running: IndexError: list index out of range'} | — | — |

### Agentic results


### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen38-27b-native-mtp score |
|---|-----------|---------|---|
| 1 | easy | baseline | — |
| 3 | medium | reasoning | — |

---

## Benchmark Results — 2026-08-17

### Timing

| Q | Difficulty | Category | qwen38-27b-nomtp:qwen38-27b-nomtp TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 15979ms | 17.3s | 8.5 |
| 3 | medium | reasoning | 914ms | 8.2s | 8.1 |

### Agentic results


### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen38-27b-nomtp score |
|---|-----------|---------|---|
| 1 | easy | baseline | — |
| 3 | medium | reasoning | — |