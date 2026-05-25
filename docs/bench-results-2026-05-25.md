# Benchmark Results — 2026-05-25

Hardware: MacBook Pro M5 32GB · Ollama 0.24.0  
Questions: Q6-Q9 (agentic tool-use suite)  
Note: Q7 result reflects the post-prompt-fix re-run ("include filename, line number, and full comment text").

---

## Model: gemma4:26b-mlx

### Timing

| Q | Difficulty | Category | TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | — | 35.9s | — |
| 7 | hard | agentic-multi-step | 16513ms | 51.2s | 31.4 |
| 8 | hard | agentic-read-reason | 27212ms | 72.1s | 11.7 |
| 9 | expert | agentic-task-done | — | 44.2s | — |

### Agentic results

| Q | Category | Expected calls | Actual calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell ×7 | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | run_shell ×3 | YES |

### Quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Score | Notes |
|---|-----------|---------|---|---|
| 6 | hard | agentic-single-tool | 1 | Correct answer (3393 lines) but used 7 sequential calls; task asked for a single pipeline |
| 7 | hard | agentic-multi-step | 2 | Correct grouped markdown (`### file`, `- Line N: text`), full comment text, all files |
| 8 | hard | agentic-read-reason | 2 | Accurate explanation with hash/repeat/diverged code quotes |
| 9 | expert | agentic-task-done | 1 | File created and verified, task_done fired; wrote to `tmp/` (project root) not `/tmp/` |

**Total: 6/8**

---

## Model: qwen3.6:35b-mlx

### Timing

| Q | Difficulty | Category | TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | 11464ms | 62.3s | 4.2 |
| 7 | hard | agentic-multi-step | 31041ms | 120.2s | 17.2 |
| 8 | hard | agentic-read-reason | 26606ms | 96.3s | 14.2 |
| 9 | expert | agentic-task-done | 21719ms | 72.9s | 14.3 |

### Agentic results

| Q | Category | Expected calls | Actual calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell ×4 | no |
| 7 | agentic-multi-step | 2 | run_shell ×5, list_files, run_shell, search_files, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | write_file, run_shell ×8, read_file ×2, write_file, run_shell | YES (forced at step 13) |

### Quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Score | Notes |
|---|-----------|---------|---|---|
| 6 | hard | agentic-single-tool | 1 | Correct answer (3393) but 4 calls, bare number with no context, no task_done |
| 7 | hard | agentic-multi-step | 2 | Correct grouped output with line numbers and content; thorough multi-strategy approach |
| 8 | hard | agentic-read-reason | 2 | Thorough explanation with full prepare-loop code block, all three guard checks |
| 9 | expert | agentic-task-done | 1 | File attempted (write_file present), task_done forced at step 13; response captured mid-task state |

**Total: 6/8**

---

## Summary

| Metric | gemma4:26b-mlx | qwen3.6:35b-mlx |
|---|---|---|
| Quality score | **6/8** | **6/8** |
| Q6 — count lines | 1/2 (7 calls, correct answer) | 1/2 (4 calls, correct answer) |
| Q7 — find TODOs | 2/2 (2 calls, grep\|awk pipeline) | 2/2 (8 calls, exhaustive search) |
| Q8 — explain code | 2/2 (1 read, clear explanation) | 2/2 (1 read, more thorough) |
| Q9 — file + task_done | 1/2 (wrong path: tmp/ not /tmp/) | 1/2 (forced summary, mid-task state) |
| Avg wall time | **48.4s** | 87.9s |
| task_done compliance | 2/4 (Q6, Q9) | 1/4 (Q9 forced) |

**Both models score 6/8.** Key differentiation: gemma4 is 1.8× faster on wall time; qwen3.6 produces more thorough output on Q7/Q8. Shared weakness: neither model uses a single pipeline for Q6, and both fail clean task_done on Q9 (wrong path or forced summary).
