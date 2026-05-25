# Benchmark Results — 2026-05-25

Hardware: MacBook Pro M5 32GB · Ollama 0.24.0  
Questions: Q6-Q9 (agentic tool-use suite)  
Note: Q7 reflects post-prompt-fix re-run; Q9 reflects post-sandbox-fix re-run (/tmp/ now accessible via run_shell); task_done compliance reflects post-orchestrator-fix re-run (text-response exit now first-class).

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
| 9 | agentic-task-done | 3 | run_shell ×2 | YES |

### Quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Score | Notes |
|---|-----------|---------|---|---|
| 6 | hard | agentic-single-tool | 1 | Correct answer (3393 lines) but used 7 sequential calls; task asked for a single pipeline |
| 7 | hard | agentic-multi-step | 2 | Correct grouped markdown (`### file`, `- Line N: text`), full comment text, all files |
| 8 | hard | agentic-read-reason | 2 | Accurate explanation with hash/repeat/diverged code quotes |
| 9 | expert | agentic-task-done | 2 | `/tmp/mira_bench_test.txt` created via run_shell, correct content, task_done fired |

**Total: 7/8**

---

## Model: qwen3.6:35b-mlx

### Timing

| Q | Difficulty | Category | TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | 11464ms | 62.3s | 4.2 |
| 7 | hard | agentic-multi-step | 31041ms | 120.2s | 17.2 |
| 8 | hard | agentic-read-reason | 26606ms | 96.3s | 14.2 |
| 9 | expert | agentic-task-done | 7065ms | 59.5s | 2.8 |

### Agentic results

| Q | Category | Expected calls | Actual calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell ×4 | no |
| 7 | agentic-multi-step | 2 | run_shell ×5, list_files, run_shell, search_files, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | run_shell ×1 | YES |

### Quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Score | Notes |
|---|-----------|---------|---|---|
| 6 | hard | agentic-single-tool | 1 | Correct answer (3393) but 4 calls, bare number with no context |
| 7 | hard | agentic-multi-step | 2 | Correct grouped output with line numbers and content; thorough multi-strategy approach |
| 8 | hard | agentic-read-reason | 2 | Thorough explanation with full prepare-loop code block, all three guard checks |
| 9 | expert | agentic-task-done | 2 | `/tmp/mira_bench_test.txt` created correctly, content correct, task_done YES (scaffolding fix) |

**Total: 7/8**

---

## Summary

| Metric | gemma4:26b-mlx | qwen3.6:35b-mlx |
|---|---|---|
| Quality score | **7/8** | **7/8** |
| Q6 — count lines | 1/2 (7 calls, correct answer) | 1/2 (4 calls, correct answer) |
| Q7 — find TODOs | 2/2 (2 calls, grep\|awk pipeline) | 2/2 (8 calls, exhaustive search) |
| Q8 — explain code | 2/2 (1 read, clear explanation) | 2/2 (1 read, more thorough) |
| Q9 — file + task_done | 2/2 (correct /tmp/ path, 2 calls, task_done) | 2/2 (correct /tmp/ path, 1 call, task_done via scaffolding) |
| Avg wall time | **48.4s** | 87.9s |
| task_done compliance | 2/4 explicit (Q6, Q9) | 1/4 explicit (Q9 via text-exit path) |

**Both models score 7/8.** gemma4 is 1.8× faster on wall time; qwen3.6 produces more thorough output on Q8. Shared weakness: neither model uses a single pipeline for Q6. Q9 task_done for qwen3.6 is now first-class (scaffolding recognizes text-response-after-tool-use as implicit task completion).

---

## Q11–Q12 Results — 2026-05-25

Questions: Q11 (agentic-write-file), Q12 (agentic-edit-file) — first run of new file-tool questions.

### Timing

| Q | Difficulty | Category | TTFT | wall | t/s | Model |
|---|-----------|---------|---|---|---|---|
| 11 | hard | agentic-write-file | — | 35.4s | — | gemma4:26b-mlx |
| 12 | hard | agentic-edit-file | — | 32.6s | — | gemma4:26b-mlx |
| 11 | hard | agentic-write-file | 7311ms | 42.3s | 3.5 | qwen3.6:35b-mlx |
| 12 | hard | agentic-edit-file | 5897ms | 47.0s | 4.3 | qwen3.6:35b-mlx |

### Agentic results

| Q | Category | Expected calls | gemma4 calls | gemma4 task_done | qwen3.6 calls | qwen3.6 task_done |
|---|---------|----------------|---|---|---|---|
| 11 | agentic-write-file | 2 | write_file, read_file | YES | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | run_shell, write_file, edit_file, read_file | YES | write_file, edit_file, read_file | YES |

### Quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4 score | gemma4 notes | qwen3.6 score | qwen3.6 notes |
|---|-----------|---------|---|---|---|---|
| 11 | hard | agentic-write-file | 2 | write_file used, both lines correct, verified | 2 | write_file used, both lines correct, verified |
| 12 | hard | agentic-edit-file | 2 | edit_file used correctly; extra run_shell probe (+1 call) | 2 | Perfect 3 calls: write_file → edit_file → read_file |

**Both models score 4/4 on Q11–Q12.** File tools work correctly. gemma4 added a superfluous `run_shell` probe on Q12 (+1 call vs expected 3); qwen3.6 matched the expected call sequence exactly.