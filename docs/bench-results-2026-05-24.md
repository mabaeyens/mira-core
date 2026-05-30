> Archived bench log. See `model-comparison-m5-macbook.md` for the current verdict.

# Benchmark Results — 2026-05-24

**Hardware:** MacBook Pro M5 32GB · Ollama 0.24.0  
**Protocol:** Clean `ollama stop` + `sudo purge` before each model run. gemma4 was warm for Q1 (reused from earlier smoke test); qwen3.6 was cold for Q1 (first query after purge + model switch).  
**Server:** Mira 0.1.33 at `http://localhost:8000` · context = 64K · `OLLAMA_KV_CACHE_TYPE=q8_0`

---

## Timing

| Q | Difficulty | Category | gemma4 TTFT | gemma4 wall | gemma4 t/s | qwen3.6 TTFT | qwen3.6 wall | qwen3.6 t/s |
|---|-----------|---------|------------|------------|-----------|-------------|-------------|------------|
| 1 | easy | baseline | 429ms ¹ | 0.5s | 59.2 | 4,810ms ² | 4.8s | 39.8 |
| 2 | easy | code-no-tools | 3,308ms | 14.9s | 36.5 | 3,300ms | 10.6s | 41.9 |
| 3 | medium | reasoning | 394ms | 29.0s | 35.4 | 273ms | 48.0s | 41.9 |
| 4 | medium | long-output | 589ms | 23.7s | 35.2 | 346ms | 47.1s | 42.1 |
| 5 | medium | thinking-toggle | 24,374ms ³ | 54.0s | 89.2 ³ | 6,517ms | 49.2s | 49.8 |
| 6 | hard | agentic-single-tool | 14,698ms | 20.3s | 61.5 | 9,121ms | 16.3s | 41.5 |
| 7 | hard | agentic-multi-step | 14,821ms | 19.9s | 66.5 | 3,839ms | 40.8s | — |
| 8 | hard | agentic-read-reason | 6,161ms | 9.8s | 63.0 | 2,958ms | 59.7s | — |
| 9 | expert | agentic-task-done | 409ms | 1.3s | 35.5 | 497ms | 8.3s | 27.9 |
| 10 | expert | multi-turn-long-context | 10,483ms ⁴ | 34.0s | 67.3 | 6,570ms ⁴ | 94.6s | — |

¹ gemma4 warm — carried over from earlier smoke test, not a cold load.  
² qwen3.6 cold load after `sudo purge` + model switch (21GB weights from disk).  
³ gemma4 Q5: TTFT of 24s is the thinking phase. The 89.2 t/s is inflated — thinking tokens counted in output_tokens; actual generation rate is ~35 t/s.  
⁴ Q10 TTFT is for turn-2 (retrieval question); turn-1 injects server.py (~23K chars).

---

## Agentic Tool Calls

| Q | Category | Expected | gemma4 calls | gemma4 task_done | qwen3.6 calls | qwen3.6 task_done |
|---|---------|---------|-------------|----------------|--------------|-----------------|
| 6 | agentic-single-tool | run_shell (1) | github_list_repos, github_search_code, github_list_repos | no | (none recorded) | no |
| 7 | agentic-multi-step | run_shell (2+) | github_search_code, github_list_repos, github_search_code | no | github_search_code ×2, github_list_files, github_clone_repo | no |
| 8 | agentic-read-reason | read_file (1) | github_list_repos ×2, github_search_code ×2 | no | github_list_files ×2 | no |
| 9 | agentic-task-done | run_shell (2) + task_done | (no tools) | no | (no tools) | no |

**Q6–Q8:** Both models chose GitHub MCP tools over `run_shell` — expected, since GitHub tools are available and match the task description. To specifically test `run_shell`, reformulate prompts to tasks GitHub tools can't handle (e.g., "count bytes in /tmp", "append to a local file").

**Q9 failure (both models):** Neither model used tools to create and verify the file. Both gave text-only responses. This points to a system prompt gap — the current prompt doesn't strongly enough incentivize tool use for concrete local actions.

---

## Head-to-Head Summary

| Metric | gemma4:26b-mlx | qwen3.6:35b-mlx | Winner |
|--------|---------------|----------------|--------|
| Cold TTFT | ~2,600ms (prior runs) | 4,810ms | gemma4 |
| Warm TTFT avg (Q3, Q4) | ~490ms | ~310ms | **qwen3.6** |
| Sustained t/s avg | ~35–39 t/s | ~41–42 t/s | **qwen3.6 (+14%)** |
| Thinking TTFT (Q5) | ~24s | ~6.5s | **qwen3.6 (3.7×)** |
| Agentic wall time (Q6–Q8) | 9.8–20.3s | 16.3–59.7s | gemma4 (more concise) |
| Long-context retrieval (Q10) | 34s total | 94.6s total | gemma4 |
| Memory footprint | 17GB (13GB headroom ✅) | 21GB (8GB headroom ⚠️) | gemma4 |

**Wall time paradox:** qwen3.6's higher t/s doesn't translate to faster completions. For Q3 (reasoning), gemma4 finished in 29s vs qwen3.6's 48s — qwen3.6 generated ~2× as many tokens. For interactive chat, this is a mixed result: more thorough answers but longer waits.

**Thinking (Q5):** qwen3.6's thinking TTFT of 6.5s vs gemma4's 24s is a material UX win. For reasoning tasks where thinking is explicitly enabled, qwen3.6 is strongly preferred.

**Long context (Q10):** qwen3.6's turn-2 took 82s vs gemma4's 15s. qwen3.6 appears to have generated a substantially longer answer after reading server.py, which may indicate better comprehension or simply more verbosity.

---

## Manual Quality Scores

*Scale: 0 = wrong/broken · 1 = partially correct · 2 = fully correct*

| Q | Difficulty | Category | gemma4 | qwen3.6 | Notes |
|---|-----------|---------|--------|---------|-------|
| 1 | easy | baseline | 2 | 2 | Both returned "4". |
| 2 | easy | code-no-tools | 2 | 2 | Both correct parsers. qwen3.6 adds datetime validation. |
| 3 | medium | reasoning | 2 | 2 | Both solid. qwen3.6 more structured (17 steps + code examples); gemma4 more concise. |
| 4 | medium | long-output | 2 | 2 | Both working sqlite3 context managers with rollback + LOG_SQL. qwen3.6 uses `@contextmanager` + logging module; gemma4 uses class-based `__enter__`/`__exit__`. |
| 5 | medium | thinking-toggle | 1 | 2 | gemma4: first sentence says "553 Service Unavailable" (wrong — SMTP code, not HTTP). Rest correct. qwen3.6: correct throughout, no regression vs Q3. **Delta: thinking helped qwen3.6, hurt gemma4.** |
| 6 | hard | agentic-single-tool | 0 | 0 | Benchmark design flaw — see note below. |
| 7 | hard | agentic-multi-step | 0 | 0 | Benchmark design flaw — see note below. |
| 8 | hard | agentic-read-reason | 0 | 0 | Benchmark design flaw — see note below. |
| 9 | expert | agentic-task-done | 0 | 0 | Benchmark design flaw — see note below. |
| 10 | expert | multi-turn-long-context | 1 | 0 | gemma4: correctly identified guard is in `core.orchestrator` not `server.py` — partial credit. qwen3.6: kept trying to search GitHub for the file rather than reasoning from injected code. |

**Q6–Q9 root cause (not a model failure):** The benchmark used `conversation_id=__bench__` which creates a bare conversation with no project. Without a project association, Mira's `_active_tools` filter removes all local tools (`run_shell`, `read_file`, etc.) — neither model ever had access to them. Both correctly fell back to what was available (GitHub tools for Q6–Q8; "filesystem unavailable" for Q9). This is a benchmark runner design flaw, not a capability gap. Fixed in v2: `bench_compare.py` now accepts `--project-name` and creates project-scoped conversations for agentic questions. Q6–Q8 prompts also rewritten with explicit absolute paths.

---

## Raw data

- `scripts/bench_raw_2026-05-24_gemma4_26b-mlx.jsonl`
- `scripts/bench_raw_2026-05-24_qwen3.6_35b-mlx.jsonl`


---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | — | 36.6s | — |
| 7 | hard | agentic-multi-step | 48702ms | 66.4s | 44.1 |
| 8 | hard | agentic-read-reason | 28979ms | 84.1s | 10.8 |
| 9 | expert | agentic-task-done | — | 24.2s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | write_file, list_files, run_shell, write_file, read_file | no |

### Manual quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score | Notes |
|---|-----------|---------|---|---|
| 6 | hard | agentic-single-tool | 0 | 15× run_shell, hit MAX_AGENT_STEPS, no answer produced |
| 7 | hard | agentic-multi-step | 1 | Output present but matched noise (yaml/jsonl files), no real TODO/FIXME in .py files |
| 8 | hard | agentic-read-reason | 2 | Correct explanation, accurate code quotes, proper structure |
| 9 | expert | agentic-task-done | 1 | File created (correct content), wrong path (project root not /tmp), no task_done, no text summary |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | qwen3.6:35b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: wall-clock timeout after 120s (0 tool calls) | — | — |
| 7 | hard | agentic-multi-step | ERR: wall-clock timeout after 120s (0 tool calls) | — | — |
| 8 | hard | agentic-read-reason | ERR: wall-clock timeout after 120s (0 tool calls) | — | — |
| 9 | expert | agentic-task-done | ERR: wall-clock timeout after 120s (0 tool calls) | — | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6:35b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | ERR | — |

### Manual quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:35b-mlx score | Notes |
|---|-----------|---------|---|---|
| 6 | hard | agentic-single-tool | N/A | Invalid run — mira.yaml pointed to gemma4; all results are ERR |
| 7 | hard | agentic-multi-step | N/A | Invalid run — see above |
| 8 | hard | agentic-read-reason | N/A | Invalid run — see above |
| 9 | expert | agentic-task-done | N/A | Invalid run — see above |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | qwen3.6:35b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Stopped: run_shell×7 — same tool called too many times in one turn. | — | — |
| 7 | hard | agentic-multi-step | — | 93.5s | — |
| 8 | hard | agentic-read-reason | 27136ms | 65.5s | 12.7 |
| 9 | expert | agentic-task-done | — | 31.4s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6:35b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, list_files, run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | write_file, list_files, write_file, run_shell | no |

### Manual quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:35b-mlx score | Notes |
|---|-----------|---------|---|---|
| 6 | hard | agentic-single-tool | 0 | SAME_TOOL_REPEAT_LIMIT fired at run_shell×7, no answer produced |
| 7 | hard | agentic-multi-step | 0 | 6 tool calls executed, 0 chars content — model completed tools but emitted no text response |
| 8 | hard | agentic-read-reason | 2 | Correct explanation, accurate code quotes, clear structure |
| 9 | expert | agentic-task-done | 1 | File created (correct content), wrong path (project root not /tmp), no task_done, no text summary |

---

## Q6–Q9 Re-run Analysis (2026-05-24, project-scoped)

### Scores

| Q | Category | Expected calls | gemma4 score | qwen3.6 score |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | 0/2 | 0/2 |
| 7 | agentic-multi-step | 2 | 1/2 | 0/2 |
| 8 | agentic-read-reason | 1 | **2/2** | **2/2** |
| 9 | agentic-task-done | 3 | 1/2 | 1/2 |
| **Total** | | | **4/8** | **3/8** |

### Findings

**Q6 (count lines):** Both models looped. gemma4 ran 15 near-identical `find | wc` variants and hit `MAX_AGENT_STEPS` without producing a number. qwen3.6 hit `SAME_TOOL_REPEAT_LIMIT` at run_shell×7. Neither got a final answer. Correct answer was ~4,616 lines (excl. `.venv`). Both models treated the first tool result as uncertain and kept retrying — this is a prompting/trust gap, not a capability gap.

**Q7 (find TODO/FIXME):** gemma4 grep'd correctly but included false positives from `bench_questions.yaml` and `bench_raw_*.jsonl` — no actual TODO/FIXME comments exist in the .py files. qwen3.6 ran 6 tool calls successfully but emitted no closing text response (silent completion failure — model treats tool results as sufficient without summarizing).

**Q8 (explain divergence guard):** Both models read the file once and produced complete, accurate explanations with relevant code quotes. Best result in this set — read-then-reason is the sweet spot for current agentic capability.

**Q9 (create file + verify):** Both models used `write_file` (project-scoped) instead of `run_shell echo >`, writing to the project root instead of `/tmp/`. Content was correct (`May 24, 2026\nbench OK`). Neither called `task_done`, neither produced a text summary. The task was effectively completed via tools but the agent loop ended silently.

### Open issues

1. **task_done never fires:** Neither model called `task_done` after completing Q9. RULE 7 in the system prompt says to call it "when the task is complete" but both models terminate the tool loop without signaling done. Needs a stronger prompt signal or an example.
2. **Silent completions (qwen3.6 Q7, Q9):** qwen3.6 completes tool calls then emits zero tokens. Likely treating tool results as the final answer. This prevents any text response from reaching the user.
3. **Loop behavior (Q6):** Both models retry a working command with minor variations. A one-shot instruction ("run one command and stop") or capping the observation feedback would help.

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | — | 102.9s | — |
| 7 | hard | agentic-multi-step | 33450ms | 85.3s | 14.8 |
| 8 | hard | agentic-read-reason | 28529ms | 69.5s | 13.7 |
| 9 | expert | agentic-task-done | — | 29.8s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell | no |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell, run_shell | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Stopped: run_shell×7 — same tool called too many times in one turn. | — | — |
| 7 | hard | agentic-multi-step | ERR: Stopped: run_shell×7 — same tool called too many times in one turn. | — | — |
| 8 | hard | agentic-read-reason | 25875ms | 68.7s | 14.6 |
| 9 | expert | agentic-task-done | — | 33.5s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell, run_shell | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Stopped: run_shell×13 — same tool called too many times in one turn. | — | — |
| 7 | hard | agentic-multi-step | 44910ms | 144.6s | 30.1 |
| 8 | hard | agentic-read-reason | 29909ms | 69.0s | 14.5 |
| 9 | expert | agentic-task-done | — | 27.0s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | run_shell, run_shell, run_shell | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | qwen3.6:35b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | 9099ms | 58.7s | 6.5 |
| 7 | hard | agentic-multi-step | ERR: Reached 15 tool calls without a final answer. | — | — |
| 8 | hard | agentic-read-reason | 21595ms | 86.7s | 14.0 |
| 9 | expert | agentic-task-done | ERR: Reached 15 tool calls without a final answer. | — | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6:35b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell | no |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:35b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Stopped: run_shell×13 — same tool called too many times in one turn. | — | — |
| 7 | hard | agentic-multi-step | — | 137.6s | — |
| 8 | hard | agentic-read-reason | 30224ms | 65.3s | 15.1 |
| 9 | expert | agentic-task-done | ERR: wall-clock timeout after 300s (2 tool calls) | — | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Stopped: run_shell×13 — same tool called too many times in one turn. | — | — |
| 7 | hard | agentic-multi-step | 56623ms | 74.3s | 33.8 |
| 8 | hard | agentic-read-reason | 24816ms | 70.1s | 13.1 |
| 9 | expert | agentic-task-done | — | 34.0s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | write_file, run_shell, run_shell, run_shell, run_shell, run_shell | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: wall-clock timeout after 300s (4 tool calls) | — | — |
| 7 | hard | agentic-multi-step | ERR: wall-clock timeout after 300s (0 tool calls) | — | — |
| 8 | hard | agentic-read-reason | ERR: wall-clock timeout after 300s (0 tool calls) | — | — |
| 9 | expert | agentic-task-done | ERR: wall-clock timeout after 300s (0 tool calls) | — | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Stopped: run_shell×13 — same tool called too many times in one turn. | — | — |
| 7 | hard | agentic-multi-step | 39198ms | 137.3s | 28.9 |
| 8 | hard | agentic-read-reason | 29118ms | 66.2s | 11.3 |
| 9 | expert | agentic-task-done | — | 28.7s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | write_file, run_shell, run_shell | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 9 | expert | agentic-task-done | — | 30.9s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 9 | agentic-task-done | 3 | run_shell, run_shell, run_shell | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | — | 66.2s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Stopped: run_shell×13 — same tool called too many times in one turn. | — | — |
| 7 | hard | agentic-multi-step | 59504ms | 156.0s | 31.5 |
| 8 | hard | agentic-read-reason | 28878ms | 78.6s | 12.1 |
| 9 | expert | agentic-task-done | — | 32.0s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
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

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Stopped: run_shell×13 — same tool called too many times in one turn. | — | — |
| 7 | hard | agentic-multi-step | — | 132.4s | — |
| 8 | hard | agentic-read-reason | 30726ms | 76.9s | 11.0 |
| 9 | expert | agentic-task-done | — | 33.2s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
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

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Reached 15 tool calls without a final answer. | — | — |
| 7 | hard | agentic-multi-step | 66194ms | 149.9s | 40.1 |
| 8 | hard | agentic-read-reason | 30355ms | 65.6s | 13.2 |
| 9 | expert | agentic-task-done | — | 31.3s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
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

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | ERR: Reached 15 tool calls without a final answer. | — | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | — | 39.2s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4:26b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | gemma4:26b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | — | 36.5s | — |
| 7 | hard | agentic-multi-step | 69665ms | 162.9s | 41.3 |
| 8 | hard | agentic-read-reason | 30641ms | 60.4s | 13.2 |
| 9 | expert | agentic-task-done | — | 22.5s | — |

### Agentic results

| Q | Category | Expected calls | gemma4:26b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
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

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | qwen3.6:35b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | 8199ms | 69.2s | 7.3 |
| 7 | hard | agentic-multi-step | ERR: Reached 15 tool calls without a final answer. | — | — |
| 8 | hard | agentic-read-reason | 20351ms | 78.8s | 8.8 |
| 9 | expert | agentic-task-done | 2189ms | 41.2s | 6.8 |

### Agentic results

| Q | Category | Expected calls | qwen3.6:35b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | write_file, run_shell, run_shell | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:35b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | qwen3.6:35b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 7 | hard | agentic-multi-step | ERR: Reached 15 tool calls without a final answer. | — | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6:35b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 7 | agentic-multi-step | 2 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:35b-mlx score |
|---|-----------|---------|---|
| 7 | hard | agentic-multi-step | — |

---

## Benchmark Results — 2026-05-24

### Timing

| Q | Difficulty | Category | qwen3.6:35b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 7 | hard | agentic-multi-step | 18368ms | 141.2s | 15.6 |

### Agentic results

| Q | Category | Expected calls | qwen3.6:35b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, search_files, search_files, run_shell, run_shell, run_shell | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:35b-mlx score |
|---|-----------|---------|---|
| 7 | hard | agentic-multi-step | — |

---

## Benchmark Results — 2026-05-25

### Timing

| Q | Difficulty | Category | qwen3.6:35b-mlx TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | 11464ms | 62.3s | 4.2 |
| 7 | hard | agentic-multi-step | 36084ms | 147.5s | 14.5 |
| 8 | hard | agentic-read-reason | 26606ms | 96.3s | 14.2 |
| 9 | expert | agentic-task-done | 21719ms | 72.9s | 14.3 |

### Agentic results

| Q | Category | Expected calls | qwen3.6:35b-mlx calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell, run_shell, run_shell, run_shell | no |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, list_files, run_shell, search_files, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | no |
| 9 | agentic-task-done | 3 | write_file, run_shell, run_shell, read_file, run_shell, run_shell, run_shell, run_shell, run_shell, read_file, write_file, read_file, run_shell | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6:35b-mlx score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |