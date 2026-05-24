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
| 1 | easy | baseline | — | — | |
| 2 | easy | code-no-tools | — | — | |
| 3 | medium | reasoning | — | — | |
| 4 | medium | long-output | — | — | |
| 5 | medium | thinking-toggle | — | — | Compare quality delta on/off |
| 6 | hard | agentic-single-tool | — | — | Used GitHub tools, not run_shell |
| 7 | hard | agentic-multi-step | — | — | |
| 8 | hard | agentic-read-reason | — | — | |
| 9 | expert | agentic-task-done | — | — | Both failed to use tools |
| 10 | expert | multi-turn-long-context | — | — | |

---

## Raw data

- `scripts/bench_raw_2026-05-24_gemma4_26b-mlx.jsonl`
- `scripts/bench_raw_2026-05-24_qwen3.6_35b-mlx.jsonl`
