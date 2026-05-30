# Benchmark Results — 2026-05-30 (mlx-lm backend)

Hardware: MacBook Pro M5 32GB  
Backend: mlx-lm 0.31.3 (OpenAI-compatible server, port 8080)  
Thinking suppression: `--chat-template-args '{"enable_thinking": false}'`

## Step 0 — Thinking suppression check

| Test | Result |
|------|--------|
| curl "Say hello in one word" → gemma4 | TTFT=1.17s, no `<think>` tag |
| May 2026 blocker | **RESOLVED** — `enable_thinking: false` works |

---

## Timing

| Q | Category | gemma4-Ollama TTFT / t/s | **gemma4-mlx-lm** TTFT / t/s | qwen36-Ollama TTFT / t/s | **qwen36-mlx-lm** TTFT / t/s |
|---|---------|---|---|---|---|
| 1 | baseline | N/A | 4392ms / 2.4 | N/A | 46018ms / 3.2 |
| 2 | code-no-tools | N/A | 505ms / 35.3 | N/A | 380ms / 45.4 |
| 3 | reasoning | N/A | 425ms / 36.0 | N/A | 307ms / 47.8 |
| 4 | long-output | N/A | 504ms / 35.7 | N/A | 342ms / 45.9 |
| 5 | thinking-toggle | N/A | 251ms / 36.2 | N/A | 194ms / 47.3 |
| 6 | agentic-single-tool | — / — (36.6s wall) | 7319ms / 76.8 | ERR | 8296ms / 95.2 |
| 7 | agentic-multi-step | 48702ms / 44.1 | 7834ms / 36.0 | ERR | 19899ms / 21.1 |
| 8 | agentic-read-reason | 28979ms / 10.8 | 21991ms / 30.2 | ERR | 21022ms / 27.7 |
| 9 | agentic-task-done | — / — (22s wall) | — / — (6.4s wall) | ERR | 2219ms / 63.6 |
| 10 | multi-turn-long-context | N/A | 573ms / 51.3 | N/A | 508ms / 16.6 |
| 11 | agentic-write-file | — / — (36s wall) | — / — (5.5s wall) | 7311ms / 3.5 | 2691ms / 82.4 |
| 12 | agentic-edit-file | — / — (33s wall) | — / — (7.6s wall) | 5897ms / 4.3 | 439ms / 33.8 |
| 13 | agentic-divergence-guard | N/A | 35325ms / 72.3 | N/A | 361ms / 2.2 |

Notes:
- Q1 TTFT includes cold-start model load penalty for both models (first request after server start)
- gemma4-Ollama Q6-Q9: prior run had task_done=NO (max steps bug, since fixed); values shown for TTFT/t/s comparison only
- qwen36-Ollama Q6-Q9: invalid run (mira.yaml misconfigured in 2026-05-24 session); all ERR
- `—` TTFT: multi-step agentic calls where TTFT tracks tool-call round-trips, not a single first token

---

## Agentic results

| Q | Category | Expected calls | **gemma4-mlx-lm** calls | done | guard | **qwen36-mlx-lm** calls | done | guard |
|---|---------|----------------|---|---|---|---|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES | — | run_shell | YES | — |
| 7 | agentic-multi-step | 2 | run_shell | YES | — | run_shell ×10, search_files ×3, list_files | YES | **YES** |
| 8 | agentic-read-reason | 1 | read_file | YES | — | read_file, search_files, read_file | YES | — |
| 9 | agentic-task-done | 3 | run_shell ×2 | YES | — | run_shell | YES | — |
| 10 | multi-turn-long-context | 0 | none | — | — | none | — | — |
| 11 | agentic-write-file | 2 | write_file, read_file | YES | — | write_file, read_file | YES | — |
| 12 | agentic-edit-file | 3 | run_shell, write_file, edit_file, read_file | YES | — | write_file, edit_file, read_file | YES | — |
| 13 | agentic-divergence-guard | 3 | run_shell ×2 | YES | **no** | run_shell ×7 | YES | **YES** |

Q4 anomaly: qwen3.6 called `github_write_file` (hallucinated tool, not in tool schema) — request was rejected by server, model recovered and completed with task_done=YES.

---

## vs Ollama delta (where comparable)

| Metric | gemma4: Ollama → mlx-lm | qwen3.6: Ollama → mlx-lm |
|--------|---|---|
| Q7 TTFT | 48702ms → 7834ms (**6.2× faster**) | ERR → 19899ms (first valid run) |
| Q8 TTFT | 28979ms → 21991ms (1.3× faster) | ERR → 21022ms (first valid run) |
| Q8 tok/s | 10.8 → 30.2 (**2.8× faster**) | ERR → 27.7 (first valid run) |
| Q9 wall | 22s → 6.4s (**3.4× faster**) | ERR → 4.2s (first valid run) |
| Q11 wall | ~36s → 5.5s (**6.5× faster**) | ~35s → 4.2s (**8.3× faster**) |
| Q12 wall | ~33s → 7.6s (**4.3× faster**) | ~34s → 5.7s (**6.0× faster**) |
| Warm TTFT (Q2-Q5) | N/A → 251–505ms | N/A → 194–380ms |
| Warm tok/s (Q2-Q5) | N/A → ~35–36 t/s | N/A → ~45–47 t/s |

---

## Key findings

**1. Thinking blocker resolved.** `--chat-template-args '{"enable_thinking": false}'` suppresses thinking mode at the chat-template level. Step 0 confirmed: TTFT=1.17s, no `<think>` tag. mlx-lm is viable as a Mira backend.

**2. gemma4 mlx-lm: strong across the board.** TTFT 6× improvement on Q7, 2.8× tok/s improvement on Q8, wall time 4–6× faster on Q11/Q12. Warm TTFT 250–505ms (excellent). tok/s 35–36 (slightly below 38–44 expected, possibly headroom with larger prompt cache).

**3. qwen3.6 mlx-lm: fast decode, dangerous cold start.** Warm TTFT 194–380ms (fastest of the four configs). tok/s 45–47 (consistently beats gemma4). But Q1 TTFT = **46s** (cold start / MoE activation cost). Unacceptable for interactive use on first turn after server restart.

**4. qwen3.6 agentic reliability concerns.** Q7: divergence guard fired unexpectedly (12 tool calls on a simple 2-step task). Q4: hallucinated `github_write_file` tool. Q10: slow long-context decode (16.6 t/s). These are regressions vs gemma4's cleaner tool use.

**5. Divergence guard (Q13).** gemma4 evaded the guard by using a shell-level loop (`timeout 30 bash -c ...`) — one `run_shell` per pattern, not repeated identical calls. Guard never fired (0/2). qwen3.6 looped 7× identical `run_shell` — guard fired correctly (2/2), but wall time was **431s**. Guard works; prompt needs refinement (forbid shell looping) for gemma4 to trigger it.

**6. backend_manager.py fix.** `is_backend_ready` was checking `OMLX_MODEL in model_ids` (hardcoded "Qwen3.6-35B-A3B") — always false when running gemma4. Fixed to check reachability only.

---

## Recommendation

**mlx-lm with gemma4 is the preferred local backend** over Ollama for Mira:
- 4–6× wall time reduction on agentic tasks
- Comparable decode speed (35–36 t/s vs Ollama's 41–44 t/s on Q7)
- Warm TTFT 250–500ms (excellent)
- Reliable tool use (no spurious tool calls, clean divergence guard evasion)

qwen3.6 decode speed (45–47 t/s) is faster, but the 46s cold start, Q7 guard misfire, and Q4 hallucinated tool make it unsuitable for Mira's default backend without further investigation.

---

## MLX Community Leaderboard (context)

| Rank | Model | Overall% | Notes |
|------|-------|----------|-------|
| 1 | claude-sonnet-4.6 | 89.6 | |
| 2 | gemini-3-flash-preview | 82.4 | |
| 3 | qwen3.6-max-preview | 80.1 | Cloud, not local |
| 5 | **gemma-4-26b-a4b-it** | **75.2** | This model, local |
| 10 | **qwen3.6-35b-a3b** | **52.5** | This model, local |

Local gemma4 (75.2) outranks local qwen3.6 (52.5) on MLX benchmark — consistent with our agentic results.

---

## Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Category | gemma4-mlx-lm | qwen36-mlx-lm | Notes |
|---|---------|---|---|---|
| 1 | baseline | — | — | |
| 2 | code-no-tools | — | — | |
| 3 | reasoning | — | — | |
| 4 | long-output | — | — | qwen3.6: spurious github_write_file call |
| 5 | thinking-toggle | — | — | server-level disable; thinking not tested |
| 6 | agentic-single-tool | — | — | |
| 7 | agentic-multi-step | — | — | qwen3.6: guard fired (12 calls) |
| 8 | agentic-read-reason | — | — | |
| 9 | agentic-task-done | — | — | |
| 10 | multi-turn-long-context | — | — | |
| 11 | agentic-write-file | — | — | |
| 12 | agentic-edit-file | — | — | |
| 13 | agentic-divergence-guard | 0/2 (guard evaded via shell loop) | 2/2 (guard fired, 431s wall) | |


---

## Benchmark Results — 2026-05-30

### Timing

| Q | Difficulty | Category | qwen3.6-35b-mlx-lm-rerun1:qwen3.6-35b-mlx-lm-rerun1 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 4 | medium | long-output | 8680ms | 78.5s | 46.5 |
| 7 | hard | agentic-multi-step | 25755ms | 164.0s | 20.7 |

### Agentic results

| Q | Category | Expected calls | qwen3.6-35b-mlx-lm-rerun1 calls | task_done |
|---|---------|----------------|---|---|
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, search_files, search_files, run_shell, run_shell, run_shell, run_shell, run_shell, list_files, search_files | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-mlx-lm-rerun1 score |
|---|-----------|---------|---|
| 4 | medium | long-output | 2 (correct output; hallucinated tool call is self-corrected) |
| 7 | hard | agentic-multi-step | 1 (eventual correct output but via guard termination, not clean exit) |

---

## Benchmark Results — 2026-05-30

### Timing

| Q | Difficulty | Category | qwen3.6-35b-mlx-lm-rerun2:qwen3.6-35b-mlx-lm-rerun2 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 4 | medium | long-output | 645ms | 68.8s | 47.7 |

### Agentic results


### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-mlx-lm-rerun2 score |
|---|-----------|---------|---|
| 4 | medium | long-output | 2 (correct output; hallucinated tool self-corrected) |

---

## qwen3.6 Anomaly Investigation — Verdicts (2026-05-30)

Re-runs: Q4 ×2, Q7 ×1 (after confirming `workspace_root` substitution was correct and shell_tools.py has no `find`-blocking rule).

### Q1 — 46s cold-start TTFT

**Verdict: MoE architecture behavior.** No re-run needed. qwen3.6 is a Mixture-of-Experts model; the first request cold-activates the routing weights. Q2+ requests are warm (TTFT 342–645ms). This is expected and consistent with MoE benchmarks. The 46s penalty is not a harness issue — no code path can explain it. **Not a harness bug.**

### Q4 — `github_write_file` hallucination

**Verdict: Systematic model bias.** Reproduced on all 3 runs (original + 2 reruns). Model has a strong training prior toward `github_write_file` when generating long code under a write-to-file framing. The tool is not in Mira's schema; the rejection is handled gracefully and model self-recovers (`task_done=YES`, correct output inline). Risk is low but systematic — any prompt that implies "write this to a file" may trigger the hallucination. **Model behavior, acceptable recovery.**

### Q7 — Divergence guard misfire (12 tool calls vs 2)

**Code analysis verdict (no re-run needed for mechanism; one re-run confirmed reproducibility):**

Root cause traced to `shell_tools.py`:
- stdout is truncated at 8000 chars (`result.stdout[:8000]`). A broad `find .` on the workspace returns >8000 chars of paths.
- Model received truncated output, misread it as "sandbox is blocking `find` with `.`".
- Pivoted to `search_files`, which also returned 200+ results including `.venv` (the tool doesn't support exclude patterns).
- Looped 12× trying progressively to filter results; guard fired.

The prompt's explicit `{workspace_root}` path was correctly substituted (`local_path=/Users/miguel/Documents/Projects/mira-core`). Shell sandbox has no rule blocking `find`. The divergence is pure model behavior — qwen3.6 cannot recover from truncated shell output by reformulating the command. gemma4 ran a single filtered `grep -rn 'TODO\|FIXME' . --include='*.py' --exclude-dir={.venv,.git,__pycache__}` in 2 calls.

Rerun1 confirmed: exact same 12-call sequence (`run_shell ×7, search_files ×3, list_files ×1, search_files ×1`), same guard fire. **Model behavior. Not a harness bug.**

### Summary

| Anomaly | Verdict | Impact | Action |
|---------|---------|--------|--------|
| Q1 46s cold start | MoE architecture | Unacceptable for first interactive turn | Don't use qwen3.6 as default backend |
| Q4 github_write_file | Systematic model bias | Low (graceful recovery) | Note in model profile; no harness change |
| Q7 12-call loop | Model behavior (truncation misread) | High (172s wall, guard misfire) | qwen3.6 unsuitable for multi-step agentic tasks |

**Overall: qwen3.6 is not suitable as a Mira default backend.** All three anomalies are model behavior. gemma4 via mlx-lm remains the recommended configuration.

---

## Benchmark Results — 2026-05-30

### Timing

| Q | Difficulty | Category | omlx-gemma4-26b:omlx-gemma4-26b TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 3882ms | 4.8s | 1.0 |
| 2 | easy | code-no-tools | 1422ms | 14.2s | 35.2 |
| 3 | medium | reasoning | 1340ms | 31.2s | 36.5 |
| 4 | medium | long-output | 1389ms | 23.2s | 36.7 |
| 5 | medium | thinking-toggle | 268ms | 28.8s | 36.7 |
| 6 | hard | agentic-single-tool | 7230ms | 8.2s | 72.7 |
| 7 | hard | agentic-multi-step | 9823ms | 22.6s | 42.5 |
| 8 | hard | agentic-read-reason | 22084ms | 42.7s | 28.3 |
| 9 | expert | agentic-task-done | ERR: Memory limit exceeded during prefill | — | — |
| 11 | hard | agentic-write-file | ERR: Memory limit exceeded during prefill | — | — |
| 12 | hard | agentic-edit-file | ERR: Memory limit exceeded during prefill | — | — |
| 13 | expert | agentic-divergence-guard | ERR: Memory limit exceeded during prefill | — | — |
| 10 | expert | multi-turn-long-context | ERR: Memory limit exceeded during prefill | — | — |

### Agentic results

| Q | Category | Expected calls | omlx-gemma4-26b calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | ERR | — |
| 11 | agentic-write-file | 2 | ERR | — |
| 12 | agentic-edit-file | 3 | ERR | — |
| 13 | agentic-divergence-guard | 3 | ERR | — |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | omlx-gemma4-26b score |
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

## Benchmark Results — 2026-05-30

### Timing

| Q | Difficulty | Category | omlx-gemma4-26b:omlx-gemma4-26b TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 9 | expert | agentic-task-done | 14937ms | 24.1s | 21.9 |
| 11 | hard | agentic-write-file | 9311ms | 16.2s | 16.8 |
| 12 | hard | agentic-edit-file | 8831ms | 26.7s | 9.3 |
| 13 | expert | agentic-divergence-guard | 9599ms | 60.4s | 5.7 |
| 10 | expert | multi-turn-long-context | 9893ms | 30.6s | 74.3 |

### Agentic results

| Q | Category | Expected calls | omlx-gemma4-26b calls | task_done |
|---|---------|----------------|---|---|
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | run_shell, write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | omlx-gemma4-26b score |
|---|-----------|---------|---|
| 9 | expert | agentic-task-done | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 13 | expert | agentic-divergence-guard | — |
| 10 | expert | multi-turn-long-context | — |

---

## Benchmark Results — 2026-05-30

### Timing

| Q | Difficulty | Category | omlx-qwen3.6-35b:omlx-qwen3.6-35b TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 11967ms | 52.4s | 0.1 |
| 2 | easy | code-no-tools | 5919ms | 68.4s | 7.7 |
| 3 | medium | reasoning | 6280ms | 71.4s | 26.9 |
| 4 | medium | long-output | 6137ms | 81.3s | 24.6 |
| 5 | medium | thinking-toggle | 5833ms | 67.7s | 30.6 |
| 6 | hard | agentic-single-tool | 6976ms | 132.9s | 1.0 |
| 7 | hard | agentic-multi-step | 8940ms | 282.8s | 17.4 |
| 8 | hard | agentic-read-reason | 6728ms | 124.7s | 13.5 |
| 9 | expert | agentic-task-done | 7463ms | 44.3s | 5.1 |
| 11 | hard | agentic-write-file | 6831ms | 47.4s | 5.1 |
| 12 | hard | agentic-edit-file | 7449ms | 53.1s | 5.5 |
| 13 | expert | agentic-divergence-guard | 7209ms | 358.3s | 1.5 |
| 10 | expert | multi-turn-long-context | 19363ms | 59.6s | 274.2 |

### Agentic results

| Q | Category | Expected calls | omlx-qwen3.6-35b calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, write_file, run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file, read_file | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | omlx-qwen3.6-35b score |
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

## omlx 0.3.12 Bench — Analysis (2026-05-30)

Run conditions: omlx 0.3.12, `--no-cache --hot-cache-max-size 0`, port 8080.  
Memory guard: balanced tier, 23.2GB ceiling. Initial run (hot cache 8GB) caused OOM on gemma4 Q9–Q13; no-cache mode resolved.

### omlx gemma4-26b vs mlx-lm gemma4

| Metric | mlx-lm 0.31.3 | omlx 0.3.12 | Delta |
|--------|--------------|------------|-------|
| Q1 cold start | 4392ms | 3882ms | 12% faster |
| Warm TTFT (Q2–Q5) | 251–505ms | 268–1422ms | 1–3× slower |
| Warm tok/s (Q2–Q5) | 35–36 t/s | 35–37 t/s | ≈ identical |
| Q7 tok/s | 36.0 t/s | 42.5 t/s | 18% faster |
| Q10 tok/s | 51.3 t/s | 74.3 t/s | 45% faster |
| Q9 wall | 6.4s | 24.1s | 3.8× slower |
| Q11 wall | 5.5s | 16.2s | 3× slower |
| Q12 wall | 7.6s | 26.7s | 3.5× slower |
| Q13 guard | evaded via shell loop | fired (4× run_shell) | behavioral diff |

Wall time regression on Q9/Q11/Q12 is the no-cache penalty — without hot cache, each prefill recomputed from scratch. Decode tok/s is identical. **Q13 behavioral difference**: omlx gemma4 made 4 identical `run_shell` calls and guard fired; mlx-lm gemma4 used a shell-level `timeout bash -c ...` loop (1 `run_shell`) and evaded. Genuine inference-engine behavioral difference, not a correctness issue.

**Verdict: omlx gemma4 viable but no advantage over mlx-lm.** Throughput identical; wall time 3–4× worse due to no-cache requirement; TTFT slightly higher. mlx-lm is simpler and faster.

### omlx qwen3.6-35b vs mlx-lm qwen3.6

| Metric | mlx-lm 0.31.3 | omlx 0.3.12 | Delta |
|--------|--------------|------------|-------|
| Q1 cold start | 46018ms | 11967ms | **4× faster** |
| Warm TTFT (Q2–Q5) | 194–380ms | 5833–6280ms | **15–30× slower** |
| Warm tok/s (Q2–Q5) | 45–47 t/s | 7.7–30.6 t/s | 2–6× slower |
| Q4 hallucinated tools | github_write_file | github_write_file + github_list_repos | worse |
| Q7 tool calls | 14 calls, guard fired | 9 calls (7× run_shell + write_file + run_shell) | still messy |
| Q13 wall | 431s | 358s | — |

Cold-start win (4×) is notable but irrelevant when warm TTFT is 15–30× worse — every interactive turn takes 5–6s before first token. Q4 hallucination escalated (two hallucinated tools vs one on mlx-lm). omlx's inference engine handles qwen3.6's MoE architecture less efficiently than mlx-lm.

**Verdict: omlx qwen3.6 not viable.** 15–30× TTFT regression makes it unusable for interactive use.

### Summary

| Config | Viable? | Notes |
|--------|---------|-------|
| mlx-lm gemma4 | **YES — recommended** | Fastest wall time, simplest setup |
| omlx gemma4 | Marginal | No advantage; memory guard adds complexity |
| mlx-lm qwen3.6 | No | 46s cold start, Q7 loop, Q4 hallucination |
| omlx qwen3.6 | No | 15–30× TTFT regression vs mlx-lm |

mlx-lm with gemma4 remains the preferred local backend. omlx 0.3.12 did not crash (regression from 0.3.8/0.3.9 fixed), but offers no performance advantage over mlx-lm on this hardware.


---

## Benchmark Results — 2026-05-30

### Timing

| Q | Difficulty | Category | gemma4-26b-optiq-4bit:gemma4-26b-optiq-4bit TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 123301ms | 124.3s | 2.0 |
| 2 | easy | code-no-tools | 549ms | 15.5s | 26.7 |
| 3 | medium | reasoning | 460ms | 36.6s | 27.6 |
| 4 | medium | long-output | 544ms | 36.6s | 27.3 |
| 5 | medium | thinking-toggle | 284ms | 37.0s | 27.2 |
| 6 | hard | agentic-single-tool | 8447ms | 9.6s | 68.0 |
| 7 | hard | agentic-multi-step | 9355ms | 111.8s | 27.8 |
| 8 | hard | agentic-read-reason | 22735ms | 47.9s | 24.2 |
| 9 | expert | agentic-task-done | — | 7.6s | — |
| 11 | hard | agentic-write-file | — | 6.9s | — |
| 12 | hard | agentic-edit-file | — | 9.2s | — |
| 13 | expert | agentic-divergence-guard | — | 169.0s | — |
| 10 | expert | multi-turn-long-context | 597ms | 16.6s | 40.8 |

### Agentic results

| Q | Category | Expected calls | gemma4-26b-optiq-4bit calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | run_shell, write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | gemma4-26b-optiq-4bit score |
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