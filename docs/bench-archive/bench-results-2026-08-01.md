# Benchmark Results — 2026-08-01

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Benchmark Results — 2026-08-01

### Timing

| Q | Difficulty | Category | mlx-0.32.0-qwen3.6-35b-a3b-4bit:mlx-0.32.0-qwen3.6-35b-a3b-4bit TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 10672ms | 10.7s | — |
| 2 | easy | code-no-tools | 2957ms | 6.7s | — |
| 3 | medium | reasoning | 2948ms | 31.2s | — |
| 4 | medium | long-output | — | 4.1s | — |
| 5 | medium | thinking-toggle | 4079ms | 25.2s | 3.9 |
| 6 | hard | agentic-single-tool | 6685ms | 6.8s | — |
| 7 | hard | agentic-multi-step | 9996ms | 21.6s | — |
| 8 | hard | agentic-read-reason | 25055ms | 63.5s | — |
| 9 | expert | agentic-task-done | 4495ms | 11.2s | — |
| 11 | hard | agentic-write-file | 7050ms | 7.7s | — |
| 12 | hard | agentic-edit-file | 8363ms | 9.0s | — |
| 13 | expert | agentic-divergence-guard | 4645ms | 39.8s | — |
| 10 | expert | multi-turn-long-context | 49843ms | 101.6s | — |

### Agentic results

| Q | Category | Expected calls | mlx-0.32.0-qwen3.6-35b-a3b-4bit calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | Score | Note |
|---|-----------|---------|---|---|
| 1 | easy | baseline | 2 | |
| 2 | easy | code-no-tools | 2 | |
| 3 | medium | reasoning | 2 | |
| 4 | medium | long-output | 1 | called `task_done` with a summary instead of writing the code, see below |
| 5 | medium | thinking-toggle | 2 | |
| 6 | hard | agentic-single-tool | 2 | 8319 is the exact line count |
| 7 | hard | agentic-multi-step | 2 | |
| 8 | hard | agentic-read-reason | 2 | |
| 9 | expert | agentic-task-done | 2 | |
| 10 | expert | multi-turn-long-context | 2 | correctly refused a false premise, see below |
| 11 | hard | agentic-write-file | 2 | |
| 12 | hard | agentic-edit-file | 2 | |
| 13 | expert | agentic-divergence-guard | 2 | guard fired, terminated gracefully in 39.8s (<120s budget) |

**25 / 26.**

## Verdict: mlx 0.32.0 shows no regression

This run is the live end-to-end check on the mlx 0.31.2 to 0.32.0 upgrade (`5ba9c71`). Everything
that was verified before the upgrade with a `PYTHONPATH` shim holds against the real production
model on the real server.

- **No reasoning leaked into any answer.** Zero occurrences of `<think>`, `</think>` or
  `<|channel|>` across all 13 responses. This is the first live confirmation of the pre-open fix
  (`81e7564`); before it, every Qwen3 turn with thinking on served its reasoning as the answer.
- **Agentic behaviour matches the 2026-07-18 baseline.** Same tool families, same `task_done`
  results, and the divergence guard still fires on Q13. Q6, Q7, Q8 and Q9 differ from that
  baseline only in tool-call *count* (one `run_shell` where there were two, and the reverse),
  which is the run-to-run variance this suite has always shown on agentic questions.
- **TTFT is unchanged** on the comparable questions: Q2 2957ms vs 2847ms, Q3 2948ms vs 2836ms.
  Q1's 10672ms is the cold first request against a freshly restarted server with an empty prompt
  cache, not a like-for-like number.

Throughput is not measured here and is not expected to move. The 0.32 gain comes from the
`qmv_wide` small-batch kernel, which only engages from about eight concurrent sequences; Mira
serves one user at a time. See `notes/mlx-032-upgrade-analysis.md` for the micro-benchmarks.

### Two answers worth reading before trusting the table

**Q4 did not deliver what was asked, and it is not an mlx problem.** The question wants a sqlite3
context manager written out in full. The model instead called `task_done` with the one-line
summary "Provided a complete Python context manager for sqlite3 ...", and the user never sees any
code: zero `token` events and zero `thinking` events reached the stream. Replaying the same
prompt three times against the same build produced the answer once and the empty summary twice, so
this is model choice under a non-deterministic sample, not a build regression. It is also not the
thinking fix: Q4 runs with thinking off, which leaves `ThinkingStripper` in its pre-`81e7564`
behaviour, and a swallowed answer would have surfaced as thinking characters anyway. What it does
show is that the agentic loop can short-circuit a question that declares `tools: false`, which is
worth fixing on its own terms. Logged in `BACKLOG.md`.

**Q10 answered correctly by contradicting the question.** Turn 2 asks where `server.py` implements
the divergence guard and to quote it. `server.py` does not implement it; `core/orchestrator.py`
does. The model said so and declined to quote code that is not there. The question carries a false
premise, so the honest answer scores 2 and the question needs rewriting, not the model.

---

## Vision regression check: the text path with `--vision` off

Run after the optional vision work landed, to confirm the changed prefill path in the
pinned fork costs nothing when vision is not in use. `mira_mlx_vision: false`, so the
tower is never imported and `/v1/stats` reports `vision: null`.

TTFT matches the run above within noise: Q2 2986ms vs 2957ms, Q3 2937ms vs 2948ms. Q1 is
the cold first request in both cases and swings accordingly (8516ms vs 10672ms).

### Timing

| Q | Difficulty | Category | vision-off-regression:vision-off-regression TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 8516ms | 8.5s | — |
| 2 | easy | code-no-tools | 2986ms | 5.9s | — |
| 3 | medium | reasoning | 2937ms | 26.6s | — |

### Agentic results


### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | vision-off-regression score |
|---|-----------|---------|---|
| 1 | easy | baseline | 2 |
| 2 | easy | code-no-tools | 2 |
| 3 | medium | reasoning | 2 |

---

## Vision, live end-to-end (2026-08-01)

Two images with no readable text of any kind, so OCR could not have produced either
answer. `mira_mlx_vision: true`, Qwen3.6-35B-A3B-4bit on mira-mlx.

| Image | Asked | Answer | Correct |
|---|---|---|---|
| Three bars: red 120px, green 300px, blue 200px, no labels | how many shapes, what colours, which is tallest | "3 rectangular bars ... Red (leftmost) the shortest, Green (middle) the tallest, Blue (rightmost) medium height" | yes, including the ordering |
| One yellow circle on dark navy | what single shape and colour | "1 shape: a circle. Colour: yellow ... Background: dark navy / near-black" | yes |

The second image was sent as the next turn of the **same conversation**, which is the
prompt-cache collision regression: an image is N copies of one placeholder token id, so
two same-sized images produce a byte-identical prefix and a naive cache would have
answered about the first one again. It did not, because image turns skip the cache.

Cost, from `/v1/stats`: tower 0.89 GB, and against the text-only baseline of 19.59 GB
active / 20.60 GB peak, vision on measured 20.69 GB active / 21.88 GB peak. So about
1.1 GB active and 1.3 GB peak, the tower plus its forward activations. A 640x480 image
costs 300 context tokens, 1024x768 costs 768.
