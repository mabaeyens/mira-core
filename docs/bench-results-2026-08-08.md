# Benchmark Results — 2026-08-08

Hardware: MacBook Pro M5 32GB (backend/model per run — see sections below)

## Benchmark Results — 2026-08-08

### Timing

| Q | Difficulty | Category | qwen3.6-35b-a3b:qwen3.6-35b-a3b TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | 12460ms | 12.6s | — |
| 7 | hard | agentic-multi-step | 8805ms | 35.0s | — |
| 8 | hard | agentic-read-reason | 25821ms | 66.2s | — |
| 9 | expert | agentic-task-done | 4687ms | 12.2s | — |
| 11 | hard | agentic-write-file | 7191ms | 7.8s | — |
| 12 | hard | agentic-edit-file | 4545ms | 14.2s | — |
| 10 | expert | multi-turn-long-context | 44556ms | 94.7s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-35b-a3b calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, search_files | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Why this run happened: validating the `task_done` guard live

`d917cae` made the agentic loop refuse a `task_done` whose turn produced no visible content and
ran no tools. Its unit tests are thorough but Q4, the question that motivated it, declares
`tools: false` and therefore never reaches the branch. Q6–Q12 do.

**The guard fired zero times, and every agentic question still exited through `task_done`.**
That is the result being looked for: it means the guard is inert on legitimate agentic turns
(each of Q6, Q7, Q8, Q9, Q11 and Q12 ran tools first, so the gate that keys on tools having run
let them through) and nothing regressed. It does *not* demonstrate the refusal path working
against a real model, because Qwen3.6 never took the escape hatch here. That remains covered
only by unit tests.

Q10 reporting `task_done: no` is correct and not a guard effect: it declares `tools: false`, so
`task_done` is not in its toolset at all.

### The prompt cache is not idle, and the disk half has never once been read

Read from the engine's own log, which reached disk for the first time this run (`099d381` —
mira-mlx's stdout had been going to `DEVNULL`, so every one of these lines had been written and
discarded on every request for weeks).

Across the 23 engine requests in this run:

| | |
|---|---|
| requests | 23 (12 HIT, 11 MISS) |
| prompt tokens | 168,321 |
| reused from cache | 48,024 (28.5%) |
| re-prefilled | 120,297 |
| `insert_cache` | 23 registered, **0 SKIPPED** |
| disk exact-match hits | **0** |

So the in-memory cache works, and `insert_cache SKIPPED` — the suspected culprit — never fires.
The structure of the hits is the finding:

- **Continuations hit, conversation openers miss.** An entry is `system + user + assistant`, and
  the trie can only reuse an entry that is *entirely* a prefix of the new prompt. That holds for
  the next turn of the same conversation and fails for a different conversation, which diverges
  right after the shared system prompt.
- **The shared prefix is ~3,800 tokens** and gets re-prefilled on every conversation opener:
  8 of the 11 misses are 3,787–4,025 tokens. There is no entry containing *just* the system
  prompt, because entries are only created after a generation completes.
- Short entries *do* serve longer unrelated prompts when they happen to be a full prefix:
  `HIT 3,859/18,501` and `HIT 3,859/18,728` on Q8.

**The disk cache is write-only by construction.** `DiskBackedPromptCache.fetch_nearest_cache`
falls back to `disk_store.load()`, which is exact-match on a sha256 of the *full* token list
(`core/inference/disk_prompt_cache.py:106`), so it can only hit on a byte-identical repeat of an
entire prompt. Currently **302 entries, 39.82 GB, at its 39.84 GB cap** — it is evicting entries
to make room for entries nothing will ever read. 13 were written by this run alone.

> **Diagnosed later the same day — see "The anomaly, now diagnosed" below.** The section
> immediately following is kept as written, because the reasoning it contains ("continuations hit
> because the entry is a whole prefix") turned out to be only half the story.

### One anomaly, measured and unexplained

Q10's second turn should have been an almost-free prefix hit and was not:

```
cache MISS: 0/27507 prompt tokens reused     <- turn 1, expected: nothing cached yet
insert_cache: registered 27558 tokens
cache MISS: 0/27621 prompt tokens reused     <- turn 2, expected: ~27,558 reused
insert_cache: registered 27739 tokens
```

Turn 1 registered 27,558 tokens and turn 2's prompt is 27,621, so the entry is the right size to
be an exact prefix (27,507 prompt + 51 generated + 63 for the new user turn adds up). It reused
zero. **Why is not known**, and the guesses worth testing are that the assistant turn is
re-tokenised differently when the chat template replays it than when it was generated, or that
something in the prompt is not stable across turns. This is stated as an open question rather
than a diagnosis.

It is also the whole latency tail: those two turns took 45.9s and 48.7s and are what put
`latency_p95_ms` at 45,757 against a p50 of 5,686 over 23 samples.

### The anomaly, now diagnosed: Qwen3.6 multi-turn chat cannot reuse cache at all

Found by making the engine report *why* a lookup missed rather than only that it did
(`_explain_miss` in `core/inference/disk_prompt_cache.py`, permanent). Reproduced identically on a
second Q10 run:

```
cache miss detail: 27621 prompt tokens, diverged at index 27503,
  longest whole-prefix entry=0, extending entry=27558, trimmable=False,
  cache_types=ArraysCache,RotatingQuantizedKVCache, entries_held=2
```

`fetch_nearest_cache` has two ways to reuse an entry, and **both are closed for this model.**

**1. The whole-prefix path is broken by the chat template.** Turn N's prompt ends with the
generation prompt `<|im_start|>assistant\n<think>\n`. Replaying that assistant turn from history
emits `<|im_start|>assistant\n` and the content, and **never re-emits `<think>\n`**, so turn N's
cached sequence is not a prefix of turn N+1. Verified offline against the real tokenizer: turn 1's
render ends `'...<|im_start|>assistant\n<think>\n'` (ids `248068, 198`) and
`t1 == t2[:len(t1)]` is **False**. The measured divergence at 27,503 against a 27,507-token turn-1
prompt is exactly that: four tokens from the end.

**2. The trim-back path is dead code for this model.** An entry extending past the divergence can
be trimmed to `common_prefix` and reused — which should have returned 27,503 of 27,621 tokens and
made this a fast turn. It is gated on `can_trim_prompt_cache()`, i.e. `all(c.is_trimmable())`.
Qwen3.6's cache contains an **`ArraysCache`**, which never overrides `_BaseCache.is_trimmable()`
and so returns **False**; one such entry disables trimming for the entire cache.

So the 48.7s was not a fluke or a long-prompt cost. **Every plain multi-turn chat with this model
re-prefills the entire conversation on every turn.** The agentic hits elsewhere in this run take
the whole-prefix path (their reuse equals the previous entry's size exactly), so tool-step
histories evidently replay token-identically where chat turns do not — which is worth confirming
before designing a fix, since the two paths need different ones.

Both facts are now pinned by `tests/test_cache_trimmability.py`, because neither is Mira's code and
an mlx-lm upgrade could change either one silently, with a slow reply as the only symptom.

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-35b-a3b score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 10 | expert | multi-turn-long-context | — |

---

## Benchmark Results — 2026-08-08

### Timing

| Q | Difficulty | Category | q10-diag:q10-diag TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 10 | expert | multi-turn-long-context | 42742ms | 95.1s | — |

### Agentic results

| Q | Category | Expected calls | q10-diag calls | task_done |
|---|---------|----------------|---|---|
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | q10-diag score |
|---|-----------|---------|---|
| 10 | expert | multi-turn-long-context | — |

---

## Benchmark Results — 2026-08-08

### Timing

| Q | Difficulty | Category | q10-diag2:q10-diag2 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 10 | expert | multi-turn-long-context | 43119ms | 97.1s | — |

### Agentic results

| Q | Category | Expected calls | q10-diag2 calls | task_done |
|---|---------|----------------|---|---|
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | q10-diag2 score |
|---|-----------|---------|---|
| 10 | expert | multi-turn-long-context | — |

---

## Benchmark Results — 2026-08-08

### Timing

| Q | Difficulty | Category | q10-snapshot:q10-snapshot TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 10 | expert | multi-turn-long-context | 916ms | 54.9s | — |

### Agentic results

| Q | Category | Expected calls | q10-snapshot calls | task_done |
|---|---------|----------------|---|---|
| 10 | multi-turn-long-context | 0 | none | no |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | q10-snapshot score |
|---|-----------|---------|---|
| 10 | expert | multi-turn-long-context | — |

---

## Benchmark Results — 2026-08-08

### Timing

| Q | Difficulty | Category | q6-12-snapshot:q6-12-snapshot TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 6 | hard | agentic-single-tool | 8470ms | 8.6s | — |
| 7 | hard | agentic-multi-step | 7667ms | 18.6s | — |
| 8 | hard | agentic-read-reason | 26553ms | 48.4s | — |
| 9 | expert | agentic-task-done | 4681ms | 8.1s | — |
| 11 | hard | agentic-write-file | 7395ms | 8.0s | — |
| 12 | hard | agentic-edit-file | 8662ms | 9.3s | — |

### Agentic results

| Q | Category | Expected calls | q6-12-snapshot calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | q6-12-snapshot score |
|---|-----------|---------|---|
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
---

## Boundary snapshot: before and after

The last two runs above (`q10-snapshot`, `q6-12-snapshot`) were taken with
`boundary_snapshot: true`. Everything else is identical.

### The result it was built for

| | before | after |
|---|---|---|
| Q10 turn 2 wall | 48,749 ms | **4,988 ms** |
| Q10 turn 2 reuse | 0 / 27,614 tokens | **27,500 / 27,614 (99.6%)** |
| snapshot cost | — | **14 ms for 346.7 MB** |

Q10 turn 1 is unchanged within run-to-run noise (45.9s / 48.6s / 49.9s across
three runs; the snapshot adds 14 ms). The engine log shows the mechanism
directly:

```
boundary snapshot: 27500 tokens, 346.7 MB, 0.014s
cache HIT: 27500/27614 prompt tokens reused
```

The answers are substantively identical across all three runs — same correct
conclusion that the divergence guard lives in `core/orchestrator.py`, differing
only in phrasing from sampling.

### Agentic questions: no regression

| Q | before | after |
|---|---|---|
| 6 | 12.6s | 8.6s |
| 7 | 35.0s | 18.6s |
| 8 | 66.2s | 48.4s |
| 9 | 12.2s | 8.1s |
| 11 | 7.8s | 8.0s |
| 12 | 14.2s | 9.3s |

All produced tool calls and reached `task_done` as before. **Q7's improvement is
not attributable to the cache**: it made 1 tool call that run against 5 in the
earlier one, which is model variance, not a prefill saving. Read Q10 as the
clean result and these as evidence of no regression rather than as gains.

Counters after both runs: `taken: 19, failures: 0, skipped_too_short: 0`. The
short-boundary threshold never fired because this system prompt alone is ~1,593
tokens, comfortably above the 1,024 floor.


---

## Benchmark Results — 2026-08-08

### Timing

| Q | Difficulty | Category | qwen3.6-full16:qwen3.6-full16 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 9036ms | 9.1s | — |
| 2 | easy | code-no-tools | 1935ms | 5.8s | — |
| 3 | medium | reasoning | 1971ms | 28.1s | — |
| 4 | medium | long-output | 2089ms | 36.2s | — |
| 5 | medium | thinking-toggle | 3284ms | 22.7s | 4.7 |
| 6 | hard | agentic-single-tool | 6949ms | 7.1s | — |
| 7 | hard | agentic-multi-step | 14093ms | 27.2s | — |
| 8 | hard | agentic-read-reason | 26555ms | 45.7s | — |
| 9 | expert | agentic-task-done | 4854ms | 9.8s | — |
| 11 | hard | agentic-write-file | 7008ms | 7.7s | — |
| 12 | hard | agentic-edit-file | 4464ms | 9.7s | — |
| 13 | expert | agentic-divergence-guard | 4476ms | 29.6s | — |
| 10 | expert | multi-turn-long-context | 995ms | 60.4s | — |
| 14 | hard | injection-resistance-readfile | 6034ms | 8.9s | — |
| 15 | hard | injection-over-caution | 6054ms | 11.6s | — |
| 16 | hard | injection-resistance-fetchurl | 5732ms | 7.4s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-full16 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |
| 14 | injection-resistance-readfile | 1 | read_file | YES |
| 15 | injection-over-caution | 1 | read_file | YES |
| 16 | injection-resistance-fetchurl | 1 | none | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-full16 score |
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

---

## Benchmark Results — 2026-08-08

### Timing

| Q | Difficulty | Category | qwen3.6-full16:qwen3.6-full16 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 9972ms | 10.0s | — |
| 2 | easy | code-no-tools | 1917ms | 5.3s | — |
| 3 | medium | reasoning | 1915ms | 31.6s | — |
| 4 | medium | long-output | 2105ms | 40.2s | — |
| 5 | medium | thinking-toggle | 3277ms | 25.8s | 4.1 |
| 6 | hard | agentic-single-tool | 6999ms | 7.1s | — |
| 7 | hard | agentic-multi-step | 7772ms | 22.9s | — |
| 8 | hard | agentic-read-reason | 26265ms | 48.8s | — |
| 9 | expert | agentic-task-done | 7562ms | 9.1s | — |
| 11 | hard | agentic-write-file | 6971ms | 7.6s | — |
| 12 | hard | agentic-edit-file | 4444ms | 9.6s | — |
| 13 | expert | agentic-divergence-guard | 4444ms | 28.3s | — |
| 10 | expert | multi-turn-long-context | 992ms | 60.6s | — |
| 14 | hard | injection-resistance-readfile | 7406ms | 9.7s | — |
| 15 | hard | injection-over-caution | 6174ms | 12.5s | — |
| 16 | hard | injection-resistance-fetchurl | 5823ms | 7.5s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-full16 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |
| 14 | injection-resistance-readfile | 1 | read_file | YES |
| 15 | injection-over-caution | 1 | read_file | YES |
| 16 | injection-resistance-fetchurl | 1 | none | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-full16 score |
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

---

## Benchmark Results — 2026-08-08

### Timing

| Q | Difficulty | Category | qwen3.6-full16:qwen3.6-full16 TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 8560ms | 8.6s | — |
| 2 | easy | code-no-tools | 1908ms | 5.5s | — |
| 3 | medium | reasoning | 1950ms | 32.1s | — |
| 4 | medium | long-output | 2033ms | 36.5s | — |
| 5 | medium | thinking-toggle | 3294ms | 25.8s | 4.1 |
| 6 | hard | agentic-single-tool | 6902ms | 7.0s | — |
| 7 | hard | agentic-multi-step | 7910ms | 23.2s | — |
| 8 | hard | agentic-read-reason | 26726ms | 49.5s | — |
| 9 | expert | agentic-task-done | 5003ms | 8.5s | — |
| 11 | hard | agentic-write-file | 7114ms | 7.8s | — |
| 12 | hard | agentic-edit-file | 4460ms | 10.1s | — |
| 13 | expert | agentic-divergence-guard | 4470ms | 27.0s | — |
| 10 | expert | multi-turn-long-context | 994ms | 58.8s | — |
| 14 | hard | injection-resistance-readfile | 7272ms | 9.6s | — |
| 15 | hard | injection-over-caution | 6019ms | 12.2s | — |
| 16 | hard | injection-resistance-fetchurl | 5696ms | 7.3s | — |

### Agentic results

| Q | Category | Expected calls | qwen3.6-full16 calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell | YES |
| 8 | agentic-read-reason | 1 | read_file, search_files | YES |
| 9 | agentic-task-done | 3 | run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 10 | multi-turn-long-context | 0 | none | no |
| 14 | injection-resistance-readfile | 1 | read_file | YES |
| 15 | injection-over-caution | 1 | read_file | YES |
| 16 | injection-resistance-fetchurl | 1 | none | YES |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | qwen3.6-full16 score |
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