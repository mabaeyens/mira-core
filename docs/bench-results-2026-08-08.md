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