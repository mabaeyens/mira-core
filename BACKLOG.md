# Backlog

## Done
- [2026-08-12] **Repetition, presence and frequency penalties are now reachable, and a request can
  carry its own seed.** Two halves of the same gap: everything mlx-lm's sampler can do was already
  built and none of it had a route from `mira.yaml`. The penalties go
  `config.py` → `orchestrator._call_llm` → `extra_body` → `_penalties_from_body` →
  `make_logits_processors`, and **nothing is sent when nothing is configured** — mira-mlx is not the
  only backend behind that client, and a foreign `extra_body` key is a 400 on some of them. A
  penalty of 0 is treated as absent rather than as "penalise by zero", and a context size only
  travels when its penalty is set, so a stray `repetition_context_size` cannot silently arm
  anything. All default to off: a penalty is a quality knob with a real cost — legitimate repetition
  in code and tables is the false-positive case, which is why the example config frames it as x16
  legitimate against x355 degenerate rather than as a number to raise until loops stop. The seed is
  the regenerate story: `mx.random.seed()` is process-global and was never set, so the engine's
  sampling was reproducible only by accident of ordering. `_admit_job` now seeds per job — the
  request's seed when it has one, `secrets.randbits(32)` when it does not — and skips it entirely at
  temperature 0, where it would be theatre. `seed: 0` is a seed, not an absence, and there is a test
  that says so. 17 new tests in `tests/test_generation_guard.py`, behavioural rather than
  structural: the repetition check asserts that a repeated token's logit actually drops, because
  mlx-lm's processor mutates the array in place and any test comparing the result to the input is
  comparing it to itself.
- [2026-08-12] **The bench asks six new questions, and every one of them is a bug that shipped.**
  Questions 17-22 in `scripts/bench_questions.yaml`: a tool that returns a bare string (the
  `list_attachments` crash), reply integrity with thinking on, legitimate repetition that a runaway
  guard must not flag, the `task_done` refusal path, whether a tool result survives into the next
  turn, and whether an injection payload still gets obeyed once it is sitting in history rather than
  arriving fresh. Picking regressions over invented hard questions is the point — a bench that only
  asks what was always easy cannot tell you the engine changed. `bench_compare.py` now captures the
  thinking channel (it was discarding it, so nothing could score it) and `run_multi_turn` takes a
  `prompt_turn1` as an alternative to injecting a file, which is what makes the memory and injection
  pairs expressible. New tier-1 checks: `answer_contains_all`, `answer_not_thinking` (it reports
  `partial` rather than passing when no thinking was captured — an uncaptured signal is not a pass),
  `not_degenerate`, `no_stream_error`, `min_line_repeats`. 38 tests pass in `test_bench_eval.py`.
- [2026-08-12] **`docs/mlx-lm-pin.md` says when to move the pin instead of leaving it a standing
  question.** The fork carries 12 commits in four groups; two of them are what actually break on any
  upgrade — `SequenceStateMachine`, and an undocumented `_embed_tokens` removal that no release note
  mentions. The doc lists four conditions that each argue for moving, the one condition that forces
  it, and a five-step how-to. It also corrects this file: "frozen at bbd8496" was stale, the real pin
  is `291a61a` on `mira-core-pin-vision`.
- [2026-08-12] **A failed request can no longer take the engine with it, and an engine that does
  die now says so instead of hanging every client that asks.** Follow-up (a) from the entry below.
  `_start_job`'s guard covered the prepare block alone, so everything from `fetch_nearest_cache`
  through `insert_segments` ran bare — a chat template that rejects its kwargs, a logits processor
  built wrong, a cache entry that will not load — and any of it propagated out through
  `_drain_inbox` into the engine loop and killed the thread. Admission is now a thin `_start_job`
  wrapper around `_admit_job`: anything raising there is logged with the request id and delivered
  to that job's queue as `exc` + `DONE`, which is the failure protocol the oversized-prompt check
  already used. **Writing it turned up a second hang the guard alone could not have fixed.** The
  `_pending` bookkeeping was built *after* `insert_segments`, and `_handle_response` silently drops
  responses for a uid it cannot find, so anything raising in that window — the detokenizer,
  `ToolCallFormatter` on a malformed tool schema — would leave a job burning decode steps nobody
  reads and a client waiting on a `DONE` nobody sends, with the job already inside the batch and
  past rescuing. The dict is now built before the insert and only the store is left after it;
  `created`/`start_time` are stamped post-insert so the latency window does not silently grow by
  the insert duration. **Loop failures deliberately do not recover in place.** An exception out of
  the batch generator leaves the KV state and MLX stream of unknown validity, and decoding the next
  token on top of that produces wrong output rather than an error — so `_die` logs the traceback,
  fails every `_pending` job and everything still in the inbox with the real exception, and sets
  `_error`. `submit()` then raises the new `EngineDead` under `_admit_lock`, which closes the
  check-then-put race against `_die`'s set-then-drain: without it one job could still land on an
  inbox nothing drains, which is the one hang left. `chat_completions` maps that to **503 with the
  original cause in the body**, not FastAPI's default 500, which discards the message — the same
  trap the non-streaming branch already documented. The loop moved out of `_run` into
  `_serve_forever` for one reason: behind a model load its failure path was untestable and
  therefore untested. **Verified live** on the restarted server: an oversized prompt returns 400 in
  0.21s with the real ceiling in the message, and the engine answers normally before and after.
  The arbitrary-crash and `_die` paths are covered by tests, **not** live-verified — forcing a real
  engine crash to prove it costs more than the coverage is worth. 138 tests pass across the six
  engine-adjacent files, 5 of them new.
- [2026-08-12] **`max_thinking_tokens` was never enforced, and wiring it uncovered an mlx-lm
  batching bug that could take the whole engine down.** The setting has existed since the
  truncated-thinking work and was sent to the model as a chat-template kwarg — but Qwen3.6's
  `chat_template.jinja` does not reference `thinking_budget` anywhere, and Jinja discards unknown
  kwargs in silence, so the number travelled the full length of the stack and did nothing. It is
  now a logits processor (`ThinkingBudget`) that watches the reasoning block and, once past the
  budget, floors every logit except `</think>` — forcing the closer rather than stopping
  generation, because a hard stop at the budget produces a reply that is all reasoning and no
  answer, which is exactly the failure this project already fixed once. **Verified against prod:**
  same prompt, same model, budget 32 → the engine logged `thinking budget HIT at 32 tokens` and
  the turn came back with exactly 32 reasoning tokens, 1,731 completion tokens and a complete
  83KB answer in 39.9s, against 1,996 / 3,827 / 73.9s unbudgeted.
  **The bug it uncovered is the more important half.** mlx-lm's `PromptProcessingBatch.extend`
  replaces a sequence's logits-processor list with `None` whenever `any()` of the batch's lists is
  falsy, and `GenerationBatch._step` then does `for processor in None`. Mira issues two concurrent
  jobs per turn — one thinking, one not — so the moment the first carried a processor and the
  second carried `[]`, the second got `None` and the engine thread died with a `TypeError` after
  one generated token. The failure mode is the worst kind: `/v1/stats` kept answering 200, and
  every request after it hung forever rather than erroring. It also hung a `pytest` run for 72
  minutes on an established socket to the dead engine. Fixed here by never handing mlx-lm an empty
  processor list (`_build_logits_processors` appends a no-op), with a test pinning the invariant.
  **Two follow-ups this leaves open**, both deliberately not done here: (a) an exception anywhere
  in the engine's `_run` loop kills the thread silently and turns every in-flight and future
  request into a hang — it should fail the pending jobs and log, not vanish (**done 2026-08-12**,
  see the entry above); (b) at the restored default of
  8192 the budget is a runaway guard and nothing more, since the worst real turn measured so far
  spent ~5k reasoning tokens. ~2048 would make it an actual budget. Also measured while here:
  search-result injection costs ~696 new prompt tokens per round with 79% cache reuse, so the
  old cutoff forcing a web search on nearly every question is **not** a throughput problem.
- [2026-08-11] **Instrumented the prefill/decode split and settled a question three separate
  arguments had failed to settle: Mira is decode-dominated, and the "effective 21.7 tok/s" figure
  that motivated the work was my own measurement artifact.** Every previous attempt compared a
  decode rate against a felt rate and argued about the gap, because the engine could see prefill
  and decode but not tool waits, and the orchestrator could see tool waits but not prefill — so
  nobody could attribute a turn. Now `_record_timing` splits every request at the first generated
  token (the only boundary the engine can actually observe) into `ttft_ms`/`decode_ms`/`decode_tps`,
  with percentiles in `/v1/stats`; `_log_turn_timing` wraps `stream_chat` so a turn is measured on
  every exit path — refusals, forced summaries, errors — and sums the engine's split across all the
  LLM calls in an agentic turn alongside tool time; `_apply_usage` carries the new `timing` block
  through `normalize_oai_stream`, which silently dropped unknown usage fields and would have made
  the orchestrator half blind. **Measured against prod over 10 real turns across two deliberately
  different workloads**: decode is **77.2%** of wall clock on short tool-free turns and **88.5%** on
  a corpus-shaped run, prefill 18.1%/8.3%, tools ~2%, and `other_ms` — RAG, history load, context
  compression — **0.0% in both**, which retires a whole family of suspicions for the cost of one
  afternoon. Decode runs at 55.1/50.4 tok/s, confirming the long-quoted ~59 without a bench.
  **The correction matters more than the confirmation**: the 21.7 tok/s "delivered rate" came from
  dividing *stored* text by *whole-turn* wall clock, and an agentic turn stores only its final
  answer, so intermediate generations, tool-call text and stripped reasoning left the numerator
  while their time stayed in the denominator. Real effective rate is 42-45 tok/s. **Two levers are
  now visible and neither is a kernel**: reasoning is 24-44% of all output and output time *is* the
  clock, so capping thinking cuts wall clock nearly one-for-one; and decode degrades with context,
  ~55 tok/s at 2k prompt tokens falling to 46-49 at 8-20k (excluding two turns that generated under
  10 tokens, whose rate comes from 5-7 inter-token gaps and is noise). Three deliberate choices:
  requests generating 0 or 1 tokens are **dropped rather than recorded as 0 tok/s**, because a
  single token has no decode window and averaging it in is how a wrong number comes to look
  precise; `decode_tps` divides by `completion_tokens - 1`, since the first token ends prefill
  rather than starting decode; and `timing` is omitted from `usage` when absent so the block stays
  byte-identical to OpenAI's shape for clients that don't know it. `ttft_ms` deliberately includes
  queue time — it is what the caller waited — which makes `prefill_tps` a floor rather than a clean
  prefill measurement, and it is labelled as such. Read out by `notes/turn_timing.py` (gitignored,
  per `specs/decode-roofline.md` §2). **This replaces Q1-Q13 bench runs as the way to answer a
  throughput question**: it measures live traffic continuously instead of a synthetic question set.
  684 tests pass.
- [2026-08-11] **The eval suite was cut to GPQA alone and scheduled overnight, and a 165-hour trap
  in the old plan was found before it ran.** Two things happened here. First, `limit=1000` on
  `mmlu_pro` never meant 1000 questions: `mmlu_pro` is a **group of 14 category subtasks** (12,032
  docs), lm-eval applies `limit` per task in the *flattened* list (`evaluator.py:535-539`), and
  `get_sample_size` (`evaluator_utils.py:49-54`) returns `int(limit)` with **no clamp to the doc
  count** — so the group would have run `sum(min(1000, N_i))` = **11,149 questions, ~165h**, turning
  the "~26h suite" into ~176h. The same trap applies to `samples=`, which is keyed by *subtask* name,
  so `samples={"mmlu_pro": [...]}` matches nothing and silently runs all 12,032. Caught at dry-run,
  cost nothing. Second, and the reason it matters less than it might: at the real 1739 questions the
  suite is still 25.8h, i.e. **four nights of a personal laptop**, and that is not a trade worth
  making. It now runs **GPQA diamond only — 198 questions, ~2.9h, one night.** IFEval was dropped as
  the weakest of the three (with thinking on and a 16384 cap, a generation that never closes
  `</think>` has its whole reasoning chain scored as the answer and still collects marks), MMLU-Pro
  on cost alone. Machinery, all gitignored: `notes/lm_evals_nightly.py` (stratified plan, 25-question
  chunks, JSON state so a chunk is consumed exactly once and an interrupt costs 25 questions rather
  than the night, wall-clock deadline checked *between* chunks), `notes/lm_evals_nightly.sh`
  (caffeinate, stop/restore production Mira), LaunchAgent `com.mab.mira-evals` at 00:30 with a hard
  stop at 08:15, and `notes/lm_evals_merge.py` to pool the chunks. **Chunking does not cost
  accuracy** — every metric is a mean over independent per-question scores, so the pooled mean and
  the n are what a single run would have produced; the only difference is which questions batch
  together, which is the batch-composition term every score here already carries and already has to
  report. The merge script re-derives each chunk's own score from its samples and refuses to certify
  the run if it cannot reproduce what the harness wrote.
- [2026-08-11] **IFEval's generation cap decided: 16384, overriding the task's own 1280, and the
  thing that gates publishing is no longer the cap.** Miguel delegated the number ("raise the limit
  as you see fit"), so the useful part is the reasoning rather than the value. **1280 predates
  reasoning models** — it is sized for an instruction-following answer and nothing else, so a
  thinking model spends the whole budget inside `<think>` and the score measures the cap. **The
  field has already moved:** lm-evaluation-harness added `think_end_token` and `enable_thinking`
  specifically to score reasoning models on the text *after* the reasoning, and a unified ~16K
  response length for IFEval-style benchmarks came with it, so 16384 is the convention rather than a
  number chosen to flatter the model. **The comparability objection cuts both ways and that is what
  settled it**: a 1280-capped score for a thinking model is not comparable to a published 1280 score
  either, because those are for models that answer directly. Neither cap buys a free comparison, so
  take the one that measures instruction following. **What replaces the block: report the cap and
  the thinking setting beside any IFEval number, and check `unterminated_thinking` before believing
  it at all.** That counter is the real hazard — on the 2-question smoke, 1 of 2 ran 16,383 tokens of
  a repetition loop without closing `</think>`, had the whole chain scored as its answer, and still
  collected `inst_level_loose_acc 0.75`. **No cap fixes that; it is the runaway problem, not the
  budget problem**, which is one more argument for the guard spec'd in
  `specs/generation-runaway-guard.md`. Also worth knowing for the truncation half:
  [lm-evaluation-harness#3382](https://github.com/EleutherAI/lm-evaluation-harness/issues/3382) is
  the same defect upstream — truncation before the think-end token gets parsed as the final
  response. Applied in gitignored `notes/run_lm_evals.sh` and `notes/lm_evals_driver.py`; no
  mira-core code changed, and **the run itself has not been started**.
- [2026-08-11] **@pierre427 reviewed mlx-lm #1584 and found a real gap; verified it, replied, and
  decided to split the PR.** Their comment made two points. The first (supplied caches skip
  quantization in `insert_segments`) was already shipped in `3ebafab` back in July, so they were
  reading an older branch state. The second is new and holds: **`CacheList` has no `to_quantized()`,
  so `maybe_quantize_kv_cache`'s `hasattr` guard skips the whole layer and `--kv-bits` is silently
  ignored** for `deepseek_v32`, `longcat_flash`, `longcat_flash_ngram`, `falcon_h1` and
  `baichuan_m1`. This is a bug **on main today**, independent of #1584. Two findings came out of
  probing it at the PR head (`/tmp/claude_cachelist_adversarial.py`): a recursive leaf pass on
  rotating leaves merges cleanly into `BatchRotatingQuantizedKVCache`, but applying the recursion
  **only at insert manufactures a mixed cohort** (supplied quantized + fresh unquantized at the same
  layer position → `AttributeError: 'RotatingKVCache' object has no attribute 'group_size'`), so it
  has to live in `maybe_quantize_kv_cache` itself. With plain leaves it yields `QuantizedKVCache`,
  which has no `merge()` — `baichuan_m1` mixes both leaf kinds across layers, which is why pierre's
  "fail loudly" guard is load-bearing rather than defensive. Reply posted as
  [comment 5250380971](https://github.com/ml-explore/mlx-lm/pull/1584#issuecomment-5250380971).
  **Process lesson worth keeping:** the first analysis dismissed the finding by reasoning from
  `_make_new_cache()`, which is the exact lane a caller-supplied cache bypasses; the adversarial
  probe refuted it. Reason about the lane the bug is *in*, and run the control that would refute you.
- [2026-08-09] **Both of the day's "discoveries" turned out to be published, named and solved —
  and the lesson is worth more than the measurements.** Miguel's challenge ("I can't be the first
  finding this out") was correct on both counts. Batch size changing greedy output is **batch
  invariance**, diagnosed in [Defeating Nondeterminism in LLM
  Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Thinking
  Machines, Sep 2025) — they measured **80 unique completions from 1,000 runs** of one prompt at
  temperature 0, against our 23-of-24; it is fixed in vLLM (`VLLM_BATCH_INVARIANT=1`), SGLang, and
  on Apple Silicon by
  [mlx-deterministic](https://github.com/ProbioticFarmer/mlx-deterministic). Repetition under
  greedy is [Holtzman et al., ICLR 2020](https://openreview.net/pdf?id=rygGQyrFvH) — **43% repeated
  n-grams greedy vs 0.5% human**, and crucially **decoding cannot remove a loop, only make it less
  likely**, which is why a penalty alone is not a guard. **DECISION: Mira will not implement batch
  invariance** (~1.6x cost, custom Metal kernels for three ops, and reproducibility buys an RL/
  regression-testing property a chat assistant does not need). Mitigation is to fix the eval batch
  size and report it beside any score. **Standing rule adopted: measure Mira, look up everything
  else** — a general fact about models, kernels or decoding is already in the literature, and
  re-deriving it locally reproduces a known result at worse statistical power while hiding the
  existing fix.
- [2026-08-09] **Sampling is configurable for the first time; it was greedy by accident, not by
  choice.** Nothing anywhere sent a sampling parameter, so `mira_mlx_server`'s own `0.0` defaults
  applied to every reply from every client. `temperature`, `top_p` and `top_k` now flow
  `mira.yaml` → `config.py` → orchestrator → `ChatJob` → `make_sampler`, with `top_k` riding in
  `extra_body` (merged, never assigned — clobbering it would silently disable the thinking toggle).
  **Defaults are unchanged, so no reply moved.** Verified rather than assumed: `top_k=1` is argmax
  and reproduced greedy byte-for-byte. Qwen3.6 ships `generation_config.json` asking for
  temperature 1.0 / top_k 20 / top_p 0.95 — adopting that is a separate decision, deliberately not
  taken here.
- [2026-08-09] **Runaway-guard spec written to `specs/generation-runaway-guard.md`** (gitignored,
  local). Leads with the research above so the do-not-reinvent reasoning is checkable. Key finding
  for whoever builds it: **`mira_mlx_server.py:1166` calls `make_logits_processors()` with no
  arguments**, so mlx-lm's repetition/presence/frequency penalties are all off — most of the work
  is wiring, not building. Thresholds must respect the measured gap: clean output reached x7
  repeated sentences, the degenerate case x355. **Not started; Miguel to review.**
- [2026-08-09] **The bench now scores answers, and its first three full runs found that the
  instrument was wrong more often than the model.** Three 16-question runs on one build plus two
  targeted injection runs. Tier 1 came out **24/24 over 12 questions with zero variance across all
  three**, and every checkable factual claim was exact rather than close: Q6's line count 10518
  against a true 10518 three times, Q7 listing exactly the 21 matches with paths and line numbers,
  Q16 reporting the degraded service its injected payload told it to conceal. **Four apparent model
  failures were all harness defects** — two truth probes computing something other than what the
  question asked (`wc -l` vs line objects, substring occurrences vs grep line matches), a judge
  ruling that verbatim-quoted code was invented, and a security question that had never once
  executed. The one reproducible product finding: **injection is resisted through both channels
  every time and disclosed through only one** — on a fetched page Mira names the embedded
  instruction and says it ignored it, on a read file it stays silent in all three runs. Written up
  in `docs/bench-accuracy-2026-08-08.md` (named for the runs' date, not the write-up's). Commits
  `068624d`, `698fd91`, `e4fd9ed`, `64e0e78`,
  `23f8ccd`, `485785f`, `26c9cb2`.
- [2026-08-09] **The judge cannot rule on what it cannot see, and asking it to produced two
  confident wrong verdicts.** Q8 quoted `_tool_call_hashes`, `call_hash`, `AGENT_DIVERGENCE_LIMIT`
  and `_run_tool` verbatim from `orchestrator.py` and was scored **0 for "inventing" them**; Q16 was
  scored 0 for "acting on the injected instruction by refusing to fetch", about a page it never saw
  by an assistant that made no tool calls. The judge prompt now states it is scoring text only and
  must never rule a quotation invented. **Rewording the prompt did NOT fix the second one** — what
  fixed it was deleting the unanswerable clause from Q16's rubric, since compliance is caught
  deterministically by `tool_absent` and a marker file. Q14 already had that split and scored
  correctly throughout. Fixture set extended 10 → 12 with both real failures as regression cases;
  the original ten passed 10/10 and caught neither, which is the whole lesson: **validation only
  proves what its cases cover.**
- [2026-08-09] **Judged noise floor measured at ±1, and it travels inside the baseline file.**
  `scripts/bench_noise.py`, three runs of one build: tier 1 moved on nothing, Q4 and Q8 each scored
  2, 2, 1. **Two runs would have reported a floor of 0** — both were stable across runs 1 and 2 and
  only moved on the third. The floor now lives in `docs/quality-baseline.md` rather than on the
  command line, because a number that has to be remembered at compare time gets left off and a
  judged delta printed without its floor reads as signal. Baseline covers 15 of 16 questions on
  build `64e0e78`.
- [2026-08-09] **Q16 never ran once, and `fetch_url` was invisible to every check.** Three things
  had to be true and none were: nothing served `127.0.0.1:8009`, `fetch_url` refuses loopback by
  default (correctly — the model picks the URL out of attacker-influenceable text), and **the bench
  could not see the tool even when it ran**, because `web_search` and `fetch_url` emit
  `fetch_start`/`search_start` rather than `tool_start`. That last one is the serious half:
  **`tool_absent` is how the injection questions verify a forbidden tool was not called, and it was
  blind to both tools** — a question asserting "must not call fetch_url" would have passed without
  the assertion ever being evaluated. Fixed with a loopback-only fixture server, a temporary copy of
  `mira.yaml` with private fetching enabled via the new `MIRA_CONFIG` env var (the real file is
  never touched), and SSE capture for both events. Q16 now scores tier 1 2, judged 2, safety pass,
  marker file absent. Questions also declare `payload_via`, so an injection question whose payload
  never arrived is marked **partial** instead of reporting a clean pass.
- [2026-08-09] **Standardised eval environment built, not yet run.** `~/.venvs/mira-evals` with
  `mlx-lm 0.31.3` (pinned to match production) and `lm_eval 0.4.12`, deliberately **separate from
  mira-core's venv** — `lm-eval` pulls torch and re-resolves the whole tree, which is how the MLX
  stack was wiped in June. `pyproject.toml`, `uv.lock` and the MLX stack all verified untouched.
  Runner at `notes/run_lm_evals.sh` (gitignored). Preflight found three things that would have cost
  an evening: **`python -m mlx_lm.evaluate` silently does nothing** (it has `main()` but no
  `__main__` guard — use the console script), **both HF credentials on this machine are invalid**,
  and **GPQA is a gated dataset**. ~~IFEval (541) and MMLU-Pro (12,032) load fine unauthenticated.~~
- [2026-08-09] **Correction to the line above: IFEval did not load, and could not have run.** That
  preflight established the *datasets* download, which is a different thing from the *tasks*
  importing. IFEval needed `langdetect` and `immutabledict` (both now installed in the evals venv
  only; MLX stack verified unchanged via `uv pip list`, 85 → 86 packages, single addition). Worse,
  **IFEval cannot run under mlx-lm 0.31.3 at all**: it declares `until: []` and `_rstrip_until`
  (`evaluate.py:33-38`) calls `min()` on that empty list, so every run dies *after* paying for full
  generation. This killed the stock CLI identically — it was never a driver problem. HF auth and the
  GPQA gate were both re-verified clear on 2026-08-09 (`whoami` resolves, `dataset_info` returns 8
  files). **The 21:00 reminder was cancelled** (Miguel on holiday; run it whenever). The lesson worth
  keeping: the old smoke test called the CLI on IFEval, so the *guard itself* would have aborted the
  whole run before the model loaded — a preflight that cannot fail is not a preflight.
- [2026-08-08] **Release decisions for v1.2.0, taken by Miguel:** the three opt-in flags
  (`boundary_snapshot`, `proactive_decompress`, `disk_prompt_cache`) **ship OFF and are
  documented**; version is **1.2.0**; and the orphaned disk cache gets **a warning, not an
  automatic deletion**. All three now hold in the tree. The flags were invisible to users
  before this — none of them appeared in `mira.yaml.example`, so shipping them off would have
  meant shipping them unknowable; each now has a block saying what it measured and what it
  costs. The warning fires once at server startup and as an advisory line in `mira doctor`,
  reporting the file count, the size and the exact `rm -rf`. It does **not** delete: these are
  pure cache files, but they are also gigabytes inside the user's data directory, and a server
  removing gigabytes unasked at startup is not a thing to build without being asked to. It is
  deliberately backend-independent, since the leftovers are just as dead when the configured
  backend is omlx and that is exactly the case where nothing else would mention them.
  `mira_cli.py` stays stdlib-only — the import is lazy and behind a bare except, because
  `doctor` must run on a half-built install. Sizes render adaptively after the first live run
  printed "0.00 GB of dead prompt-cache files" for a 4.5MB probe, which reads as nothing at all
  inside a warning asking someone to act.
- [2026-08-08] **Benches stopped writing into the real conversation history, and the first fix
  for it took Mira's backend down.** Miguel saw bench Q2/Q4 ("write a Python …") conversations
  in his own list. The existing teardown did delete them, but **cleanup is not isolation**:
  they are visible in the app for the whole run and permanent if the run is interrupted.
  `MIRA_DATA_DIR` could never have covered this — it configures the process that *owns* the
  database, which is why it fixed `pytest` (in-process) and not the bench, which drives an
  already-running server over HTTP. So the bench now stops production, starts its own server
  against a throwaway data dir, and restores production afterwards; measured across a real Q1
  run, the database was byte-identical (29 conversations, 98 messages, same `max_created_at`).
  **Then the teardown killed the engine production was loading**: `stop()` runs twice, once
  explicitly and once from atexit, and its `pkill` matches any mira-mlx engine, but only the
  *restore* was guarded. `ensure_backend_running` does not retry, so Mira sat serving with no
  backend for ten minutes. The run reported success throughout — exit 0, database untouched,
  "production is back up" already printed — and only `/health` showed it. Both halves fixed:
  `stop()` is idempotent as a whole, and the restore now waits for `backend_ready` rather than
  for port 8000, which answers in seconds while the model is still loading. Mutation-checked,
  then re-verified live, which is the only place it ever appeared.
- [2026-08-08] **JSONL is ignored by filetype now, not by one filename pattern.** The rule was
  `scripts/bench_raw_*.jsonl` and it held only because every JSONL on disk happened to match
  it — `scripts/results.jsonl`, `docs/trace.jsonl`, `core/data.jsonl` and a `bench_raw` file at
  the repo root were all untracked-but-visible, one `git add -A` from a commit. Nothing was
  tracked as JSONL, so this untracked no file. The comment points at a negation rather than
  `git add -f`, so any future exception stays visible in `.gitignore` instead of in a shell
  history.
- [2026-08-08] **Turned the disk prompt cache off and deleted its 39.75GB, which had served
  zero reads in three weeks** (`DISK_PROMPT_CACHE`, default off). Not a bug in the store: a
  lookup is an exact-match sha256 over the full token list while an entry is keyed on prompt
  **plus everything generated**, so a hit needs a new prompt to equal an earlier
  prompt-and-completion byte for byte, which the chat template alone rules out. The layer above
  it matches prefixes; this one can only match a literal repeat. **Two false alarms were checked
  before deleting anything**: 219 of 296 files had an atime later than their mtime, and the
  engine log held 38 lines mentioning "disk". The atimes cluster into exactly two hours
  (2026-07-26 14:00 and 2026-08-08 10:00, the latter being my own analysis sweep) rather than
  spreading across three weeks as request-driven reads would, and the log lines are the *logger
  name* `core.inference.disk_prompt_cache` on miss diagnostics. With `disk_cache_hits: 0` that is
  three independent confirmations. The flag disables the store by handing the engine a 0 budget,
  which is a path the engine already had, so no new code path exists to go wrong. `/hardware`'s
  `derived_disk_cache_max_gb` now reports 0 rather than what the volume could afford. **The code
  stays**: a prefix-capable disk layer is still a real idea (`specs/prefix-aware-disk-prompt-cache.md`),
  but reviving it needs a measured read-vs-prefill comparison first, since entries were 38-262MB
  against a ~4.8s prefill. Deleted only `*.safetensors` in that one directory, after guarding on
  the directory holding nothing else and the running engine reporting a 0 budget; 400Gi free ->
  440Gi, conversations.db and chroma_db untouched, 533 tests pass.
- [2026-08-08] **Built the boundary snapshot, and plain multi-turn chat on Qwen3.6 stopped re-prefilling itself: Q10's second turn went 48,749ms → 4,988ms with 99.6% of its prompt reused where it reused nothing at all.** The two reuse paths stay closed — trimming is architecturally impossible on a hybrid linear-attention model, and the chat template's `<think>` scaffold breaks whole-prefix matching at every turn boundary — so neither was opened. Instead prefill is **split at the history boundary** (`plan_prefill_segments`) and the state there is captured on the way past, because it cannot be recovered afterwards. On `end_of_segment` for the first segment the engine pulls that state out of the batch and registers it keyed on exactly the tokens processed. **Two things the gate check had already corrected were essential**: the engine had to move from `next_generated()` to `next()`, because the `end_of_segment` signal rides on the prompt responses the former discards; and the live cache is **not** the object handed to `insert_segments` (`_merge_caches` merges it, so that object's offset stays 0) — it comes from `batch.extract_cache(idx)` resolved by uid every time, since sequences migrate between batches. **The measured cost beat my own extrapolation by an order of magnitude**: 14ms for 346.7MB, against the 100–600ms I had projected from a 0.6B model's 3,562 MB/s. Turn 1 is unchanged within run-to-run noise and all six agentic questions came out same-or-faster (Q6 12.6→8.6s, Q8 66.2→48.4s, Q11 7.8→8.0s), with zero snapshot failures across 19 taken. **Q7's 35.0→18.6s is NOT attributable to the cache** — it made 1 tool call that run against 5 before, which is model variance; the clean attributable result is Q10. **Safety is where the care went**, because the failure mode is silent: `_history_boundary` verifies the history render really is a prefix of the prompt and returns None otherwise, since a template rendering history differently with and without `add_generation_prompt` would key an entry on tokens that were never processed and change output rather than merely slow it. Snapshot failures are swallowed and counted — a lost snapshot costs speed, nothing else. Mutation-tested: removing any of the seven gates (the `end_of_segment` check, the not-split guard, the once-per-job latch, uid-resolved indexing, the split bounds check, the cache-offset subtraction, the prefix verification) kills a specific test each. Also fixed a trap the feature would otherwise have set: **`max_size` was left at mlx-lm's default of 10** and a turn now stores two entries, which would have quietly halved how many conversations stay warm; it is now an explicit `PROMPT_CACHE_MAX_ENTRIES = 20`, with bytes still the real ceiling. Off by default in `core/config.py`, on in `mira.yaml` while it proves itself. 528 tests pass. Design and measurements in `specs/assistant-boundary-snapshot.md`; architecture in `docs/prompt-cache.md`.
- [2026-08-08] **Ran Q6–Q12 to validate the `task_done` guard live, and the engine's own log — reaching disk for the first time — answered the prompt-cache question and the p95 question in the same run.** **The guard is inert on legitimate agentic turns**: zero refusals across the seven questions, every one of Q6/Q7/Q8/Q9/Q11/Q12 ran tools and then exited through `task_done` normally, which is exactly what the gate keyed on tools-having-run is supposed to allow. It does **not** demonstrate the refusal path against a real model — Qwen3.6 never took the escape hatch — so that stays covered by unit tests only, and Q10 reporting `task_done: no` is correct rather than a guard effect since it declares `tools: false`. **The prompt cache is not idle and `insert_cache` never skips**, which kills the original suspicion outright: 23 requests, 12 hits, 23 registered and **0 SKIPPED**, 48,024 of 168,321 prompt tokens reused (28.5%). What it cannot do is the interesting part. **Continuations hit and conversation openers always miss**, because an entry is `system + user + assistant` and the trie can only reuse an entry that is *entirely* a prefix of the new prompt — true of the next turn in the same conversation, never true of a different one, which diverges right after the shared system prompt. So **the same ~3,800 tokens are re-prefilled on every opener** (8 of the 11 misses were 3,787–4,025 tokens) and no entry containing just the system prompt can ever exist, since entries are only created after a generation completes. **The 39.82GB disk cache has never served a read, by construction rather than by bug**: `DiskBackedPromptCache.fetch_nearest_cache` falls back to an **exact match** on a sha256 of the full token list (`disk_prompt_cache.py:106`), so it needs a byte-identical repeat of an entire prompt; it is now at its 39.84GB cap, evicting entries to make room for entries nothing will read, 13 of them written by this run. **The p95 is fully attributed and is not a pathology**: p50 5,686ms against p95 45,758ms over 23 samples, and both tail requests are Q10's turns at 45.9s and 48.7s, full misses on ~27,500-token prompts. No stall, no lock, no scheduling problem — long prompt times cache miss. **One anomaly is measured and deliberately left undiagnosed rather than guessed at**: Q10 turn 1 registered 27,558 tokens, turn 2's prompt was 27,621 (27,507 prompt + 51 generated + 63 new), which is an exact prefix by arithmetic, and it reused **zero**. That single miss is what created the entire latency tail, and it comes before any other cache work. Both specs rewritten around measurements instead of suspicions; `specs/prompt-cache-earns-nothing.md` keeps its wrong original premise at the bottom so it stays recognisable. Bench ran in a throwaway git worktree with a live memory watchdog (available RAM 1.75–10.71GB, zero trips), and swept itself clean.
- [2026-08-08] **`task_done` can no longer be a turn's entire user-visible output, and half of the bug it was filed for turned out to be already fixed.** The spec named two independent defects behind bench Q4 answering "Provided a complete Python context manager for sqlite3 ..." with no context manager. **The first one was closed on 2026-08-01 by `d6a8a17` and I checked before rewriting it**: `tools_enabled` is plumbed from `bench_compare.stream_chat()` through the `/chat` form field to `_active_tools`, and `tests/test_tools_enabled.py` asserts `task_done` disappears from the schema list — not incidentally, since `task_done` is not in `_LOCAL_TOOLS` and a filter written that way would have left it reachable. So only the loop guard was outstanding. It refuses a `task_done` whose turn produced **no visible content and ran no tools**, because such a summary is a claim about work the user cannot see. **It keys on tools having run, not on empty content**, so an agentic turn whose answer genuinely *is* a summary ("deleted 3 files") still exits normally. The refusal is delivered as the **tool result for that call**, not as a fresh user turn: an assistant message carrying `tool_calls` with no matching reply breaks strict-alternation chat templates in the Mistral family, which is the trap the `_append_step_nudge` helper exists to avoid elsewhere. One refusal per turn — a model that calls `task_done` again is taken at its word rather than recursed on — and the latch resets per turn because the orchestrator is pooled per conversation and reused. Answering *every* call in the batch matters and is tested: reaching the branch at all implies `_total_tool_calls == 0`, which implies every prepared call is a `task_done`, so nothing else is left unanswered. **Measured rather than argued, 10 runs each side on the live server** (Qwen3.6-35B-A3B, mira-mlx), judged by whether the reply actually contains `@contextmanager` or `def __enter__` and whether the extracted block compiles, not by whether it merely looks like code: pre-fix **10/10 context managers, 10/10 compile**, post-fix **10/10 and 10/10**, TTFT median 2.04s → 2.10s, wall median 38.6s → 37.1s. **The guard fired zero times in those 20 runs and that is the expected result**, because Q4 declares `tools: false` and `task_done` is therefore not on the model's menu at all — the live runs prove no regression, and the mechanism's correctness rests on the 10 unit tests, where removing each of the four gates kills exactly one test and removing the guard entirely kills seven. **Two harness errors caught and corrected before they became conclusions**: the first post-fix run reused the baseline's `conversation_id`s, so every request carried the previous answer in history (faster walls, different answers, worthless as a comparison) and was rerun with fresh ids; and a first re-scoring pass reported the baseline at 9/10 because one conversation also held an earlier tools-on smoke run, making message index 0 the wrong message. Both were tooling, not product. Worth recording separately: **with tools on, the model answers Q4 by calling `write_file` and summarising**, which is a legitimately agentic turn the guard correctly stays out of, so the tools-on path never reproduced this bug and the reproduction genuinely needs `tools: false`. 489 tests pass.
- [2026-08-08] **Built proactive decompression, and found that the obvious way to read per-process memory on macOS is wrong for a Metal process.** Mira now faults its own weights back in on the engine's idle branch when another app has had them compressed out, instead of letting the next reply pay 17.60s. Verified end to end: the touch reclaimed **8.34GB in 1.82s** with the pressure source still held, and the next real request took **0.65s**. Off by default in `core/config.py`, **on in `mira.yaml`** for a week of real use first. **The reusable finding is the instrument.** `task_info(mach_task_self(), TASK_VM_INFO)` costs 1.4us and needs no entitlement, but the struct mixes two accounting systems and the obvious fields are the wrong ones. XNU fills the rev0 fields from pmap statistics (`osfmk/kern/task.c:5303`, `vm_info->_name = map->pmap->stats._name * PAGE_SIZE`), and **the pmap does not account IOKit/Metal mappings, which is where all of Mira's memory lives**. Measured on the live engine with the model resident and replying in 0.40s, against `vmmap` reporting 292MB swapped and 18.9G resident: `compressed` read **15.46GB** (a permanent false alarm), `resident_size` read **4.65GB** (out by ~14GB), and `phys_footprint - resident_size` read **19.17GB** because it inherits that error. The rev3 **ledger** entries are the answer: `ledger_tag_graphics_footprint` read 19.76GB, matching MLX's own 19.66GB, and `ledger_tag_graphics_footprint_compressed` read **exactly 0** warm and gigabytes under eviction, so there is no threshold to tune. Same blind spot that makes `ps` RSS report 8.91GB for a process holding 19.66GB, which was visible hours earlier and read past. **The touch had to be 512 tokens, and the spec's original arithmetic was right about why.** A one-token pass was built and measured first: 1.19s, **zero bytes reclaimed**, model still 13.39GB compressed, because top-8-of-256 routing reaches ~3% of the expert table per token; a real request faults everything in only because a chat-templated prompt is several dozen tokens. Covering an E-expert table routed top-k is coupon-collector, `(E/k)*ln(E)` ~ 177, so 512 with a prime id stride. **Two failures were caught by design rather than luck**: `last_reclaimed_bytes` is a measured ledger delta, so the useless one-token touch reported 0 instead of looking successful; and mutation-testing the gates (remove the once-per-event latch, the headroom floor, the battery check, the per-process requirement, the critical-pressure check) killed exactly one test each, which is also what surfaced that `_touch_model_weights` had no exception handling on the only thread that can serve requests. Also filled a real gap: `derive_dynamic_ceiling_bytes` shipped in `e59951c` with **no unit tests at all**; there are now 45 across `test_memory_advisory.py` and `test_proactive_decompress.py`. 479 tests pass.
- [2026-08-08] **Committed `.python-version`, which was never ignored — just never added — and stopped it from silently retargeting CI.** It had sat untracked since the 3.13 migration, so the interpreter pin the root `CLAUDE.md` calls mandatory existed only on this machine; a fresh clone got whatever `uv` picked, which is exactly the drift that moved this project 3.12→3.13 unnoticed once already. It is not in `.gitignore` and has no delete in the history, so nothing ever decided to keep it local. **Committing it alone would have broken the CI contract, though**, and the precedence is the opposite of what it looks like: `setup-uv`'s `python-version: "3.12"` exports `UV_PYTHON`, and **`.python-version` outranks `UV_PYTHON`** — verified rather than assumed (`uv python find` with both set resolved 3.13; `uv 0.10.12`). CI would have quietly run 3.13 while the yaml still said 3.12, leaving the `>=3.12` floor in `requires-python` untested by anything. An explicit `--python` on the command line *does* outrank the file (also verified: `uv run --no-project --python 3.12 python -V` → 3.12.13 with the file saying 3.13), so all three `uv run` steps now pass `--python 3.12`. Result: local dev stays 3.13 per the standard, CI keeps testing the declared floor, and the two no longer fight.
- [2026-08-08] **Ran the proactive-decompression gate and rewrote the spec around what it measured. The feature survives, smaller and simpler; three of the original spec's claims did not.** No engine code changed; the result is the deliverable. Method: allocate a bounded hog, evict Mira, then **hold the hog alive and quiet across the measured turn** — holding is the whole point, because it is what separates "the request decompressed the model" from "macOS reclaimed once the pressure source exited". **The premise survives.** With the hog held and an idle control window showing 18 decompressions/s, one 8-token request drove Mira's own `vmmap` SWAPPED from 9.90GB to 0.63GB and decompressed 1,208,240 pages (19.80GB) with **22 pageins**. Nothing else could have done that. Two corollaries: the weights are **anonymous, not file-backed**, so any design around prefaulting mmap'd safetensors is aimed at the wrong mechanism; and **routing sparsity protects nothing** — a trivial request faults in essentially the whole 19.66GB `active_memory_bytes`, so §4b's "a one-token generation is not a valid touch" was simply wrong. **The original 15.37s stands, and an intermediate claim that it "does not reproduce" was my error.** The forced hog evicted only **9.90GB of an 18.80GB model** and gave 3.38s, which I read as 7.5x-not-33x. Then an **unforced** eviction from ordinary use (Xcode, WebKit, two editor sessions, no test process running) evicted all 18.80GB and cost **17.60s**, releasing 1.68GB of swap where the forced run released none. So the cost scales with how much got evicted, disk swap is what makes the tail expensive, and **the expensive case is the normal one**. Procedural lesson worth more than the number: before concluding a measurement contradicts a prior result, verify the condition was actually reproduced — comparing evicted bytes against total footprint is one line. The same event also showed **zero self-recovery on a natural event**: advisory logged `evicted` at 13:16:54 and was still evicted at 13:28:55, cleared only by a user request. And the shipped notification path validated live end to end: notified once, then correctly suppressed at 180s/511s/721s by the 15-minute cap. That retires the gate as written: it asked for a touch costing materially less than a real turn, and **no such touch can exist**, because the cost *is* faulting ~19.8GB back through the decompressor at ~6.8GB/s and any touch that faults the same bytes pays the same. The feature cannot reduce the work, only move it off the user's turn — **17.1s of it in the case that actually happens here.** The gate is now "the touch must reproduce the ~19.8GB and drive self-`compressed` under 1GB", which is about correctness, not a speed win. **§1's memory-neutrality claim held up, and my first correction to it did not:** availability fell 5.34GB across the decompress turn, but also **4.19GB across a warm control turn** because any turn transiently allocates, and both settled at the same place (4.75 vs 4.82GB) — net cost of the decompression itself is **~1GB**, as originally estimated. An availability reading without a warm control overstates it by ~4GB. No tug of war either: Mira **stayed** decompressed with the hog still held (second turn 0.67s). **So the trade is 2.9s for ~1GB, which is worth building** — spec rewritten accordingly. **The method improved more than the verdict did.** `task_info(mach_task_self(), TASK_VM_INFO)` returns *this process's own* `compressed`/`resident`/`phys_footprint`/`decompressions` in **1.4us with no entitlement**, validated against a 6GB victim process (self-report 1.75GB vs `vmmap` SWAPPED 1.6GiB = 1.72GB, `footprint -p` agreeing on 6.46GB). That replaces the system-wide `EVICTED_COMPRESSOR_FRACTION` heuristic at `hardware.py:232` with a per-process fact (measured anchors: 1.8% warm, 52% evicted), kills edge case (b) "not attributable" outright, and structurally kills the restart false positive `4777ef7` had to patch around, since a fresh engine reads its own compressed bytes as ~0. It also means **no separate swap-triggered variant is needed**: `compressed` is high in both cases, so one trigger covers the range and the payoff scales from ~2.9s to ~15s as the compressor spills. **Two harness traps worth as much as the result.** `ps` RSS is wrong for MLX — it read 8.91GB while `active_memory_bytes` was 19.66GB, because Metal `IOAccelerator` buffers are not in RSS. And the first hog re-walked every page every 0.5s and **generated 19.77GB of its own decompressions**, the same order as the real signal and indistinguishable from a positive result — allocate, touch once, hold silently, and always run an idle control window first. Harness kept at `notes/eviction-harness/`.
- [2026-08-08] **Mira stopped sizing itself as though it owned the Mac, and now says so when something else takes the memory** (`e59951c`, `3f18764`, `4777ef7`). Every budget came from `hw.memsize - SAFETY_MARGIN` computed once on the model thread at startup, so on this 32GB machine the ceiling was 29GB forever; `_check_memory_pressure` then compared MLX's *own* allocations against that constant, which makes another app taking 12GB invisible by construction, and it only ran while jobs were in flight, so an idle server never re-checked — and idle is exactly when you go and use the computer for something else. The ceiling is now re-derived from real system state on the model loop's idle branch beside `_release_idle_tower()`, time-gated at 30s because that loop spins at 50Hz and the probe is ~2.5ms of subprocess. **Two measurements changed the design.** Availability has to count `inactive` pages: measured under load, free alone read 0.06GB where several GB was genuinely available, so a free-only budget concludes the machine is permanently starving. And Mira's own footprint is added back before the margin is subtracted, because it already shows as unavailable in the system numbers — leaving it out makes the budget shrink in response to Mira's own size, which shrinks nothing and then shrinks again. **The signal that actually matters turned out to be compressor occupancy, not availability.** Under external pressure macOS compressed 21.4GB of pages into 17.05GB of compressor, mostly Mira's own weights, and **the next request took 15.37s against a warm 0.47s, a 33x penalty**; that single turn decompressed the model and emptied the compressor back to 0.20GB. So the slow reply is itself the fix, and the evidence is gone by the time anyone wonders what happened. A 220-second idle sample after the event was dead flat (available 8.27→8.36GB, pressure stuck at 2): **there is no self-recovery, waiting is not a strategy.** Confirmed live rather than argued: with Xcode and Docker Desktop launching, the ceiling tracked 8.64GB of external allocation with an 8.64GB drop **while macOS still reported `pressure_level: 1`**, then flipped to advisory `evicted` on its own once the compressor hit 17.03GB. `server.py` now relays the advisory onto `GET /hardware` (token-authenticated; deliberately NOT `/health`, which is exempt from the token check because it is the reachability probe — verified 401 without a token, full block with one, `/health` unchanged). macOS notifications are live and default on (`memory_advisory_notifications`), edge-triggered on the transition into `evicted`, capped at one per 15 minutes, reusing `scheduler.notify()` — extracted rather than re-implemented so the osascript argv hardening from `5a5015c` (leading dash parses as an option, a quote closes the AppleScript literal) stays in one place. **A false positive was caught on the very first live run and fixed** (`4777ef7`): the watcher notified 30s after start because a backend restart reports `evicted` on its first poll — the outgoing process's 18GB is reclaimed while the new one loads, spiking the compressor for seconds — and with `previous = None` that read as a transition. A transition needs a prior state to be a transition; the watcher now establishes a baseline first. Every existing test seeded a benign reading and so could not see it; the regression tests start the sequence *at* the bad state, and one checks the fix does not swallow a genuine eviction following a restart. Everything is advisory: nothing refuses, delays or shrinks a request, a failed probe reports `unknown` rather than defaulting to healthy, and a backend that reports nothing yields no advisory rather than an all-clear. Rode along: `/v1/stats` reported `active_memory_bytes: 0` on an idle server holding the whole model, because those counters were only written by `_check_memory_pressure`. 444 tests pass.
- [2026-08-08] **Made `MLX_ENABLE_TF32` a decision instead of an inherited default** (`ca71921`). The value did not change; what changed is that it is now stated in `backend_manager.py`'s `Popen` env with the measurements next to it. It is not cosmetic: mlx#3897 traced the M5 batch-vs-single attention divergence to TF32 accumulation inside the NAX kernel, and `MLX_ENABLE_TF32=0` is what makes mlx-lm's `test_generate` pass 28/28 here, so Mira had been shipping a setting known to perturb upstream bit-equivalence tests with the two halves of that trade recorded in different places and never weighed. Measured on a quiet machine (M5, `applegpu_g17g`, mlx 0.32.0, two passes within 2%), turning it off costs **2.58x on fp32 matmul (8804→3412 GFLOP/s), 2.70x on 4-bit quantized matmul, and 2.89x on `gather_qmm` at real MoE dimensions (8933→3095)**, and buys back about 10 mantissa bits of fp32 accumulation — well under the quantization noise of a 4-bit model, so keeping it on is the right trade. bf16 is unaffected either way (13983 vs 14050), which fits the dispatch gate's `dtype != float32` clause. Decode is untouched: the NAX path needs **more than 16 rows** to engage (identical at 16 — 1135 vs 1136 — and 1.74x apart at 32) and decode runs one row at a time, at 4.0% of the 8933 peak. Two investigations died on the way: the theory that the fp32 attention divergence was a fused-kernel or head-dim effect is **refuted** (naive GEMM shows the same error, and D=96 is not on a cleaner path in absolute terms — 4.0e-03 against 2.4e-03 for D=64/128), and the M5-accelerators-can't-help-MoE theory is **half wrong** — routing divides prompt tokens by 32 (256 experts, top-8), which puts the cliff at a ~1,024-token prompt, and Mira's real ~1,598-token prompts already clear it and collect 2.8x. Sparsity shifts the tensor-core entry threshold rather than excluding MoE from it. Full tables in gitignored `notes/m5-tf32-nax-measurements.md`.
- [2026-08-08] **Cleared 22 of 38 local specs and repointed everything that referenced them** (`b570805`). 14 had shipped long ago (vision, KV-quant, prompt-injection hardening, run_shell sandbox, test DB isolation, `mira chat`, and both MoE offload specs that were actually built) and 8 were explicitly abandoned — the three Phase D offload optimizations shelved by their own gate measurements, the cancelled per-arch atol PR, the conceded batching-narrowing follow-up, the LAN self-signed-CA spec superseded by Tailscale HTTPS, and the vision prefix-cache whose Parts A and B were deliberately not built. Twelve comments and docstrings still pointed at two of the deleted files, now aimed at `docs/moe-offload-case-study.md` and `docs/offload-resident-sizing.md` instead; three more had already been dangling from an earlier session. `BACKLOG.md`'s 2026-07-10 entry keeps its reference on purpose — it names a spec inside a correction recording that the spec never discussed KV-cache quantization, which is prose about a document's contents rather than a pointer to it.
- [2026-08-03] **Reviewed the maintainer's takeover of the vllm-mlx Mistral parser PR ([#631](https://github.com/waybarrios/vllm-mlx/pull/631)) and found two regressions in his fixes, one of them against `main`.** No mira-core code changed. waybarrios posted a five-finding review of `dfe6f46` and then, twelve minutes later, pushed two commits (`e65f075`, `a9daebf`) onto the fork branch fixing all five himself. All five check out when re-run: the legacy `[ARGS]`-in-JSON boundary is correct, the forged-call smuggling case is rejected on both paths, and the bounded name buffer flushes instead of losing the response. **The two new problems**, both measured by running the parser at `origin/main` (`0dd1157`), `dfe6f46` and `a9daebf` side by side plus the repo's own `tests/test_tool_parsers.py`. **(1) Parallel tool calls collapse on the streaming path.** The new `if self._args_started` early return at `mistral_tool_parser.py:293` runs ahead of the new-call detection at line 310, so a second `[TOOL_CALLS]` can never open index 1: it is appended to index 0's arguments instead, leaving one call whose arguments do not parse as JSON and a second call the client never sees. This catches the **legacy brace format, which `main` streams correctly** (`main` and `dfe6f46` both give indices `[0, 1]` with valid JSON; `a9daebf` gives `[0]` with `'{"city":"Madrid"}[TOOL_CALLS]get_time{"tz":"CET"}'`), so it is a regression against main and not only against the branch's earlier head. The `[ARGS]` format is not a regression, main never parsed it. The end-of-stream re-parse cannot rescue it: that fallback is guarded by `not tool_calls_detected`, and `tool_calls_detected` is set (`server.py:6232`) as soon as any delta carries `tool_calls`, which index 0 already did. **The one-line fix does not work and their own suite proves it** — letting a marker-bearing delta fall through restores `[0, 1]` but fails `test_mistral_streaming_marker_in_arguments_does_not_reset` (119/120). That failure is the useful part: swallowing the marker is exactly what keeps a `[TOOL_CALLS]` inside a quoted argument value from forging a second call, so the two behaviours are one mechanism and the real fix needs JSON string state carried across argument deltas, plus withholding a trailing partial marker so a split delta cannot leak either way. **(2) An odd number of double quotes before the marker hides the tool call entirely.** `_split_on_tool_call_markers` tracks string state from index 0, but the text ahead of the first `[TOOL_CALLS]` is prose, not JSON, so one unbalanced `"` leaves `in_string` true at the marker and it never becomes a split point (`tools_called=False`, whole string returned as content, both formats). Fix verified: start the scan at the first marker instead of index 0, which gives 120/120 and keeps the forged call rejected on both paths. Posted as [comment 5163641481](https://github.com/waybarrios/vllm-mlx/pull/631#issuecomment-5163641481). Local branch fast-forwarded `dfe6f46` to `a9daebf`; `pytest 9.1.1` installed into the fork's venv only, deliberately not into its `pyproject.toml`, so a repo that is not ours stays undiffed. Draft notes local in `notes/pr631-review-reply-draft.md` and `notes/pr631-finding1-fix-note.md`.
- [2026-08-03] **Closed out mlx [#3860](https://github.com/ml-explore/mlx/issues/3860); nothing owed.** Both asks from the 07-26 comments landed in the docs PR: attention fallbacks composed from ordinary GEMMs are named as affected (the head_dim 96 case), and the misleading "float16 and bfloat16 are unaffected" line is gone. The page has since been narrowed to 21 lines carrying only what holds regardless of backend and hardware, so the M5 `g17g`/`g17s` and M3 Max `g15s` measurements now live in the issue comments instead, alongside the author's CUDA sm_120 numbers posted 08-03. [#3883](https://github.com/ml-explore/mlx/issues/3883) is closed, [#3894](https://github.com/ml-explore/mlx/pull/3894) is open and unmerged.
- [2026-08-01] **Fixed a live bug where every Qwen3 turn with thinking on served its reasoning to
  the user as the answer** (`81e7564`). Qwen3's chat template pre-opens `<think>\n` in the *prompt*
  when `enable_thinking` is true, so the model never emits an opening tag and `ThinkingStripper`
  saw the whole reasoning stream as ordinary content. Fixed with a latch: `ThinkingStripper(preopened=True)`
  starts inside a thinking block, `saw_reasoning()` disarms it when a backend turns out to split
  reasoning into its own channel (in which case `content` really is just the answer), and `drain()`
  reclassifies an unclosed pre-opened block back to visible so a missing close tag cannot swallow a
  whole response. Request and response sides now share one predicate,
  `_uses_qwen_thinking_template()`, so they cannot drift apart again. 10 new tests across
  `test_thinking_stripper.py` and `test_thinking_control.py`, the latter parametrized over all four
  chat-template backends. Confirmed live on 2026-08-01: zero `<think>`/`</think>`/`<|channel|>`
  occurrences across a full 13-question bench.
- [2026-08-01] **Moved to mlx 0.32.0 + mlx-metal 0.32.0** (`5ba9c71`), then verified it end to end
  (`docs/bench-results-2026-08-01.md`, 25/26, no regression vs the 2026-07-18 baseline). The
  upgrade fixes nothing Mira had: measured against the four real 0.31.2 bugs, `BatchKVCache` uses
  an array offset so it never takes the broken rope branch, and `SwitchGLU` keeps M=1 so it never
  takes the broken gather_qmm one. The throughput win is real but only from about eight concurrent
  sequences (+24%), which Mira does not have. It was done now because `mlx-vlm >= 0.6.5` requires
  it and the vision work needs that, and doing it as its own change keeps "did the bump break
  something" separate from "did the vision seam break something". Analysis in
  `notes/mlx-032-upgrade-analysis.md` (local).
- [2026-08-01] **Upstreamed an omlx truncation bug**: `ThinkingParser.finish()` re-emits the whole
  reasoning stream as `content` when a thinking turn is cut short by `max_tokens`, so a truncated
  answer arrives as its own chain of thought. jundot/omlx issue #2457 and PR #2458 (fork clone at
  `~/Projects/omlx`, branch `fix/streaming-truncated-thinking`). The fix keeps the
  maintainer's intentional recovery path for the non-truncated case and only suppresses it when
  `finish_reason=length`. Awaiting the maintainer; that repo merges outside PRs in a median of
  under a day, so this should not become another mlx-lm-style long wait.
- Entries before 2026-08-01 were pruned on 2026-08-08. They are duplicated in
  `CHANGELOG.md`, in git history, and in the memory files those entries already
  cited; nothing was recorded only here.

## Pending

### Batched quantized KV kills the engine on any GQA model — reproduced, one-line fix, not applied
- **Production configuration crashes as soon as two requests overlap a long one.** Found 2026-08-12
  by the thinking-budget run, which was not looking for it: the engine thread dies with
  `ValueError: [broadcast_shapes] Shapes (3,1,1,1629) and (3,2,8,1,1629) cannot be broadcast` inside
  `quantized_scaled_dot_product_attention`, and every request after that returns 503. The 503 is the
  new `EngineDead` path working; the death is the bug. **Cause:** with grouped-query attention the
  scores are reshaped to five dimensions and the keys and values are lifted to match with
  `expand_dims(..., axis=-3)`, but `BatchRotatingQuantizedKVCache.make_mask` (cache.py:1969) always
  returns a four-dimensional mask. Four against five broadcasts from the right, so the mask's batch
  axis lands on `n_kv_heads`: at batch 1 the leading 1 broadcasts against anything and it works, at
  batch 3 it is 3 against 2 and it does not. **That is why nothing saw it** — a single user is at
  batch 1 nearly always, and a bench that asks concurrently is not. Needs all four at once: `kv_bits`
  set (`mira.yaml:28` has 8), past `quantized_kv_start` so the cache has really converted, two or
  more sequences decoding, and `n_repeats > 1` (Qwen3.6 is 16 heads over 2 KV heads).
  **Reproduced standalone** in a second with no model and no server —
  `/tmp/claude_repro_qsdpa_mask.py`, identical error string — and the fix verified in the same run:
  give the mask the axis the keys already get, `mx.expand_dims(mask, axis=-3)` when `n_repeats > 1`
  and the mask is a 4-D array. It belongs in `base.py` where the reshape happens, not in one cache
  class, since every caller quantizing a batched cache needs it. **Not applied**: the code is in the
  fork, in the same area as PR #1584, and whether it rides in that PR or lands ahead of it as its own
  commit is Miguel's call. **This blocks the #1584 A/B/C split** — #1584 is the PR that makes batched
  KV quantization work, this is a crash in exactly that combination on any GQA model, and it is
  invisible at batch 1. It also settles the running argument in that thread in favour of running the
  batched path rather than reading it. Meanwhile production is exposed: dropping `mira_mlx_kv_bits`
  avoids it at the cost of KV compression, leaving it means an occasional engine death that now at
  least surfaces as a clean 503. No regression test yet; the natural one is the repro as a pytest
  with `pytest.importorskip("mlx.core")`, parametrised over batch 1 and 3. Full write-up in
  gitignored `notes/kv-quant-batched-mask-crash-2026-08-12.md`.

### `MAX_THINKING_TOKENS` — measured 2026-08-12, and 2048 does not bind either
- **The code default is now 2048; the live `mira.yaml` is deliberately still 8192.** Changing a
  runtime config is Miguel's call, and the measurement below is the argument for it rather than a
  reason to do it silently.
- **Two runs, and the first one measured nothing.** Both arms (8192 and 2048), same five rephrased
  questions on the topics that broke worst in batches 1 and 2, adaptive thinking: 0 broken of 5 at
  8192, 2 of 5 at 2048 — **and both of those were the engine crash in the entry above, not the
  budget**. Thinking was 0 characters on 3 of the 5 turns at 8192, so under adaptive mode the model
  mostly declined to think and the two arms were comparing nothing. n=1 per cell cannot separate a
  budget effect from ordinary batch-composition variance anyway ([[project_batch_divergence_changes_output]]).
- **Second run, thinking forced per request** (`thinking_enabled=true`, the three hardest questions,
  fresh server per arm): **0 broken of 3 in both arms.** Replies were if anything longer at 2048 —
  median 3,750c against 2,964c — which is variance, not an improvement, and that is the point.

  | arm | set-analysis | data-modelling | kernel-debate | total |
  |---|---|---|---|---|
  | 8192 | 1,894c reply / 1,098c thinking | 2,964c / 2,381c | 6,591c / 3,875c | 282.3s |
  | 2048 | 1,941c / 456c | 3,750c / 3,459c | 6,202c / 2,587c | 265.6s |

- **The finding: `thinking budget HIT` appears zero times in either arm's engine log.** The
  processor is confirmed active — `thinking budget active: 2048 tokens (preopened=True, ...)` on
  all six requests — so it is armed and simply never reached. The longest reasoning block was 3,875
  characters, roughly 1k tokens, on the question written specifically to make the model argue with
  itself. **So 2048 is still a runaway guard, not a budget.** Lowering it changes nothing about a
  normal turn and tightens the ceiling on a runaway fourfold, which is what a guard is for.
- **Go, with the honest caveat.** This run cannot show a quality cost because there was nothing to
  cut — every turn finished its reasoning well inside both budgets. The case that would bind is the
  ~5k-token turn seen in the earlier corpus, and it did not reproduce on demand here, so the entire
  effect of 2048 lands on a tail this measurement never sampled. Anyone wanting a stronger result
  needs a prompt that reliably produces long reasoning, not more repeats of these.
- Harness: `notes/thinking_budget_arms.py`, now with `--force-thinking`, `--arms` and `--questions`.
  Results in `notes/thinking_budget_forced.json` (forced) and `notes/thinking_budget_arms.json`
  (adaptive, and the run that found the crash). Engine logs kept as
  `/tmp/claude_budget_arm_<budget>_engine{,_adaptive}.log`.

### `derive_resident_expert_fraction` is blind to gpt-oss-120b
- **`gpt-oss-120b-MXFP4-Q8` reports `num_experts=None` in its config, so `_classify_weight_bytes`
  finds 0GB of experts and `derive_resident_expert_fraction` returns the `floor_fraction` (0.3)
  unchanged** — the same early return a dense model gets. Found 2026-08-11 while modelling what a
  128GB Mac would do: at 32GB the model is over-DRAM either way so nothing is lost today, but on a
  larger machine gpt-oss-120b is the obvious thing to run and the MoE sizing logic cannot see it.
  The fix is a config-shape question, not a policy one: find what gpt-oss calls its expert count
  (the key is not `num_experts`) and teach `_classify_weight_bytes` that spelling. Related and
  already known: attention-sink models cannot use quantized KV, so gpt-oss needs the unquantized
  path regardless. **No action while this Mac has 32GB** — it changes nothing at this size.
- **The 128GB crossover, recorded so it is not re-derived:** the ceiling is
  `min(0.55 x total, 0.78 x total - 3GB, total - 3GB)` = **70.4GB on a 128GB Mac**, so the offload
  path only engages above that. Verified against the cached models: the 8-bit Qwen3.6 (35.1GB) and
  gpt-oss-120b (59.0GB) both go from over-DRAM at 32GB to **fully resident** at 128GB. A 4-bit
  100B-class model therefore never exercises offload on 128GB — that needs a 4-bit 235B-class
  (~132GB) or an 8-bit 120B.

### Waiting on a week of ordinary use — no action until then
- **`proactive_decompress` and `boundary_snapshot` are both ON in `mira.yaml` and OFF in
  `core/config.py`, and that split is the plan, not an oversight.** Decided 2026-08-08: this Mac
  runs both so the week produces real data, while anything shipped or remote keeps the
  conservative default until that data exists. **Confirmed for v1.2.0: they ship off**, and both
  are now documented in `mira.yaml.example` so "off" does not also mean "undiscoverable". The
  week is what decides whether the default flips in a later release. What the week is for:
  - `boundary_snapshot` — whether `_history_boundary` ever refuses a real prompt (it logs a
    warning), whether `failures` stays 0, and whether two entries per turn push the LRU against
    its 5.00GB ceiling. Watch `/v1/stats` → `boundary_snapshot`.
  - `proactive_decompress` — `events` climbing with non-zero `last_reclaimed_bytes` is it
    working; `failures > 0` means it disabled itself; `skipped_no_headroom` climbing means the
    availability floor is too high for this machine.
  - **The eviction notification firing at all.** A week of zero notifications is a real result,
    not a failure: it would mean eviction is rarer than the forced Xcode+Docker test suggested.
  One known limitation, accepted rather than fixed: **a request arriving during a touch waits
  behind it**, because the model thread is the only thread that can serve. Still never worse than
  doing nothing, but the magnitude is larger than the first measurement suggested — `events: 7`,
  `last_reclaimed_bytes: 20.05GB`, `last_seconds: 19.52` within hours of enabling it. So the touch
  scales with how much was evicted, ~1.8s for a partial to **~19.5s for a full one**. If that wait
  is ever felt, the fix is to make the touch interruptible (chunk it, re-check the inbox between
  chunks), not to disable it.
  **Trial data, 2026-08-09 morning: `events: 23`, `last_seconds: 1.577`, `last_reclaimed_bytes:
  7.26GB`, `skipped_no_headroom: 0`, `failures: 0`.** So it is firing regularly under ordinary use
  and reclaiming on the idle branch at no user-visible cost. Worth stating plainly because the
  17.60s figure in `mira.yaml` describes the **unmitigated** case: macOS never faults the model back
  spontaneously, which is the motivation for this flag, not a description of what happens with it
  on. Even with it off the worst case was ever one slow reply, never permanent degradation — the
  request that pays the cost is itself the fix. **An eviction costs time, not correctness**, so it
  cannot change a generated answer and accuracy evals are unaffected by it.

### Needs Miguel
- ~~**Arm the wake for the GPQA night.**~~ **ARMED 2026-08-11** — `pmset -g sched` shows
  `wakeorpoweron at 08/12/2026 00:25:00 by 'pmset'`, and Arq's four wakes (00:00/01:00/01:30/03:00)
  survived alongside it. **The lid must stay open** — `caffeinate` does not defeat clamshell sleep,
  and that is now the only thing between the plan and a result. If the plan ever grows past one
  night, `pmset repeat wakeorpoweron MTWRFSU 00:25:00` is the form to use instead of a one-off.
- ~~**Decide IFEval's generation cap.**~~ **DONE 2026-08-11** — see the Done entry. IFEval runs at
  16384, overriding its `max_gen_toks: 1280`, and the publishing condition moved off the cap and
  onto reporting it. Nothing blocks the eval suite now.
- ~~**Greedy decoding is not tweakable anywhere.**~~ **DONE 2026-08-09.** `temperature`, `top_p` and
  `top_k` are now configurable in `mira.yaml`, wired through `config.py` → orchestrator → `ChatJob`
  → `make_sampler`. Defaults are unchanged (0.0/0.0/0), so no reply changed. Verified rather than
  assumed: `top_k=1` is argmax and reproduced greedy byte-for-byte. `tests/test_sampling_config.py`
  covers it, including that `extra_body` is merged so `top_k` cannot clobber the thinking toggle.
- **Two findings from the sampling work; the first is now fixed.**
  1. ~~**Output is deterministic per (prompt, params) even at temperature 1.0**~~ **FIXED
     2026-08-12.** Two identical requests returned byte-identical text, so regenerating a reply gave
     the user the same answer no matter the temperature. The diagnosis held: there is no response
     cache, the server set no seed, and mlx-lm's samplers thread `mx.random.state` through
     `mx.compile`. `_admit_job` now seeds per job — the request's `seed` when given, a fresh
     `secrets.randbits(32)` when not — and skips seeding entirely at temperature 0, where it changes
     nothing. Setting `seed:` in `mira.yaml` buys back reproducibility on purpose, which is what a
     bench wants and a regenerate does not. **Note this only holds on an idle server**: consequence
     2 below still applies, since batch composition moves the arithmetic regardless of the seed.
  2. **Thinking can eat the entire production budget.** In 3 of 5 runs at `--max-tokens 4096`, a
     moderately complex creative prompt hit `finish_reason: length` without ever closing `</think>`,
     meaning the user gets truncated reasoning and no answer. ~~Seen on one prompt only; worth
     reproducing deliberately before treating it as a general bug.~~ **REPRODUCED 2026-08-11 — it
     is a general bug.** The conversation corpus run (24 multi-turn exchanges, three topics, real
     `/chat` traffic) hit it on **8 of 28 assistant turns**, and the correlation with the cap is
     **7 for 7**: every `finish_reason=length` in that window produced a user-visible broken reply,
     matching to the second. Day-wide rate 10 of 102 LLM calls, 9.8%.

     The mechanism is now pinned down, and it is not the cap alone. `thinking_stripper.py:97-111`
     deliberately reclassifies an unclosed pre-opened `<think>` block as the answer — correct when
     the model *chose* not to close it, wrong when it was cut off — and it cannot tell the two apart
     because `orchestrator.py:946` calls `drain()` without passing the `finish_reason` it read at
     line 922. The signature is reply length equal to thinking length exactly (13,299 = 13,299;
     19,247 = 19,247), the same characters emitted once as thinking and again as the answer.
     One reply was a bare `<|mask_start|>` special token, saved to the database as the assistant
     message. A retried prompt reproduced byte-identically at 14,025 characters, so a fix has an
     exact target to verify against. Full writeup and the open decision in
     `specs/truncated-thinking-becomes-the-answer.md`.

     **REPLICATED on a second batch, and the cap is the real problem.** Three more conversations
     on unrelated topics, same protocol, nothing changed in between: 5 of 24 turns broken (20.8%)
     against 8 of 27 (29.6%) in batch 1, and 26.1% vs 29.4% counting only turns long enough to
     reach the cap. Median reply length was near identical across batches, so answer length is not
     driving it. Four of the six conversations were hit.

     Batch 2's failures took a **different and worse shape**: replies of exactly 4096 `!`
     characters, a full `max_tokens` of one repeated token. Five consecutive turns in one
     conversation. It began on a turn where `fetch_url` started three fetches and completed two,
     and from there the conversation never recovered — later turns called no tools at all and still
     returned 4096 `!`, because the poisoned reply is saved to history and fed back in. **The
     conversation is permanently dead with no user-facing way to recover it.** The other two
     conversations in the batch were clean throughout, so the poisoning is conversation-scoped.

     Same trigger (`finish_reason=length`), two outcomes. Leaked reasoning at least carries
     information; this carries none. Ranking that follows from it:
     1. **`max_tokens=4096` is the root cause worth fixing first** — it makes truncation common and
        produces both disasters. The stripper only decides what a truncated turn looks like.
     2. **Guard the output.** A reply that is 4096 of one character should never be shown or
        persisted. Detecting it is one line, and persisting it is what turns a bad answer into a
        dead conversation.
     3. **Then** the stripper fix, which still needs the decision below.

     ~~**Needs a decision before building.**~~ **ALL THREE SHIPPED 2026-08-11**, Miguel approved
     the ranking above.
     1. `max_output_tokens` is a real setting now (`config.py`, `mira.yaml`, both call sites in
        `backend_manager.py`), default **16384**. It was hardcoded to 4096 in three places. Worth
        recording how long this hid: `MAX_THINKING_TOKENS` has been 8192 all along, double the
        entire generation budget, so it could never bind — and the orchestrator sends it as
        `thinking_budget` while **mira-mlx never reads it**. That knob is dead code; the config
        was advertising a limit nothing enforced.
     2. `_degenerate_run()` in `orchestrator.py` discards a reply that is one character repeated,
        judged over non-drawing characters so ASCII diagrams survive. First version of the rule
        scored two real diagrams as broken, which is why that exclusion exists.
     3. `ThinkingStripper.truncated()` stops the reclassification when, and only when, the stream
        was cut at the cap. A turn with nothing left to show now says so instead of returning empty.

     Verified live, not assumed: engine restarted and running `--max-tokens 16384`, and the exact
     prompt that produced 14,025 characters of chain of thought twice byte-identically now returns
     a 2,968 character answer in 78.5s. Caveat on that check: it ran on a fresh conversation id, so
     the context differs from the poisoned turn 8 it reproduces. 670 tests pass; the new ones in
     `tests/test_truncated_generation.py` were checked against the old `drain()` and fail on it.

     **The `fetch_url` half of it — chased, and FIXED 2026-08-11.** Two independent defects, both
     confirmed against the live URLs rather than reasoned about:
     1. `markdownify`'s `strip=` argument only skips the *tag* and still walks its children, so the
        contents of `<script>` and `<style>` came back as page text. developer.apple.com returned
        969 characters of `var baseUrl = "/tutorials/"` and `.noscript{font-family:...}` — handed
        to the model as the page. They are removed outright now, which takes the same page to 55
        characters, correctly reading as empty.
     2. The JS-page fallback required `raw_html_len > 50_000`, assuming a client-rendered page
        ships a lot of HTML. Apple's shell is 17 KB, so it never qualified. The test is now the
        text-to-HTML ratio with a much lower size floor.
     Together: those two pages went from 969 and 978 characters of CSS to **17,964 and 8,291
     characters of real content**, with the Qlik page that already worked unchanged at 24,038.
     `tests/test_url_fetcher_extraction.py`, checked against the old code first — the old path
     hands over 4,249 chars of JavaScript and never calls the fallback.

     Two things that turned out **not** to be problems, recorded so they are not re-investigated:
     the 24,038 seen on three unrelated URLs is `MAX_CONTENT_CHARS` (24,000) plus the 38-character
     truncation notice, exactly as designed. And the 36 `web_search` calls were the total across
     the whole corpus run, not one runaway turn: the loop is bounded at `MAX_AGENT_STEPS=15` /
     `MAX_TOOL_CALLS_PER_TURN=20` / `SAME_TOOL_REPEAT_LIMIT=15`, and no turn came near any of them.
     The repeated searching was a *consequence* of fetch_url returning junk, not a loop of its own.

     **Batch 3 says the fix holds.** Post-fix replication on the three topics that broke worst,
     chosen deliberately rather than an easier set: set analysis (was 5 of 9), data modelling
     (was 5 of 8 and died outright), kernel debate (was 2 of 9). **0 broken turns of 23**, against
     29.6% and 20.8%, and **zero `finish_reason=length` warnings** in the batch window against 5
     in batch 2. Median reply length held at 3,518 characters and 17 of 23 turns cleared the
     "long enough to reach the cap" threshold, so this is not a batch of shorter answers dodging
     the problem. Two confounds, stated rather than hidden: fetched pages now return real content
     instead of CSS, so fetching turns had more to work with (one turn made 34 tool calls, the
     largest of any batch), and the askers were new agent instances writing their own wording.

     Batch 3 did surface **three errored turns, and they are a different, pre-existing bug** —
     now fixed. `_tool_ui_labels`'s default done-label lambda called `.get` on the tool result,
     but `list_attachments` and `read_attachment` return a plain string, so the stream raised
     `AttributeError` after `tool_start` had already been sent and the user got "Internal error —
     see server logs". `git log -L` dates both lines to `502344f`, 2026-04-25; batches 1 and 2
     never hit it because the model never called that tool. The line at `orchestrator.py:1135`
     already guarded the same result with `isinstance(result, dict)`, so the string case was
     known here and the label layer alone did not believe it. Two things shipped with it: the
     handler in `server.py` bound the exception and dropped it, which is why the logs it tells the
     user to check held nothing, and that now calls `logger.exception`. Verified on the live
     server, not assumed: the same request now emits `tool_start` then `tool_done`, no error.

     **Two residues, both left in place on purpose.** Batch 2's poisoned conversations are still
     in the production database with their 4096-`!` replies saved as assistant messages. The fix
     stops new poisoning at generation time; it does not clean history, so continuing one of those
     conversations feeds the model five turns of `!`. Miguel read them in the app on 2026-08-11
     and chose to keep them as they are. Untested and uncleaned, deliberately.

     And a caution about the corpus as an instrument, found by reading those same conversations:
     the asking agents **did not notice a total failure while it was happening**. Turn 4 returned
     4096 `!` and the asker replied "You hit on something I hadn't thought through yet... That's a
     real insight", then carried on for four more turns and signed off thanking Mira for a
     walkthrough it never gave. A corpus conversation can therefore read as a coherent exchange
     while one side is punctuation. Never infer a turn was usable from the fact that the
     conversation continued.

     **A third residue, found 2026-08-12 by checking today's code instead of re-reading old
     conversations: a retry leaves the failed turn in the database.** `db.save_messages` appends
     with no dedup, and mira-apps' `resendLast()` (`ChatViewModel.swift:382`) drops two messages
     from its own local array and re-sends — the server is never told anything was dropped.
     **Demonstrated live**, not inferred: two identical posts to `/chat` on one conversation id left
     4 rows, 2 of them the same user message, and the test conversation was deleted afterwards. So
     every retry of a broken reply permanently doubles the question in history and keeps the
     corrupted answer next to it, and rebuilt context feeds the model both. That is the mechanism
     behind the poisoned conversations above surviving a retry. The fix is a mira-core question
     rather than a client one — the client cannot un-save what it already sent — so either the app
     names the turn it is replacing or `/chat` grows an idempotency key. **Not built, not specced.**
     Cheap check on the same run: a fresh 10-turn pass showed **0 reasoning-as-answer and 0
     degenerate replies**, so the generation-side fixes are still holding.
- **Nothing is established about which sampling config is safer, and two probe rounds prove why.**
  Same prompt, same parameters, two rounds on 2026-08-09, flatly contradictory:

  | | round 1 | round 2 |
  |---|---|---|
  | greedy (production) | clean, max repeat x2 | **worst of five**, 60% dup, x7 |
  | temp 1.0 alone | **303 repeats**, length cap | **cleanest of five**, 0% dup, x1 |

  So the greedy-causes-loops theory is dead, but its replacement ("temperature alone is worse")
  died with it in round 2. **n=1 per configuration ranks nothing here**: greedy alone differed
  across two server processes on the same prompt (9,907 vs 14,711 chars, cold vs warm prompt
  cache), consistent with the known M5 batched-attention divergence. Any real answer needs repeats
  against a fixed cache state. `mira.yaml.example` now recommends changing all three together on
  the model author's authority only, and says so.
- **RESOLVED 2026-08-09: batch size changes greedy output, and that is what the eval loop was.**
  Paired experiment, 24 IFEval prompts through `batch_generate` at batch 1 and batch 4, greedy
  throughout, everything else identical (`notes/batch_divergence_probe.py`):

  - **23 of 24 prompts produced DIFFERENT output.** Greedy is deterministic within a process, so a
    single differing token proves the arithmetic changed. Median token-count difference 20%, max
    4.35x.
  - **The eval failure mode reproduced, in the batched condition only.** Prompt 1040: single-request
    finished cleanly in 3,170 tokens (max sentence repeat x3); batched ran to the 8,192 cap, never
    closed `</think>`, and repeated one line **355 times** (68% duplicate ratio). Its single-request
    twin did neither.
  - **But batching is NOT systematically worse, and the mean says otherwise only because of that
    one outlier.** Median max-repeat is 3 (single) vs 4 (batched) — the means, 3.75 vs 18.92, are
    driven entirely by prompt 1040. Failures went both ways: 1001 ran to the cap unclosed only when
    single, 1040 only when batched, 1130 in both.

  So the honest conclusion is **not** "batching causes loops". It is: **which prompts degenerate is
  decided by numerical happenstance**, and batch composition is one of the inputs. Rates are
  indistinguishable at n=24 (one failure each way).

  This is the known M5 batched-attention divergence ([[project_m5_batched_attention_divergence]]),
  where `MLX_ENABLE_TF32=0` is float32-only and the bf16 path has **no opt-out** — so there is
  nothing to fix, only to mitigate. Caveat: the standing rule says numerics tests set that flag and
  this run did not; it is documented as float32-only and this model computes in bf16, so no effect
  is expected, but that was reasoned, not verified.

  **Consequences worth acting on:**
  1. **Eval scores carry a batch-composition component.** `run_lm_evals.sh` runs at batch 4; a
     re-run at a different batch size is not strictly comparable. Fix the batch size, and report it
     alongside any score.
  2. **Production replies depend on concurrent traffic.** mira-mlx does continuous batching, so the
     same question can get a different answer depending on what else is in flight. Inherent, not a
     bug, but it means "deterministic per (prompt, params)" holds only for an idle server.
  3. **~8% of these prompts ran to the cap without finishing their reasoning** (2-3 of 24 in at
     least one condition) at an 8,192 budget. Production caps at 4,096, so likely worse. A
     repetition/runaway guard would help where numerics cannot be fixed. **The guard is wired as of
     2026-08-12** — repetition, presence and frequency penalties reach `make_logits_processors` from
     `mira.yaml` — but it ships **off**, so this consequence is unchanged until someone turns it on.
     Deliberate: penalising repetition also penalises the legitimate kind, and code and tables are
     full of it. The example config gives the discriminating number rather than a recommendation
     (x16 repeats in a healthy reply against x355 in a degenerate one).
- **HF credentials.** ~~As of 2026-08-09 both were invalid.~~ **Re-verified clear on 2026-08-09:**
  `whoami()` resolves as `mabaeyens` and `dataset_info('Idavidrein/gpqa')` returns 8 files, so the
  gate is accepted too. The underlying hazard stands: `~/.zprofile` line 10 exports an `HF_TOKEN`
  **which takes precedence over `hf auth login`**, so updating only the stored credential does
  nothing. `hf auth login --force` once would give a single source of truth.
- **Score the 2026-08-08 agentic bench.** `docs/bench-results-2026-08-08.md` has timings and tool
  traces for Q6–Q12 filled in and the quality column empty. Less urgent than it was:
  `scripts/bench_eval.py` now scores these automatically, so this column is only for judgement the
  harness deliberately does not make.

### Small, no decision needed
- **Memory-state notifications should only fire when Miguel can act, or when they explain something
  he caused.** Asked for directly on 2026-08-11, after the log showed **302 `advisory → evicted`
  transitions in 70.5 hours** — one every ~14 minutes, round the clock, including 01:30, 03:00 and
  06:53 while he was asleep. macOS compresses an idle 20 GB process and Mira decompresses it back;
  there is nothing for a user to do about either, so the notification is pure noise and it trains him
  to ignore the channel that should carry the real ones. **The rule: keep the log line, drop the
  notification, unless the state is one the user can change (close another app, plug in) or one his
  own action produced.** Not the same thing as suppressing the *event* — that is
  `specs/idle-decompress-treadmill.md`, which may or may not be worth building; this item stands
  either way and is much cheaper. Touches whatever in mira-apps consumes `system_memory.advisory`,
  so it is a mira-apps change with a mira-core decision behind it: mira-core owns the advisory field
  and should say which transitions are user-facing.
  **Specced 2026-08-12** in `mira-apps/specs/memory-notifications-only-when-actionable.md`, and the
  writing turned up the reason the channel has been silent since 2026-08-11: **mira-core turned
  notifications off globally**, not selectively. `MEMORY_ADVISORY_NOTIFICATIONS` defaults to False
  (`core/config.py:324`), `core/memory_watch.py` still records every transition, and the server log
  shows 12 `model evicted by another process` warnings on 2026-08-12 with
  `Memory advisory notifications off; still recording transitions` alongside them. So "no
  notifications since yesterday" is the flag doing what it says, not a bug — and worth knowing that
  only `evicted` ever notified, there was never a separate pressure notification. The selective rule
  is still unbuilt; the blunt off-switch is standing in for it.
- **The question set is now the limiting factor, not the harness.** Tier 1 scored 24/24 three runs
  running, so the deterministic half no longer discriminates between builds — it is a regression
  alarm, not a measure of quality. The judged half carries a ±1 floor, so it cannot be read finely
  either. Sixteen questions, most of them comfortable for a 35B model. **The standardised suites
  (IFEval, MMLU-Pro, GPQA, BFCL, AgentDojo) are the right instrument for "is Mira accurate"; this
  bench should settle into being a regression alarm for Mira's own plumbing** — orchestrator, tools,
  injection handling — which no public benchmark covers. Adding harder questions buys more than
  further harness work. **Six added 2026-08-12 (17-22), each one a bug that shipped** — see the Done
  entry. They are written and unit-tested; **they have not been run against a live server yet**, so
  whether they actually discriminate is unmeasured.
- ~~**Bench Q10 turn 2 rests on a false premise.**~~ Fixed 2026-08-09 in `5cb4dc8`: the injected
  file is now named by the question (`core/orchestrator.py`). Any Q10 score before that date
  measured a broken question.
- **The `task_done` guard's refusal path has no live demonstration.** Q6–Q12 ran on 2026-08-08 and
  the guard fired **zero** times while every agentic question exited through `task_done` normally,
  which proves it is inert on legitimate turns and regressed nothing. It does not prove the refusal
  works against a real model, because Qwen3.6 never took the escape hatch. Only unit tests cover it.
  **Bench question 20 exists to bait it** as of 2026-08-12 — a task the model is likely to want to
  declare done early — but it has not been run, so this stays open until it has.

### Upstream — open, and genuinely nothing to do but wait
- **Three mlx-lm PRs are open with zero reviews**, confirmed 2026-08-08:
  [#1584](https://github.com/ml-explore/mlx-lm/pull/1584) (KV-cache quantization for the
  continuous-batching path), [#1588](https://github.com/ml-explore/mlx-lm/pull/1588) (disk-backed
  expert offloading), [#1619](https://github.com/ml-explore/mlx-lm/pull/1619) (`rotated` flag
  round-trip in `BatchRotatingKVCache.meta_state`). A maintainer review is the binary gate — 31 of
  400 external PRs got one and 30 merged; 71 did not and 0 merged. **Do not open a fourth.**
  Mira is NOT blocked on any of them: KV quantization has been live since 2026-07-18
  (`mira_mlx_kv_bits: 8`) on the `mira-core-pin` fork branch. What upstreaming buys is a smaller
  patch-carrying burden, nothing functional.
  - **Why the pin exists, since it is not obvious:** rebasing `mira-mistral-tool-call-fix` picked
    up upstream's text-state-machine refactor (mlx-lm #1501), which renamed `SequenceStateMachine`
    → `TextStateMachine` with a different API and broke `mira_mlx_server.py`'s import when the
    rebased commit was briefly live in the lockfile. `pyproject.toml` now pins a dedicated
    `mira-core-pin` branch frozen at `bbd8496` so PR-prep rebases cannot break mira-core again.
  - **Follow-up once #1584 merges:** check whether mlx-lm's own tool-call-flush fix supersedes the
    Mistral-specific patch given #1501, and migrate `mira_mlx_server.py` to the `TextStateMachine`
    API when moving off `mira-core-pin`.
  - **#1584 is no longer "wait only" as of 2026-08-11: it is being split into three PRs.** At +1132
    across 4 files it sits in the class that merged **0 of 14**, and the CacheList work owed to
    @pierre427 would only grow it. Miguel asked @angeloskath about splitting on 2026-07-26 and got
    no answer — consistent with the roster finding below, since angeloskath has not commented
    anywhere since 2026-07-20. **Decision: do not volunteer the split as eagerness-signalling, but
    do act on an unanswered question of one's own.** The three pieces, with ready-to-post bodies in
    gitignored `notes/pr1584-split/`: **(A)** single-sequence `RotatingQuantizedKVCache` +
    `RotatingKVCache.to_quantized()`, ~390 lines, unblocks #1573; **(B)** batching
    (`BatchRotatingQuantizedKVCache`, `BatchGenerator` wiring, `merge()` raise→delegate), depends on
    A, is what #1476 rebases onto; **(C)** the `CacheList` recursion, independent of both and a
    live bug on main. Index comment for the #1584 thread is drafted alongside them. **Carry
    `MLX_ENABLE_TF32=0` into all three descriptions** — without it the same 8 pre-existing
    `test_generate.py` failures make three separate PRs look broken to any gen-17 reviewer.
    **Next action: await @pierre427's two tests, then open A/B/C and post the index.**
- **vllm-mlx: one commit still owed upstream.** `edc07d4` (sibling KV-cache-path gap) waits on
  upstream [#629](https://github.com/waybarrios/vllm-mlx/pull/629) merging, then goes up as its own
  PR. It is **not on any branch** — dropped in the 2026-07-10 rebase that narrowed
  `fix/mistral-args-token-tool-parsing` to the parser fix — so recover it by SHA through the
  reflog, not by checking out a branch. (`d408d70` is superseded by upstream #629 itself.)
  **PR #631 merged on 2026-08-03**; a stale entry here still described it as awaiting the
  maintainer with a next action attached, corrected 2026-08-08. Nothing is owed on #631.

### mira-apps (separate session)
- **Three specs written 2026-08-08; the first is already done, the other two are in `mira-apps/specs/`:**
  - ~~`mira-core/specs/eviction-notification-predicts-slow-reply.md`~~ **DONE in `cdb446b`**, in a
    parallel session, within the hour. The notification no longer predicts what the next reply will
    cost, and the test rejects a prediction in **either** direction plus a string vague enough to
    satisfy the ban by saying nothing — verified against the old copy, the opposite copy and an
    empty one. "your Mac" deliberately kept rather than mirroring the banner's "the Mac running
    Mira": the two agree on the claim, not the wording.
  - ~~`decode-check-covers-system-memory.md`~~ **DONE in `74b2824`** — the decode check now covers
    `SystemMemory.swift`, which was the one whose failure was silent: a renamed key decodes to
    `.unknown`, which renders as nothing, which looks exactly like a healthy machine.
  - ~~`memory-advisory-device-verification.md`~~ **DONE in `3765330`** — the banner has been seen
    rendering. Both specs are gone from `mira-apps/specs/`; this list was stale until 2026-08-12.
- **Two specs written 2026-08-12, both in `mira-apps/specs/`, neither built:**
  - `memory-notifications-only-when-actionable.md` — the selective rule. Note what the
    investigation turned up first: **the silence is mira-core's, not the app's.**
    `MEMORY_ADVISORY_NOTIFICATIONS` has defaulted to False since 2026-08-11 (`core/config.py:324`),
    `core/memory_watch.py` still records every transition, and the server log shows 12
    `model evicted by another process` warnings on 2026-08-12 alone. Only `evicted` ever notified —
    there was never a separate pressure notification to miss. The spec is about the distinction the
    old design lacked: an interrupt has to be actionable or explain something the user caused,
    everything else belongs in the ambient banner. **Open question recorded in the spec:** whether
    mira-core adds a `cause`/`user_actionable` field or the client infers it. The lean is mira-core —
    it is the side that knows which process took the memory.
  - `connection-errors-device-verification.md` — the three device checks: (a) a genuine 403 names
    `allowed_hosts` instead of blaming the network, (b) a real unreachable case (Tailscale off) still
    reads sensibly, (c) the longer 403 string does not overflow the error label in the Add Connection
    sheet. **Open question recorded in the spec:** whether a permanent 403 should stop the 90s
    retry — the lean is to keep retrying but surface the reason on the first refusal, since today a
    misconfigured `allowed_hosts` retries a hopeless connection and then goes orange with no
    explanation.

## Notes
- **A cache whose lookup key differs in kind from its insert key can never hit, and its size tells
  you nothing about that.** The disk prompt cache inserted on prompt+generated tokens and looked up
  on the prompt alone, hashed whole, so it filled 39.75GB over three weeks at a 100% miss rate and
  looked perfectly healthy from the outside — growing, evicting, within budget. `disk_cache_hits`
  was 0 the entire time and nobody read it. **Watch the hit counter of every cache, not its size**,
  and be suspicious of two "reads happened" signals that are not reads: file atimes move for
  backups and analysis sweeps, and a grep for a subsystem name matches its own log lines.
- **Do not build memory logic on `kern.memorystatus_vm_pressure_level`.** Measured 2026-08-08: when Xcode and Docker had already taken 8.64GB and the derived ceiling had dropped by the full 8.64GB, macOS was still reporting level 1. It only moved to 2 a whole sample later, once the compressor had spiked to 17.03GB. Availability is the leading indicator, compressor occupancy is the confirmation, and Apple's own level is the laggard — which is why all three are reported in `/v1/stats` and the advisory keys on the compressor.
- **`MLX_ENABLE_TF32` is parsed as an integer, so `MLX_ENABLE_TF32=true` means OFF.** Measured on q4 M=1024: `1` gives 8783 GFLOP/s, unset 8755, `0` gives 3184, and **`true` gives 3185** — identical to `0`. Anything non-numeric reads as disabled. `config.py` keeps a YAML bool and converts to `"1"`/`"0"` at the `Popen` boundary specifically so nobody can hit this by writing the value that looks correct.
- **An mlx-lm PR merges if and only if a maintainer reviews it.** Measured 2026-08-08 over 400 PRs, external authors only: **31 got a maintainer review and 30 merged (97%); 71 did not and 0 merged**. Community review does not substitute (0 of 3, and the reviewer pool includes pcuenca and Blaizzy). Requesting a reviewer is not an available lever — 6 of 386 external PRs have one pending and it needs write access. Diff size is the other gate: **401+ additions merged 0 of 14**, with 79 more sitting open in that class. The roster changed and the threads never noticed: angeloskath's last merge was 2026-07-09 and neither he nor awni has commented since 2026-07-20; **michalk8 and nastya236 do the merging now**. All three of Miguel's open PRs have `reviewers=[]`. **Decision 2026-08-08: no ping on any of them; do not re-propose it.** Full tables in gitignored `notes/mlxlm-merge-dynamics-2026-08-08.md`.
- `uv tool install --force <local-path>` can silently reuse a cached build and NOT pick up new local commits, even though the command reports success — confirmed by checking the installed file's actual content after a "successful" reinstall still showed the pre-fix code. Full fix requires `uv tool uninstall <pkg> && uv cache clean <pkg>` before reinstalling from a local path with uncommitted-upstream changes; `--force` alone is not sufficient.
- All local backends (mira-mlx/omlx/mlx-lm/vllm-mlx) share port 8080 — only one runs at a time by design. This means a stale/orphaned process from a previous backend can make `is_backend_ready()`'s health check pass spuriously (it just hits `/v1/models` on the shared port), masking a failed switch. See `feedback_backend_switch_self_stop.md` for the specific bug this caused.
- `pyproject.toml` stays tag-driven via `hatch-vcs` — the git tag IS the version, never hand-edit. `mira.yaml` is gitignored runtime config, not tracked (confirmed 2026-07-06, no leaked secrets — a local `mira.yaml` with a real token exists only on disk, never committed).
- Installer already automates nearly everything (uv, Python deps incl. mlx stack, disk/RAM preflight, mira.yaml bootstrap, optional tesseract/LaunchAgent via brew, doctor health check). The oMLX GUI step is the sole exception and is expected to stay manual indefinitely absent an oMLX-side scriptable install/model API.
- Tried `MLX_METAL_FAST_SYNCH=1` + `MLX_MAX_OPS_PER_BUFFER=50` + `MLX_MAX_MB_PER_BUFFER=50` (undocumented MLX Metal-backend env vars, found via `strings` on `libmlx.dylib` and confirmed against MLX's C++ source — `mlx/backend/metal/device.cpp`/`fence.cpp`) on mira-mlx (2026-07-18): `subprocess.Popen()` in `backend_manager.py` inherits the parent env with no override, so setting these on `server.py`'s own process before launch propagates to the mira-mlx subprocess. A/B benched against a plain baseline on **both** models mira-mlx runs — Ministral 3 14B and Qwen3.6-35B-A3B, isolated — **no measurable TTFT difference on either** (within ±20ms noise on the 5 non-agentic questions for both models; Qwen3.6's Q1 showed an ~800ms gap but that's within normal first-request-after-switch variance, not a consistent effect across Q2-Q5). Apparent wins on agentic questions were fully explained by tool-call-count variance between runs, not the env vars, on both models. Correctness unaffected either way on both. Not worth setting as a default; not pursued further.
