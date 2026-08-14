# Backlog

## Pending

### Confirm the overnight-eviction culprit from the sampler, then decide if any tuning is warranted
- The `notes/mem-samples.log` sampler is running (started 2026-08-13). After a night or two, read it
  and check which process's footprint climbs alongside the compressor at the moment of the next
  eviction. **If external (Xcode/lldb/browser), no model or config change is warranted** — the fix is
  behavioural (quit Xcode when idle). If it's Mira's *own* cache growth, the lever is the engine's
  prompt-cache cap (`mira_mlx_server.py:2114`, defaults to 12G) or the context window — trim peak
  footprint on this box without changing the model or its quality. Do not touch anything until the log
  says which. Stop the sampler with `pkill -f notes/mem-sampler.sh` once the question is answered.

### ~~Batched quantized KV kills the engine on any GQA model~~ — FIXED 2026-08-12, kept for the lesson
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
  `/tmp/claude_repro_qsdpa_mask.py`, identical error string.
- **Whose bug: the trigger is ours, the fix is upstream's, and nothing was deleted.** Worth checking
  the ownership before writing any code, because it changed the answer completely. (a) The crashing
  cache class `BatchRotatingQuantizedKVCache` **does not exist upstream** — it arrived with
  `bbd8496`, the commit behind PR #1584, so the batched quantized path is Mira-side. (b) The function
  that crashes is upstream's, and `base.py` differs from upstream by exactly two lines with the fork
  on the lacking side. (c) Those two lines are the fix, and upstream merged them on **2026-07-09** as
  `a790972`, *"Fix broadcast crash in quantized SDPA with GQA + batched padding mask (batch >= 2)"*
  (PR #1467) — `if n_repeats > 1 and mask.ndim > 3: mask = mx.expand_dims(mask, -3)`, plus a test.
  The pin is 24 commits behind and that is one of them; the installed copy contains **zero**
  occurrences of the guard. So the fork branched before #1467 landed and has carried the pre-fix
  version ever since. **This is the failure mode a pin exists to cause: it freezes the bugs along
  with the API.**
- **PR #1584 was never affected, contrary to what this entry first said.** `git branch --contains
  a790972` lists `kv-cache-quant-batching`: the PR branch picked the fix up through its merge of
  `main` on 2026-07-09 and is only 3 commits behind upstream. **The exposure was Mira's alone**,
  because the pin branches (`mira-core-pin`, `mira-core-pin-vision`) are the two that never took it.
  Checking which branches contain a commit costs one command and would have saved a wrong claim —
  "the PR that added the feature must be the PR that broke it" was an assumption, not a finding.
- **FIXED 2026-08-12, and production is verified.** `a790972` cherry-picked onto
  `mira-core-pin-vision` as `9721b95` (clean, original authorship kept, upstream's own regression
  test came with it and passes). Pushed as a fast-forward — no history rewrite, the pin branch is
  never force-pushed. `pyproject.toml` bumped `291a61a` → `9721b95` and the venv resynced; the
  installed `mlx_lm/models/base.py` now carries the guard, checked rather than assumed.
  **End-to-end check on the restarted server: three concurrent ~1,600-word requests — the same shape
  as the 1,629-token crash — all answered, zero broadcast or `engine loop died` lines in the engine
  log.** One trap worth recording: `uv sync` from a worktree silently re-points the editable
  `mira-core` install at that worktree, so production would have started importing branch code.
  Restored with `uv pip install -e <main checkout> --no-deps`; check `_editable_impl_mira_core.pth`
  after any sync run from a worktree.
- **PR #1584 was also rebased onto current upstream `main`** (5 commits, linear, merge commit
  dropped, no conflicts; `tests/test_prompt_cache.py` + `tests/test_models.py` = 104 passed).
  **Local only** — pushing it force-pushes an open public PR, which is a separate decision. The
  pre-rebase tip is tagged `pre-rebase-1584-2026-08-12` (`443dd99`).
- Full write-up in gitignored `notes/kv-quant-batched-mask-crash-2026-08-12.md`; the pin-move
  reasoning is in `docs/mlx-lm-pin.md`.

### `MAX_THINKING_TOKENS` — measured 2026-08-12, and 2048 does not bind either
- **The code default is now 2048; the live `mira.yaml` is deliberately still 8192.** Changing a
  runtime config is my call, and the measurement below is the argument for it rather than a
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
    **2026-08-14: caught the opposite — a false positive.** The cause classifier keyed
    `external_pressure` on the OS pressure level alone, but compressing Mira's own ~20GB idle model
    is itself enough to raise pressure to warn, so the treadmill fired the notification seven times
    in one morning with other apps holding under 1GB. Fixed: the cause now attributes the compressor
    per-process (`compressor_bytes - self_compressed_bytes`) and stays `idle_reclaim` unless the
    non-Mira remainder clears `EXTERNAL_COMPRESSED_MIN_BYTES`. Lands on the next engine restart.
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

### To do
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

     ~~**Needs a decision before building.**~~ **ALL THREE SHIPPED 2026-08-11**, I approved
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
     conversations feeds the model five turns of `!`. I read them in the app on 2026-08-11
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
     behind the poisoned conversations above surviving a retry. The fix had to be mira-core's rather
     than the client's — the client cannot un-save what it already sent.

     **FIXED 2026-08-12 on the server side.** `/chat` takes `retry: bool` (default false); when set
     it calls the new `db.drop_last_turn()` and trims `orch.conversation_history` back to before the
     last user message, under the conversation lock and before the turn starts. **Both stores**,
     because the database is what a later session reloads and the in-memory history is what this
     turn prompts with — fixing only one would have looked correct in a test and wrong in use. It
     deletes from the last `user` row forward rather than a fixed two rows, since a turn can persist
     tool messages too. Off by default on purpose: asking the same question twice deliberately is
     legitimate and must not silently delete history. **Verified live**, not only unit-tested: three
     sends on one conversation gave 2 rows, then 4 — the control, proving the append still
     reproduces without the flag — then 4 again rather than 6. 12 tests in
     `tests/test_retry_replaces_turn.py`; 729 in the suite. Harness: `notes/retry_live_check.py`.
     **The app half is not done.** Until mira-apps' `resendLast()` sends `retry=true`, nothing
     changes for a real user — spec in `mira-apps/specs/retry-replaces-the-failed-turn.md`.
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
- **Score the 2026-08-08 agentic bench.** `docs/bench-archive/bench-results-2026-08-08.md` has timings and tool
  traces for Q6–Q12 filled in and the quality column empty. Less urgent than it was:
  `scripts/bench_eval.py` now scores these automatically, so this column is only for judgement the
  harness deliberately does not make.

### Documentation leftovers from the 2026-08-12 sweep
- ~~**The legacy `.github/ISSUE_TEMPLATE.md` is dead weight now that a directory shadows it.**~~
  **Deleted with approval 2026-08-12** in mira-apps `caad472`. Only mira-apps ever had one — the
  earlier note here said "one per repo" and that was wrong, mira-core has only ever had the
  directory form. Its content was not lost: `bug_report.md` was rewritten from it in `ef6e20b`.
- **mira-apps has no feature request template, and now nothing else to fall back on.** The deleted
  file was bug-and-feature both, opening with a Type selector; `bug_report.md` is bug-only by
  design. So a feature request or a question currently lands on a form asking for steps to reproduce
  and an expected-vs-actual. Either a second template or a `config.yml` with a discussions link
  closes it — small, and worth doing before anyone files the next one.
- **The three settings the sweep made configurable for the first time deserve a look now that
  someone can actually set them.** `shell_sandbox`, `shell_sandbox_allow_network` and
  `url_fetch_allow_private` were read by `config.py` and absent from `mira.yaml.example`, so their
  non-default paths have never been exercised by anyone deliberately choosing them. Documenting a
  security control is not the same as having reviewed what happens when it is switched off.
- **`scripts/benchmark.py` and `scripts/bench_standard.py` still carry ollama** — fifteen references
  in the first. ollama was retired 2026-08-01, so the harness can name a backend the server cannot
  start. Left alone deliberately during the doc pass: it is code, not prose, and belongs in its own
  small change rather than folded into a markdown commit.
- **`docs/moe-offload-lazy-load-design.md` is a design doc for something that shipped.** The measured
  outcome lives in the case study; this is the plan that preceded it, and the two now overlap. Fold
  whatever is still true into the case study rather than maintaining both.

### Small, no decision needed
- **Memory-state notifications should only fire when I can act, or when they explain something
  I caused.** Asked for directly on 2026-08-11, after the log showed **302 `advisory → evicted`
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
    @pierre427 would only grow it. I asked @angeloskath about splitting on 2026-07-26 and got
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
- **An mlx-lm PR merges if and only if a maintainer reviews it.** Measured 2026-08-08 over 400 PRs, external authors only: **31 got a maintainer review and 30 merged (97%); 71 did not and 0 merged**. Community review does not substitute (0 of 3, and the reviewer pool includes pcuenca and Blaizzy). Requesting a reviewer is not an available lever — 6 of 386 external PRs have one pending and it needs write access. Diff size is the other gate: **401+ additions merged 0 of 14**, with 79 more sitting open in that class. The roster changed and the threads never noticed: angeloskath's last merge was 2026-07-09 and neither he nor awni has commented since 2026-07-20; **michalk8 and nastya236 do the merging now**. All three of my open PRs have `reviewers=[]`. **Decision 2026-08-08: no ping on any of them; do not re-propose it.** Full tables in gitignored `notes/mlxlm-merge-dynamics-2026-08-08.md`.
- `uv tool install --force <local-path>` can silently reuse a cached build and NOT pick up new local commits, even though the command reports success — confirmed by checking the installed file's actual content after a "successful" reinstall still showed the pre-fix code. Full fix requires `uv tool uninstall <pkg> && uv cache clean <pkg>` before reinstalling from a local path with uncommitted-upstream changes; `--force` alone is not sufficient.
- All local backends (mira-mlx/omlx/mlx-lm/vllm-mlx) share port 8080 — only one runs at a time by design. This means a stale/orphaned process from a previous backend can make `is_backend_ready()`'s health check pass spuriously (it just hits `/v1/models` on the shared port), masking a failed switch. See `feedback_backend_switch_self_stop.md` for the specific bug this caused.
- `pyproject.toml` stays tag-driven via `hatch-vcs` — the git tag IS the version, never hand-edit. `mira.yaml` is gitignored runtime config, not tracked (confirmed 2026-07-06, no leaked secrets — a local `mira.yaml` with a real token exists only on disk, never committed).
- Installer already automates nearly everything (uv, Python deps incl. mlx stack, disk/RAM preflight, mira.yaml bootstrap, optional tesseract/LaunchAgent via brew, doctor health check). The oMLX GUI step is the sole exception and is expected to stay manual indefinitely absent an oMLX-side scriptable install/model API.
- Tried `MLX_METAL_FAST_SYNCH=1` + `MLX_MAX_OPS_PER_BUFFER=50` + `MLX_MAX_MB_PER_BUFFER=50` (undocumented MLX Metal-backend env vars, found via `strings` on `libmlx.dylib` and confirmed against MLX's C++ source — `mlx/backend/metal/device.cpp`/`fence.cpp`) on mira-mlx (2026-07-18): `subprocess.Popen()` in `backend_manager.py` inherits the parent env with no override, so setting these on `server.py`'s own process before launch propagates to the mira-mlx subprocess. A/B benched against a plain baseline on **both** models mira-mlx runs — Ministral 3 14B and Qwen3.6-35B-A3B, isolated — **no measurable TTFT difference on either** (within ±20ms noise on the 5 non-agentic questions for both models; Qwen3.6's Q1 showed an ~800ms gap but that's within normal first-request-after-switch variance, not a consistent effect across Q2-Q5). Apparent wins on agentic questions were fully explained by tool-call-count variance between runs, not the env vars, on both models. Correctness unaffected either way on both. Not worth setting as a default; not pursued further.
- **A fact that must stay current belongs in exactly one file; everywhere else links to it.** The
  mlx-lm pin SHA was written out in four places and went stale in three of them within hours of
  moving — `docs/packaging.md` was still showing `branch = "mira-mistral-tool-call-fix"`, a branch
  that no longer exists, in precisely the form `docs/mlx-lm-pin.md` was written to forbid. That
  document had predicted the failure in its own closing paragraph and then suffered it, which is the
  useful part: a warning is not a mechanism. **The SHA now appears in `pyproject.toml` and
  `docs/mlx-lm-pin.md` and nowhere else**, stated as a rule inside the doc so the next copy is a
  visible violation rather than an oversight. Same rule now applies to settings: `docs/configuration.md`
  is the one reference, README and `docs/architecture.md` point at it instead of keeping short lists
  that silently fall behind `config.py`.
- **`_get()` in `core/config.py` is the definition of what a setting is.** Anything read through it
  is configurable and must appear in `docs/configuration.md`; anything not read through it is a
  constant, however much it looks like a knob. Written down because the drift it caused was invisible
  from either end — seven keys the code honoured had never been offered to a user, three of them
  security controls, while `docs/architecture.md` advertised `reranker_model` as source-only weeks
  after it became a config field.
- **GitHub silently ignores `.github/ISSUE_TEMPLATE.md` once a `.github/ISSUE_TEMPLATE/` directory
  exists.** No warning, no precedence note in the repo, nothing in the UI. This is how a
  well-written template in mira-apps went unseen for its entire life while GitHub's unedited default
  — browser, iPhone 6, "Version [e.g. 22]" — collected every bug report for a macOS and iOS app.
  If a template ever looks like it is not being used, check for the directory before rewriting it.
