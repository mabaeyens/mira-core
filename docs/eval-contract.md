# Mira eval contract

Why this exists: my evals have an observer-effect problem. Batch size changes the
generated tokens (23/24 prompts differ batch-1 vs batch-4), TF32 and M5 batched
attention flip outputs, and turning on MTP is distribution-preserving but not
bit-identical. So the *apparatus* I measure with is entangled with the result, and
the trap is optimizing against a fast proxy, shipping it, and having production (a
different apparatus) behave differently. This contract pins the apparatus, defines
what "quality" means as behavior, and gives one decision rule for shipping. It
formalizes the discipline already in `scripts/bench_eval.py` (deterministic tier
first, judge advisory until a noise floor earns it veto power) and adds the
two-instrument split and the behavioral gate.

## 1. Two instruments, strict roles, never mixed

| Instrument | Apparatus | Job | What I trust |
|---|---|---|---|
| **Proxy** (fast) | no-think, pinned concurrency, fixed subset | rank candidate levers cheaply | the *relative order* only, never the absolute number |
| **Gate** (slow) | production's exact batch / TF32 / MTP / sampling / model, full set, thinking as shipped | the ship / no-ship decision | the number, and only this one |

`scripts/eval_gpqa_fast.py` is the proxy. The gate is the same runner with
`--gate` (thinking on, full 198, concurrency pinned to the production decode
regime). One runner, two modes, one recorded apparatus block, so the apparatus
cannot drift by accident.

## 2. Trust deltas, not absolutes

The absolute score wobbles with the apparatus; the delta between A and B measured
back-to-back under one held-fixed apparatus does not. Every comparison is paired
and interleaved under a single apparatus (as the sort-threshold A/B already did).
A lone absolute score is a data point, never a verdict.

## 3. Prove the proxy does not lie

Before I trust the proxy to filter levers, and again whenever the model or engine
apparatus changes: run ~5 points through BOTH proxy and gate and confirm the
proxy's *ranking* predicts the gate's *ranking*. If a lever wins on the proxy and
the gate disagrees, the proxy's apparatus is hiding the effect that matters — fix
the proxy, do not ship the lever. This is the same discipline as
`bench_eval.py --validate-judge`, applied to the proxy↔gate relationship.

## 4. Measure behavior, not tokens

Token-level jitter is where the observer effect lives; aggregate behavioral
metrics are nearly immune to it. "Did it truncate?" over n=100 is stable while
individual tokens flip. So the gate is behavioral pass/fail, not brittle
string-match, and I never diagnose from a single output — always the aggregate.

## 5. What "quality" means (the goal the levers move toward)

North star: **a real conversation turn is reliably complete, coherent, and
on-instruction.** Not "win GPQA." GPQA is a floor guard, not the target.

| Metric | Kind | Threshold |
|---|---|---|
| **Completeness** — no truncated-thinking turns (was 8/28 on 2026-08-11) | hard invariant | 0 failures |
| **Non-degeneration** — no empty / looping turns | hard invariant | 0 failures |
| **Coherence + instruction-following** — judge score on a fixed real-turn set drawn from `conversations.db` | target metric | ≥ baseline in `docs/quality-baseline.md` |
| **Reasoning floor** — GPQA/MATH at gate apparatus | guard | not regressed below baseline band |
| **Safety** (Q14/Q15/Q16 in `bench_questions.yaml`) | hard invariant | pass/fail, never averaged in |

The hard invariants come from real transcripts, not the corpus JSONL (which hides
real failures). Read `conversations.db`.

## 6. The decision rule

> A lever ships only if it moves a **behavioral metric** by a **paired delta**
> under the **gate apparatus** that clears the measured noise band, and does not
> trip a hard invariant or the reasoning-floor guard. The proxy only decides which
> levers are worth running through the gate.

Observer-effect-proof by construction: apparatus fixed and equal to production,
signal is a delta not an absolute, metric is behavioral so token jitter cannot
fake it.

## 7. Apparatus is recorded, always

Every run records: model, MTP on/off + draft settings, kv_bits, `MLX_ENABLE_TF32`,
concurrency (= effective batch), think mode, sampling params, subset id + hash,
git SHA. A score without its apparatus block is uninterpretable and is discarded.
This is "report batch size with the scores," made mandatory.

## 8. Cost discipline

- Proxy: minutes. Runs as often as I like; filters candidates.
- Gate: reserved for a ship decision, on a fresh pinned engine (production stopped
  first, per the bench protocol).
- Judge scoring is offline against stored jsonl (`bench_eval.py`), so it is free to
  repeat and never on the critical path.
