# KV-cache quantization in mlx-lm: three PRs and what they cannot do yet

Mira runs on a pinned `mabaeyens/mlx-lm` fork, so upstream's KV-quantization story
is directly Mira's. As of 2026-07-27 that story is spread across three open PRs by
three people, none of which has a maintainer review. This note records how they
divide, and more usefully, the two constraints that only showed up when somebody
tried to make them work together.

I am writing it down because both constraints are the kind that get rediscovered
expensively. One of them silently disables continuous batching on the default
server invocation.

## The three PRs

| PR | Author | Owns |
|---|---|---|
| [#1353](https://github.com/ml-explore/mlx-lm/pull/1353) | soobrosa | Server CLI flags (`--kv-bits`, `--kv-group-size`, `--quantized-kv-start`, `--max-kv-size`) and startup validation |
| [#1584](https://github.com/ml-explore/mlx-lm/pull/1584) | me | `RotatingKVCache.to_quantized()`, `RotatingQuantizedKVCache`, `BatchRotatingQuantizedKVCache`, plus `BatchGenerator` wiring |
| [#1618](https://github.com/ml-explore/mlx-lm/pull/1618) | PhilipJohnBasile | Cache capability API (`can_quantize_prompt_cache()` / `is_quantizable()`) and runtime validation |

The split settled on 2026-07-26 and everyone agreed to it in writing. #1353 is the
only one that is independently mergeable: after its refactor it touches
`server.py` and `tests/test_server.py` and nothing else, so it does not depend on
or conflict with the other two.

Before that refactor, #1353 caught `NotImplementedError` per layer and left
unquantizable layers at full precision without saying so. On Gemma 4 26B-A4B that
is 25 of 30 layers (25 `RotatingKVCache` against 5 `KVCache`, counted on the real
model rather than from the config), so `--kv-bits 8` delivered almost nothing and
reported success. It now probes every cache entry at load time and refuses to
start, naming the class and the count.

## Constraint 1: `--max-kv-size` and `--kv-bits` are incompatible in the sequential path

`make_prompt_cache(model, max_kv_size=N)` builds `RotatingKVCache(max_size=N, keep=4)`
at `cache.py:37`. The `keep` is hardcoded. #1584's `to_quantized()` raises on
`keep > 0` at `cache.py:565`, because sink tokens are not implemented in the
quantized rotating cache.

So that flag combination is unquantizable both before and after #1584, for any
model that does not provide its own `make_cache`. Gemma 4 only escapes because it
passes `keep=0`.

This was soobrosa's find, from writing the probe. I had shipped #1584 believing it
closed the rotating-cache gap, and it closes most of it, but not the path the
`--max-kv-size` flag documents.

**Update 2026-07-28: this may never need implementing.** soobrosa filed
[#1631](https://github.com/ml-explore/mlx-lm/issues/1631) and found that
`cache.py:37` is the only nonzero `keep` in the package, so the generic fallback
is the single producer of a shape two other subsystems refuse. If awni decides the
`keep=4` default is superseded rather than deliberate (he added it in #1015), a
one-line change there removes the case instead of anyone building sink tokens.
Verified the count at `ff1e837` and posted it: 20 `RotatingKVCache(` construction
sites package-wide, exactly one nonzero. Seventeen are in model files, five passing
`keep=0` explicitly (`cohere2`, `exaone4`, `exaone_moe`, `gemma3n`,
`gemma4_text`) and twelve omitting the argument entirely (`olmo3`,
`recurrent_gemma`, `gemma3_text`, `mellum`, `mimo_v2_flash`, `llama`,
`iquestloopcoder`, `step3p5`, `gpt_oss`, `afmoe`, `ministral3`, `baichuan_m1`).
The other two are `generate.py:1777` and `cache.py:1690`, both defaulting.

`cache.py:1690` is worth its own line: it sits in `BatchRotatingKVCache.extract()`,
which rebuilds a single sequence out of a batch as `RotatingKVCache(self.max_size)`
with no `keep` passed. Paired with `merge()` at `cache.py:1707`, which validates
`max_size` across caches and never looks at `keep`, a `keep=4` cache that goes into
a batch and comes back out returns as `keep=0`. Sink tokens are already dropped on
the batching path today, quantized or not.

## Constraint 2: it inverts on the batched path

`BatchGenerator._make_new_cache()` builds `RotatingKVCache(max_size=self.max_kv_size)`
with no `keep` argument, and the default is `keep=0` (`cache.py:426`). So after
#1584 the same flag combination quantizes fine there.

The startup probe is still right today, because #1353 forces every request through
the sequential path whenever `--kv-bits` is set, and sequential goes through
`make_prompt_cache`. But the probe's verdict is a property of one code path, not of
the flag, and the error message should not harden into claiming otherwise.

## Constraint 3: `--quantized-kv-start` defaults to 5000, and the batched path refuses it

This is the one nobody had raised on any of the three threads.

`--quantized-kv-start` defaults to `DEFAULT_QUANTIZED_KV_START`, which is 5000
(`generate.py:56`, `cache_prompt.py:14`). #1584's `BatchGenerator.__init__` raises
`NotImplementedError` when `kv_bits` is set and `quantized_kv_start != 0`, because
per-job caches are created once, empty, at insertion time, and there is no per-step
re-check for the offset threshold to ever trigger. Only immediate quantization
works on that path.

Which means a plain `mlx_lm.server --kv-bits 8`, no other flags, no `--max-kv-size`,
lands on a config the batched path cannot serve. Anyone narrowing the batching
disable by looking at `max_kv_size` alone would ship exactly that bug.

**Correction 2026-07-28: the raise is structural, not a missing re-check.** The
comment I wrote at `generate.py:1595` gives "no per-step re-check" as the reason,
and that undersells it badly enough that I nearly offered on #1618 to just remove
the raise. Checked before posting, and the offer would have been wrong three times
over. Per-job caches exist only until the batch forms:
`PromptProcessingBatch.__init__` calls `_merge_caches(caches)` at
`generate.py:1120` and collapses them into one cache per layer for the whole batch,
so afterwards there is no per-sequence object left to convert. Rows in a batch also
sit at different offsets, so a per-sequence threshold has nothing to attach to on a
shared cache. And a cache quantized mid-flight could not rejoin a batch anyway,
because `_merge_caches` dispatches on `caches[0][i].merge(...)` (`generate.py:867`)
and so needs one class across all jobs, while `QuantizedKVCache` (`cache.py:245`)
defines no `merge` at all.

Real deferral therefore means batch-level threshold semantics plus a
`BatchKVCache.to_quantized()`, and no plain `BatchQuantizedKVCache` exists (only
`BatchRotatingQuantizedKVCache`, `cache.py:1758`). Unimplemented rather than
forbidden, but a feature, not plumbing. Do not describe it as cheap.

## What the follow-up owes

#1353 currently disables batching outright whenever `--kv-bits` is set:

```python
if self.cli_args.kv_bits is not None:
    return False
```

Correct as things stand, since `BatchGenerator` on main takes no `kv_bits` at all,
so routing a quantized request there would ignore the quantization silently. I
argued for narrowing it to `kv_bits is not None and max_kv_size is None` and was
wrong.

After #1584 the parameters exist (it adds `kv_bits`, `kv_group_size` and
`quantized_kv_start` to `BatchGenerator.__init__`, quantizes in `_make_new_cache()`
while the cache is still empty, and quantizes externally supplied caches in
`insert_segments()`). The real narrowing is then closer to:

```python
if self.cli_args.kv_bits is not None and self.cli_args.quantized_kv_start != 0:
    return False
```

with `max_kv_size` not in the condition at all, per constraint 2. That is a
server-side change in soobrosa's file, so it belongs in a follow-up after both
#1353 and #1584 merge, not bolted onto a cache PR that is already +1157 across
four files with no review on it.

I committed to opening that follow-up on 2026-07-27. **PhilipJohnBasile claimed it
back the next day** and it is now tracked against #1618, on the argument that the
probe has to know which generator will actually serve a request rather than
validating one cache shape at load time, which makes it a capability-API concern
rather than a `server.py` one. That argument is sound and I conceded it rather than
contest the attribution. What I handed over instead were the two constraints above,
since both are properties of my file that his API would otherwise harden around.

The piece underneath all of this was sink-token support in
`RotatingQuantizedKVCache`, the only thing that fixes constraint 1 rather than
routing around it. soobrosa filed it as #1631 on 2026-07-28 and, as recorded under
constraint 1, it may be answered by deleting `keep=4` rather than by implementing
anything. That decision is awni's.

## Where the discussion is

- The position above was posted to #1353 on 2026-07-27:
  [comment 5088038211](https://github.com/ml-explore/mlx-lm/pull/1353#issuecomment-5088038211)
- The ordering agreement with PhilipJohnBasile is on #1618:
  [comment 5082713105](https://github.com/ml-explore/mlx-lm/pull/1618#issuecomment-5082713105)
- The original silent-skip objection is on #1353:
  [comment 5083332147](https://github.com/ml-explore/mlx-lm/pull/1353#issuecomment-5083332147)
- Conceding the narrowing follow-up, with the constraint-3 correction, is on #1618
  (2026-07-28):
  [comment 5100971172](https://github.com/ml-explore/mlx-lm/pull/1618#issuecomment-5100971172)
- The `keep` construction-site count is on #1631 (2026-07-28):
  [comment 5108906145](https://github.com/ml-explore/mlx-lm/issues/1631#issuecomment-5108906145)

A note on #1573, which is adjacent and easy to conflate: it is **not** fixed by any
of these. chrislyons' crash comes from mlx-vlm, which vendors its own 90KB copy of
`cache.py` carrying the identical `RotatingKVCache Quantization NYI`, and
`mlx_lm.server` has never had a `--kv-bits` flag at all. PhilipJohnBasile confirmed
that independently on 2026-07-28. The equivalent fix has to be filed against
`Blaizzy/mlx-vlm`, and as of this writing nobody has done so.

Everything in this note was checked against the two diffs and the local checkout
rather than recalled, which is how constraint 3 turned up. Line references are to
the fork at the time of writing.
