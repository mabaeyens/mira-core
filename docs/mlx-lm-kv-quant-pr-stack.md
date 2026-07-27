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

I committed to opening that follow-up. It is gated on two merges that have zero
maintainer reviews between them.

The unaddressed piece underneath all of this is sink-token support in
`RotatingQuantizedKVCache`. That is the only thing that fixes constraint 1 rather
than routing around it, and I am not attempting it in #1584. It deserves its own
issue.

## Where the discussion is

- The position above was posted to #1353 on 2026-07-27:
  [comment 5088038211](https://github.com/ml-explore/mlx-lm/pull/1353#issuecomment-5088038211)
- The ordering agreement with PhilipJohnBasile is on #1618:
  [comment 5082713105](https://github.com/ml-explore/mlx-lm/pull/1618#issuecomment-5082713105)
- The original silent-skip objection is on #1353:
  [comment 5083332147](https://github.com/ml-explore/mlx-lm/pull/1353#issuecomment-5083332147)

Everything in this note was checked against the two diffs and the local checkout
rather than recalled, which is how constraint 3 turned up. Line references are to
the fork at the time of writing.
