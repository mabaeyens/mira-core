# Draft reply to soobrosa on mlx-lm #1353

Status: **POSTED 2026-07-27**, https://github.com/ml-explore/mlx-lm/pull/1353#issuecomment-5088038211
Context: soobrosa refactored #1353 on 2026-07-26
(commits 62d092d, f669e9e) in response to my comment, dropped the silent skip,
added a load-time probe, and asked me to choose where the batching narrowing
lands (#1584 or a follow-up).

Every claim below was verified against the two diffs and the local checkout, not
from memory. Verification notes at the bottom.

---

You are right and I was wrong on the batching condition. On main today
`BatchGenerator.__init__` takes no `kv_bits`, so my narrowing would have sent a
quantized request down a path that ignores the quantization, which is the exact
failure I came here to complain about. Blanket disable is correct as this PR
stands, and naming the cost in `--help` is the right trade.

One correction going the other way, in case you only read the `cache.py` half of
#1584: the `generate.py` half already adds `kv_bits`, `kv_group_size` and
`quantized_kv_start` to `BatchGenerator.__init__`, quantizes in `_make_new_cache()`
while the cache is still empty, and quantizes externally supplied caches in
`insert_segments()`. So the parameters you are waiting for arrive with #1584
itself, not after it.

On where the narrowing goes: **a follow-up, after both are in.** #1584 is already
+1157 across four files with no review on it, and bolting `server.py` onto a cache
PR would make the thing harder to review for no gain, since the wiring cannot be
tested until your file lands anyway. I will open the follow-up once #1353 and
#1584 are both merged, or earlier if a maintainer would rather see it stacked.

Two things the follow-up has to get right, both of which I only found while
checking your probe.

The first is `--quantized-kv-start`. It defaults to `DEFAULT_QUANTIZED_KV_START`,
5000, in your argparse block, and #1584's `BatchGenerator` refuses
`quantized_kv_start != 0` when `kv_bits` is set. Per-job caches are built once,
empty, at insertion time, so there is no per-step re-check for the offset
threshold to ever trigger; only immediate quantization works on that path. Which
means a plain `mlx_lm.server --kv-bits 8`, no other flags, would hit that raise if
the narrowing only looked at `max_kv_size`. The real condition is closer to
`kv_bits is not None and quantized_kv_start != 0`, with `max_kv_size` not in it at
all, and here is why.

The second is your `keep=4` catch, which is correct and which I confirmed
(`cache.py:37` hardcodes `keep=4`, and #1584 raises on `keep > 0`, so
`--max-kv-size` with `--kv-bits` is unquantizable in the sequential path before
and after my PR). But it inverts on the batched path.
`BatchGenerator._make_new_cache()` builds `RotatingKVCache(max_size=...)` with the
default `keep=0`, so after #1584 that combination quantizes fine there. Your probe
is right today, because `--kv-bits` forces sequential and sequential goes through
`make_prompt_cache`. The moment the narrowing lands, the probe would refuse to
start on a config the batched path can actually serve. Not asking you to change
anything now (it would be dead code until the wiring exists), just flagging it so
the `keep=4` assumption does not get baked into the error message as a permanent
truth about the flag.

Sink tokens in `RotatingQuantizedKVCache` are the fix for the sequential half and
I am not attempting them in #1584. Worth its own issue at some point.

Nothing here should hold you up. This PR is two files now, does not touch
`cache.py` or `generate.py`, and closes #1043, #615 and half of #1308 on its own.
It should go in whenever a maintainer gets to it, ahead of mine if that is what
happens.

---

## Verification notes (local, not for posting)

- `--quantized-kv-start` default: `gh pr diff 1353` argparse block, line ~142,
  `default=DEFAULT_QUANTIZED_KV_START`; value 5000 at `mlx_lm/generate.py:56` and
  `mlx_lm/cache_prompt.py:14`.
- #1584 `BatchGenerator.__init__` raise: `gh pr diff 1584`, `generate.py` hunk at
  1587, `if kv_bits is not None and quantized_kv_start != 0`.
- #1584 adds the three params: same hunk, `generate.py +58 -9`.
- `_make_new_cache()` uses `RotatingKVCache(max_size=self.max_kv_size)` with no
  `keep`, and `RotatingKVCache.__init__(self, max_size, keep=0)` at
  `cache.py:426`.
- `keep=4`: `cache.py:37`. #1584's raise: `cache.py:565`.
- #1584 size: +1157 -14 across 4 files, 0 reviews as of 2026-07-26T12:34Z.
