# Sizing the resident expert set for offload

`enable_offload(resident_slots=..., fetch_fn=...)` keeps a fixed number of
experts resident and pages the rest from disk on a router miss. The resident
count is the one knob that trades throughput for memory, and it is easy to set
wrong in a way that only shows up as an out-of-memory crash mid-run. This note
covers how to size it.

## The resident set is not a free knob

`resident_slots` (or the `resident_fraction` a caller derives it from) has to be
sized against two moving quantities, not set once as a constant:

- **The expert table.** A fraction that is safe on a 256-expert table can OOM on
  a table with more or larger experts, because the resident set and the
  transient gather stack both grow with per-expert size.
- **Available RAM.** The resident cache, the activations, and the page cache all
  share unified memory. A fraction chosen on a 32 GB machine does not carry to a
  larger table on the same machine: `lBroth`'s validation on Qwen3-235B (a
  ~118 GB table at ~2.6x their RAM) OOM'd at fraction 0.3, a fraction that is
  fine on smaller tables.

The practical rule: pick the resident set so that the resident bytes plus the
worst-case transient gather stack plus room for activations and page cache all
fit, and re-derive it whenever the model or the machine changes. It is not a
portable constant.

## Budgeting in bytes, not slots

`resident_slots` is a count, which is the wrong unit once experts are not all the
same size: heterogeneous experts, or a mixed-precision table where different
experts carry different quantization. For those, pass `resident_bytes` instead:

```python
switch_linear.enable_offload(resident_bytes=6 * 1024**3, fetch_fn=fetch_fn)
```

The slot count is derived by measuring one expert through the same `fetch_fn` a
cold miss uses and floor-dividing the budget by it, so the figure is the true
resident footprint of a cache entry rather than an estimate off the on-disk
layout. If both `resident_slots` and `resident_bytes` are given, the tighter of
the two wins, so a byte budget can only lower a slot count, never raise it past
what the caller already allowed.

## Reserve page-cache headroom

When you budget bytes, do not spend all of free RAM on the resident set. The
buffered reads that `fetch_fn` issues on a miss go through the kernel page cache,
and squeezing that cache slows every miss. On this machine, per-miss cold reads
rose from about 240 to 280 µs as the resident fraction grew and the page cache
shrank, a ~15% penalty that scales with miss rate, so it lands hardest exactly
in the diverse-prefill regime where the miss rate is already high.

The direction is what matters here: leave explicit headroom for the page cache
rather than sizing the resident set right up to free RAM. A concrete reserve
figure is rig-, block-size-, and filesystem-specific; measure your own miss
latency across a couple of resident fractions and reserve enough that it stays
flat, rather than transplanting a number from another setup.

> The larger throughput-collapse figure that circulated during the #1438
> discussion (an ~800→180 MB/s drop under page-cache pressure) traces to a
> single-rig, unstated-block-size measurement in a third-party source, not to
> anything measured here, and should not be cited as a general rule. The
> 240→280 µs direction above is the measurement this note stands on.
