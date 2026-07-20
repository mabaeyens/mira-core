#!/usr/bin/env python3
"""A1/A2 coalescing ceiling on OUR reader, measured on the real 8bit shard.

Gate 2's premise is doramirdor's I/O model (mlx-lm#1438): collapsing an expert's
{weight,scales,biases} x {gate,up,down} into one contiguous read cuts
per-cold-expert I/O 2.25x (A1, 3 reads/expert) / 3.0x (A2, 1 read/expert). That
model was built against a reader doing open+seek+read per slice at QD1. OUR
reader already uses shared fds + os.pread (no open() per slice) and an 8-way
fetch pool, so a chunk of what his 9->1 op reduction buys is already banked.
This measures what is actually left on OUR I/O path.

Units. One MISS as the offload counters define it = one module (one projection)
for one expert = 3 preads (weight, scales, biases). A1 collapses those 3 into 1
and needs no dispatch change. A2 collapses all 9 slices of an expert (3
projections x 3 tensors) into 1 and DOES need fetch-at-dispatch, since the three
projections are separate modules with separate LRUs.

Cache discipline (the thing that makes or breaks this bench). F_NOCACHE stops a
read from POPULATING the page cache; it does not bypass pages already resident.
So re-reading the same real expert offsets across arms or runs silently turns
the second read into a RAM read and the comparison becomes cold-vs-warm. Both
arms therefore read never-before-touched random regions of the real shard, with
the real slice SIZES (one ~1MB weight + two ~32KiB scales/biases -- that size mix
is the whole point, since the tiny reads are pure latency floor). Bytes are
bytes: what is being measured is N scattered reads of size S versus 1 read of
size N*S at equal cache state, which is doramirdor's own proxy.

RESULT (2026-07-20, Qwen3.6-35B-A3B-8bit, M5/32GB) and the decision it drove.

    lever                      I/O speedup   decode @0.3   decode @0.45
    A1  3 slices -> 1 read        1.61x         +11.0%         +9.0%
    A2  9 slices -> 1 read        2.34x         +17.6%        +14.3%

Read share of the decode token, from the resident-fraction sweep in
oracle_prefetch_ceiling.py: 26.1% at fraction 0.3, 21.8% at 0.45. The marginal
per-miss wall is NOT constant across fractions (240us at 0.3->0.4, 258us at
0.4->0.45, 280us at 0.45->0.5): a larger resident set leaves less RAM for the
page cache, so the misses that remain are colder. Extrapolating a fixed wall
from the 0.3 anchor understates the high fractions by about two points.

Both levers require a coalesced side-file, which breaks the current design of
reading straight from the model's own safetensors shards (no repack step, no
second artifact to keep in sync per model, nothing to invalidate on re-quant).
At the RAM-aware fraction this build actually ships (~0.45) that costs the
no-repack property for +14.3%, so:

    DECISION: A2 is NO-GO. A1 is not a candidate at all (+9.0% is well under
    the bar).

REVISIT TRIGGER TESTED AND RETIRED (2026-07-20). The trigger above was "a model
far enough over DRAM to force a much smaller resident fraction, pushing the read
share back up". Ran it on gpt-oss-120b-MXFP4-Q8 (56.3GB expert table, 1.8x RAM,
mxfp4, different architecture):

    model                      read share   A1 I/O   A2 I/O   A2 end-to-end
    Qwen 8bit @0.45 (shipped)     21.8%     1.61x    2.34x       +14.3%
    gpt-oss @0.30                 39.3%     1.08x    1.29x        +9.7%

The read share rose as predicted and A2 still got WORSE. Coalescing is an
IOPS/latency lever and gpt-oss reads are already bandwidth-bound: its module is
4050K + 253K (one read = 94% of the bytes, scattered already 5.8 GB/s) where
Qwen's is 1024K + 32K + 32K (two tiny reads = pure per-op latency floor,
scattered only 2.5 GB/s). The two effects are ANTI-CORRELATED: a bigger model
forces a smaller fraction (raising read share, good) and has larger slices
(bandwidth-bound reads, bad), and the second wins. There is no "bigger model"
that rescues coalescing. Do not re-open on that reasoning.

Run against another model with:
    --model <repo> --module model.layers.N.mlp.experts.gate_proj --read-share X

Discussion and full numbers: ml-explore/mlx-lm#1438 (and PR #1588 for the
offload primitive itself).
"""
import argparse
import fcntl
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.inference.disk_expert_cache import DiskExpertCacheStore  # noqa: E402

MODEL = "mlx-community/Qwen3.6-35B-A3B-8bit"
F_NOCACHE = 48
# 'biases' are quant zero-points and are absent in mxfp4 (gpt-oss), where a
# module-miss is 2 slices rather than 3. slice_sizes() skips whatever is missing.
ATTRS = ["weight", "scales", "biases"]


def _open(path, nocache):
    fd = os.open(path, os.O_RDONLY)
    if nocache:
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        except OSError:
            pass
    return fd


def slice_sizes(store, module_path):
    """Real per-expert byte size of each attr for one projection module."""
    out = []
    for a in ATTRS:
        found = store._resolve(module_path, a)
        if found is None:
            continue
        _, _, meta = found
        start, end = meta["data_offsets"]
        out.append((end - start) // meta["shape"][0])
    return out


def measure(path, size, sizes, trials, nocache, rng):
    """Median wall for (a) len(sizes) scattered reads vs (b) 1 read of the same
    total bytes. Every read lands on a fresh random offset so neither arm can
    inherit the other's page-cache state."""
    total = sum(sizes)
    scattered, coalesced = [], []
    for _ in range(trials):
        offs = [rng.randrange(0, size - s - 1) for s in sizes]
        fd = _open(path, nocache)
        t0 = time.perf_counter()
        for off, nb in zip(offs, sizes):
            os.pread(fd, nb, off)
        scattered.append((time.perf_counter() - t0) * 1e6)
        os.close(fd)

        off = rng.randrange(0, size - total - 1)
        fd = _open(path, nocache)
        t0 = time.perf_counter()
        os.pread(fd, total, off)
        coalesced.append((time.perf_counter() - t0) * 1e6)
        os.close(fd)
    return statistics.median(scattered), statistics.median(coalesced), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--module", default="model.layers.10.mlp.switch_mlp.gate_proj")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--read-share", type=float, default=0.261,
                    help="measured cold-read share of the decode token at fraction 0.3 "
                         "(oracle_prefetch_ceiling.py sweep; use 0.218 for fraction 0.45)")
    args = ap.parse_args()

    rng = random.Random(0)
    from huggingface_hub import snapshot_download
    store = DiskExpertCacheStore(Path(snapshot_download(args.model)))

    sizes1 = slice_sizes(store, args.module)
    base = args.module.rsplit(".", 1)[0]
    sizes9 = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        sizes9 += slice_sizes(store, f"{base}.{proj}")

    shard = store._resolve(args.module, "weight")[0]
    path = str(store.model_path / shard)
    size = os.path.getsize(path)

    share = args.read_share
    print(f"model : {args.model}")
    print(f"shard : {shard}  ({size/1e9:.1f} GB)")
    print(f"slices: A1 {[f'{s/1024:.0f}K' for s in sizes1]}  "
          f"A2 {len(sizes9)} slices, {sum(sizes9)/1024:.0f} KiB total")
    print(f"read share of decode token: {share*100:.1f}%\n")

    for label, nocache in (("COLD (F_NOCACHE, fresh offsets)", True),
                           ("warm-ish (page cache allowed)", False)):
        print(f"{label}:")
        for tag, sizes, note in (
            ("A1  3 slices -> 1 read", sizes1, "no dispatch change"),
            ("A2  9 slices -> 1 read", sizes9, "needs fetch-at-dispatch"),
        ):
            sc, co, total = measure(path, size, sizes, args.n, nocache, rng)
            ratio = sc / co if co else float("nan")
            e2e = 1.0 / ((1 - share) + share / ratio) if ratio > 0 else 1.0
            print(f"  {tag:<24} scattered {sc:7.1f} us   coalesced {co:7.1f} us   "
                  f"I/O {ratio:5.2f}x   decode {e2e:.3f}x ({(e2e-1)*100:+5.1f}%)   "
                  f"[{total/1024:.0f} KiB, {note}]")
        print()


if __name__ == "__main__":
    main()
