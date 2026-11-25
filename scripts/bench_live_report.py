#!/usr/bin/env python3
"""Score and compare two live-with bench runs (see bench_live.py).

    python scripts/bench_live_report.py A.jsonl B.jsonl [--corpus DIR] [--pairs OUT.json]

Deterministic tier: exact/contains match for code, long and unit4 items, plus a
leak check on unit4 (fetched URLs or search queries that look like the benchmark's
answer sources). Judged tier (trip, session) is exported as blind A/B pairs for the
Judge, an independent read, and an adversarial agent: this script does not judge.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

LEAK = re.compile(r"huggingface\.co/datasets/gaia-benchmark|gaia-benchmark|agents-course.*unit4|"
                  r"/datasets/.*gaia", re.I)
SETS = {"code": "code_set.jsonl", "long": "long_set.jsonl", "unit4": "unit4_set.jsonl",
        "trip": "trip_set.jsonl", "session": "session_set.jsonl"}


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def norm(s):
    return re.sub(r"[^a-z0-9.\- ]", "", str(s).lower()).strip()


def correct(item, row):
    text = row.get("content") or ""
    if item["set"] in ("code", "long"):
        return all(norm(e) in norm(text) for e in item["expected_contains"])
    if item["set"] == "unit4":
        ref = norm(item["reference"])
        return bool(ref) and ref in norm(text)
    return None


def leaked(row):
    blob = " ".join(row.get("fetch_urls", []) + row.get("search_queries", []))
    return bool(LEAK.search(blob))


def stats(rows):
    w = [r["wall_ms"] / 1000 for r in rows if r.get("wall_ms")]
    t = [r["ttft_ms"] / 1000 for r in rows if r.get("ttft_ms")]
    k = [r["eval_tokens"] for r in rows if r.get("eval_tokens")]
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(len(v) * p))] if v else None
    return {"n": len(rows), "wall_p50": q(w, .5), "wall_p90": q(w, .9), "ttft_p50": q(t, .5),
            "tok_mean": statistics.mean(k) if k else None,
            "errors": sum(1 for r in rows if r.get("error")),
            "peak_gb": max((r.get("peak_footprint_gb") or 0 for r in rows), default=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--corpus", default="notes/live_with_2026-11-25")
    ap.add_argument("--pairs", default="")
    args = ap.parse_args()
    corpus = Path(args.corpus)
    items = {}
    for fn in SETS.values():
        for line in (corpus / fn).open():
            it = json.loads(line)
            items[it["id"]] = it
    runs = {"A": load(args.a), "B": load(args.b)}

    for name, rows in runs.items():
        print(f"\n== {name} ({args.a if name == 'A' else args.b})")
        for s in SETS:
            sub = [r for r in rows.values() if r["set"] == s]
            if not sub:
                continue
            st = stats(sub)
            line = f"{s:8} n={st['n']:2} wall p50={st['wall_p50']:.0f}s p90={st['wall_p90']:.0f}s " \
                   f"ttft p50={st['ttft_p50'] or 0:.0f}s tok={st['tok_mean'] or 0:.0f} err={st['errors']}"
            sc = [correct(items[r["id"]], r) for r in sub if r["id"] in items]
            if sc and sc[0] is not None:
                line += f" correct={sum(1 for x in sc if x)}/{len(sc)}"
            if s == "unit4":
                line += f" leaks={sum(leaked(r) for r in sub)}"
            print(line)
        print(f"peak footprint {stats(list(rows.values()))['peak_gb']:.1f} GB")

    # blind pairs for the judged sets
    rng = random.Random(20261125)
    pairs = []
    for iid, it in items.items():
        if it["set"] not in ("trip", "session"):
            continue
        a, b = runs["A"].get(iid), runs["B"].get(iid)
        if not (a and b):
            continue
        swap = rng.random() < 0.5
        x, y = (b, a) if swap else (a, b)
        pairs.append({"id": iid, "prompt": it["prompt"], "rubric": it["rubric"],
                      "first": x["content"], "second": y["content"],
                      "_key": {"first": "B" if swap else "A", "second": "A" if swap else "B"}})
    if args.pairs:
        Path(args.pairs).write_text(json.dumps(pairs, indent=1))
        print(f"\nwrote {len(pairs)} blind pairs to {args.pairs} (strip _key before judging)")


if __name__ == "__main__":
    main()
