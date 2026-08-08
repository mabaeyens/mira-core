#!/usr/bin/env python3
"""Measure how much a score moves between runs of the SAME build.

The gate treats judged deltas as advisory until this number exists, and the
reason is that a gate on a noisy signal fires on nothing, gets muted, and takes
the harness down with it. So the floor has to be measured rather than assumed,
and it has to be measured the only way it can be: run the same build more than
once and look at what moved anyway.

Usage:
    python scripts/bench_noise.py run1.jsonl run2.jsonl run3.jsonl

All runs must carry the same label, since a floor derived across models measures
the models. Questions whose captured TRUTH differs between runs are reported
separately and excluded from the floor: a probe that changed, or a workspace that
changed under it, is not the model being unstable.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_eval as be  # noqa: E402


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if len(paths) < 2:
        print("need at least two runs of the same build")
        return 2
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("no such run: " + ", ".join(str(p) for p in missing))
        return 2

    questions = be.load_questions()
    judge = be.Judge(be.DEFAULT_JUDGE_URL)

    labels, per_run, truths = set(), [], defaultdict(list)
    for p in paths:
        records = be.load_run(p)
        labels.add(next((r.get("model") for r in records if r.get("model")), "unknown"))
        for r in records:
            truths[r["id"]].append(r.get("truth"))
        scores = {s.qid: s for s in be.score_run(records, questions, judge, 1)}
        per_run.append((p.name, scores))
        print(f"scored {p.name}")

    if len(labels) > 1:
        print(f"\nREFUSING: runs carry different labels {sorted(labels)}. A floor "
              f"measured across models measures the models.")
        return 2

    qids = sorted({q for _, s in per_run for q in s})
    moved_t1, moved_judged, shifted_truth, stable = [], [], [], []

    for qid in qids:
        t1 = [s[qid].tier1 for _, s in per_run if qid in s]
        jd = [s[qid].judged for _, s in per_run if qid in s]
        tr = truths.get(qid, [])

        if len({str(t) for t in tr}) > 1:
            shifted_truth.append((qid, tr))
            continue

        t1_span = _span(t1)
        jd_span = _span(jd)
        if t1_span:
            moved_t1.append((qid, t1, t1_span))
        if jd_span:
            moved_judged.append((qid, jd, jd_span))
        if not t1_span and not jd_span:
            stable.append(qid)

    n = len(per_run)
    print(f"\n{n} runs, label {labels.pop()!r}\n")

    if shifted_truth:
        print("Excluded, captured truth differs between runs (not model noise):")
        for qid, tr in shifted_truth:
            print(f"  Q{qid}: truth {tr}")
        print()

    print(f"tier 1 unstable:  {len(moved_t1)} question(s)")
    for qid, vals, span in moved_t1:
        print(f"  Q{qid}: {vals}  span {span}")
    print(f"judged unstable:  {len(moved_judged)} question(s)")
    for qid, vals, span in moved_judged:
        print(f"  Q{qid}: {vals}  span {span}")
    print(f"identical across all {n} runs: {len(stable)} question(s)")

    judged_floor = max((s for _, _, s in moved_judged), default=0)
    print(f"\nmeasured judged noise floor: +/-{judged_floor}")
    if judged_floor == 0:
        print(f"Nothing moved. With only {n} runs that is a ceiling on what was "
              f"observed, not proof of determinism.")
    if moved_t1:
        print("\nWARNING: tier 1 moved between runs of the same build. Tier 1 is "
              "meant to be exact and it gates the build, so this is a harness "
              "defect, not a noise floor to be tolerated.")
    return 0


def _span(values: list) -> int:
    real = [v for v in values if v is not None]
    if len(real) < 2:
        return 0
    return max(real) - min(real)


if __name__ == "__main__":
    raise SystemExit(main())
