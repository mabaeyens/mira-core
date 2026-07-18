#!/usr/bin/env python3
"""Analyze a JSONL expert-activation log (core/inference/expert_profiler.py)
and answer the go/no-go question for specs/moe-expert-offload-01-profiling.md:
is expert activation skewed/correlated enough to make a resident-expert cache
worthwhile (per Eliseev & Mazur's Mixtral-offloading paper and Alizadeh et
al.'s "LLM in a Flash"), or close to uniform-random?

Two numbers per layer, each reported alongside its THEORETICAL UNIFORM-RANDOM
BASELINE (inferred from num_experts/top_k observed in the data) — absolute
percentages are meaningless on their own for a wide router (e.g. Qwen3.6-35B-A3B
routes top-8-of-256 experts, so even perfectly uniform selection puts ~20% of
calls in the "top 20%" bucket by construction; the question is the RATIO to
that baseline, not the raw number):
  - top-k concentration: fraction of activation calls that hit the
    most-frequently-selected K% of experts for that layer. BuddyMoE reports
    ~60-70%+ absolute for comparable MoE models at the top 20% — but the
    number that actually matters here is concentration / uniform_baseline.
  - adjacent-token expert-set overlap: Jaccard similarity of the selected
    expert set between consecutive token positions within the same forward
    call, vs. the expected Jaccard of two independent uniform-random draws.
    High overlap-over-baseline validates "LLM in a Flash"'s windowing premise
    (reuse across adjacent tokens, not just across time).

Stdlib only (no pandas/numpy dependency) — this is a one-off analysis script,
not something worth adding a new project dependency for.

Usage:
    python scripts/analyze_expert_profile.py <path-to-jsonl> [--top-fraction 0.2]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_entries(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def top_k_concentration(layer_expert_counts: Counter, top_fraction: float) -> float:
    """Fraction of all activation events attributable to the most-frequent
    `top_fraction` of distinct experts seen for this layer."""
    if not layer_expert_counts:
        return 0.0
    n_experts = len(layer_expert_counts)
    n_top = max(1, round(n_experts * top_fraction))
    total = sum(layer_expert_counts.values())
    top_total = sum(count for _, count in layer_expert_counts.most_common(n_top))
    return top_total / total if total else 0.0


def adjacent_overlap(entries_for_layer: list) -> float:
    """Mean Jaccard similarity of the selected-expert set between
    consecutive rows (token positions) within the same call, averaged
    across all calls for this layer."""
    jaccards = []
    for entry in entries_for_layer:
        expert_ids = entry["expert_ids"]  # shape: [..., seq_or_batch, top_k]
        rows = _flatten_to_rows(expert_ids)
        for a, b in zip(rows, rows[1:]):
            set_a, set_b = set(a), set(b)
            union = set_a | set_b
            if union:
                jaccards.append(len(set_a & set_b) / len(union))
    return sum(jaccards) / len(jaccards) if jaccards else 0.0


def uniform_baselines(n_experts: int, top_k: int, top_fraction: float) -> tuple:
    """Expected concentration/overlap under uniform-random top_k-of-n_experts
    selection — the null hypothesis every offloading paper's skew claim is
    implicitly measured against. Approximates E[Jaccard] via E[intersection]/
    E[union] rather than a true expectation-of-ratio (exact for concentration,
    an approximation for overlap — close enough for a go/no-go baseline given
    top_k << n_experts here)."""
    n_top = max(1, round(n_experts * top_fraction))
    baseline_concentration = n_top / n_experts
    expected_intersection = (top_k * top_k) / n_experts
    expected_union = 2 * top_k - expected_intersection
    baseline_overlap = expected_intersection / expected_union if expected_union else 0.0
    return baseline_concentration, baseline_overlap


def _flatten_to_rows(nested: list) -> list:
    """expert_ids from mx.array.tolist() can be nested (batch, seq, top_k) or
    (seq, top_k) depending on prefill vs decode — flatten to a flat list of
    per-token-position expert-id lists regardless of leading batch dims."""
    if not nested:
        return []
    first = nested[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        rows = []
        for batch_row in nested:
            rows.extend(_flatten_to_rows(batch_row))
        return rows
    return nested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.2, help="fraction of experts counted as 'hot' (default 0.2 = top 20%%)")
    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"error: {args.log_path} does not exist", file=sys.stderr)
        sys.exit(1)

    by_layer = defaultdict(list)
    total_calls = 0
    for entry in load_entries(args.log_path):
        by_layer[entry["layer_idx"]].append(entry)
        total_calls += 1

    if total_calls == 0:
        print("no entries found in log — nothing to analyze", file=sys.stderr)
        sys.exit(1)

    # Infer num_experts/top_k from the data itself (not hardcoded) so this
    # works unmodified across models (Qwen3.6-35B-A3B: 256 experts/top-8;
    # Gemma4 or others may differ). num_experts is a lower bound (max observed
    # id + 1) — undercounts only if some expert never fired in this sample,
    # which would make the uniform baseline slightly conservative, not wrong.
    max_expert_id = -1
    top_k = None
    for entries in by_layer.values():
        for entry in entries:
            for row in _flatten_to_rows(entry["expert_ids"]):
                if row:
                    max_expert_id = max(max_expert_id, max(row))
                    top_k = top_k or len(row)
    n_experts = max_expert_id + 1
    baseline_concentration, baseline_overlap = uniform_baselines(n_experts, top_k, args.top_fraction)

    print(f"loaded {total_calls} activation-call records across {len(by_layer)} layers")
    print(f"inferred: num_experts={n_experts} top_k={top_k}")
    print(
        f"uniform-random baseline: top-{int(args.top_fraction*100)}%-concentration="
        f"{baseline_concentration:.1%}  adjacent-token overlap={baseline_overlap:.1%}\n"
    )
    print(f"{'layer':>6}  {'concentration':>14}  {'  vs baseline':>13}  {'overlap':>9}  {'  vs baseline':>13}")

    all_concentrations = []
    all_overlaps = []
    for layer_idx in sorted(by_layer):
        entries = by_layer[layer_idx]
        expert_counts = Counter()
        for entry in entries:
            for row in _flatten_to_rows(entry["expert_ids"]):
                expert_counts.update(row)
        concentration = top_k_concentration(expert_counts, args.top_fraction)
        overlap = adjacent_overlap(entries)
        all_concentrations.append(concentration)
        all_overlaps.append(overlap)
        c_ratio = concentration / baseline_concentration if baseline_concentration else float("nan")
        o_ratio = overlap / baseline_overlap if baseline_overlap else float("nan")
        print(f"{layer_idx:>6}  {concentration:>13.1%}  {c_ratio:>11.1f}x  {overlap:>8.1%}  {o_ratio:>11.1f}x")

    mean_concentration = sum(all_concentrations) / len(all_concentrations)
    mean_overlap = sum(all_overlaps) / len(all_overlaps)
    mean_c_ratio = mean_concentration / baseline_concentration if baseline_concentration else float("nan")
    mean_o_ratio = mean_overlap / baseline_overlap if baseline_overlap else float("nan")
    print(
        f"\n{'mean':>6}  {mean_concentration:>13.1%}  {mean_c_ratio:>11.1f}x  "
        f"{mean_overlap:>8.1%}  {mean_o_ratio:>11.1f}x"
    )

    print("\n--- go/no-go (ratio to uniform-random baseline, not absolute %) ---")
    if mean_c_ratio >= 2.0 or mean_o_ratio >= 3.0:
        print(
            f"SIGNAL FOUND: mean concentration is {mean_c_ratio:.1f}x uniform-random, "
            f"overlap is {mean_o_ratio:.1f}x uniform-random. Activation is skewed/correlated "
            "enough to justify prototyping spec 02 (moe-expert-offload-02-runtime-cache.md). "
            "Record this table in BACKLOG.md."
        )
    else:
        print(
            f"NO STRONG SIGNAL: mean concentration is only {mean_c_ratio:.1f}x uniform-random, "
            f"overlap {mean_o_ratio:.1f}x. Before concluding no-go, verify sample size/diversity "
            "is representative — otherwise record the result and close out spec 02 in BACKLOG.md."
        )


if __name__ == "__main__":
    main()
