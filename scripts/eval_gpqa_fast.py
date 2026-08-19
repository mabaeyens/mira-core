#!/usr/bin/env python3
"""eval_gpqa_fast.py — Lever 0 of the eval contract: the fast GPQA proxy.

The proxy's job (per docs/eval-contract.md) is to RANK candidate levers cheaply,
not to produce a shippable number. It gets its speed from two things the old
serial 12-14h GPQA run threw away:

  1. Concurrency. It fires many questions at once at the inference backend
     (:8080) so mira-mlx's continuous batching keeps the GPU saturated, instead
     of one-at-a-time through the orchestrator on :8000.
  2. No-think. --proxy runs with enable_thinking=False, so each answer is a
     few hundred tokens, not a 19k-character chain of thought.

Two modes, one apparatus block, so the apparatus can never drift by accident:

  --proxy  (default) : thinking OFF, fixed stratified n-subset, high concurrency.
                       Minutes. Trust only the RELATIVE order between levers.
  --gate             : thinking ON, full 198 Diamond, concurrency pinned low to
                       match the production decode regime. The ship decision.

Scoring here is the deterministic tier only (flexible-extract letter match) plus
behavioral flags (empty / no-letter / truncated). Judged coherence scoring stays
offline in scripts/bench_eval.py against the raw jsonl this writes.

Everything the observer-effect note warns about is recorded in the apparatus
block: model, think mode, concurrency (= effective batch), max_tokens,
temperature, MLX_ENABLE_TF32, subset id + sha256, git SHA. A score without its
apparatus is discarded.

Usage:
  # rank a lever, fast:
  MLX_ENABLE_TF32=0 python scripts/eval_gpqa_fast.py --proxy --tag baseline
  # ...flip the lever in mira.yaml, restart engine, then:
  MLX_ENABLE_TF32=0 python scripts/eval_gpqa_fast.py --proxy --tag with-lever
  # the ship decision, once a lever wins the proxy:
  MLX_ENABLE_TF32=0 python scripts/eval_gpqa_fast.py --gate --tag with-lever
  # rank a LONG-CONTEXT lever (kv_bits, kv_group_size, prefill_step): pad each
  # question to a long prompt so the lever actually bites, concurrency forced to 1:
  MLX_ENABLE_TF32=0 python scripts/eval_gpqa_fast.py --proxy --context-pad 32000 --n 40 --tag kv4

On a 32GB Mac the served model holds ~20GB wired, leaving ~4GB, and the engine's
working set grows ~16MB per request. A run that starts from an engine that has
already been serving all session trips the 3GB RAM floor around question ~70. Start
n=100 from a FRESHLY restarted + warmed engine (/mira-server restart, then wait for a
real generation) so headroom is at its peak. The live watchdog aborts with no verdict
rather than risk the machine, so a truncated run is safe, just inconclusive.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "scripts" / "eval_fixtures"
GPQA_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/*/gpqa_diamond.csv"
)

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_SEED = 20260818  # fixed: the subset and every option order must be stable across runs
RAM_FLOOR_GB = 3.0  # abort if free memory drops below this (a bench once rebooted the Mac)

DOMAIN_COL = "High-level domain"


# ─── RAM guard ────────────────────────────────────────────────────────────────

def free_gb() -> float:
    """Approx free+inactive+speculative memory in GB from vm_stat (16384 B pages on Apple)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = {}
    for line in out.splitlines():
        m = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line)
        if m:
            pages[m.group(1)] = int(m.group(2))
    free_pages = sum(pages.get(k, 0) for k in ("free", "inactive", "speculative"))
    return free_pages * 16384 / 1e9


# ─── Fixture: fixed stratified subset, built once, versioned on disk ──────────

def _load_diamond() -> list[dict]:
    matches = glob.glob(GPQA_GLOB)
    if not matches:
        sys.exit(f"ERROR: GPQA Diamond CSV not found at {GPQA_GLOB}")
    return list(csv.DictReader(open(matches[0])))


def build_fixture(n: int, seed: int) -> Path:
    """Deterministic stratified subset of GPQA Diamond, proportional by high-level domain.

    Written to scripts/eval_fixtures/gpqa_diamond_sub{n}.jsonl. Each item stores the
    pre-shuffled A-D option order and the resulting gold letter, so the exact same
    multiple-choice problem is posed on every run regardless of dict ordering.
    """
    path = FIXTURE_DIR / f"gpqa_diamond_sub{n}.jsonl"
    if path.exists():
        return path
    rows = _load_diamond()
    total = len(rows)
    # bucket by domain, deterministic order within each bucket
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r.get(DOMAIN_COL, "Unknown"), []).append(r)
    rng = random.Random(seed)
    for dom in buckets:
        buckets[dom].sort(key=lambda r: r["Question"])  # stable base order
        rng.shuffle(buckets[dom])
    # proportional allocation, largest-remainder so the counts sum to exactly n
    quotas = {d: n * len(rs) / total for d, rs in buckets.items()}
    alloc = {d: int(q) for d, q in quotas.items()}
    while sum(alloc.values()) < n:
        d = max(quotas, key=lambda d: (quotas[d] - alloc[d], len(buckets[d])))
        alloc[d] += 1

    # round-robin interleave across domains so ANY prefix (e.g. --limit 12) stays
    # stratified, not clustered by domain.
    selected = {d: buckets[d][: alloc[d]] for d in sorted(buckets)}
    ordered: list[tuple[str, dict]] = []
    rank = 0
    while any(rank < len(selected[d]) for d in selected):
        for d in sorted(selected):
            if rank < len(selected[d]):
                ordered.append((d, selected[d][rank]))
        rank += 1

    items = []
    for idx, (dom, r) in enumerate(ordered):
        opts = [
            r["Correct Answer"].strip(),
            r["Incorrect Answer 1"].strip(),
            r["Incorrect Answer 2"].strip(),
            r["Incorrect Answer 3"].strip(),
        ]
        # option order seeded by the question itself, so it is stable no matter where
        # the item lands in the interleaved sequence.
        qseed = seed + int.from_bytes(hashlib.sha256(r["Question"].encode()).digest()[:6], "big")
        order = list(range(4))
        random.Random(qseed).shuffle(order)
        shuffled = [opts[i] for i in order]
        gold = "ABCD"[order.index(0)]  # where the correct answer landed
        items.append(
            {
                "id": idx,
                "domain": dom,
                "subdomain": r.get("Subdomain", ""),
                "question": r["Question"].strip(),
                "options": shuffled,
                "gold": gold,
            }
        )
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path


def fixture_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ─── Long-context padding ─────────────────────────────────────────────────────
# A short GPQA prompt cannot rank a long-context lever (kv_bits, kv_group_size,
# prefill_step): quant error accrues over the sequence and is negligible at ~200
# tokens, so every long-ctx lever returns a null delta on the bare proxy — not
# because it is safe, but because the proxy never enters the regime where it acts.
# --context-pad injects a large neutral passage BETWEEN the question and the
# answer instruction, forcing the model to attend back across a long (quantized)
# KV span to the question. The passage is deterministic per (seed, item id) so
# kv4 and kv8 runs see byte-identical prompts. Extraction runs on the model's
# OUTPUT, never the prompt, so filler content can never pollute the score.

CHARS_PER_TOKEN = 6.4  # measured for _FILLER_WORDS against the Qwen3.6 tokenizer (common words
                       # tokenize ~1 token each); achieved prompt_tokens is recorded from usage
_FILLER_WORDS = (
    "region river valley meadow harbor village orchard cabinet lantern compass "
    "signal harvest merchant journey pattern surface distance measure balance "
    "current season weather timber granite meadow prairie current custom market "
    "council ledger parcel courier notice matter subject ordinary quiet distant "
    "steady gentle narrow modest common plain sober careful patient slower older "
    "toward across beside beyond within around before after during between under"
).split()


def make_filler(target_tokens: int, seed: int) -> str:
    """Deterministic neutral prose of ~target_tokens tokens, seeded so it is stable.

    Built to a CHARACTER budget (target_tokens × CHARS_PER_TOKEN) and biased to
    undershoot: overshooting the requested context risks blowing past the served
    window (truncation/rejection) or the Metal ceiling (OOM). The achieved length
    is recorded from usage.prompt_tokens, so the operator always sees the real number.
    """
    rng = random.Random(seed)
    char_budget = int(target_tokens * CHARS_PER_TOKEN)
    out, sent, length = [], [], 0
    step = rng.randint(8, 16)
    while length < char_budget:
        w = rng.choice(_FILLER_WORDS)
        sent.append(w)
        length += len(w) + 1
        if len(sent) >= step:
            sent[0] = sent[0].capitalize()
            out.append(" ".join(sent) + ".")
            sent, step = [], rng.randint(8, 16)
    if sent:
        sent[0] = sent[0].capitalize()
        out.append(" ".join(sent) + ".")
    return " ".join(out)


# ─── Prompt + scoring ─────────────────────────────────────────────────────────

def build_messages(item: dict, context_pad: int = 0, seed: int = 0) -> list[dict]:
    labels = "ABCD"
    choices = "\n".join(f"({labels[i]}) {opt}" for i, opt in enumerate(item["options"]))
    ask_line = (
        "What is the correct answer? Reply with only a single letter: A, B, C, or D. "
        "Do not explain."
    )
    if context_pad and context_pad > 0:
        filler = make_filler(context_pad, seed ^ item["id"])
        user = (
            "Below is a multiple-choice question, followed by an unrelated passage. "
            "Read the question, then ignore the passage, and answer at the end.\n\n"
            f"QUESTION:\n{item['question']}\n\n{choices}\n\n"
            f"PASSAGE (irrelevant, ignore it):\n{filler}\n\n"
            f"Now recall the multiple-choice QUESTION above. {ask_line}"
        )
    else:
        user = f"{item['question']}\n\n{choices}\n\n{ask_line}"
    return [{"role": "user", "content": user}]


def extract_letter(text: str) -> str | None:
    """flexible-extract-style A-D pull, matching notes/b1_elicitation_gpqa.py."""
    if not text:
        return None
    t = text.strip()
    for pat in (
        r"answer\s+is\s*:?\s*\(?([A-D])\)?",
        r"answer\s*:?\s*\(?([A-D])\)?\s*$",
        r"\\boxed\{\s*\(?([A-D])\)?\s*\}",
        r"final\s+answer\s*:?\s*\(?([A-D])\)?",
    ):
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    parens = re.findall(r"\(([A-D])\)", t)
    if parens:
        return parens[-1].upper()
    toks = re.findall(r"\b([A-D])\b", t)
    if toks:
        return toks[-1].upper()
    return None


# ─── Backend call ─────────────────────────────────────────────────────────────

async def ask(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    item: dict,
    think: bool,
    max_tokens: int,
    api_key: str,
    context_pad: int = 0,
    seed: int = 0,
) -> dict:
    payload = {
        "model": model,
        "messages": build_messages(item, context_pad, seed),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.perf_counter()
    r = await client.post(
        f"{base_url}/v1/chat/completions", json=payload, headers=headers
    )
    sec = round(time.perf_counter() - t0, 2)
    base = {"id": item["id"], "domain": item["domain"], "gold": item["gold"], "sec": sec}
    if r.status_code != 200:
        # A stall/wedge (504) or any non-200 is an APPARATUS failure, not a wrong
        # answer — record it separately so it never silently scores as incorrect.
        detail = ""
        try:
            detail = str(r.json().get("detail", ""))[:200]
        except Exception:
            detail = r.text[:200]
        stalled = r.status_code in (503, 504)
        return {**base, "picked": None, "correct": False, "finish_reason": None,
                "empty": True, "no_letter": True, "truncated": False,
                "errored": True, "stalled": stalled, "http_status": r.status_code,
                "prompt_tokens": None,
                "err": f"HTTP {r.status_code}: {detail}", "answer": ""}
    body = r.json()
    choice = body["choices"][0]
    msg = choice.get("message", {})
    answer = msg.get("content", "") or ""
    finish = choice.get("finish_reason")
    picked = extract_letter(answer)
    prompt_tokens = (body.get("usage") or {}).get("prompt_tokens")
    return {
        **base,
        "picked": picked,
        "correct": picked == item["gold"],
        "finish_reason": finish,
        "empty": not answer.strip(),
        "no_letter": picked is None,
        "truncated": finish == "length",
        "errored": False,
        "stalled": False,
        "prompt_tokens": prompt_tokens,
        "answer": answer,
    }


async def run(items, base_url, model, think, max_tokens, concurrency, api_key,
              context_pad=0, seed=0):
    sem = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(items)
    aborted = {"flag": False, "fired": False}

    async def watchdog():
        while not aborted["flag"]:
            if free_gb() < RAM_FLOOR_GB:
                aborted["flag"] = True
                aborted["fired"] = True
                print(f"\nABORT: free memory below {RAM_FLOOR_GB} GB floor", file=sys.stderr)
                return
            await asyncio.sleep(2)

    timeout = httpx.Timeout(600.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        wd = asyncio.create_task(watchdog())

        async def worker(i, item):
            async with sem:
                if aborted["flag"]:
                    return
                try:
                    results[i] = await ask(
                        client, base_url, model, item, think, max_tokens, api_key,
                        context_pad, seed,
                    )
                except Exception as e:  # noqa: BLE001 — record, keep the batch going
                    results[i] = {
                        "id": item["id"], "gold": item["gold"], "picked": None,
                        "correct": False, "err": f"{type(e).__name__}: {e}",
                        "empty": True, "no_letter": True, "truncated": False,
                        "errored": True, "stalled": False, "prompt_tokens": None,
                        "domain": item["domain"], "answer": "",
                    }
                done = sum(1 for r in results if r is not None)
                if done % 10 == 0 or done == len(items):
                    ok = sum(1 for r in results if r and r.get("correct"))
                    print(f"  {done}/{len(items)}  correct={ok}", end="\r", flush=True)

        await asyncio.gather(*(worker(i, it) for i, it in enumerate(items)))
        aborted["flag"] = True
        await wd
    print()
    # A watchdog abort no longer discards the completed work: keep whatever finished
    # and flag the run inconclusive. On a RAM-starved host (32GB, long-context pad) a
    # near-complete batch is still worth its partial data, provided it is never treated
    # as a ship-grade number — the caller stamps aborted=True on the summary.
    done_results = [r for r in results if r is not None]
    return done_results, aborted["fired"]


# ─── Apparatus ────────────────────────────────────────────────────────────────

def probe_model(base_url: str, api_key: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key not in (None, "none") else {}
        r = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=5)
        r.raise_for_status()
        return r.json()["data"][0]["id"]
    except Exception:
        return "unknown"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--proxy", action="store_true", help="fast ranking apparatus (default): thinking OFF, subset, high concurrency")
    mode.add_argument("--gate", action="store_true", help="ship apparatus: thinking ON, full 198, low concurrency")
    ap.add_argument("--n", type=int, default=100, help="fixture subset size (proxy; gate defaults to full 198)")
    ap.add_argument("--full", action="store_true", help="use all 198 Diamond items regardless of mode")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items to the first N of the fixed order; pair --proxy/--gate with the same --limit to validate")
    ap.add_argument("--context-pad", type=int, default=0,
                    help="target total prompt tokens: inject a neutral passage between the "
                         "question and the answer so long-ctx levers (kv_bits, kv_group_size, "
                         "prefill_step) actually bite. Forces concurrency to 1 by default (a 32k "
                         "prefill per request; several at once OOM a 32GB Mac). 0 = off (short proxy).")
    ap.add_argument("--concurrency", type=int, default=None, help="override effective batch (default 8 proxy / 2 gate / 1 when --context-pad)")
    ap.add_argument("--max-tokens", type=int, default=None, help="override (default 16 proxy / 20480 gate)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=None, help="default: probe /v1/models")
    ap.add_argument("--api-key", default=os.environ.get("MIRA_ENGINE_API_KEY", "none"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tag", default="untagged", help="label for this run (e.g. the lever under test)")
    ap.add_argument("--out", default=None, help="raw jsonl path (default scripts/eval_runs/<ts>_<tag>.jsonl)")
    args = ap.parse_args()

    is_gate = args.gate
    think = is_gate
    padded = args.context_pad and args.context_pad > 0
    if args.concurrency is not None:
        concurrency = args.concurrency
    elif padded:
        concurrency = 1  # a padded prefill is ~all the RAM headroom; don't stack them
    else:
        concurrency = 2 if is_gate else 8
    if padded and concurrency > 2:
        print(f"WARNING: --context-pad with concurrency={concurrency}: several long-context "
              "prefills at once can OOM a 32GB Mac. The watchdog will abort if it must.")
    # proxy: the letter-only prompt answers in ~1 token, so a tiny cap keeps it fast;
    # gate: thinking is on and counts against the cap, so it needs real room.
    max_tokens = args.max_tokens if args.max_tokens is not None else (20480 if is_gate else 16)

    # preflight RAM
    fg = free_gb()
    print(f"preflight free mem ~{fg:.1f} GB")
    if fg < RAM_FLOOR_GB + 1:
        sys.exit(f"ABORT: only {fg:.1f} GB free before start")

    # item source. --full (or a bare --gate, the ship default) uses all 198 in fixed
    # seeded order; otherwise the sub-N fixture. --limit caps how many are actually run,
    # from the front of that fixed order — so `--proxy --limit K` and `--gate --limit K`
    # score the IDENTICAL K items (the proxy-vs-gate validation of the contract).
    use_full = args.full or (is_gate and args.limit is None and args.n == 100)
    if use_full:
        rows = _load_diamond()
        items = []
        for i, r in enumerate(rows):
            opts = [r["Correct Answer"].strip(), r["Incorrect Answer 1"].strip(),
                    r["Incorrect Answer 2"].strip(), r["Incorrect Answer 3"].strip()]
            qseed = args.seed + int.from_bytes(hashlib.sha256(r["Question"].encode()).digest()[:6], "big")
            order = list(range(4))
            random.Random(qseed).shuffle(order)
            gold = "ABCD"[order.index(0)]
            items.append({"id": i, "domain": r.get(DOMAIN_COL, ""), "subdomain": r.get("Subdomain", ""),
                          "question": r["Question"].strip(), "options": [opts[j] for j in order], "gold": gold})
        subset_id, subset_hash = "gpqa_diamond_full198", "n/a"
    else:
        fpath = build_fixture(args.n, args.seed)
        items = [json.loads(l) for l in open(fpath)]
        subset_id, subset_hash = fpath.name, fixture_sha(fpath)
    if args.limit is not None:
        items = items[: args.limit]
        subset_id = f"{subset_id}[:{args.limit}]"

    model = args.model or probe_model(args.base_url, args.api_key)

    apparatus = {
        "mode": "gate" if is_gate else "proxy",
        "tag": args.tag,
        "model": model,
        "think": think,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "context_pad": args.context_pad,
        "temperature": 0.0,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "unset"),
        "subset_id": subset_id,
        "subset_sha": subset_hash,
        "n": len(items),
        "seed": args.seed,
        "base_url": args.base_url,
        "git_sha": git_sha(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print("=== apparatus ===")
    print(json.dumps(apparatus, indent=2))
    if apparatus["mlx_enable_tf32"] != "0":
        print("WARNING: MLX_ENABLE_TF32 is not 0 — the contract mandates TF32 off for reproducibility.")

    t0 = time.time()
    results, aborted = asyncio.run(run(items, args.base_url, model, think, max_tokens, concurrency,
                                       args.api_key, args.context_pad, args.seed))
    elapsed = time.time() - t0
    if aborted:
        print(f"WATCHDOG ABORT: {len(results)}/{len(items)} items completed before the RAM floor. "
              "Partial results below are INCONCLUSIVE (never a ship number).", file=sys.stderr)
    if not results:
        sys.exit("No items completed — nothing to report.")

    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    errored = sum(1 for r in results if r.get("errored"))
    stalled = sum(1 for r in results if r.get("stalled"))
    completed = n - errored  # answered without an apparatus failure
    by_dom: dict[str, list[int]] = {}
    for r in results:
        b = by_dom.setdefault(r["domain"], [0, 0])
        b[1] += 1
        b[0] += int(r["correct"])
    ptoks = sorted(r["prompt_tokens"] for r in results if r.get("prompt_tokens"))
    ptok_stats = None
    if ptoks:
        ptok_stats = {
            "min": ptoks[0], "median": ptoks[len(ptoks) // 2], "max": ptoks[-1],
            "mean": round(sum(ptoks) / len(ptoks)),
        }
    summary = {
        "apparatus": apparatus,
        "aborted": aborted,  # True = RAM watchdog fired; n < requested, treat as inconclusive
        "requested_n": len(items),
        "n": n,
        "correct": correct,
        "prompt_tokens": ptok_stats,  # achieved context length (from usage) when --context-pad
        "accuracy": round(correct / n, 4),  # raw: apparatus failures count as wrong
        "completed": completed,
        "accuracy_completed": round(correct / completed, 4) if completed else None,  # excludes stalls/errors
        "errored_turns": errored,
        "stalled_turns": stalled,
        "empty_turns": sum(1 for r in results if r.get("empty")),
        "no_letter_turns": sum(1 for r in results if r.get("no_letter")),
        "truncated_turns": sum(1 for r in results if r.get("truncated")),
        "by_domain": {d: {"correct": c, "n": t, "acc": round(c / t, 3)} for d, (c, t) in by_dom.items()},
        "elapsed_sec": round(elapsed, 1),
        "elapsed_min": round(elapsed / 60, 2),
        "throughput_q_per_min": round(n / (elapsed / 60), 1),
    }

    # persist raw + summary
    out = Path(args.out) if args.out else REPO / "scripts" / "eval_runs" / (
        f"{datetime.now().strftime('%Y-%m-%dT%H%M%S')}_{apparatus['mode']}_{args.tag}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "apparatus"}, indent=2))
    print(f"\nraw → {out}")
    acc_c = summary["accuracy_completed"]
    print(
        f"accuracy {summary['accuracy']:.1%} on n={n} "
        f"({apparatus['mode']}, {elapsed/60:.1f} min, {summary['throughput_q_per_min']} q/min). "
        + (f"completed {completed}/{n} → {acc_c:.1%} on completed. " if errored else "")
        + "Flags: "
        f"stalled={stalled} errored={errored} empty={summary['empty_turns']} "
        f"no_letter={summary['no_letter_turns']} truncated={summary['truncated_turns']}."
    )
    if errored:
        print("Apparatus failures present (stalls/errors) — they are NOT wrong answers; "
              "read accuracy_completed and fix the apparatus before trusting the number.")
    if not is_gate:
        print("Proxy number — trust the DELTA vs another proxy run, not this absolute (docs/eval-contract.md).")


if __name__ == "__main__":
    main()
