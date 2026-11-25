#!/usr/bin/env python3
"""GPQA Diamond through Mira's own harness (isolated server, real /chat path).

Why this exists: a GPQA run that calls mlx-lm directly measures the model, not
Mira. Mira adds a system prompt, a thinking toggle, a thinking budget that forces
an answer, repetition penalty and its own token cap, so the same model can score
differently here. If Mira does worse than the raw model, that is a Mira problem.

Same questions for every model (seeded sample), same thinking budget, same
output cap. Anything not answered inside that counts as wrong.

    python scripts/bench_gpqa_mira.py --label qwen36 --n 40 --out notes/x.jsonl
    python scripts/bench_gpqa_mira.py --label inkling --n 40 --out notes/y.jsonl \\
        --config-override '{"model": "/path", "mira_mlx_mtp_enabled": false}'
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench_compare", HERE / "bench_compare.py")
bc = importlib.util.module_from_spec(spec)
sys.modules["bench_compare"] = bc
spec.loader.exec_module(bc)

FIXTURE = HERE / "eval_fixtures" / "gpqa_diamond_sub198.jsonl"


def letter(text: str):
    body = re.sub(r"<[^>]*think[^>]*>", " ", text or "")
    for pat in (r"answer\s+is\s*:?\s*\(?([A-D])\)?", r"\(([A-D])\)", r"\b([A-D])\b"):
        m = re.findall(pat, body, re.I)
        if m:
            return m[-1].upper()
    return None


def engine_pid():
    out = subprocess.run(["pgrep", "-f", "core.inference.mira_mlx_server"],
                         capture_output=True, text=True).stdout.split()
    return out[0] if out else None


class FootprintSampler(threading.Thread):
    """Peak physical footprint of the engine process, GPU memory included."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_gb = None
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.wait(8):
            pid = engine_pid()
            if not pid:
                continue
            out = subprocess.run(["footprint", "-p", pid], capture_output=True, text=True).stdout
            m = re.search(r"Footprint:\s*([\d.]+)\s*(GB|MB|KB)", out)
            if not m:
                continue
            v = float(m.group(1)) / {"GB": 1, "MB": 1024, "KB": 1024 ** 2}[m.group(2)]
            self.peak_gb = max(self.peak_gb or 0, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--budget", type=int, default=8192, help="max_thinking_tokens, same for every model")
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-override", default="")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.fixture)]
    sample = random.Random(args.seed).sample(items, args.n)
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open()}
    todo = [it for it in sample if it["id"] not in done]
    print(f"{args.label}: {len(sample)} sampled, {len(done)} done, {len(todo)} to run, "
          f"budget {args.budget}", flush=True)

    overrides = {"max_thinking_tokens": args.budget}
    if args.config_override:
        overrides.update(json.loads(args.config_override))
    bc.CONFIG_OVERRIDES.update(overrides)

    server = bc.IsolatedServer()
    sampler = FootprintSampler()
    try:
        server.start()
        sampler.start()
        for it in todo:
            choices = "\n".join(f"({'ABCD'[i]}) {o}" for i, o in enumerate(it["options"]))
            prompt = (f"{it['question']}\n\n{choices}\n\nWhat is the correct answer? "
                      f"Reply with only a single letter: A, B, C, or D. Do not explain.")
            gold = it.get("answer") or it.get("gold")
            conv = bc.create_bench_conversation(None)
            res = bc.stream_chat(prompt, args.label, True, False, conv)
            content = res.get("content", "")
            pred = letter(content) if not res.get("error") else None
            row = {
                "id": it["id"], "gold": gold, "pred": pred,
                "ok": pred is not None and pred == gold,
                "answered": pred is not None,
                "ttft_ms": res.get("ttft_ms"), "wall_ms": res.get("wall_ms"),
                "eval_tokens": res.get("eval_tokens"),
                "thinking_chars": len(res.get("thinking") or ""),
                "error": res.get("error"), "content_tail": content[-200:],
                "peak_footprint_gb": sampler.peak_gb,
            }
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {it['id']}: gold={gold} pred={pred} ok={row['ok']} "
                  f"tok={row['eval_tokens']} wall={round((row['wall_ms'] or 0)/1000)}s "
                  f"ttft={round((row['ttft_ms'] or 0)/1000)}s", flush=True)
    finally:
        sampler.stop_flag.set()
        server.stop()


if __name__ == "__main__":
    main()
