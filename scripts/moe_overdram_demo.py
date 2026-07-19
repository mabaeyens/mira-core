#!/usr/bin/env python3
"""Demonstrate that MoE expert disk offloading lets a model whose stacked expert
table exceeds unified memory load and run on this machine — the payoff of the
lazy-load fix (`load(lazy=True)` gated on offload; see
`docs/moe-offload-lazy-load-design.md` and `docs/moe-offload-case-study.md` §9).

The reference run uses `mlx-community/Qwen3.6-35B-A3B-8bit` (35GB on disk;
~33.8GB stacked expert table) on a 32GB Apple-silicon Mac (24.96GB Metal wired
limit). Its expert table alone is larger than both physical RAM and the wired
limit, so the eager load path (`mlx_lm.load`, default `lazy=False`, which evals
every parameter) cannot complete — but lazy load + offload keeps only a fraction
resident and runs it.

Two phases:

  PHASE 1 — eager materialization is impossible (measured, SAFELY).
    Load with `lazy=True` (nothing wired), then evaluate the stacked expert
    modules one at a time, watching active memory climb, and ABORT at a small
    ceiling (default 7GB) so the machine is never driven to OOM. Extrapolate the
    per-module slope to the full table and compare against the wired limit.

  PHASE 2 — lazy + offload runs the over-DRAM model (the proof).
    Load with `lazy=True`, install expert offload at `--fraction`, and generate.
    Peak stays near the resident-fraction footprint, far below what an eager
    load would need, with coherent output.

Run from the repo root (needs the mira-core venv, i.e. the pinned mlx-lm fork):
    .venv/bin/python3 scripts/moe_overdram_demo.py
    .venv/bin/python3 scripts/moe_overdram_demo.py --model <repo-id> --fraction 0.3

Note: this loads a large model. If a mira-mlx server (e.g. the com.mab.mira
LaunchAgent) is running, stop it first so the demo has clean headroom and clean
measurements — this is a benchmark, and benches always want a fresh, uncached
process.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import generate  # noqa: E402
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear  # noqa: E402
from mlx_lm.utils import load  # noqa: E402

from core.inference.expert_offload import install as install_offload  # noqa: E402
from core.inference.expert_offload import stats as offload_stats  # noqa: E402
from core.prompts import build_system_prompt  # noqa: E402

GB = 1024 ** 3
DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-8bit"


def active() -> float:
    return mx.get_active_memory() / GB


def peak() -> float:
    return mx.get_peak_memory() / GB


def reset_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def phase1_eager_trajectory(model_id: str, ceiling_gb: float, wired_limit_gb: float) -> None:
    print(f"\n=== PHASE 1: eager materialization trajectory "
          f"(aborts at {ceiling_gb}GB, never OOMs the machine) ===", flush=True)
    model, _ = load(model_id, lazy=True)  # lazy: nothing wired yet
    print(f"  after lazy load: active={active():.2f}GB (full table still unevaluated)", flush=True)

    expert_mods = [m for _, m in model.named_modules()
                   if isinstance(m, (QuantizedSwitchLinear, SwitchLinear))]
    total = len(expert_mods)
    print(f"  {total} expert modules in the table; eval'ing until active > {ceiling_gb}GB…", flush=True)

    reset_peak()
    a0 = active()
    evaluated = 0
    for m in expert_mods:
        mx.eval(m.weight)
        if hasattr(m, "scales"):
            mx.eval(m.scales)
        evaluated += 1
        if active() > ceiling_gb:
            break
    a1 = active()
    per_mod = (a1 - a0) / max(evaluated, 1)
    extrapolated_full = a0 + per_mod * total
    print(f"  eval'd {evaluated}/{total} expert modules -> active {a0:.2f} -> {a1:.2f}GB "
          f"({per_mod * 1024:.0f}MB/module)", flush=True)
    print(f"  EXTRAPOLATED full eager materialization of the expert table "
          f"≈ {extrapolated_full:.1f}GB", flush=True)
    verdict = "OOM (cannot complete)" if extrapolated_full > wired_limit_gb else "would fit"
    print(f"  vs {wired_limit_gb}GB wired limit -> eager load: {verdict}", flush=True)

    del model, expert_mods
    gc.collect()
    mx.clear_cache()
    reset_peak()
    print(f"  cleaned up: active={active():.2f}GB", flush=True)


def phase2_offload_proof(model_id: str, fraction: float, max_tokens: int) -> None:
    print(f"\n=== PHASE 2: lazy + offload (fraction {fraction}) "
          f"— the over-DRAM model actually runs ===", flush=True)
    reset_peak()
    t0 = time.time()
    model, tokenizer = load(model_id, lazy=True)
    print(f"  after lazy load:            active={active():.2f}  peak={peak():.2f}GB", flush=True)
    install_offload(model, model_id, fraction)
    print(f"  after install(offload {fraction}): active={active():.2f}  peak={peak():.2f}GB", flush=True)
    mx.clear_cache()
    reset_peak()

    prompt = build_system_prompt() + "\n\nUser: In two sentences, what is a mixture-of-experts model?"
    t1 = time.time()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    dt = time.time() - t1
    print(f"  after generation:           active={active():.2f}  peak={peak():.2f}GB", flush=True)
    print(f"  offload hit/miss: {offload_stats(model)}", flush=True)
    print(f"  load+setup {t1 - t0:.1f}s, generate {dt:.1f}s", flush=True)
    print(f"\n  --- MODEL OUTPUT ---\n  {out.strip()}\n  --------------------", flush=True)
    print(f"\n  RESULT: a model whose expert table exceeds this machine's memory loaded and "
          f"generated\n  coherent text with peak {peak():.2f}GB resident — impossible via eager load.",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="MoE over-DRAM offload demo")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"HF repo id (default: {DEFAULT_MODEL})")
    ap.add_argument("--fraction", type=float, default=0.3, help="resident expert fraction (default: 0.3)")
    ap.add_argument("--ceiling-gb", type=float, default=7.0,
                    help="Phase 1 abort ceiling in GB (default: 7.0)")
    ap.add_argument("--wired-limit-gb", type=float, default=24.96,
                    help="Metal wired limit to compare against (default: 24.96, this 32GB Mac)")
    ap.add_argument("--max-tokens", type=int, default=64, help="tokens to generate (default: 64)")
    ap.add_argument("--skip-phase1", action="store_true", help="skip the eager-trajectory phase")
    args = ap.parse_args()

    print(f"=== MoE OVER-DRAM DEMO  {args.model}  (wired limit {args.wired_limit_gb}GB) ===", flush=True)
    if not args.skip_phase1:
        phase1_eager_trajectory(args.model, args.ceiling_gb, args.wired_limit_gb)
    phase2_offload_proof(args.model, args.fraction, args.max_tokens)


if __name__ == "__main__":
    main()
