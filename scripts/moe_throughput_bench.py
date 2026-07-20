#!/usr/bin/env python3
"""Throughput bench for MoE expert offload: prefill tok/s (cold + warm) and
decode tok/s, measured against the mira-mlx BACKEND directly (not the /chat
orchestrator — that path can't report token counts, since the mira-mlx backend
doesn't emit prompt_eval_count/eval_count, so decode/prefill tok/s must be timed
against the backend stream here).

Each config gets its own FRESH backend process. Default configs:
  1. <4bit-model> offload-off  — baseline, all experts resident (eager load)
  2. <4bit-model> offload-on   — offload at --fraction (model fits; the pure
     offload penalty)
  3. <8bit-model> offload-on   — offload at --fraction (over-DRAM: eager load
     is impossible on a 32GB Mac)

Prefill tok/s: a ~2-3k-token prompt, max_tokens=1 (non-stream); wall ~= prefill
  time, so prefill_tps = prompt_tokens / wall. Measured COLD (first request,
  experts fetched from disk) then WARM (repeat). For offload the two are close
  because a diverse prefill touches nearly every expert and the resident
  fraction can't hold that working set.
Decode tok/s: short prompt, max_tokens=200 (stream); decode_tps =
  completion_tokens / (wall - ttft), after a warmup.

Reference result (2026-07-19, Qwen3.6-35B-A3B, 32GB M5, fraction 0.3):
  4bit off:  prefill 616/957 cold/warm, decode 57.1 t/s, peak 19.31GB
  4bit on :  prefill  77/ 76 cold/warm, decode 10.8 t/s, peak  7.31GB
  8bit on :  prefill  59/ 56 cold/warm, decode  6.6 t/s, peak 13.05GB  (over-DRAM)
See docs/moe-offload-case-study.md.

Stop any running mira-mlx server first (this launches its own on other ports).
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from core.prompts import build_system_prompt  # noqa: E402

PYTHON = str(REPO / ".venv/bin/python3")
LOGDIR = Path(tempfile.mkdtemp(prefix="mira-tbench-"))

PREFILL_PROMPT = build_system_prompt() + "\n\n" + (
    "Consider the following engineering scenario in detail and hold it in mind. " * 120
) + "\n\nUser: Acknowledge in one word."
DECODE_PROMPT = ("Write a detailed technical explanation, in flowing prose with no lists, "
                 "of how a mixture-of-experts transformer routes tokens to experts and why "
                 "that makes inference memory-bound rather than compute-bound.")


def launch(model, fraction, port, kv_bits=8):
    args = [
        PYTHON, "-m", "core.inference.mira_mlx_server",
        "--model", model, "--host", "127.0.0.1", "--port", str(port),
        "--max-tokens", "4096", "--prefill-step-size", "1024",
        "--prompt-cache-max-bytes", "1000000000", "--max-kv-size", "128000",
        "--fix-mistral-regex",
    ]
    # Models with attention sinks (gpt-oss) cannot use a quantized KV cache:
    # mlx-lm raises "Quantized SDPA does not support attention sinks" from the
    # generation thread, which leaves requests hanging until the client times
    # out rather than returning an error. Pass --kv-bits 0 for those.
    if kv_bits:
        args += ["--kv-bits", str(kv_bits), "--kv-group-size", "64"]
    if fraction is not None:
        args += ["--resident-expert-fraction", str(fraction)]
    logf = open(LOGDIR / f"tbench_{port}.log", "w")
    return subprocess.Popen(args, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT)


def wait_ready(port, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def _req(port, prompt, max_tokens, stream):
    payload = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "stream": stream}).encode()
    return urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")


def prefill_once(port):
    """Non-stream, max_tokens=1. Returns wall_s (prefill dominates)."""
    t0 = time.time()
    with urllib.request.urlopen(_req(port, PREFILL_PROMPT, 1, False), timeout=300) as r:
        json.load(r)
    return time.time() - t0


def decode_once(port, max_tokens=200):
    """Stream. Returns (completion_tokens, ttft_s, gen_s)."""
    t0 = time.time()
    ttft = None
    ntok = 0
    with urllib.request.urlopen(_req(port, DECODE_PROMPT, max_tokens, True), timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                break
            try:
                ev = json.loads(d)
            except json.JSONDecodeError:
                continue
            delta = (ev.get("choices") or [{}])[0].get("delta", {}).get("content", "")
            if delta:
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
    wall = time.time() - t0
    return ntok, ttft, wall - (ttft or wall)


def stats(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/stats", timeout=5) as r:
            s = json.load(r)
        ec = s.get("expert_cache") or {}
        return round((s.get("peak_memory_bytes") or 0) / 1024**3, 2), ec.get("hit_rate")
    except Exception:
        return None, None


def count_prompt_tokens(model):
    from mlx_lm.utils import hf_repo_to_path
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_repo_to_path(model))
    return len(tok.apply_chat_template([{"role": "user", "content": PREFILL_PROMPT}],
                                       add_generation_prompt=True)["input_ids"])


def run_config(name, model, fraction, port, p_tok, kv_bits=8):
    print(f"\n### {name}  ({model.split('/')[-1]}, fraction={fraction}, kv_bits={kv_bits or 'off'}) ###",
          flush=True)
    proc = launch(model, fraction, port, kv_bits)
    try:
        if not wait_ready(port):
            print("  NOT READY", flush=True)
            return {"name": name, "error": "not ready"}
        cold = prefill_once(port)
        warm = prefill_once(port)
        print(f"  prefill: {p_tok} tok | cold {cold:.2f}s = {p_tok/cold:.1f} t/s | "
              f"warm {warm:.2f}s = {p_tok/warm:.1f} t/s", flush=True)
        decode_once(port, 32)  # warmup
        ntok, ttft, gen_s = decode_once(port, 200)
        dec = round(ntok / gen_s, 1) if gen_s > 0 else None
        print(f"  decode: {ntok} tok in {gen_s:.2f}s = {dec} t/s (ttft {ttft:.2f}s)", flush=True)
        peak, hit = stats(port)
        print(f"  peak={peak}GB  hit_rate={hit}", flush=True)
        return {"name": name, "prompt_tokens": p_tok,
                "prefill_cold_tps": round(p_tok/cold, 1), "prefill_warm_tps": round(p_tok/warm, 1),
                "decode_tps": dec, "peak_gb": peak, "hit_rate": hit}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
        time.sleep(4)


def main():
    ap = argparse.ArgumentParser(description="MoE offload throughput bench")
    ap.add_argument("--model-4bit", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    ap.add_argument("--model-8bit", default="mlx-community/Qwen3.6-35B-A3B-8bit")
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--base-port", type=int, default=8131)
    ap.add_argument("--skip-8bit", action="store_true", help="skip the over-DRAM 8-bit config")
    ap.add_argument("--skip-4bit", action="store_true",
                    help="skip both 4-bit configs; bench only the over-DRAM model. Use when "
                         "--model-8bit is a foreign model with no 4-bit sibling to compare against.")
    ap.add_argument("--label-8bit", default="8bit-offload-on",
                    help="summary label for the over-DRAM config (rename when it is not an 8-bit)")
    ap.add_argument("--kv-bits", type=int, default=8,
                    help="KV cache quantization bits; 0 disables. Must be 0 for models with "
                         "attention sinks (gpt-oss) -- mlx-lm cannot do quantized SDPA with sinks.")
    args = ap.parse_args()

    if args.skip_4bit and args.skip_8bit:
        ap.error("--skip-4bit and --skip-8bit together leave nothing to bench")

    configs = []
    if not args.skip_4bit:
        configs += [
            ("4bit-offload-off", args.model_4bit, None, args.base_port),
            ("4bit-offload-on", args.model_4bit, args.fraction, args.base_port + 1),
        ]
    if not args.skip_8bit:
        configs.append((args.label_8bit, args.model_8bit, args.fraction, args.base_port + 2))

    print("=== MoE OFFLOAD THROUGHPUT BENCH ===", flush=True)
    # Tokenize with the first model actually benched: prompt-token count is the
    # numerator of every prefill t/s below, so borrowing another model's
    # tokenizer silently skews the result.
    p_tok = count_prompt_tokens(configs[0][1])
    print(f"prefill prompt = {p_tok} tokens (chat-templated, {configs[0][1].split('/')[-1]} "
          f"tokenizer); logs in {LOGDIR}", flush=True)

    results = [run_config(n, m, f, p, p_tok, args.kv_bits) for (n, m, f, p) in configs]
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'config':<18}{'prefill_cold':>13}{'prefill_warm':>13}{'decode':>8}{'peak_gb':>8}{'hit':>6}",
          flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<18} ERROR: {r['error']}", flush=True)
            continue
        print(f"{r['name']:<18}{r['prefill_cold_tps']:>13}{r['prefill_warm_tps']:>13}"
              f"{str(r['decode_tps']):>8}{str(r['peak_gb']):>8}{str(r['hit_rate']):>6}", flush=True)


if __name__ == "__main__":
    main()
