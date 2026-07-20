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

# NOTE: prefill and decode are measured from DIFFERENT prompts. The prefill
# figure comes from a long prompt (PREFILL_REPEATS scenario sentences, ~3015
# tokens at the default 120); the decode figure comes from the short prompt
# below. So a decode t/s here is "decode after a SHORT prompt", which is what
# to compare against someone else's steady-state number -- and it is also why
# the lifetime hit_rate is dragged down by the long cold prefills that run
# first. Both prompt lengths are printed at run time; quote them with any
# result.
PREFILL_REPEATS_DEFAULT = 120


def build_prefill_prompt(repeats):
    return build_system_prompt() + "\n\n" + (
        "Consider the following engineering scenario in detail and hold it in mind. " * repeats
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


def prefill_once(port, prefill_prompt):
    """Non-stream, max_tokens=1. Returns wall_s (prefill dominates)."""
    t0 = time.time()
    with urllib.request.urlopen(_req(port, prefill_prompt, 1, False), timeout=300) as r:
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
        # Two different metrics, do NOT compare across them: hit_rate is blended
        # over the process lifetime so a cold prefill drags it down, while
        # decode_hit_rate is the steady-state number a residency change moves.
        # See mira_mlx_server.py:372-374.
        return (round((s.get("peak_memory_bytes") or 0) / 1024**3, 2),
                ec.get("hit_rate"), ec.get("decode_hit_rate"))
    except Exception:
        return None, None


def count_tokens(model, text):
    from mlx_lm.utils import hf_repo_to_path
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_repo_to_path(model))
    return len(tok.apply_chat_template([{"role": "user", "content": text}],
                                       add_generation_prompt=True)["input_ids"])


def run_config(name, model, fraction, port, p_tok, d_tok, prefill_prompt,
               kv_bits=8, decode_tokens=200, skip_prefill=False):
    print(f"\n### {name}  ({model.split('/')[-1]}, fraction={fraction}, kv_bits={kv_bits or 'off'}, "
          f"prefill_prompt={p_tok} tok, decode_prompt={d_tok} tok, gen={decode_tokens} tok) ###",
          flush=True)
    proc = launch(model, fraction, port, kv_bits)
    try:
        if not wait_ready(port):
            print("  NOT READY", flush=True)
            return {"name": name, "error": "not ready"}
        cold = warm = None
        if skip_prefill:
            # The long prefills are what drag the LIFETIME hit_rate down. Skipping
            # them makes hit_rate and decode_hit_rate directly comparable, which is
            # the right shape for matching someone else's short-prompt claim.
            print("  prefill: SKIPPED (--skip-prefill), so hit_rate is decode-dominated",
                  flush=True)
        else:
            cold = prefill_once(port, prefill_prompt)
            warm = prefill_once(port, prefill_prompt)
            print(f"  prefill: {p_tok} tok | cold {cold:.2f}s = {p_tok/cold:.1f} t/s | "
                  f"warm {warm:.2f}s = {p_tok/warm:.1f} t/s", flush=True)
        decode_once(port, 32)  # warmup
        ntok, ttft, gen_s = decode_once(port, decode_tokens)
        dec = round(ntok / gen_s, 1) if gen_s > 0 else None
        print(f"  decode: {ntok} tok in {gen_s:.2f}s = {dec} t/s (ttft {ttft:.2f}s) "
              f"after a {d_tok}-tok prompt", flush=True)
        peak, hit, dec_hit = stats(port)
        print(f"  peak={peak}GB  hit_rate={hit} (lifetime)  "
              f"decode_hit_rate={dec_hit} (steady state)", flush=True)
        return {"name": name, "prompt_tokens": p_tok, "decode_prompt_tokens": d_tok,
                "gen_tokens": decode_tokens,
                "prefill_cold_tps": round(p_tok/cold, 1) if cold else None,
                "prefill_warm_tps": round(p_tok/warm, 1) if warm else None,
                "decode_tps": dec, "peak_gb": peak,
                "hit_rate": hit, "decode_hit_rate": dec_hit}
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
    ap.add_argument("--decode-tokens", type=int, default=200,
                    help="tokens to generate for the decode measurement. Raise it when comparing "
                         "against a 'steady state' figure -- a short run is still partly warming "
                         "the LRU, which reads as slower than steady state.")
    ap.add_argument("--kv-bits", type=int, default=8,
                    help="KV cache quantization bits; 0 disables. Must be 0 for models with "
                         "attention sinks (gpt-oss) -- mlx-lm cannot do quantized SDPA with sinks.")
    ap.add_argument("--prefill-repeats", type=int, default=PREFILL_REPEATS_DEFAULT,
                    help="scenario-sentence repeats in the prefill prompt (default 120 ~= 3015 "
                         "tokens). Lower it to measure at a shorter prompt; the actual token "
                         "count is measured and printed either way.")
    ap.add_argument("--skip-prefill", action="store_true",
                    help="skip the two long prefills entirely. They are what drag the LIFETIME "
                         "hit_rate down, so skipping makes hit_rate decode-dominated and directly "
                         "comparable to a short-prompt steady-state claim.")
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
    prefill_prompt = build_prefill_prompt(args.prefill_repeats)
    tokenizer_model = configs[0][1]
    p_tok = count_tokens(tokenizer_model, prefill_prompt)
    d_tok = count_tokens(tokenizer_model, DECODE_PROMPT)
    print(f"tokenizer      : {tokenizer_model.split('/')[-1]}", flush=True)
    print(f"prefill prompt : {p_tok} tokens (repeats={args.prefill_repeats})"
          f"{'  [SKIPPED]' if args.skip_prefill else ''}", flush=True)
    print(f"decode prompt  : {d_tok} tokens, generating {args.decode_tokens} tokens", flush=True)
    print(f"logs in {LOGDIR}", flush=True)

    results = [run_config(n, m, f, p, p_tok, d_tok, prefill_prompt,
                          args.kv_bits, args.decode_tokens, args.skip_prefill)
               for (n, m, f, p) in configs]
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'config':<18}{'prefill_cold':>13}{'prefill_warm':>13}{'decode':>8}{'peak_gb':>8}"
          f"{'hit_life':>10}{'hit_decode':>12}", flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<18} ERROR: {r['error']}", flush=True)
            continue
        print(f"{r['name']:<18}{str(r['prefill_cold_tps']):>13}{str(r['prefill_warm_tps']):>13}"
              f"{str(r['decode_tps']):>8}{str(r['peak_gb']):>8}"
              f"{str(r['hit_rate']):>10}{str(r['decode_hit_rate']):>12}", flush=True)
    print("hit_life is blended over the process lifetime (cold prefill drags it down); "
          "hit_decode is steady state. Not comparable to each other.", flush=True)


if __name__ == "__main__":
    main()
