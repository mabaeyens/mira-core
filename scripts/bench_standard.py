"""
Standard LLM inference benchmark — emits pp/tg metrics comparable to llama-bench tables.

pp (prompt processing): prefill throughput = prompt_tokens / (ttft_s)
tg (token generation):  generation throughput = output_tokens / (gen_s)

Usage:
    python scripts/bench_standard.py
    python scripts/bench_standard.py --base-url http://localhost:8080 --model Qwen3.6-35B-A3B
    python scripts/bench_standard.py --reps 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL    = "Qwen3.6-35B-A3B"
DEFAULT_REPS     = 3

# ─── Test matrix ──────────────────────────────────────────────────────────────

PP_SIZES = [128, 512, 1024]   # target prompt token counts
TG_SIZES = [128, 512]         # target output token counts

CHARS_PER_TOKEN = 4           # conservative approximation

# Filler paragraph for building controlled-length prompts.
_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
)

# ─── Prompt builders ──────────────────────────────────────────────────────────

def _build_pp_prompt(target_tokens: int) -> str:
    target_chars = target_tokens * CHARS_PER_TOKEN
    repeated = (_FILLER * ((target_chars // len(_FILLER)) + 2))[:target_chars]
    return repeated


def _build_tg_messages(target_output_tokens: int) -> tuple[list[dict], int]:
    """Return (messages, max_tokens) for a tg cell."""
    instruction = (
        f"Write a detailed paragraph of approximately {target_output_tokens} words "
        "about the history of computing. Do not stop early."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": instruction},
    ]
    return messages, target_output_tokens


# ─── Streaming call ───────────────────────────────────────────────────────────

def _stream(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float = 180.0,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": "Bearer none",
        "Content-Type": "application/json",
    }

    ttft_ms: float | None = None
    output_tokens = 0
    prompt_tokens = 0
    char_count = 0
    t_start = time.perf_counter()

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if usage := chunk.get("usage"):
                    output_tokens = usage.get("completion_tokens", output_tokens)
                    prompt_tokens  = usage.get("prompt_tokens", prompt_tokens)
                for choice in chunk.get("choices", []):
                    content = (choice.get("delta") or {}).get("content") or ""
                    if content:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        char_count += len(content)

    wall_ms = (time.perf_counter() - t_start) * 1000

    if output_tokens == 0 and char_count > 0:
        output_tokens = max(1, char_count // CHARS_PER_TOKEN)

    return {
        "ttft_ms": round(ttft_ms) if ttft_ms is not None else None,
        "wall_ms": round(wall_ms),
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
    }


# ─── Cell runners ─────────────────────────────────────────────────────────────

def _run_pp_cell(base_url: str, model: str, size: int, reps: int) -> dict | None:
    prompt = _build_pp_prompt(size)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    results = []
    for rep in range(reps):
        print(f"  pp{size} rep {rep + 1}/{reps} ...", end=" ", flush=True)
        try:
            r = _stream(base_url, model, messages, max_tokens=16)
        except Exception as exc:
            print(f"ERROR: {exc}")
            return None
        if r["ttft_ms"] is None:
            print("no TTFT — skipping")
            return None
        # Use reported prompt_tokens if available, else character estimate.
        n_prompt = r["prompt_tokens"] if r["prompt_tokens"] > 0 else size
        pp = n_prompt / (r["ttft_ms"] / 1000)
        results.append({"pp_tps": pp, "ttft_ms": r["ttft_ms"]})
        print(f"{pp:.0f} t/s  (TTFT {r['ttft_ms']} ms)")
    return {
        "tps_median": statistics.median(x["pp_tps"] for x in results),
        "ms_values":  [x["ttft_ms"] for x in results],
    }


def _run_tg_cell(base_url: str, model: str, size: int, reps: int) -> dict | None:
    messages, max_tokens = _build_tg_messages(size)
    results = []
    for rep in range(reps):
        print(f"  tg{size} rep {rep + 1}/{reps} ...", end=" ", flush=True)
        try:
            r = _stream(base_url, model, messages, max_tokens=max_tokens)
        except Exception as exc:
            print(f"ERROR: {exc}")
            return None
        if r["ttft_ms"] is None:
            print("no TTFT — skipping")
            return None
        gen_ms = r["wall_ms"] - r["ttft_ms"]
        if gen_ms <= 0 or r["output_tokens"] == 0:
            print("generation too short — skipping")
            return None
        tg = r["output_tokens"] / (gen_ms / 1000)
        results.append({"tg_tps": tg, "wall_ms": r["wall_ms"]})
        print(f"{tg:.1f} t/s  ({r['output_tokens']} tokens, {r['wall_ms']} ms)")
    return {
        "tps_median": statistics.median(x["tg_tps"] for x in results),
        "ms_values":  [x["wall_ms"] for x in results],
    }


# ─── Table output ─────────────────────────────────────────────────────────────

def _print_table(rows: list[dict], model: str, base_url: str) -> None:
    backend_label = "omlx" if "8080" in base_url else base_url
    header = f"| {'model':<28} | {'backend':<10} | {'test':<8} | {'t/s':>8} | {'avg ms':>8} | {'std ms':>7} |"
    sep    = f"|{'-'*30}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*9}|"
    print()
    print(header)
    print(sep)
    for row in rows:
        avg_ms = statistics.mean(row["ms_values"])
        std_ms = statistics.stdev(row["ms_values"]) if len(row["ms_values"]) > 1 else 0.0
        print(
            f"| {model:<28} | {backend_label:<10} | {row['test']:<8} "
            f"| {row['tps_median']:>8.1f} | {avg_ms:>8.0f} | {std_ms:>7.0f} |"
        )
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Standard LLM inference benchmark (pp/tg)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--reps",     type=int, default=DEFAULT_REPS)
    args = parser.parse_args()

    # Health check
    try:
        httpx.get(f"{args.base_url}/v1/models", timeout=5).raise_for_status()
    except Exception:
        print(f"ERROR: server not reachable at {args.base_url}")
        print("Start omlx first: omlx-cli serve <model-path>")
        sys.exit(1)

    print(f"Benchmarking {args.model} at {args.base_url}  ({args.reps} reps each)\n")

    rows: list[dict] = []

    for size in PP_SIZES:
        print(f"[pp{size}]")
        result = _run_pp_cell(args.base_url, args.model, size, args.reps)
        if result:
            rows.append({"test": f"pp{size}", **result})

    for size in TG_SIZES:
        print(f"[tg{size}]")
        result = _run_tg_cell(args.base_url, args.model, size, args.reps)
        if result:
            rows.append({"test": f"tg{size}", **result})

    if rows:
        _print_table(rows, args.model, args.base_url)
    else:
        print("No results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
