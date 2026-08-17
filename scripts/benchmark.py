#!/usr/bin/env python3
"""
Backend performance benchmark for mira-core.

Runs a 9-cell test matrix (3 prompts × 3 session positions) across locally
installed mlx-lm models. Produces a markdown report analysed by Claude Haiku.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --skip-mlx
    uv run python scripts/benchmark.py --reps 5

Output:
    /tmp/mira_benchmark_YYYY-MM-DD.jsonl      raw timings (never committed)
    /tmp/mira_benchmark_report_YYYY-MM-DD.md  markdown report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import httpx

# ─── Endpoints ────────────────────────────────────────────────────────────────

MLX_LM_BASE = "http://localhost:8080"

# ─── Benchmark parameters ─────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {
    "short":  "What is the capital of France?",
    "medium": "Explain the difference between a mutex and a semaphore in two sentences.",
    "long": (
        "Review this Python function and suggest one improvement: "
        "`def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)`"
    ),
}

# Filler Q&A used to build up session history for warm/re-warm positions.
FILLER_USER = "What is 2+2?"
FILLER_ASST = "4."

SYSTEM_PROMPT = "You are a helpful assistant. Be concise."

SESSION_POSITIONS = ["cold", "warm", "re-warm"]
# cold    → position 1 in session (no prior context)
# warm    → position 2 in session (1 prior filler exchange — system prompt cached)
# re-warm → position 5 in session (4 prior filler exchanges — long cached prefix)

DEFAULT_REPS = 3


# ─── Model discovery ──────────────────────────────────────────────────────────


def discover_mlx_models() -> list[str]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.exists():
        return []
    models = []
    for p in hub.iterdir():
        if p.name.startswith("models--mlx-community--"):
            repo = p.name.removeprefix("models--").replace("--", "/")
            models.append(repo)
    return models


def mlx_lm_is_running() -> bool:
    try:
        httpx.get(f"{MLX_LM_BASE}/health", timeout=2).raise_for_status()
        return True
    except Exception:
        try:
            httpx.get(f"{MLX_LM_BASE}/v1/models", timeout=2).raise_for_status()
            return True
        except Exception:
            return False


def get_loaded_mlx_models() -> list[str]:
    try:
        resp = httpx.get(f"{MLX_LM_BASE}/v1/models", timeout=5)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []


# ─── Streaming call ───────────────────────────────────────────────────────────

def stream_openai(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "none",
    timeout: float = 120.0,
) -> dict:
    """
    Send a streaming chat completion and return timing metrics.

    Returns dict with: ttft_ms, wall_ms, output_tokens, tok_per_s
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 512,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    ttft_ms: float | None = None
    output_tokens = 0
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
                for choice in chunk.get("choices", []):
                    content = choice.get("delta", {}).get("content") or ""
                    if content:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        char_count += len(content)

    wall_ms = (time.perf_counter() - t_start) * 1000

    # Fallback: estimate token count from chars if usage not returned
    if output_tokens == 0 and char_count > 0:
        output_tokens = max(1, char_count // 4)

    gen_ms = wall_ms - (ttft_ms or 0)
    tok_per_s = (output_tokens / gen_ms * 1000) if gen_ms > 0 and output_tokens > 0 else 0.0

    return {
        "ttft_ms": round(ttft_ms) if ttft_ms is not None else None,
        "wall_ms": round(wall_ms),
        "output_tokens": output_tokens,
        "tok_per_s": round(tok_per_s, 1),
    }


# ─── Session position helpers ─────────────────────────────────────────────────

def build_messages(prompt: str, position: str) -> list[dict]:
    """Build the messages array for the requested session position."""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    filler_pairs = {"cold": 0, "warm": 1, "re-warm": 4}.get(position, 0)
    for _ in range(filler_pairs):
        msgs.append({"role": "user", "content": FILLER_USER})
        msgs.append({"role": "assistant", "content": FILLER_ASST})

    msgs.append({"role": "user", "content": prompt})
    return msgs


# ─── Ingest phase ─────────────────────────────────────────────────────────────

def run_cell(
    *,
    base_url: str,
    backend: str,
    model: str,
    prompt_id: str,
    prompt_text: str,
    position: str,
    rep: int,
    no_think: bool = False,
    jsonl_path: Path,
) -> dict:
    text = prompt_text + " /no_think" if no_think else prompt_text
    messages = build_messages(text, position)

    try:
        metrics = stream_openai(base_url, model, messages)
        error = None
    except Exception as exc:
        metrics = {"ttft_ms": None, "wall_ms": None, "output_tokens": 0, "tok_per_s": 0.0}
        error = str(exc)[:200]

    result = {
        "model": model,
        "backend": backend,
        "prompt_id": prompt_id,
        "position": position,
        "rep": rep,
        "no_think": no_think,
        **metrics,
        "error": error,
    }

    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


def run_ingest(
    *,
    mlx_models: list[str],
    skip_mlx: bool,
    reps: int,
    jsonl_path: Path,
) -> None:
    tasks: list[dict] = []

    if not skip_mlx:
        if not mlx_lm_is_running():
            print("  mlx-lm server not running — skipping mlx-lm cells")
        else:
            loaded = get_loaded_mlx_models() or mlx_models[:1]
            for model in loaded:
                is_qwen3 = "qwen3" in model.lower()
                for prompt_id, prompt_text in PROMPTS.items():
                    for position in SESSION_POSITIONS:
                        for rep in range(1, reps + 1):
                            tasks.append(dict(
                                base_url=MLX_LM_BASE, backend="mlx-lm", model=model,
                                prompt_id=prompt_id, prompt_text=prompt_text,
                                position=position, rep=rep, no_think=False,
                            ))
                # Qwen3 /no_think cold variant only
                if is_qwen3:
                    for prompt_id, prompt_text in PROMPTS.items():
                        for rep in range(1, reps + 1):
                            tasks.append(dict(
                                base_url=MLX_LM_BASE, backend="mlx-lm", model=model,
                                prompt_id=prompt_id, prompt_text=prompt_text,
                                position="cold", rep=rep, no_think=True,
                            ))

    total = len(tasks)
    print(f"\n▶ {total} cells  ({reps} reps × {len(PROMPTS)} prompts × {len(SESSION_POSITIONS)} positions)\n")

    for i, task in enumerate(tasks, 1):
        nt = " /no_think" if task.get("no_think") else ""
        label = (
            f"[{i:>{len(str(total))}}/{total}]  "
            f"{task['backend']}/{task['model']}  "
            f"{task['prompt_id']:6}  {task['position']:7}  rep{task['rep']}{nt}"
        )
        print(f"  {label}", end="  ", flush=True)

        result = run_cell(jsonl_path=jsonl_path, **task)

        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            ttft = f"{result['ttft_ms']}ms" if result["ttft_ms"] is not None else "—"
            print(f"TTFT={ttft}  wall={result['wall_ms']}ms  tok/s={result['tok_per_s']}")


# ─── Analysis phase (Haiku) ────────────────────────────────────────────────────

_HAIKU_PROMPT = """\
You are a benchmark analyst. Below is raw JSONL from a latency benchmark.

Each line has: model, backend, prompt_id (short/medium/long),
position (cold/warm/re-warm), rep (1–N), no_think (bool),
ttft_ms, wall_ms, output_tokens, tok_per_s, error (null or string).

Instructions:
1. Skip rows where error is not null.
2. Group by (model, backend, prompt_id, position, no_think).
   For each group compute: median ttft_ms, median wall_ms, median tok_per_s.
3. Produce a markdown table:
   | Model | Backend | Prompt | Position | no_think | TTFT (ms) | Total (ms) | Tok/s |
   Sort by Model, Backend, Prompt, Position.
4. Write exactly three paragraphs with these headings:
   ## Cache behavior
   Compare warm vs cold TTFT for mlx-lm. How much does a cached prefix help,
   and by how much (use numbers)?

   ## Thinking overhead
   How much latency does Qwen3 thinking add over gemma4, and does /no_think
   close the gap? Use numbers.

   ## Recommendation
   Which model + backend + thinking config should be the default, and when to switch?

Return ONLY the markdown. No preamble.

JSONL:
"""


def run_analysis(jsonl_path: Path) -> str:
    raw = jsonl_path.read_text().strip()
    if not raw:
        return "_No benchmark data collected._"

    try:
        result = subprocess.run(
            ["claude", "--model", "claude-haiku-4-5-20251001", "-p", _HAIKU_PROMPT + raw],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        return "_Haiku analysis unavailable: `claude` CLI not found in PATH_"
    except subprocess.TimeoutExpired:
        return "_Haiku analysis timed out (120 s)_"

    if result.returncode != 0:
        return f"_Haiku analysis failed (exit {result.returncode}): {result.stderr[:300]}_"
    return result.stdout.strip()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="mira-core backend benchmark")
    parser.add_argument("--skip-mlx",   action="store_true", help="Skip mlx-lm backend")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help=f"Repetitions per cell (default {DEFAULT_REPS})")
    args = parser.parse_args()

    today = date.today().isoformat()
    jsonl_path   = Path(f"/tmp/mira_benchmark_{today}.jsonl")
    report_path  = Path(f"/tmp/mira_benchmark_report_{today}.md")

    print("=== mira-core Backend Benchmark ===")
    print(f"JSONL  → {jsonl_path}")
    print(f"Report → {report_path}")

    # Discovery
    mlx_models    = [] if args.skip_mlx   else discover_mlx_models()
    mlx_running   = not args.skip_mlx and mlx_lm_is_running()

    print(f"\nmlx-lm  ({MLX_LM_BASE}): {mlx_models or '(none)'}"
          + ("" if mlx_running else "  [server not running]"))

    if not mlx_running:
        print("\nNo backends available. Start mlx-lm.server and retry.")
        sys.exit(1)

    # Ingest
    run_ingest(
        mlx_models=mlx_models,
        skip_mlx=args.skip_mlx,
        reps=args.reps,
        jsonl_path=jsonl_path,
    )

    # Analysis
    print(f"\n▶ Analysing with Claude Haiku…")
    report = run_analysis(jsonl_path)

    # Report
    header = f"## Benchmark Results — {today}\n\n"
    report_path.write_text(header + report + "\n")

    print(f"\n{'=' * 60}")
    print(header + report)
    print(f"\n{'=' * 60}")
    print(f"Report → {report_path}")
    print(f"JSONL  → {jsonl_path}")


if __name__ == "__main__":
    main()
