#!/usr/bin/env python3
"""
bench_compare.py — Mira model benchmark runner.

Usage:
    python scripts/bench_compare.py --model gemma4:26b-mlx --project-name mira-core
    python scripts/bench_compare.py --model qwen3.6:35b-mlx --project-name mira-core
    python scripts/bench_compare.py --model gemma4:26b-mlx --model qwen3.6:35b-mlx --project-name mira-core

--project-name is required for agentic questions (Q6–Q9) to make run_shell and read_file available.
Without it, local tools are filtered out and agentic questions will fail.

Outputs:
    docs/bench-results-YYYY-MM-DD.md   (created/appended)
    bench_raw_YYYY-MM-DD_MODEL.jsonl   (raw timing per question, in scripts/)
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests
import yaml

BASE_URL = "http://localhost:8000"
SCRIPTS_DIR = Path(__file__).parent
DOCS_DIR = SCRIPTS_DIR.parent / "docs"
QUESTIONS_FILE = SCRIPTS_DIR / "bench_questions.yaml"
SERVER_PY = SCRIPTS_DIR.parent / "server.py"


def get_project_id(project_name: str) -> str:
    """Look up a project by name and return its ID. Raises if not found."""
    resp = requests.get(f"{BASE_URL}/projects", timeout=5)
    resp.raise_for_status()
    projects = resp.json()
    for p in projects:
        if p["name"].lower() == project_name.lower():
            return p["id"]
    names = [p["name"] for p in projects]
    raise ValueError(f"Project '{project_name}' not found. Available: {names}")


def create_bench_conversation(project_id: str | None = None) -> str:
    """Create a fresh conversation (optionally project-scoped) and return its ID."""
    payload = {"project_id": project_id or ""}
    resp = requests.post(f"{BASE_URL}/conversations", json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()["id"]


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE) as f:
        data = yaml.safe_load(f)
    return data["questions"]


def make_prompt(q: dict) -> str | None:
    """Return the prompt string for a single-turn question, or None for multi-turn (handled separately)."""
    if q.get("turns", 1) > 1:
        return None
    return q["prompt"]


def stream_chat(prompt: str, model: str, thinking: bool, tools: bool, conversation_id: str) -> dict:
    """
    POST /chat (multipart/form-data) and consume the SSE stream.
    Returns:
        ttft_ms       — ms to first content token
        wall_ms       — total wall time ms
        content       — full response text
        tool_calls    — list of tool names called
        task_done     — bool, whether task_done event arrived
        eval_tokens   — token count from done event (if present)
        eval_tps      — tokens/sec from done event (if present)
    """
    # /chat takes Form fields, not JSON
    form_data = {
        "message": prompt,
        "conversation_id": conversation_id,
        "thinking_enabled": str(thinking).lower(),
    }

    t_start = time.perf_counter()
    ttft_ms = None
    content_parts = []
    tool_calls = []
    task_done_fired = False
    eval_tokens = None
    eval_tps = None

    try:
        with requests.post(f"{BASE_URL}/chat", data=form_data, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "token" and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000

                if event_type == "token":
                    content_parts.append(event.get("content", ""))

                elif event_type == "tool_start":
                    tool_name = event.get("tool", "unknown")
                    if tool_name == "task_done":
                        task_done_fired = True
                    else:
                        tool_calls.append(tool_name)

                elif event_type == "done":
                    # done event carries no timing metadata in current server — wall time is our measure
                    pass

                elif event_type == "stats":
                    eval_tokens = event.get("output_tokens")

    except requests.exceptions.Timeout:
        return {"error": "timeout after 300s"}
    except Exception as e:
        return {"error": str(e)}

    wall_ms = (time.perf_counter() - t_start) * 1000

    # Derive t/s from wall time and token count (TTFT to end = generation phase)
    if eval_tokens and ttft_ms:
        gen_s = (wall_ms - ttft_ms) / 1000
        eval_tps = round(eval_tokens / gen_s, 1) if gen_s > 0 else None

    return {
        "ttft_ms": round(ttft_ms) if ttft_ms else None,
        "wall_ms": round(wall_ms),
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "task_done": task_done_fired,
        "eval_tokens": eval_tokens,
        "eval_tps": eval_tps,
    }


def run_multi_turn(q: dict, model: str) -> dict:
    """Run a 2-turn question (Q10). Uses a fresh conversation."""
    conv_id = create_bench_conversation()

    # Turn 1: inject server.py contents
    if not SERVER_PY.exists():
        return {"error": f"server.py not found at {SERVER_PY}"}
    server_content = SERVER_PY.read_text()
    turn1_prompt = (
        f"I'm going to share a Python file with you. Here are its contents:\n\n"
        f"```python\n{server_content}\n```\n\n"
        f"Acknowledge that you've read it."
    )

    print(f"    Turn 1 (inject server.py, {len(server_content):,} chars)...", end="", flush=True)
    r1 = stream_chat(turn1_prompt, model, thinking=False, tools=False, conversation_id=conv_id)
    if "error" in r1:
        return r1
    print(f" {r1['wall_ms']}ms")

    # Turn 2: actual question
    print(f"    Turn 2 (retrieval)...", end="", flush=True)
    r2 = stream_chat(q["prompt_turn2"], model, thinking=False, tools=False, conversation_id=conv_id)
    if "error" in r2:
        return r2
    print(f" {r2['wall_ms']}ms")

    return {
        "ttft_ms": r2["ttft_ms"],
        "wall_ms": r1["wall_ms"] + r2["wall_ms"],
        "content": r2["content"],
        "tool_calls": [],
        "task_done": False,
        "eval_tokens": r2["eval_tokens"],
        "eval_tps": r2["eval_tps"],
        "multi_turn": True,
        "turn1_wall_ms": r1["wall_ms"],
        "turn2_wall_ms": r2["wall_ms"],
    }


def run_benchmark(model: str, questions: list[dict], project_id: str | None = None) -> list[dict]:
    results = []
    for q in questions:
        qid = q["id"]
        category = q["category"]
        print(f"  Q{qid} [{category}]...", end="", flush=True)

        if q.get("turns", 1) > 1:
            result = run_multi_turn(q, model)
        else:
            # Fresh conversation per question; scoped to project for agentic questions
            use_project = project_id if q.get("tools") else None
            conv_id = create_bench_conversation(use_project)
            result = stream_chat(
                q["prompt"],
                model,
                thinking=q.get("thinking", False),
                tools=q.get("tools", False),
                conversation_id=conv_id,
            )

        result["id"] = qid
        result["model"] = model

        if "error" in result:
            print(f" ERROR: {result['error']}")
        else:
            tps_str = f" @ {result['eval_tps']} t/s" if result["eval_tps"] else ""
            tools_str = f" tools={result['tool_calls']}" if result["tool_calls"] else ""
            done_str = " task_done=YES" if result["task_done"] else ""
            print(f" TTFT={result['ttft_ms']}ms wall={result['wall_ms']}ms{tps_str}{tools_str}{done_str}")

        results.append(result)
    return results


def format_markdown_table(all_results: dict[str, list[dict]], questions: list[dict]) -> str:
    models = list(all_results.keys())
    today = date.today().isoformat()

    lines = [
        f"## Benchmark Results — {today}",
        "",
        "### Timing",
        "",
    ]

    # Header
    header = "| Q | Difficulty | Category |"
    sep = "|---|-----------|---------|"
    for m in models:
        short = m.split(":")[0].replace("gemma4", "gemma4").replace("qwen3.6", "qwen3.6")
        tag = m.split(":")[-1] if ":" in m else m
        header += f" {short}:{tag} TTFT | wall | t/s |"
        sep += "---|---|---|"
    lines.append(header)
    lines.append(sep)

    q_map = {q["id"]: q for q in questions}
    for q in questions:
        qid = q["id"]
        row = f"| {qid} | {q['difficulty']} | {q['category']} |"
        for m in models:
            res = next((r for r in all_results[m] if r["id"] == qid), None)
            if res is None or "error" in res:
                err = res["error"] if res else "missing"
                row += f" ERR: {err} | — | — |"
            else:
                ttft = f"{res['ttft_ms']}ms" if res["ttft_ms"] else "—"
                wall = f"{res['wall_ms']/1000:.1f}s"
                tps = f"{res['eval_tps']}" if res["eval_tps"] else "—"
                row += f" {ttft} | {wall} | {tps} |"
        lines.append(row)

    lines += ["", "### Agentic results", ""]
    agentic_qs = [q for q in questions if q.get("tools") or q.get("turns", 1) > 1]
    if agentic_qs:
        lines.append("| Q | Category | Expected calls |" + "".join(f" {m} calls | task_done |" for m in models))
        lines.append("|---|---------|----------------|" + "".join("---|---|" for _ in models))
        for q in agentic_qs:
            qid = q["id"]
            row = f"| {qid} | {q['category']} | {q.get('expected_tool_calls', '?')} |"
            for m in models:
                res = next((r for r in all_results[m] if r["id"] == qid), None)
                if res is None or "error" in res:
                    row += " ERR | — |"
                else:
                    calls = ", ".join(res["tool_calls"]) if res["tool_calls"] else "none"
                    done = "YES" if res["task_done"] else "no"
                    row += f" {calls} | {done} |"
            lines.append(row)

    lines += [
        "",
        "### Manual quality scores (fill in after review)",
        "",
        "Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct",
        "",
    ]
    lines.append("| Q | Difficulty | Category |" + "".join(f" {m} score |" for m in models))
    lines.append("|---|-----------|---------|" + "".join("---|" for _ in models))
    for q in questions:
        row = f"| {q['id']} | {q['difficulty']} | {q['category']} |"
        for _ in models:
            row += " — |"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Mira model benchmark runner")
    parser.add_argument("--model", action="append", required=True, help="Model tag(s) to benchmark")
    parser.add_argument("--questions", type=str, default=None, help="Comma-separated question IDs (default: all)")
    parser.add_argument("--project-name", type=str, default=None,
                        help="Project name to scope agentic questions (enables run_shell/read_file). "
                             "Must match a project in Mira's project list.")
    args = parser.parse_args()

    questions = load_questions()
    if args.questions:
        ids = {int(x) for x in args.questions.split(",")}
        questions = [q for q in questions if q["id"] in ids]

    # Check server is up
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: Mira server not reachable at {BASE_URL}: {e}")
        sys.exit(1)

    # Resolve project for agentic questions
    project_id = None
    if args.project_name:
        try:
            project_id = get_project_id(args.project_name)
            print(f"Project '{args.project_name}' → id={project_id}")
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        agentic_qs = [q for q in questions if q.get("tools")]
        if agentic_qs:
            print("WARNING: agentic questions present but --project-name not set. "
                  "Local tools (run_shell, read_file) will be unavailable.")

    today = date.today().isoformat()
    raw_dir = SCRIPTS_DIR
    docs_dir = DOCS_DIR
    docs_dir.mkdir(exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    for model in args.model:
        print(f"\nBenchmarking {model}...")
        results = run_benchmark(model, questions, project_id=project_id)
        all_results[model] = results

        raw_path = raw_dir / f"bench_raw_{today}_{model.replace(':', '_').replace('/', '_')}.jsonl"
        with open(raw_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  Raw results: {raw_path}")

    # Write markdown results
    md = format_markdown_table(all_results, questions)
    results_path = docs_dir / f"bench-results-{today}.md"

    if results_path.exists():
        existing = results_path.read_text()
        results_path.write_text(existing + "\n\n---\n\n" + md)
    else:
        header = f"# Benchmark Results — {today}\n\nHardware: MacBook Pro M5 32GB · Ollama 0.24.0\n\n"
        results_path.write_text(header + md)

    print(f"\nResults written to {results_path}")
    print("\nNext: fill in manual quality scores in the results file.")


if __name__ == "__main__":
    main()
