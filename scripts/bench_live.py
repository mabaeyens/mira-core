#!/usr/bin/env python3
"""Live-with bench runner: real work through Mira's own harness, one model per run.

See specs/live-with-bench.md. The corpus (code explanation, trips, unit4 tasks,
self-contained session questions, one long-context prompt) lives in a gitignored
notes/ folder because part of it comes from private sessions.

    python scripts/bench_live.py --label qwen --out notes/x/qwen.jsonl
    python scripts/bench_live.py --label inkling --out notes/x/ink.jsonl \\
        --config-override '{"model": "/path", "mira_mlx_mtp_enabled": false}'

Same prompts, same order, same thinking budget for every model. Thinking is left
to Mira's own per-turn decision, so the toggle is part of what is measured.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_compare as bc  # noqa: E402
from bench_gpqa_mira import FootprintSampler  # noqa: E402

REPO = HERE.parent
DEFAULT_CORPUS = REPO / "notes" / "live_with_2026-11-25"
SET_FILES = {
    "code": "code_set.jsonl",
    "trip": "trip_set.jsonl",
    "unit4": "unit4_set.jsonl",
    "session": "session_set.jsonl",
    "long": "long_set.jsonl",
}


def load_items(corpus: Path, sets: list[str]) -> list[dict]:
    items = []
    for name in sets:
        path = corpus / SET_FILES[name]
        items += [json.loads(line) for line in path.open()]
    return items


def build_prompt(item: dict, corpus: Path, workspace_root: str) -> str:
    prompt = item["prompt"]
    if item.get("file_name"):
        prompt += (f"\n\nThe attached file is at {workspace_root}/unit4_files/{item['file_name']}. "
                   f"Read it with your tools.")
    if item.get("inject_file"):
        text = (REPO / item["inject_file"]).read_text()
        prompt = f"{prompt}\n\n```python\n{text}\n```"
    return prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--sets", default="code,trip,unit4,session,long")
    ap.add_argument("--budget", type=int, default=8192, help="max_thinking_tokens, same for every model")
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    ap.add_argument("--config-override", default="")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    items = load_items(corpus, args.sets.split(","))
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.label}: {len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)

    overrides = {"max_thinking_tokens": args.budget}
    if args.config_override:
        overrides.update(json.loads(args.config_override))
    bc.CONFIG_OVERRIDES.update(overrides)

    server = bc.IsolatedServer()
    sampler = FootprintSampler()
    source_repo = REPO
    tmp_base = wt = project_id = None
    try:
        server.start()
        bc.sweep_orphaned_bench_projects()
        tmp_base, wt = bc.make_worktree(source_repo)
        project_id, workspace_root = bc.register_throwaway_project(wt)
        (Path(workspace_root) / "unit4_files").mkdir(exist_ok=True)
        for f in (corpus / "unit4_files").glob("*"):
            shutil.copy(f, Path(workspace_root) / "unit4_files" / f.name)
        sampler.start()

        for n, item in enumerate(todo):
            conv = bc.create_bench_conversation(project_id)
            prompt = build_prompt(item, corpus, workspace_root)
            tools = bool(item.get("tools"))
            t0 = time.time()
            res = bc.stream_chat(prompt, args.label, None, tools, conv)
            row = {
                "id": item["id"], "set": item["set"], "label": args.label,
                "cold": n == 0 and not done,
                "ttft_ms": res.get("ttft_ms"), "wall_ms": res.get("wall_ms"),
                "content": res.get("content", ""),
                "thinking_chars": len(res.get("thinking") or ""),
                "tool_calls": res.get("tool_calls", []),
                "tool_labels": res.get("tool_labels", []),
                "fetch_urls": res.get("fetch_urls", []),
                "search_queries": res.get("search_queries", []),
                "eval_tokens": res.get("eval_tokens"), "eval_tps": res.get("eval_tps"),
                "error": res.get("error"),
                "peak_footprint_gb": sampler.peak_gb,
                "started_at": t0,
            }
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {item['id']}: tok={row['eval_tokens']} wall={round((row['wall_ms'] or 0)/1000)}s "
                  f"ttft={round((row['ttft_ms'] or 0)/1000)}s tools={len(row['tool_calls'])} "
                  f"err={row['error']}", flush=True)
    finally:
        sampler.stop_flag.set()
        bc.teardown(source_repo, tmp_base, wt, project_id)
        server.stop()


if __name__ == "__main__":
    main()
