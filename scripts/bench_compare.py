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
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import requests
import yaml

BASE_URL = "http://localhost:8000"
SCRIPTS_DIR = Path(__file__).parent
SOURCE_REPO = SCRIPTS_DIR.parent  # the mira-core repo root (a git repo)
DOCS_DIR = SCRIPTS_DIR.parent / "docs"
QUESTIONS_FILE = SCRIPTS_DIR / "bench_questions.yaml"
SERVER_PY = SCRIPTS_DIR.parent / "server.py"


def _auth_headers() -> dict:
    """Bearer header from mira.yaml's auth_token, when the server has one configured."""
    try:
        cfg = yaml.safe_load((SOURCE_REPO / "mira.yaml").read_text())
        token = cfg.get("auth_token", "")
    except Exception:
        token = ""
    return {"Authorization": f"Bearer {token}"} if token else {}


HEADERS = _auth_headers()

# Conversation ids created during a run, deleted in teardown (see feedback_bench_cleanup).
_created_convs: list[str] = []


def get_project_id(project_name: str) -> tuple[str, str]:
    """Look up a project by name. Returns (project_id, local_path). Raises if not found."""
    resp = requests.get(f"{BASE_URL}/projects", headers=HEADERS, timeout=5)
    resp.raise_for_status()
    payload = resp.json()
    projects = payload["projects"] if isinstance(payload, dict) else payload
    for p in projects:
        if p["name"].lower() == project_name.lower():
            return p["id"], p.get("local_path", "")
    names = [p["name"] for p in projects]
    raise ValueError(f"Project '{project_name}' not found. Available: {names}")


def create_bench_conversation(project_id: str | None = None) -> str:
    """Create a fresh conversation (optionally project-scoped) and return its ID.

    Force-titled "bench-<ts>-<id6>" immediately (never left as "New conversation") so any
    conversation that survives a crashed run — teardown never ran, no project_id to key off —
    is still trivially identifiable as bench debris on the next sweep. See [[feedback_bench_cleanup]].
    """
    payload = {"project_id": project_id or ""}
    resp = requests.post(f"{BASE_URL}/conversations", json=payload, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    conv_id = resp.json()["id"]
    _created_convs.append(conv_id)
    title = f"bench-{int(time.time())}-{conv_id[:6]}"
    try:
        requests.patch(f"{BASE_URL}/conversations/{conv_id}", json={"title": title}, headers=HEADERS, timeout=5)
    except Exception as e:
        print(f"  warning: failed to force-title bench conversation {conv_id}: {e}")
    return conv_id


# ── Throwaway git worktree (isolate agentic runs from the live repo) ────────────

def make_worktree(source_repo: Path) -> tuple[Path, Path]:
    """Create a detached git worktree at HEAD in a temp dir, outside the source repo.

    Returns (tmp_base, worktree_path). The worktree reflects the *committed* HEAD —
    uncommitted working-tree changes are intentionally excluded (reproducible).
    """
    tmp_base = Path(tempfile.mkdtemp(prefix="mira-bench-wt-"))
    wt = tmp_base / "wt"
    subprocess.run(
        ["git", "-C", str(source_repo), "worktree", "add", "--detach", str(wt), "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return tmp_base, wt


def sweep_orphaned_bench_projects() -> None:
    """Delete leftover bench debris (conversations + bench-wt-* projects) from prior runs
    that never reached teardown (e.g. SIGKILL, terminal closed, crash before the outer
    try/finally).

    Runs at the very start of main(), before this run creates anything of its own — so
    every conversation titled "bench-*" and every "bench-wt-*" project seen here belongs
    to a previous, unfinished run and is safe to delete unconditionally.
    """
    try:
        conversations = requests.get(f"{BASE_URL}/conversations", headers=HEADERS, timeout=5).json()["conversations"]
    except Exception as e:
        print(f"  sweep: could not list conversations, skipping: {e}")
        conversations = []

    orphan_convs = [c for c in conversations if c["title"].startswith("bench-")]
    if orphan_convs:
        print(f"Sweeping {len(orphan_convs)} orphaned bench conversation(s) from interrupted runs...")
        for c in orphan_convs:
            try:
                requests.delete(f"{BASE_URL}/conversations/{c['id']}", headers=HEADERS, timeout=5)
            except Exception as e:
                print(f"  sweep: failed to delete conversation {c['id']}: {e}")

    try:
        projects = requests.get(f"{BASE_URL}/projects", headers=HEADERS, timeout=5).json()["projects"]
    except Exception as e:
        print(f"  sweep: could not list projects, skipping: {e}")
        return

    orphan_projects = [
        p for p in projects
        if p["name"].startswith("bench-wt-") and not Path(p["local_path"]).exists()
    ]
    if not orphan_projects:
        return

    print(f"Sweeping {len(orphan_projects)} orphaned bench-wt-* project(s) from interrupted runs...")
    for p in orphan_projects:
        for c in conversations:
            if c.get("project_id") == p["id"]:
                try:
                    requests.delete(f"{BASE_URL}/conversations/{c['id']}", headers=HEADERS, timeout=5)
                except Exception as e:
                    print(f"  sweep: failed to delete conversation {c['id']}: {e}")
        try:
            requests.delete(f"{BASE_URL}/projects/{p['id']}", headers=HEADERS, timeout=5)
        except Exception as e:
            print(f"  sweep: failed to delete project {p['id']}: {e}")


def register_throwaway_project(local_path: Path) -> tuple[str, str]:
    """Register a temporary project pointing at the worktree. Returns (id, local_path)."""
    name = f"bench-wt-{int(time.time())}"
    resp = requests.post(
        f"{BASE_URL}/projects",
        json={"name": name, "local_path": str(local_path)},
        headers=HEADERS,
        timeout=5,
    )
    resp.raise_for_status()
    p = resp.json()
    return p["id"], p.get("local_path", str(local_path))


def teardown(source_repo: Path | None, tmp_base: Path | None, wt: Path | None,
             project_id: str | None) -> None:
    """Best-effort cleanup — each step is isolated so one failure doesn't block the rest.

    Order matters: the worktree folder is removed *before* the project is deleted, because
    DELETE /projects refuses to drop an entry whose local_path still exists on disk.
    """
    for cid in _created_convs:
        try:
            requests.delete(f"{BASE_URL}/conversations/{cid}", headers=HEADERS, timeout=5)
        except Exception as e:
            print(f"  teardown: failed to delete conversation {cid}: {e}")
    _created_convs.clear()

    if source_repo and wt:
        try:
            subprocess.run(
                ["git", "-C", str(source_repo), "worktree", "remove", "--force", str(wt)],
                check=True, capture_output=True, text=True,
            )
        except Exception as e:
            print(f"  teardown: failed to remove worktree {wt}: {e}")
    if tmp_base:
        shutil.rmtree(tmp_base, ignore_errors=True)
    if source_repo:
        try:
            subprocess.run(
                ["git", "-C", str(source_repo), "worktree", "prune"],
                check=True, capture_output=True, text=True,
            )
        except Exception:
            pass

    if project_id:
        # Folder is now gone, so the local_path-exists guard in DELETE /projects passes.
        try:
            requests.delete(f"{BASE_URL}/projects/{project_id}", headers=HEADERS, timeout=5)
        except Exception as e:
            print(f"  teardown: failed to delete project {project_id}: {e}")


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE) as f:
        data = yaml.safe_load(f)
    return data["questions"]


def make_prompt(q: dict) -> str | None:
    """Return the prompt string for a single-turn question, or None for multi-turn (handled separately)."""
    if q.get("turns", 1) > 1:
        return None
    return q["prompt"]


MAX_TOOL_CALLS = 25
WALL_TIMEOUT_S = 600  # hard wall-clock limit per question (10 min)


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
    # /chat takes Form fields, not JSON.
    # `tools` was declared per question for a year and never sent, so every
    # question ran with the full agentic toolset and a `tools: false` question
    # could still short-circuit itself by calling task_done (bench Q4, 2026-08-01).
    form_data = {
        "message": prompt,
        "conversation_id": conversation_id,
        "thinking_enabled": str(thinking).lower(),
        "tools_enabled": str(tools).lower(),
    }

    t_start = time.perf_counter()
    ttft_ms = None
    content_parts = []
    tool_calls = []
    task_done_fired = False
    divergence_guard_fired = False
    eval_tokens = None
    eval_tps = None

    try:
        with requests.post(f"{BASE_URL}/chat", data=form_data, headers=HEADERS, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if time.perf_counter() - t_start > WALL_TIMEOUT_S:
                    return {"error": f"wall-clock timeout after {WALL_TIMEOUT_S}s ({len(tool_calls)} tool calls)"}
                if len(tool_calls) > MAX_TOOL_CALLS:
                    return {"error": f"too many tool calls ({len(tool_calls)}), likely infinite loop"}
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

                elif event_type == "divergence_guard":
                    divergence_guard_fired = True

                elif event_type == "error":
                    wall_ms = round((time.perf_counter() - t_start) * 1000)
                    return {"error": event.get("message", "server error"), "tool_calls": tool_calls, "wall_ms": wall_ms, "divergence_guard_fired": divergence_guard_fired}

                elif event_type == "done":
                    # Capture content from done event as fallback (e.g. forced summary injected by orchestrator)
                    done_content = event.get("content", "")
                    if done_content and not content_parts:
                        content_parts.append(done_content)
                    if event.get("task_done"):
                        task_done_fired = True

                elif event_type == "stats":
                    eval_tokens = event.get("output_tokens")

    except requests.exceptions.Timeout:
        return {"error": "timeout after 60s (no data from server)"}
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
        "divergence_guard_fired": divergence_guard_fired,
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


def run_benchmark(model: str, questions: list[dict], project_id: str | None = None, workspace_root: str = "") -> list[dict]:
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
            prompt = q["prompt"].replace("{workspace_root}", workspace_root) if workspace_root else q["prompt"]
            result = stream_chat(
                prompt,
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
            guard_str = " divergence_guard=YES" if result.get("divergence_guard_fired") else ""
            print(f" TTFT={result['ttft_ms']}ms wall={result['wall_ms']}ms{tps_str}{tools_str}{done_str}{guard_str}")

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
                        help="[--no-worktree only] Existing project to scope agentic questions "
                             "to its live local_path. Ignored in the default worktree mode.")
    parser.add_argument("--source-repo", type=str, default=str(SOURCE_REPO),
                        help=f"Repo to snapshot into the throwaway worktree (default: {SOURCE_REPO}). "
                             "The worktree reflects committed HEAD, not uncommitted changes.")
    parser.add_argument("--no-worktree", action="store_true",
                        help="Disable worktree isolation and run agentic questions against the "
                             "live --project-name repo (WILL mutate it). Explicit opt-in only.")
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

    sweep_orphaned_bench_projects()

    has_agentic = any(q.get("tools") or q.get("turns", 1) > 1 for q in questions)

    # Resolve the workspace for agentic questions.
    project_id = None
    workspace_root = ""
    source_repo = tmp_base = wt = None  # for teardown
    if has_agentic and not args.no_worktree:
        # Default: isolate in a throwaway git worktree — the live repo is never touched.
        source_repo = Path(args.source_repo).expanduser().resolve()
        try:
            tmp_base, wt = make_worktree(source_repo)
            project_id, workspace_root = register_throwaway_project(wt)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: git worktree add failed: {e.stderr or e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: could not provision throwaway worktree: {e}")
            teardown(source_repo, tmp_base, wt, project_id)
            sys.exit(1)
        print(f"Throwaway worktree: {wt}  (project {project_id}, from {source_repo}@HEAD)")
        if args.project_name:
            print(f"  Note: --project-name '{args.project_name}' ignored in worktree mode.")
    elif args.no_worktree and args.project_name:
        try:
            project_id, workspace_root = get_project_id(args.project_name)
            print(f"--no-worktree: running against LIVE project '{args.project_name}' → id={project_id} "
                  f"(this WILL mutate {workspace_root})")
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    elif has_agentic:
        print("WARNING: agentic questions present but --no-worktree set without --project-name. "
              "Local tools (run_shell, read_file) will be unavailable.")

    try:
        today = date.today().isoformat()
        raw_dir = SCRIPTS_DIR
        docs_dir = DOCS_DIR
        docs_dir.mkdir(exist_ok=True)

        all_results: dict[str, list[dict]] = {}

        for model in args.model:
            print(f"\nBenchmarking {model}...")
            results = run_benchmark(model, questions, project_id=project_id, workspace_root=workspace_root)
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
            header = f"# Benchmark Results — {today}\n\nHardware: MacBook Pro M5 32GB (backend/model per run — see sections below)\n\n"
            results_path.write_text(header + md)

        print(f"\nResults written to {results_path}")
        print("\nNext: fill in manual quality scores in the results file.")
    finally:
        # Always clean up: bench conversations, throwaway project, and the worktree.
        # (Skip project/worktree teardown in --no-worktree mode where they're the live repo.)
        teardown(source_repo, tmp_base, wt, project_id if not args.no_worktree else None)


if __name__ == "__main__":
    main()
