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
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import requests
import yaml

# Which server the bench drives. It defaults to the live one, which is the whole
# problem this knob exists to solve: a bench run's conversations are written by
# whichever server serves them, so pointing the bench at the production server
# puts bench traffic in the user's own history for the duration of the run. The
# teardown deletes them afterwards, but "deleted afterwards" is not "never shown"
# — they are visible in the app for the whole run, and they survive permanently
# if the run is interrupted before teardown.
#
# MIRA_DATA_DIR (core/config.py) cannot fix that from here: it configures the
# process that owns the database, and the bench does not own it — it drives an
# already-running server over HTTP. The fix is to point this at a server started
# with its own MIRA_DATA_DIR, which is what --server is for.
DEFAULT_BASE_URL = "http://localhost:8000"
BASE_URL = os.getenv("MIRA_BENCH_SERVER", DEFAULT_BASE_URL)
SCRIPTS_DIR = Path(__file__).parent
SOURCE_REPO = SCRIPTS_DIR.parent  # the mira-core repo root (a git repo)
DOCS_DIR = SCRIPTS_DIR.parent / "docs"
QUESTIONS_FILE = SCRIPTS_DIR / "bench_questions.yaml"
# (SERVER_PY removed 2026-08-08: the multi-turn question now names its own
# inject_file, because hardcoding server.py here is what made Q10 ask about code
# that was not in the file it shared.)


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

LAUNCH_AGENT = "com.mab.mira"
LAUNCH_AGENT_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT}.plist"
ENGINE_PATTERN = "core.inference.mira_mlx_server"


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for(predicate, timeout: float, interval: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class IsolatedServer:
    """Run the bench against its own Mira server, with production stopped.

    Why production has to stop rather than run alongside: `server.py` binds :8000
    unconditionally, both instances would drive the single inference backend on
    :8080, and a second API process loads its own embedding and reranker models
    on a machine that is already holding a ~19GB LLM. One server at a time is the
    only configuration that is both correct and measurable.

    What this buys over the teardown that already existed: the bench's
    conversations are written to a throwaway MIRA_DATA_DIR and never touch the
    real conversations.db at all. The old cleanup deleted them afterwards, which
    left them visible in the app for the whole run and left them permanently if
    the run was interrupted before teardown. Isolation does not depend on the
    run finishing.

    Restoring production is registered with atexit and on SIGINT/SIGTERM, because
    the failure this must not have is leaving Mira down after a crashed bench.
    """

    def __init__(self) -> None:
        self.data_dir: str | None = None
        self.proc: subprocess.Popen | None = None
        self.agent_was_loaded = False
        self._restored = False
        self._stopped = False

    # -- production ----------------------------------------------------------
    @staticmethod
    def _agent_loaded() -> bool:
        return subprocess.run(["launchctl", "list", LAUNCH_AGENT],
                              capture_output=True).returncode == 0

    def _stop_production(self) -> None:
        self.agent_was_loaded = self._agent_loaded()
        if self.agent_was_loaded:
            print(f"  stopping production ({LAUNCH_AGENT})...")
            subprocess.run(["launchctl", "unload", str(LAUNCH_AGENT_PLIST)], check=False)
        else:
            print(f"  production ({LAUNCH_AGENT}) was not loaded; leaving it that way")

        # The engine is a child of the server process and can outlive it. Kill it
        # so the bench server cold-loads: a bench must never inherit a warm cache
        # it did not create, or the first question measures someone else's state.
        subprocess.run(["pkill", "-f", ENGINE_PATTERN], check=False)
        if not _wait_for(lambda: not _port_is_open(8000) and not _port_is_open(8080), timeout=60):
            raise RuntimeError(
                "port 8000 or 8080 still open after stopping production — refusing to "
                "start a second server on top of whatever is holding it"
            )

    def _restore_production(self) -> None:
        if self._restored:
            return
        self._restored = True
        if not self.agent_was_loaded:
            print("  production was not running before the bench; not starting it")
            return
        print(f"  restoring production ({LAUNCH_AGENT})...")
        subprocess.run(["launchctl", "load", str(LAUNCH_AGENT_PLIST)], check=False)

        # Wait for the BACKEND, not just the port. :8000 answers within seconds
        # while the model is still loading, so returning there reports success
        # over a Mira that cannot answer anything — and leaves the process free
        # to run more teardown while production is at its most fragile.
        def backend_up() -> bool:
            try:
                r = requests.get(f"{DEFAULT_BASE_URL}/health", timeout=3)
                return r.status_code == 200 and r.json().get("backend_ready") is True
            except Exception:
                return False

        if backend_up() or _wait_for(backend_up, timeout=300, interval=3):
            print("  production is back up, backend loaded")
        elif _port_is_open(8000):
            print("  WARNING: production is serving but its backend never became ready — "
                  "check `tail /tmp/com.mab.mira.log`")
        else:
            print("  WARNING: production did not come back up — check `launchctl list com.mab.mira`")

    # -- bench server --------------------------------------------------------
    def start(self) -> None:
        # Registered before anything is stopped: if the very next call fails
        # halfway, the handler still puts production back.
        atexit.register(self.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._on_signal)

        self._stop_production()

        self.data_dir = tempfile.mkdtemp(prefix="mira_bench_data_")
        env = os.environ.copy()
        env["MIRA_DATA_DIR"] = self.data_dir
        print(f"  bench data dir: {self.data_dir}  (thrown away afterwards)")

        self.proc = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=str(SOURCE_REPO), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("  starting bench server (cold model load, this takes a minute)...")

        def ready() -> bool:
            if self.proc.poll() is not None:
                raise RuntimeError(f"bench server exited early (code {self.proc.returncode})")
            try:
                r = requests.get(f"{DEFAULT_BASE_URL}/health", timeout=3)
                return r.status_code == 200 and r.json().get("backend_ready") is True
            except Exception:
                return False

        if not _wait_for(ready, timeout=420, interval=3):
            raise RuntimeError("bench server never reported backend_ready")
        print("  bench server ready")

    def _on_signal(self, signum, frame):  # noqa: ARG002
        self.stop()
        sys.exit(130)

    def stop(self) -> None:
        # Idempotent, and the guard is load-bearing rather than tidy. stop() is
        # called explicitly at the end of the run AND again by atexit, and the
        # pkill below is not specific to this script's engine — it matches any
        # mira-mlx process. On 2026-08-08 the second call fired moments after
        # production had been restored and killed the engine production was in
        # the middle of loading, leaving Mira up with no backend and no retry.
        # _restored alone did not cover it: only the restore was guarded.
        if self._stopped:
            return
        self._stopped = True

        if self.proc is not None and self.proc.poll() is None:
            print("  stopping bench server...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        # Same reason as on the way in: never hand a warm engine to whatever runs next.
        subprocess.run(["pkill", "-f", ENGINE_PATTERN], check=False)
        if self.data_dir:
            shutil.rmtree(self.data_dir, ignore_errors=True)
            self.data_dir = None
        self._restore_production()


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
    """Run a 2-turn question (Q10). Uses a fresh conversation.

    The injected file comes from the question's `inject_file`, not from a
    constant. It was hardcoded to server.py while Q10 asked about the divergence
    guard, which lives in core/orchestrator.py — so a correct model had to
    contradict the question and the item could not distinguish good retrieval
    from a confident hallucination.
    """
    conv_id = create_bench_conversation()

    rel = q.get("inject_file")
    if not rel:
        return {"error": f"Q{q['id']} is multi-turn but declares no inject_file"}
    target = (SOURCE_REPO / rel).resolve()
    if not target.exists():
        return {"error": f"inject_file not found: {target}"}
    server_content = target.read_text()
    turn1_prompt = (
        f"I'm going to share a Python file with you. Here are its contents:\n\n"
        f"```python\n{server_content}\n```\n\n"
        f"Acknowledge that you've read it."
    )

    print(f"    Turn 1 (inject {rel}, {len(server_content):,} chars)...", end="", flush=True)
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


ARTIFACT_MAX_BYTES = 8192


def _truth_core_py_line_count(workspace: Path) -> int:
    """Total lines across core/**/*.py, excluding __pycache__ — Q6's ground truth."""
    total = 0
    for p in sorted((workspace / "core").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            with open(p, "rb") as fh:
                total += sum(1 for _ in fh)
        except OSError:
            continue
    return total


def _truth_todo_fixme_count(workspace: Path) -> int:
    """TODO/FIXME occurrences under the workspace — Q7's reference count."""
    skip = {".venv", ".git", "__pycache__", "node_modules"}
    count = 0
    for p in workspace.rglob("*"):
        if not p.is_file() or skip & set(p.parts):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        count += text.count("TODO") + text.count("FIXME")
    return count


# Computed in Python rather than by shelling out: the constraint against
# shell=True is repo-wide, and piping find into wc is also the exact command the
# model under test is being asked to produce, so deriving truth the same way
# would score the model against its own approach instead of against the tree.
_TRUTH_PROBES = {
    "core_py_line_count": _truth_core_py_line_count,
    "todo_fixme_count": _truth_todo_fixme_count,
}


def _capture_truth(q: dict, workspace_root: str):
    """Ground truth for a question, computed against the tree the model saw.

    At bench time, not at eval time: "how many lines are in core/" belongs to the
    commit the run happened on. Scoring a week-old run against today's checkout
    would report the repo's growth as a model regression.
    """
    name = (q.get("check") or {}).get("truth")
    if not name or not workspace_root:
        return None
    probe = _TRUTH_PROBES.get(name)
    if probe is None:
        print(f"    warning: Q{q['id']} names unknown truth probe {name!r}")
        return None
    try:
        return probe(Path(workspace_root))
    except Exception as exc:  # noqa: BLE001 — evidence gathering must not fail a run
        print(f"    warning: truth probe {name} failed: {exc}")
        return None


def _capture_artifacts(q: dict, workspace_root: str) -> dict:
    """Read the files a question was supposed to produce, immediately after it ran.

    Scoring happens offline against the raw jsonl, but agentic questions write
    into a throwaway git worktree that teardown deletes — so by the time anything
    scores the run, the evidence is gone. Capturing here, per question, is
    cheaper and less fragile than reordering teardown around a scorer.

    A missing file records `None` rather than being omitted: "the model did not
    create it" and "nobody looked" must not read the same downstream. Contents are
    truncated because these are small declared outputs and an unbounded read would
    let a runaway question put megabytes in the results file.
    """
    checks = (q.get("check") or {}).get("files") or []
    out: dict = {}
    for entry in checks:
        rel = entry["path"]
        base = Path(workspace_root) if entry.get("in_workspace") and workspace_root else Path("/")
        target = (base / rel).expanduser() if not rel.startswith("/") else Path(rel)
        try:
            out[rel] = target.read_text()[:ARTIFACT_MAX_BYTES]
        except OSError:
            out[rel] = None
    return out


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
        result["artifacts"] = _capture_artifacts(q, workspace_root)
        result["truth"] = _capture_truth(q, workspace_root)

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
    # Declared up front: every request helper reads the module-level BASE_URL, so
    # --server assigns it rather than threading a URL through a dozen call sites.
    global BASE_URL

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
    parser.add_argument("--server", type=str, default=BASE_URL,
                        help=f"Mira server to drive (default: {BASE_URL}; env MIRA_BENCH_SERVER). "
                             "Giving this explicitly means you are pointing the bench at a server "
                             "you manage, so production is left alone and nothing is isolated for "
                             "you — the conversations land in whatever database that server owns.")
    parser.add_argument("--use-live-server", action="store_true",
                        help="Drive the already-running production server instead of starting an "
                             "isolated one. Bench conversations then land in the REAL history and "
                             "are only removed by the teardown afterwards, which does not run if "
                             "the bench is interrupted. Explicit opt-in only.")
    args = parser.parse_args()

    BASE_URL = args.server.rstrip("/")
    if BASE_URL != DEFAULT_BASE_URL:
        print(f"  server: {BASE_URL}")

    # Isolation is the default. A bench must not put its conversations in the
    # user's own history, and the only way to guarantee that is for the server
    # writing them to own a different database — cleanup afterwards leaves them
    # visible for the whole run and permanently if the run dies. Skipped when
    # --server names a server this script did not start, since stopping
    # production would then be both useless and destructive.
    isolated = None
    if not args.use_live_server and BASE_URL == DEFAULT_BASE_URL:
        isolated = IsolatedServer()
        try:
            isolated.start()
        except Exception as e:
            print(f"ERROR: could not start an isolated bench server: {e}")
            isolated.stop()
            sys.exit(1)
    elif args.use_live_server:
        print("  --use-live-server: bench conversations WILL be written to the real history")

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
        # Ordered explicitly rather than left to atexit, so the "production is
        # back up" line lands after the teardown output instead of underneath it.
        # In isolated mode the teardown above is belt-and-braces: those rows are
        # in a temp database that is about to be deleted either way.
        if isolated is not None:
            isolated.stop()


if __name__ == "__main__":
    main()
