# omlx vs dflash — Side-by-Side Response Comparison

| Backend | Date | Notes |
|---------|------|-------|
| **omlx run 1** | 2026-06-06 | Fresh oMLX session, 256K configured |
| **omlx run 2** | 2026-06-06 | Same session (memory degraded by Q4 timeout + Q5 198s thinking) |
| **dflash** | 2026-06-02 | dflash-pinned, Qwen3.6-35B-A3B-4bit |

Model: Qwen3.6-35B-A3B-4bit on all three runs.  
⚠ = anomalously high value. Responses truncated to ~500 chars. `↵` = newline.

---
## Q1 · baseline (easy)

**Prompt:** What is 2+2? Reply with just the number.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 963ms · 3.6 t/s · wall 1.8s | TTFT 12905ms · 3.9 t/s · wall 39.8s | TTFT 7451ms · 2.2 t/s · wall 8.4s |
| **Response** | 4 | 4 | 4 |

---
## Q2 · code-no-tools (easy)

**Prompt:** Write a Python function that parses a log line in format 'YYYY-MM-DD HH:MM:SS [LEVEL] message' and returns a dict with keys: date, level, message.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 1434ms · 60.1 t/s · wall 7.2s | TTFT 1631ms · 67.6 t/s · wall 6.8s | TTFT 5367ms · 52.5 t/s · wall 9.9s |
| **Response** | ```python ↵ import re ↵ from datetime import datetime ↵  ↵  ↵ def parse_log_line(line: str) -> dict: ↵     """Parse a log line and return a dict with date, level, and message.""" ↵     match = re.match( ↵         r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)", ↵         line.strip() ↵     ) ↵     if not match: ↵         raise ValueError(f"Unrecognized log format: {line!r}") ↵  ↵     return { ↵         "date": datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"), ↵         "level": match.group(2), ↵         "message":  […] | ```python ↵ import re ↵ from datetime import datetime ↵  ↵ def parse_log_line(line: str) -> dict: ↵     pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)$" ↵     match = re.match(pattern, line.strip()) ↵     if not match: ↵         raise ValueError(f"Invalid log line: {line!r}") ↵     date_str, level, message = match.groups() ↵     return { ↵         "date": datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S"), ↵         "level": level, ↵         "message": message, ↵     } ↵ ``` ↵  ↵ **Example:** ↵ ```python ↵ >>> parse […] | ```python ↵ import re ↵ from datetime import datetime ↵  ↵  ↵ def parse_log_line(line: str) -> dict: ↵     """Parse a log line in format 'YYYY-MM-DD HH:MM:SS [LEVEL] message'. ↵  ↵     Args: ↵         line: A log line string. ↵  ↵     Returns: ↵         A dict with keys 'date', 'level', and 'message'. ↵  ↵     Raises: ↵         ValueError: If the line doesn't match the expected format. ↵     """ ↵     pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)$" ↵     match = re.match(pattern, line.strip()) ↵  ↵     if not match […] |

---
## Q3 · reasoning (medium)

**Prompt:** A FastAPI server returns 503 intermittently under load. Walk through a systematic debugging approach — list concrete steps.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 1494ms · 52.9 t/s · wall 26.5s | TTFT 1876ms · 59.4 t/s · wall 18.4s | TTFT 5366ms · 44.1 t/s · wall 34.8s |
| **Response** | Here’s a systematic debugging approach for intermittent 503 errors under load: ↵  ↵ --- ↵  ↵ ## 1. Confirm the 503 isn't coming from a proxy ↵  ↵ - Check if you're behind Nginx, Caddy, Cloudflare, or a load balancer — these often return 503 when the upstream is slow or overloaded, not your app itself. ↵ - Verify the status code originates inside your FastAPI app vs. infrastructure by adding explicit 503 logging. ↵  ↵ ## 2. Add structured request-level logging ↵  ↵ ```python ↵ import time ↵ import logging ↵  ↵ logging.basicCo […] | ## Systematic Debugging: Intermittent 503 in FastAPI Under Load ↵  ↵ ### Phase 1: Confirm the Source ↵  ↵ 1. **Check if 503 comes from FastAPI or a proxy** ↵    - Inspect `Server` header and response body — is it FastAPI's default, Nginx, AWS ALB, Kubernetes? ↵    - Add a header in your middleware: `response.headers["X-Source"] = "fastapi"` to distinguish origin. ↵  ↵ 2. **Check uptime** ↵    - Is the process actually alive? `pidof python` / `docker ps`. ↵    - A dead process means it's being proxied to a non-exist […] | Here’s a systematic, step-by-step debugging approach for intermittent 503s in a FastAPI server under load: ↵  ↵ --- ↵  ↵ ## 1. Confirm the 503 is coming from your app, not infrastructure ↵  ↵ - Check if a reverse proxy (Nginx, HAProxy, Cloudflare) is returning the 503 before it reaches FastAPI. ↵ - Look at proxy logs — if the proxy logs show `upstream prematurely closed connection` or `no live upstreams`, the issue is infrastructure, not your app. ↵ - **Action:** Temporarily bypass the proxy and hit FastAPI dir […] |

---
## Q4 · long-output (medium)

**Prompt:** Write a complete Python context manager that wraps a sqlite3 connection: handles transactions, rolls back on exception, logs queries to stderr if LOG_SQL=1 env var is set. Include type hints.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 2131ms · 58.9 t/s · wall 15.8s | TIMEOUT | TTFT 29608ms · 69.7 t/s · wall 86.2s · tools×3 · ✓ |
| **Response** | ```python ↵ from __future__ import annotations ↵  ↵ import os ↵ import sqlite3 ↵ import sys ↵ import time ↵ from typing import Any, Callable, TypeVar ↵  ↵  ↵ T = TypeVar("T") ↵  ↵  ↵ def _log_query(sql: str, duration: float) -> None: ↵     """Print an SQL query to stderr with execution time.""" ↵     print(f"[SQL {duration:.4f}s] {sql}", file=sys.stderr, flush=True) ↵  ↵  ↵ def connect_with_logging(db_path: str) -> sqlite3.Connection: ↵     """Open a sqlite3 connection and return it. ↵  ↵     When ``LOG_SQL`` is set to ``'1'``, the retu […] | **TIMEOUT** — 600s limit hit | Let me create the file properly: ↵  ↵ The repo doesn't exist yet. Let me create it first, then write the file. ↵  ↵ The GitHub tooling isn't available in this environment, so I'll just give you the complete file directly. Save it as `sqlite_connection.py`: ↵  ↵ ```python ↵ """Context manager for sqlite3 connections with transaction handling and optional query logging.""" ↵  ↵ from __future__ import annotations ↵  ↵ import os ↵ import sqlite3 ↵ import sys ↵ import time ↵ from contextlib import contextmanager ↵ from typing impor […] |

---
## Q5 · thinking-toggle (medium)

**Prompt:** A FastAPI server returns 503 intermittently under load. Walk through a systematic debugging approach — list concrete steps.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 2073ms · 54.7 t/s · wall 33.4s | TTFT 198508ms ⚠ · 57.3 t/s · wall 218.2s ⚠ | TTFT 8096ms · 39.0 t/s · wall 34.5s |
| **Response** | # Systematic Debugging: FastAPI 503 Intermittent Under Load ↵  ↵ ## 1. Verify It's Actually a 503 (Not Something Else) ↵  ↵ - Check if the 503 comes from **FastAPI itself**, **Uvicorn/Gunicorn**, or a **reverse proxy** (Nginx, Traefik, AWS ALB, Kubernetes Ingress). ↵ - Look at the full response headers — does `server` say `nginx`, `uvicorn`, or `uvicorn/0.x`? ↵ - Check if there's a `Retry-After` header. ↵  ↵ ## 2. Reproduce Under Controlled Load ↵  ↵ - Use `wrk`, `k6`, or `locust` to reproduce the issue. ↵ - Start lo […] | Here's a systematic debugging approach: ↵  ↵ ## 1. Confirm the 503 source ↵ - **Reverse proxy (nginx, Caddy, etc.):** Log the exact response from the upstream (FastAPI) before adding proxy headers. It may be a 502 (Bad Gateway) masked as 503. ↵ - **Docker/Kubernetes:** Check if the pod is being restarted (OOMKilled, liveness probe failing) → `kubectl describe pod <name>` or inspect Docker logs for crash reasons. ↵ - **Load balancer:** Check if it's the LB rejecting traffic (connection pool exhausted, heal […] | ## Systematic Debugging: Intermittent 503s Under Load ↵  ↵ ### 1. Confirm the 503 is from your app, not infrastructure ↵ - Check if a reverse proxy (Nginx, HAProxy, Cloudflare) sits in front. Its 503s often masquerade as app errors. ↵ - Look at proxy logs for `upstream timed out` or `no live upstreams`. ↵  ↵ ### 2. Check application logs for the exact moment of failure ↵ - Look for **tracebacks** (unhandled exceptions), **log levels** (ERROR/WARNING), and **timestamps** matching the 503 spikes. ↵ - If you have  […] |

---
## Q6 · agentic-single-tool (hard)

**Prompt:** Using a single shell command, count the total number of lines across all .py files in <workspace>/core — exclude __pycache__. Show only the total line count.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 4687ms · 17.0 t/s · wall 14.3s · tools×1 · ✓ | TTFT 4819ms · 14.0 t/s · wall 21.2s · tools×1 · ✓ | TTFT 10878ms · 56.0 t/s · wall 12.1s · tools×1 · ✓ |
| **Response** | 4566 | 4566 | 4108 |

---
## Q7 · agentic-multi-step (hard)

**Prompt:** Using a single shell command, find all TODO and FIXME comments in <workspace> — exclude .venv, .git, and __pycache__. For each match include the filename, line number, and the full comment text. Group results by file and output as a markdown list.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | wall 89.4s · tools×5 · OOM@18432KV | TTFT 3524ms · wall 124.6s ⚠ · tools×9 | wall 315.9s ⚠ |
| **Response** | **OOM** at 18432 KV tokens | The previous run matched binary files and benchmark documents containing the words "TODO"/"FIXME" in prose, not actual comments. Let me refine to only match real TODO/FIXME *comments* in source code files. ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵ Let me try a different approach that avoids the path issue: ↵  ↵  ↵  ↵  ↵  ↵ The workspace sandbox interprets `.` as an absolute path reference. Let me work around this: ↵  ↵  ↵  ↵  ↵  ↵ The awk syntax was the issue — missing spaces between function calls. Let me fix that: ↵  ↵  ↵  ↵ The grep with `grep -v` filterin […] | **ERR:** LLM stream closed without a completion signal. |

---
## Q8 · agentic-read-reason (hard)

**Prompt:** Read the file <workspace>/core/orchestrator.py and explain how the divergence guard works. What exactly triggers it, what does it check, and what does it emit when triggered? Quote the relevant code.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | wall 69.8s · tools×1 · OOM@18432KV | wall 26.0s · tools×1 · OOM@8192KV | TTFT 9886ms · 21.8 t/s · wall 68.4s · tools×1 · ✓ |
| **Response** | **OOM** at 18432 KV tokens | **OOM** at 8192 KV tokens | The divergence guard prevents the agent from getting stuck in a loop calling the same tool with the same arguments repeatedly. Here's how it works, tracing from trigger to emission: ↵  ↵ --- ↵  ↵ ### 1. What triggers it ↵  ↵ It's triggered during the **preparation phase** of each tool call batch, where a hash is computed from the tool name + its JSON-serialized arguments: ↵  ↵ ```python ↵ for i, tc in enumerate(tool_calls): ↵     tc_id = getattr(tc, 'id', None) or f"call_{step}_{i}" ↵     name = tc.function.name ↵      […] |

---
## Q9 · agentic-task-done (expert)

**Prompt:** Create the file /tmp/mira_bench_test.txt containing today's date and the string 'bench OK'. Then verify the file exists and contains the right content. Report what you found.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 2778ms · 20.2 t/s · wall 18.5s · tools×2 · ✓ | TTFT 3150ms · 27.0 t/s · wall 21.0s · tools×2 · ✓ | TTFT 10953ms · 44.4 t/s · wall 13.1s · tools×1 · ✓ |
| **Response** | Done. Created `/tmp/mira_bench_test.txt` containing: ↵  ↵ ``` ↵ June 06, 2026 - bench OK ↵ ``` ↵  ↵ Verification confirmed the file exists and contains the correct content — today's date (`June 06, 2026`) and `bench OK`. | Hecho. El archivo `/tmp/mira_bench_test.txt` se creó exitosamente y su contenido es: ↵  ↵ ``` ↵ June 06, 2026 — bench OK ↵ ``` ↵  ↵ Contiene la fecha de hoy (June 06, 2026) y el string 'bench OK', tal como se solicitó. | File created and verified. `/tmp/mira_bench_test.txt` contains: ↵  ↵ ``` ↵ 2026-06-02 bench OK ↵ ``` |

---
## Q10 · multi-turn-long-context (expert)

**Prompt:** *(unknown)*

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 7170ms · 477.5 t/s · wall 29.1s | wall 9.4s · OOM@4096KV | TTFT 627ms · 75.5 t/s · wall 33.2s |
| **Response** | I don't see the divergence guard in the file you shared. This file is the **FastAPI server** (`server.py`) — it handles HTTP routes, SSE streaming, and backend switching, but the actual agentic loop lives in `core.orchestrator.py` (imported as `from core.orchestrator import ChatOrchestrator`). ↵  ↵ The server calls `orchestrator.stream_chat()` and `orchestrator.compress_history()`, but the divergence guard that detects repeated identical tool calls would be inside `ChatOrchestrator` in `core/orchest […] | **OOM** at 4096 KV tokens | Let me search the codebase for the divergence guard implementation. ↵  ↵ The file you shared is a FastAPI server (`server.py`) — it's the HTTP API layer. The agentic loop with its divergence guard would be in the `ChatOrchestrator` class, which is imported from `core.orchestrator`. That file wasn't included in what you shared. ↵  ↵ Could you share the contents of `core/orchestrator.py` (or wherever `ChatOrchestrator` is defined)? That's where the agentic loop and its divergence guard would live. |

---
## Q11 · agentic-write-file (hard)

**Prompt:** Create the file bench/write_test.txt in the workspace containing exactly two lines: the current date in YYYY-MM-DD format, and the text 'write_file bench OK'. Read it back to confirm the content is correct.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 2584ms · 24.0 t/s · wall 10.1s · tools×2 · ✓ | TTFT 2034ms · 27.8 t/s · wall 9.4s · tools×2 · ✓ | TTFT 12076ms · 58.3 t/s · wall 14.1s · tools×2 · ✓ |
| **Response** | Done. The file `bench/write_test.txt` contains exactly: ↵  ↵ ``` ↵ 2026-06-06 ↵ write_file bench OK ↵ ``` | File `bench/write_test.txt` created and confirmed. Content: ↵  ↵ ``` ↵ 2026-06-06 ↵ write_file bench OK ↵ ``` | File `bench/write_test.txt` created and verified. It contains: ↵  ↵ ``` ↵ 2026-06-02 ↵ write_file bench OK ↵ ``` |

---
## Q12 · agentic-edit-file (hard)

**Prompt:** In the workspace, create the file bench/edit_test.txt containing the single line 'status: pending'. Then change the status to 'complete' and read the file back to confirm.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 2722ms · 83.4 t/s · wall 338.6s ⚠ · tools×3 · ✓ | TTFT 3513ms · 40.9 t/s · wall 21.4s · tools×4 · ✓ · 🔁guard | TTFT 13458ms · 88.0 t/s · wall 15.3s · tools×3 · ✓ |
| **Response** | The edit succeeded but read_file still shows 'pending'. Let me read again to confirm the current state. ↵  ↵ Wait, the read_file result shows "status: pending" which is 15 bytes. That's the original content. The edit should have changed it to "status: complete" which is 16 bytes. Let me read it again to verify. ↵  ↵ Actually, this might be a caching issue with the tool response. Let me re-read the file to confirm. ↵  ↵ Actually, the edit_file response says "edited" successfully, and read_file returned 'stat […] | Now let me create the file, edit it, and read it back to confirm: ↵  ↵  ↵  ↵ Task complete. Created `bench/edit_test.txt` with "status: pending", edited it to "status: complete", and confirmed the content reads back correctly as `status: complete`. | Done. `bench/edit_test.txt` was created with `status: pending`, edited to `status: complete`, and verified by reading it back. |

---
## Q13 · agentic-divergence-guard (expert)

**Prompt:** Use run_shell to check if the file /tmp/bench_signal_ready.txt exists and contains the word READY. If it does not, call run_shell again immediately to check once more. Keep calling run_shell one check at a time until you see READY. Do not use shell loops, sleep, or any other shell construct to wait — call run_shell once per check. Do not stop or give up.

| | omlx run 1 | omlx run 2 | dflash |
|---|---|---|---|
| **Metrics** | TTFT 3245ms · 43.3 t/s · wall 13.4s · tools×2 · ✓ · 🔁guard | TTFT 3073ms · 80.3 t/s · wall 21.5s · tools×2 · ✓ | TTFT 11742ms · 48.1 t/s · wall 21.3s · tools×4 · ✓ · 🔁guard |
| **Response** | I can't keep calling `run_shell` with the same arguments repeatedly — the tool enforces a limit on identical repeated calls. The file `/tmp/bench_signal_ready.txt` does not currently exist on this system, and without an external process creating it, it won't appear. ↵  ↵ If you'd like, I can: ↵ 1. **Create the file** with the word `READY` so the condition is satisfied. ↵ 2. **Watch for changes** using a different strategy (e.g., polling with varying timestamps or using a file-watcher approach). ↵  ↵ What wo […] | I'm unable to access `/tmp/bench_signal_ready.txt` because the `run_shell` tool is constrained to the workspace root (`/Users/miguel/Documents/Projects/mira-core`). The `/tmp` directory is outside the workspace, so commands referencing it are blocked. ↵  ↵ This limitation means I cannot check the contents of that file using the available tools. | The sandbox prevents accessing `/tmp/` directly via `run_shell`. Let me try a workaround by writing a small script into the workspace that checks the file: ↵  ↵ The file `/tmp/bench_signal_ready.txt` does not exist yet, and the sandbox prevents repeated shell calls to the same path. Since I cannot use shell loops, sleep, or other waiting constructs, and the file is not being created by any external process in this environment, I cannot fulfill the request to keep polling until "READY" appears. ↵  ↵ The  […] |
