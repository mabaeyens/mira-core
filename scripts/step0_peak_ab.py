"""STEP 0: test lBroth's explanation of the gpt-oss peak gap (mlx-lm#1438).

Their claim: our published 20.01 GiB peak at fraction 0.3 is inflated by a
preallocated prompt cache and max-KV window, not by the offload path. Their
harness allocates no prompt cache, sets no max-KV, uses a short prompt, and
reads 19.0 GiB on a 48GB M5 Pro.

ARM B ONLY. The original two-arm version crashed this 32GB machine (hard reboot,
2026-07-20): arm A reproduced our heavy config (1GB prompt cache + 128k max-KV)
on top of a 59GB model. Arm A is unnecessary anyway, since 20.01 GiB is already
published. Arm B is the lighter config by construction and is the actual question.

Guards: refuses to start unless production is down and enough RAM is free.

NOTE --kv-bits is left off entirely: gpt-oss has attention sinks and mlx-lm raises
"Quantized SDPA does not support attention sinks" from the generation thread,
which hangs the request rather than returning an error.
"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = str(REPO / ".venv" / "bin" / "python")
MODEL = "mlx-community/gpt-oss-120b-MXFP4-Q8"
FRACTION = 0.3
DECODE_TOKENS = 200
PORT = 8152
MIN_FREE_GB = 22.0

# Short prompt, matching lBroth's stated ~106-token scale. Actual token count is
# reported in the output rather than assumed.
PROMPT = (
    "Explain briefly how a mixture-of-experts transformer routes each token to a "
    "small subset of expert feed-forward networks, and why that makes the parameter "
    "count much larger than the active compute per token. Keep it to one paragraph."
)


def free_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    free = inactive = 0
    for line in out.splitlines():
        if "Pages free" in line:
            free = int(line.split()[2].rstrip("."))
        elif "Pages inactive" in line:
            inactive = int(line.split()[2].rstrip("."))
    return (free + inactive) * 16384 / 1e9


def preflight():
    problems = []
    r = subprocess.run(["pgrep", "-f", "mira_mlx_server"], capture_output=True, text=True)
    if r.stdout.strip():
        problems.append(f"an inference server is still running (pids {r.stdout.split()})")
    g = free_gb()
    if g < MIN_FREE_GB:
        problems.append(f"only {g:.1f} GB free, need >= {MIN_FREE_GB}")
    return problems, g


def launch():
    args = [
        PYTHON, "-m", "core.inference.mira_mlx_server",
        "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
        "--max-tokens", "1024", "--prefill-step-size", "1024",
        "--resident-expert-fraction", str(FRACTION),
        "--prompt-cache-max-bytes", "0",
    ]
    logf = open(f"/tmp/step0_{PORT}.log", "w")
    return subprocess.Popen(args, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT)


def wait_ready(timeout=1200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


def decode():
    payload = json.dumps({
        "model": "x", "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": DECODE_TOKENS, "stream": False, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = json.loads(r.read())
    dt = time.time() - t0
    u = body.get("usage", {})
    out = u.get("completion_tokens", 0)
    return {"wall_s": dt, "out_tokens": out, "tok_s": out / dt if dt else 0,
            "prompt_tokens": u.get("prompt_tokens", 0)}


if __name__ == "__main__":
    problems, g = preflight()
    if problems:
        print("PREFLIGHT FAILED, refusing to start:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"preflight OK ({g:.1f} GB free)")
    print(f"model={MODEL} fraction={FRACTION} decode_tokens={DECODE_TOKENS}")
    print("arm B: no prompt cache, no max-KV\n")

    p = launch()
    try:
        if not wait_ready():
            print(f"*** never became ready; see /tmp/step0_{PORT}.log")
            sys.exit(1)
        print(f"  ready ({free_gb():.1f} GB free); warming up")
        decode()
        print("  measuring")
        d = decode()
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/stats", timeout=10) as r:
            s = json.loads(r.read())
        peak_b = s.get("peak_memory_bytes", 0)
        ec = s.get("expert_cache") or {}
        res = {
            "arm": "B (no prompt cache, no max-kv)",
            "peak_gib": peak_b / 1024**3, "peak_gb": peak_b / 1000**3,
            "decode_tok_s": d["tok_s"], "prompt_tokens": d["prompt_tokens"],
            "out_tokens": d["out_tokens"],
            "hit_rate": ec.get("hit_rate"), "decode_hit_rate": ec.get("decode_hit_rate"),
        }
        print("\n" + "=" * 66)
        print(f"  peak            {res['peak_gib']:.2f} GiB  ({res['peak_gb']:.2f} GB)")
        print(f"  decode          {res['decode_tok_s']:.2f} t/s over {res['out_tokens']} tok")
        print(f"  prompt tokens   {res['prompt_tokens']}")
        print(f"  hit_rate        {res['hit_rate']}  decode_hit_rate {res['decode_hit_rate']}")
        print("=" * 66)
        print(f"  ours published: 20.01 GiB | lBroth: 19.0 GiB | arm B: {res['peak_gib']:.2f} GiB")
        d19 = abs(res["peak_gib"] - 19.0)
        print("  -> explanation HOLDS" if d19 < 1.0 else
              f"  -> does NOT account for the gap (still {d19:.2f} GiB off lBroth)")
        Path("/tmp/step0_results.json").write_text(json.dumps(res, indent=2))
        print("\n  raw -> /tmp/step0_results.json")
    finally:
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
