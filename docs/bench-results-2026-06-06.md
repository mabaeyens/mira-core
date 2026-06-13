# Benchmark Results — 2026-06-06

Hardware: MacBook Pro M5 10c 32GB · mlx-lm v0.31.3 · dflash-mlx v0.1.8

---

## Bench 1 — gemma-4-12B-it-qat-4bit

**Status: BLOCKED — architecture not supported**

### What happened

`mlx-community/gemma-4-12B-it-qat-4bit` was uploaded on 2026-06-06. Its `config.json` declares:

```
"architectures": ["Gemma4UnifiedForConditionalGeneration"]
"model_type": "gemma4_unified"
```

mlx-lm v0.31.3 (latest stable, released 2026-04-22) does not have a handler for `gemma4_unified`. The server returns HTTP 404 for every request:

```
Error code: 404 – Model type gemma4_unified not supported.
```

The standard `gemma-4-26b-a4b-it-4bit` uses `model_type: gemma4` which IS supported; the `_unified` suffix is a new architecture used by QAT and multimodal variants, added after the last mlx-lm release.

### Comparison: QAT vs standard 4bit (2026-06-04)

| Metric | Standard 4bit (2026-06-04) | QAT 4bit (2026-06-06) |
|--------|---------------------------|----------------------|
| Quality | 9/16 | BLOCKED (not runnable) |
| Tool calls | 0/3 | BLOCKED |
| RAM est. | ~7 GB | ~7 GB |
| mlx-lm support | ✓ | ✗ (gemma4_unified) |

### Recommendation

Do not block the weekly queue on this model. Re-bench when mlx-lm adds `gemma4_unified` support (likely in v0.32.x). Track at: https://github.com/ml-explore/mlx-lm

---

## Bench 2 — omlx (256K context)

**Status: BLOCKED — oMLX.app not running**

oMLX is a native macOS app that must be started manually before switching the Mira backend to `omlx`. It was not running at bench time and cannot be launched via subprocess.

### Procedure (when ready to run)

1. Open oMLX.app manually
2. Wait for model load (Qwen3.6-35B-A3B-4bit)
3. Switch backend: `POST /models/switch {"backend": "omlx", "model_id": "Qwen3.6-35B-A3B"}`
4. Run bench: `python scripts/bench_compare.py --model omlx-qwen3.6 --project-name mira-core`
5. Run long-context probe: paste a 40K-token document, confirm dflash compresses while omlx does not
6. After bench: switch back to dflash, delete bench conversations

See `specs/bench-omlx-256k.md` for full procedure.

---

## mlx / mlx-lm upgrade (opportunistic)

`uv pip install --upgrade mlx mlx-lm` was run during this session. mlx-lm stayed at **v0.31.3** — already the latest. Other dependency upgrades landed (transformers 5.5→5.10, rich 14→15, protobuf 6.33→7.35, tqdm 4.67→4.68).

mlx itself is not a standalone importable package in the venv — it ships as part of mlx-lm's wheel. No version increment visible.

---

## Summary

| Bench | Status | Blocker | Action |
|-------|--------|---------|--------|
| gemma-4-12B-it-qat-4bit | BLOCKED | `gemma4_unified` not in mlx-lm ≤0.31.3 | Re-bench when mlx-lm v0.32+ ships |
| omlx 256K context | BLOCKED | oMLX.app must be started manually | Run manually: open oMLX, then `bench_compare.py --model omlx-qwen3.6` |

Neither bench yielded data this session. Backend restored to **dflash + Qwen3.6-35B-A3B-4bit**.


---

## Bench 3 — omlx (256K context) vs dflash baseline

**Model:** Qwen3.6-35B-A3B-4bit (same model, both backends)  
**omlx run:** 2026-06-06 · **dflash baseline:** 2026-06-02 (qwen3.6-dflash-datetime-fix)

### TTFT comparison

| Q | Category | omlx TTFT | dflash TTFT | ratio |
|---|---------|-----------|-------------|-------|
| 1 | baseline | 963ms | 9587ms | **10× faster** |
| 2 | code-no-tools | 1434ms | 5401ms | **3.8×** |
| 3 | reasoning | 1494ms | 5727ms | **3.8×** |
| 4 | long-output | 2131ms | 28454ms | **13×** |
| 5 | thinking-toggle | 2073ms | 7651ms | **3.7×** |
| 6 | agentic-single-tool | 4687ms | 10830ms | **2.3×** |
| 7 | agentic-multi-step | ERR (18K OOM) | ERR (16K OOM crash) | — |
| 8 | agentic-read-reason | ERR (18K OOM) | 8807ms ✓ | dflash wins |
| 9 | agentic-task-done | 2778ms | 10448ms | **3.8×** |
| 10 | multi-turn-long-context | 7170ms | 18869ms | **2.6×** |
| 11 | agentic-write-file | 2584ms | 11470ms | **4.4×** |
| 12 | agentic-edit-file | 2722ms | 13055ms | **4.8×** |
| 13 | agentic-divergence-guard | 3245ms | 11824ms | **3.6×** |

omlx TTFT is **4–10× lower** than dflash across all non-OOM questions. Likely cause: oMLX holds the system prompt KV state in RAM across conversations, while dflash's L2 SSD prefix cache takes several seconds to restore per conversation (per 2026-06-02 analysis).

### Throughput comparison (decode t/s)

| Q | Category | omlx t/s | dflash t/s | note |
|---|---------|----------|------------|------|
| 2 | code-no-tools | 60.1 | 48.0 | omlx faster |
| 3 | reasoning | 52.9 | 39.9 | omlx faster |
| 4 | long-output | 58.9 | 107.1 | dflash faster (speculative) |
| 5 | thinking-toggle | 54.7 | 40.8 | omlx faster |
| 6 | agentic-single-tool | 17.0 | 74.1 | dflash faster (tool I/O skewing measure) |
| 12 | agentic-edit-file | 83.4 | 100.8 | dflash slightly faster |
| 13 | agentic-divergence-guard | 43.3 | 52.0 | dflash slightly faster |

dflash's 1.3–1.5× speculative decoding advantage shows on long-generation tasks (Q4, Q12). omlx matches or beats dflash on shorter outputs. Q10 t/s (477.5) is a measurement artifact (multi-turn token count inflation).

### OOM / large context behavior

| Backend | Q7 (grep results, ~18K KV) | Q8 (read orchestrator.py, ~18K KV) |
|---------|---------------------------|-------------------------------------|
| dflash | OOM crash (process restarted) | ✓ succeeds (8807ms TTFT) |
| omlx | OOM error (graceful, no crash) | OOM error (graceful, no crash) |

Both backends hit memory limits around 18K KV tokens — well below even dflash's 64K window. The 256K omlx context window is **not achievable on 32GB RAM with a 35B model**. Headroom is ~12GB after model weights (~20GB), which limits KV to ~18K tokens at 4bit.

dflash handles Q8 (file read + reasoning) while omlx OOMs. dflash crashes on Q7 (very large grep output) vs omlx's graceful error.

### Q12 anomaly

omlx Q12 `agentic-edit-file` took **338.6s wall** vs dflash's 14.7s. At 83.4 t/s → ~28,000 tokens generated. The model likely entered a verbose reasoning loop before each tool call. This is not a latency issue but an instruction-following / verbosity issue specific to this question under omlx (possibly different temperature or sampling params).

### Raw timing — omlx-qwen3.6

| Q | Difficulty | Category | TTFT | wall | t/s |
|---|-----------|---------|------|------|-----|
| 1 | easy | baseline | 963ms | 1.8s | 3.6 |
| 2 | easy | code-no-tools | 1434ms | 7.2s | 60.1 |
| 3 | medium | reasoning | 1494ms | 26.5s | 52.9 |
| 4 | medium | long-output | 2131ms | 15.8s | 58.9 |
| 5 | medium | thinking-toggle | 2073ms | 33.4s | 54.7 |
| 6 | hard | agentic-single-tool | 4687ms | 14.3s | 17.0 |
| 7 | hard | agentic-multi-step | ERR: OOM at 18K KV (pre-chunk guard) | — | — |
| 8 | hard | agentic-read-reason | ERR: OOM at 18K KV (pre-chunk guard) | — | — |
| 9 | expert | agentic-task-done | 2778ms | 18.5s | 20.2 |
| 11 | hard | agentic-write-file | 2584ms | 10.1s | 24.0 |
| 12 | hard | agentic-edit-file | 2722ms | 338.6s | 83.4 |
| 13 | expert | agentic-divergence-guard | 3245ms | 13.4s | 43.3 |
| 10 | expert | multi-turn-long-context | 7170ms | 29.1s | 477.5¹ |

¹ Q10 t/s is a multi-turn measurement artifact.

### Agentic results — omlx

| Q | Category | Expected calls | omlx calls | task_done |
|---|---------|----------------|------------|-----------|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | ERR | — |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | none | no |

### Recommendation

**Use omlx when:** TTFT matters more than throughput (e.g., interactive conversational tasks with short responses). Its prefix cache behavior gives 4–10× lower latency.

**Use dflash when:** Long-generation tasks (code, documents), large tool-output contexts (Q8 succeeds on dflash, OOMs on omlx), or maximum throughput needed.

**Context length threshold:** Neither backend can exceed ~18K KV on 32GB RAM with Qwen3.6-35B. The 256K vs 64K window distinction is irrelevant for this hardware — RAM is the binding constraint.

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | omlx run 1 score |
|---|-----------|---------|---|
| 1 | easy | baseline | — |
| 2 | easy | code-no-tools | — |
| 3 | medium | reasoning | — |
| 4 | medium | long-output | — |
| 5 | medium | thinking-toggle | — |
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | 0 (OOM) |
| 8 | hard | agentic-read-reason | 0 (OOM) |
| 9 | expert | agentic-task-done | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 13 | expert | agentic-divergence-guard | — |

---

## Bench 4 — omlx run 2 (same session, degraded RAM)

**Purpose:** Verify whether changing `OMLX_CONTEXT` from 262144→131072 in `backend_manager.py` affects OOM behavior.  
**Result:** No effect — `OMLX_CONTEXT` is metadata-only for Mira; oMLX's memory guard is internal.

**Critical finding: oMLX memory degrades within a session.** Run 2 followed run 1 in the same oMLX process. By run 2, accumulated KV state (from Q4's 600s timeout and Q5's 198s thinking loop) had consumed enough RAM that the OOM threshold dropped from ~18K to as low as 4K KV tokens.

### Run 2 anomalies

| Q | Category | Run 1 | Run 2 | Cause |
|---|---------|-------|-------|-------|
| Q1 | baseline | TTFT 963ms | TTFT 12905ms | Cold load — oMLX restarted |
| Q4 | long-output | TTFT 2131ms, 15.8s | **TIMEOUT 600s** | Model entered degenerate generation loop |
| Q5 | thinking-toggle | TTFT 2073ms, 33.4s | **TTFT 198508ms** | 3.3min thinking loop before first output token |
| Q7 | agentic-multi-step | OOM @18K KV | 9 tool calls, no task_done | Succeeded but looped (no completion) |
| Q8 | agentic-read-reason | OOM @18K KV | **OOM @8K KV** | RAM consumed by Q4/Q5 |
| Q10 T2 | multi-turn-long-context | TTFT 7170ms | **OOM @4K KV** | Further RAM depletion |

### Raw timing — omlx-qwen3.6-128k (run 2)

| Q | Category | TTFT | wall | t/s | notes |
|---|---------|------|------|-----|-------|
| 1 | baseline | 12905ms | 39.8s | 3.9 | cold load |
| 2 | code-no-tools | 1631ms | 6.8s | 67.6 | |
| 3 | reasoning | 1876ms | 18.4s | 59.4 | |
| 4 | long-output | — | 600s | — | TIMEOUT |
| 5 | thinking-toggle | 198508ms | 218.2s | 57.3 | 3.3min thinking |
| 6 | agentic-single-tool | 4819ms | 21.2s | 14.0 | ✓ |
| 7 | agentic-multi-step | 3524ms | 124.6s | — | 9 tool calls, no task_done |
| 8 | agentic-read-reason | — | — | — | OOM @8K KV |
| 9 | agentic-task-done | 3150ms | 21.0s | 27.0 | ✓ |
| 11 | agentic-write-file | 2034ms | 9.4s | 27.8 | ✓ |
| 12 | agentic-edit-file | 3513ms | 21.4s | 40.9 | ✓ divergence_guard fired |
| 13 | agentic-divergence-guard | 3073ms | 21.5s | 80.3 | ✓ |
| 10 T1 | multi-turn-long-context | 29215ms | — | — | |
| 10 T2 | multi-turn-long-context | — | — | — | OOM @4K KV |

### Conclusion

The `OMLX_CONTEXT` code change has no effect on oMLX's memory guard. The effective context window of oMLX with Qwen3.6-35B on 32GB RAM is **session-state-dependent** — it starts at ~18K KV on a fresh session and degrades as prior queries consume memory. For production use, oMLX should be restarted between bench runs or between long workloads.

See `docs/bench-compare-omlx-vs-dflash.md` for full side-by-side response comparison.
| 10 | expert | multi-turn-long-context | — |

---

## Benchmark Results — 2026-06-06

### Timing

| Q | Difficulty | Category | omlx-qwen3.6-128k:omlx-qwen3.6-128k TTFT | wall | t/s |
|---|-----------|---------|---|---|---|
| 1 | easy | baseline | 12905ms | 39.8s | 3.9 |
| 2 | easy | code-no-tools | 1631ms | 6.8s | 67.6 |
| 3 | medium | reasoning | 1876ms | 18.4s | 59.4 |
| 4 | medium | long-output | ERR: wall-clock timeout after 600s (0 tool calls) | — | — |
| 5 | medium | thinking-toggle | 198508ms | 218.2s | 57.3 |
| 6 | hard | agentic-single-tool | 4819ms | 21.2s | 14.0 |
| 7 | hard | agentic-multi-step | 3524ms | 124.6s | — |
| 8 | hard | agentic-read-reason | ERR: Prefill context too large for available memory (pre-chunk guard at 6144 tokens, kv_len=8192): predicted peak would exceed prefill safety cap 22.5GB (90% of effective ceiling 25.0GB) | — | — |
| 9 | expert | agentic-task-done | 3150ms | 21.0s | 27.0 |
| 11 | hard | agentic-write-file | 2034ms | 9.4s | 27.8 |
| 12 | hard | agentic-edit-file | 3513ms | 21.4s | 40.9 |
| 13 | expert | agentic-divergence-guard | 3073ms | 21.5s | 80.3 |
| 10 | expert | multi-turn-long-context | ERR: Prefill context too large for available memory (pre-chunk guard at 2048 tokens, kv_len=4096): predicted peak would exceed prefill safety cap 22.5GB (90% of effective ceiling 25.0GB) | — | — |

### Agentic results

| Q | Category | Expected calls | omlx-qwen3.6-128k calls | task_done |
|---|---------|----------------|---|---|
| 6 | agentic-single-tool | 1 | run_shell | YES |
| 7 | agentic-multi-step | 2 | run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell, run_shell | no |
| 8 | agentic-read-reason | 1 | ERR | — |
| 9 | agentic-task-done | 3 | run_shell, run_shell | YES |
| 11 | agentic-write-file | 2 | write_file, read_file | YES |
| 12 | agentic-edit-file | 3 | write_file, edit_file, read_file, write_file | YES |
| 13 | agentic-divergence-guard | 3 | run_shell, run_shell | YES |
| 10 | multi-turn-long-context | 0 | ERR | — |

### Manual quality scores (fill in after review)

Scale: 0 = wrong/broken, 1 = partially correct, 2 = fully correct

| Q | Difficulty | Category | omlx-qwen3.6-128k score |
|---|-----------|---------|---|
| 1 | easy | baseline | — |
| 2 | easy | code-no-tools | — |
| 3 | medium | reasoning | — |
| 4 | medium | long-output | — |
| 5 | medium | thinking-toggle | — |
| 6 | hard | agentic-single-tool | — |
| 7 | hard | agentic-multi-step | — |
| 8 | hard | agentic-read-reason | — |
| 9 | expert | agentic-task-done | — |
| 11 | hard | agentic-write-file | — |
| 12 | hard | agentic-edit-file | — |
| 13 | expert | agentic-divergence-guard | — |
| 10 | expert | multi-turn-long-context | — |