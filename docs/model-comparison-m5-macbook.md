# Model Comparison: MacBook Pro M5 (32GB RAM) Performance Guide

> **What runs today, and how it was decided.** Trimmed on 2026-08-12: roughly 320 lines of
> Ollama, llama.cpp and dflash measurement from May–June 2026 came out, because all three
> stopped being Mira backends and the numbers had become configuration advice for software
> nobody here runs. They are recoverable from git history — the *Bench History* index at the
> bottom names every run and its date, so any single number can be found again.
>
> Verdicts are dated and kept in order, newest first. Where an older one is still true it
> says so explicitly.

## Current Verdict (updated 2026-08-01)

**Four backends, and mira-mlx is the one.** As of 2026-08-01, dflash and ollama are
retired from the codebase entirely (see `CHANGELOG.md`). What remains: **mira-mlx**
(default), **omlx** (backup, and the fallback when mira-mlx cannot serve something),
plus **mlx-lm** and **vllm-mlx**, which stay because they are cheap to keep and useful
for comparison. Nothing was lost model-wise — everything ollama served runs on
mira-mlx, and Gemma 4 is still reachable through omlx.

Also new on 2026-08-01: mira-mlx can do **real vision**, optionally, by loading the
Qwen3.6 checkpoint's own vision tower (`mira_mlx_vision: true`). Off by default. It
costs about 1.1GB active. See `docs/architecture.md`, "Vision on mira-mlx".

The 2026-07-10 verdict below still stands for the model choice and the numbers.

## Verdict of 2026-07-10 (model choice unchanged)

**Winner: mira-mlx (Mira's own inference server) — Qwen3.6-35B-A3B as the normal default model**

mira-mlx (`core/inference/mira_mlx_server.py`) was promoted to the default backend on
2026-07-09, replacing omlx as the primary path — it's bundled (no separate GUI app), has
RAM-aware sizing, a disk-backed prompt cache, and `/v1/stats` visibility. omlx remains
fully supported as an alternative backend.

**2026-07-10 bench results** (`scripts/bench_standard.py` throughput + `scripts/bench_compare.py`
13-question quality/agentic suite, both via mira-mlx):

| Model | Decode t/s | pp1024 t/s | Agentic (7 questions) |
|-------|-----------|-----------|------------------------|
| Qwen3.6-35B-A3B-4bit | 56–66 t/s | ~880 t/s | 7/7 tool calls firing correctly (after fix below) |
| Ministral-3-14B-4bit | 19.5–21 t/s | ~2900 t/s (warm) | 6/7 (one bench-harness path-scoping edge case, not a mira-mlx defect) |

Qwen3.6's decode throughput on mira-mlx is 3× Ministral's, as expected for a 3B-active MoE
vs. a 14B dense model. Both are usable defaults; Qwen3.6 stays the "normal" default model,
Ministral is the temporary default (per explicit request) while evaluating the Mistral family.

**Tool-calling regression, found and fixed same day:** Qwen3.6 initially fired **0/7** tool
calls on mira-mlx (while Ministral got 6/7 and omlx's Qwen3.6 got 7/7) — three stacked bugs:
(1) `core/orchestrator.py`'s Qwen3-thinking backend allow-list omitted `mira-mlx`, so it never
received an explicit `enable_thinking` value; (2) mira-mlx's HTTP handler silently dropped
`chat_template_kwargs` even when sent; (3) the tool-text buffering logic (needed for Mistral's
one-sided `[TOOL_CALLS]` marker) also captured Qwen's closing `</tool_call>` marker, breaking
the parser's end-anchored regex. All three fixed 2026-07-10 — see `docs/architecture.md`
"Model quirks" for details. Full writeup: `docs/bench-archive/bench-results-2026-07-10.md`.

**Current `mira.yaml`:**
```yaml
backend: mira-mlx
model: mlx-community/Qwen3.6-35B-A3B-4bit   # or mlx-community/Ministral-3-14B-Instruct-2512-4bit
host: http://localhost:8080
context_window: 65536
```

---

## Previous Verdict (2026-06-07, superseded by mira-mlx above — omlx itself is unchanged and still fully supported)

**Winner: oMLX 0.4.1 + `Qwen3.6-35B-A3B`**

| Metric | Value |
|--------|-------|
| TTFT (warm) | ~0ms (KV cache held in RAM after startup warm-up) |
| TTFT (startup warm-up) | ~5.5s (one-time per server start) |
| Sustained t/s | ~52–55 t/s |
| Memory | ~20GB (weights + KV cache) |
| Quality | 58.7% SWE-Bench; MLX Community leaderboard: 52.5% |
| Thinking | Per-request via `extra_body={"enable_thinking": False/True}`; ≤14 ms overhead |

**Why:** oMLX 0.4.1 holds the KV cache in RAM and achieves ~0ms TTFT on every new conversation after a one-time 5.5s startup warm-up. Benchmarked 2026-06-07 against the full Mira system prompt (1 488 tokens): omlx median 0ms vs ollama 0.30.6 MLX at 90ms vs dFlash at ~48s (SSD prefix-cache restore). Qwen3.6-35B-A3B is unchanged as the model — same quality, same throughput, dramatically better responsiveness.

**Rejected alternatives:**
- **dFlash** — demoted to fallback; SSD prefix-cache restore adds ~48s TTFT regardless of cache hit; best option for very large context sessions (>18K tokens) where omlx can OOM
- **ollama 0.30.6 MLX** — 90ms median TTFT (good, but 10–20× slower than omlx); viable fallback
- **omlx 0.3.12** — had 15–30× TTFT regression on qwen3.6; fixed in 0.4.1 (see `docs/omlx-ctl.md`)
- **OptiQ mixed-precision (4-bit)** — 20–25% slower decode; not worth the quality gain on this hardware
- **gemma-4-26b-a4b-it-4bit** — 42.3% SWE-Bench vs Qwen3's 58.7%; no adaptive thinking

**Current `mira.yaml`:**
```yaml
backend: omlx
model: Qwen3.6-35B-A3B
host: http://localhost:8080
context_window: 65536
```

---

> All numbers on this page were measured locally on the hardware below. Nothing is quoted from
> a vendor or a leaderboard except the *Model quality reference* table, which says so.

---

## Hardware Reference

| Field | Value |
|-------|-------|
| Model | MacBook Pro 14-inch (M5, 2025) |
| Chip | Apple M5 |
| CPU | 10-core (4 performance + 6 efficiency) |
| GPU | 10-core (with Neural Accelerators per core) |
| Neural Engine | 16-core |
| Unified Memory | 32 GB (LPDDR5X) |
| Memory Bandwidth | 153.6 GB/s |
| Storage | 1 TB SSD |
| OS | macOS 26.4.1 (Darwin 25.5.0) at the time of these benches |

---

## Quick Reference

### Recommendation Matrix

Options you can actually select today:

| Option | Sustained t/s | Warm TTFT | Memory | Quality (SWE) | Verdict |
|--------|--------------|-----------|--------|---------------|---------|
| **Qwen3.6-35B-A3B-4bit (mira-mlx)** | **56–66 t/s** | **~1.1–1.2s pp1024** | ⚠️ 18GB | **58.7%** | **⭐ The default.** See "Current Verdict" above |
| Qwen3.6-35B-A3B (omlx) | ~52–55 t/s | ~0ms warm | ⚠️ 20GB | 58.7% | Backup backend; unbeatable warm TTFT, separate GUI app |
| Ministral-3-14B-4bit (mira-mlx) | 19.5–21 t/s | ~2900 t/s pp (warm) | — | — | Dense Mistral-family alternative; much faster prefill, 3× slower decode |
| Qwen3.6-35B-A3B-4bit (mlx-lm) | ~52–55 t/s | 300–450ms | ⚠️ 18GB | 58.7% | Stock upstream server, no mira-mlx extras |
| gemma-4-26b-a4b-it-4bit (omlx) | ~36 t/s | 250–505ms | ✅ 17GB | 42.3% | Reachable through omlx; lower quality, lighter |

**Current Mira config (2026-08-01)**: mira-mlx + `Qwen3.6-35B-A3B-4bit` (port 8080). mira-mlx is built on the same mlx-lm continuous-batching primitives but as Mira's own server, with RAM-aware sizing and a disk-backed prompt cache. Gemma 4 stays available through omlx for workflows that want its leaderboard score.

Retired options that used to be in this table — Ollama's `qwen3.6:35b-mlx` and `gemma4:26b` variants, llama.cpp, dflash, and oMLX 0.3.9's OOM — came out with the 2026-08-12 trim. See *Bench History* for when each was measured.

### Backends Evaluated

| Backend | Status | Notes |
|---------|--------|-------|
| mira-mlx | ✅ Default (2026-07-09) | Mira-owned MLX server (`core/inference/mira_mlx_server.py`); Qwen3.6 and Mistral/Ministral both supported, incl. tool-calling; optional vision since 2026-08-01 |
| omlx | 🔄 Backup backend | Fully supported; ~0ms warm TTFT after startup warm-up; real vision; its own model library |
| mlx-lm | 🔄 Kept for comparison | Stock upstream server, no mira-mlx extras. Historical default (2026-05) |
| vllm-mlx | 🔄 Kept for comparison | Patched Mistral tool-call parser; used for the Ministral-3-14B evaluation |
| dflash | ❌ Retired 2026-08-01 | Was the large-context fallback; mira-mlx's disk-backed prompt cache covers that case |
| Ollama | ❌ Retired 2026-08-01 | Only ever served `ministral-3:14b`, which runs on three remaining backends |
| llama.cpp b9260 | ❌ Never integrated | Benchmarked May 2026 for comparison only; manual setup, no Mira backend ever existed |

---

## Model quality reference

Published leaderboard scores, not measured here. Backend-independent — these are properties of
the model, which is why they survived the trim while the throughput tables did not.

| Model | SWE-Bench Verified | HumanEval | MBPP | Average |
|-------|--------------------|-----------|------|---------|
| gemma4:26b | 42.3% | 74.1% | 58.2% | 58.2% |
| qwen3.6:27b | **54.1%** | **82.4%** | **68.9%** | **68.5%** |
| qwen3.6:35b-a3b | **58.7%** | **84.2%** | **71.3%** | **71.4%** |

## Bench History

Every bench run, in order. This is the index to use when you want a number that used to be in
this file: find the date, then `git log -- docs/model-comparison-m5-macbook.md` or the named
`docs/bench-archive/` file.

| Date | Key finding |
|------|-------------|
| 2026-05-24 | gemma4:26b-mlx vs Q4_K_M via Ollama — MLX eliminates cold-load penalty; t/s identical |
| 2026-05-25 | qwen3.6:35b-mlx vs gemma4:26b-mlx via Ollama — qwen3.6 +38% quality but 1.8× slower wall time |
| 2026-05-29 | omlx 0.3.12 viability test — no advantage over mlx-lm on gemma4; 15–30× TTFT regression on qwen3.6 |
| 2026-05-30 | mlx-lm promoted to default — 4–6× wall time improvement over Ollama on agentic tasks; OptiQ and unsloth rejected |
| 2026-05-31 | Latency matrix: Qwen3.6 on mlx-lm wins (307ms warm TTFT, ≤14ms thinking overhead); Ollama cache saves 4.6× on warm prefix; Qwen3 becomes default |
| 2026-06-06 | omlx 0.4.1 vs dFlash: omlx 4–10× faster TTFT (963ms–4.7s vs 5.4–29.6s); dFlash OOM-safe above 18K KV |
| 2026-06-07 | TTFT shootout: omlx 0.4.1 (0ms warm) vs ollama 0.30.6 MLX (90ms) vs dFlash (~48s); omlx becomes default |
| 2026-07-09 | mira-mlx promoted to default: bundled, RAM-aware sizing, disk-backed prompt cache, `/v1/stats` |
| 2026-07-18 | KV quantization at 8 bits and MoE expert offload measured on both models — `docs/bench-archive/bench-results-2026-07-18.md` |
| 2026-08-01 | mlx 0.32.0 verified against the real model (25/26 on the quality suite, no regression) — `docs/bench-archive/bench-results-2026-08-01.md`. dflash and ollama retired the same day |

