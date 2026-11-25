# Model Comparison: MacBook Pro M5 (32GB RAM) Performance Guide

> **What runs today, and how it was decided.** Only the current verdict and the options you can
> select today are kept. Older bench runs (Ollama, llama.cpp, dflash, omlx-as-default, and the
> bench write-ups) were removed from the tree and are recoverable from git history:
> `git log -- docs/model-comparison-m5-macbook.md docs/bench-archive`.

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

**Winner: mira-mlx (Mira's own inference server), Qwen3.6-35B-A3B as the default model**

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
"Model quirks" for details.

**Current `mira.yaml`:**
```yaml
backend: mira-mlx
model: mlx-community/Qwen3.6-35B-A3B-4bit   # or mlx-community/Ministral-3-14B-Instruct-2512-4bit
host: http://localhost:8080
context_window: 65536
```

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

Retired options that used to be in this table (Ollama's `qwen3.6:35b-mlx` and `gemma4:26b` variants, llama.cpp, dflash, and oMLX 0.3.9's OOM) are in git history.

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

---

## Inkling 2 Mini vs Qwen3.6-35B-A3B (measured 2026-11-25)

`kiteteam/Inkling-2-Mini-q4l-MLX` (15.4 GB, 3x looped middle layers) on mlx-lm PR #2086, against the
production Qwen3.6-35B-A3B-4bit-MTP with its native MTP sidecar. Same machine, same token cap per
question (everything counts, an unfinished answer is wrong). Not adopted: decode is about half of Qwen's.

| | Qwen3.6 (prod config) | Inkling 2 Mini, `low` effort |
|---|---|---|
| GPQA Diamond, 198 questions, 8192-token cap | 23% (149 unanswered, the cap) | **77%** (16 unanswered) |
| Tokens per GPQA question | 7,016 | 1,984 |
| Tool-call bench (Mira harness, 22 questions), tier 1 | 33/38 | 33/38 |
| Tool-call bench, wall time / tokens | 534 s / 11.7k | 945 s / 13.5k |
| Decode in the harness | 50 to 80 t/s (MTP) | 25 to 28 t/s |
| Peak memory during GPQA | 21.3 GB | 18.3 GB |
| Metal working-set limit | 26.8 GB | 26.8 GB |

- **The GPQA gap is mostly the cap, not knowledge.** Qwen thinks until it runs out of tokens; Inkling
  at low effort finishes. Qwen's own contract number (45% at n=100, thinking off, 16 tokens) is a
  different measurement and is not comparable.
- **Effort matters more than any other setting.** At `high` the tool bench took 3.3x the tokens and
  two questions hit the 600 s timeout. mira-mlx now defaults Inkling to `low` (`MIRA_INKLING_EFFORT`).
- **The slow decode is bandwidth.** The M5 base has 153.6 GB/s; the 45 t/s published for it needs an
  M4 Pro class chip. Looped layers visit 48 layers per token against a 3B-active MoE.
- **Harness gaps found and fixed.** Mira's per-turn thinking toggle and thinking budget only reached
  Qwen, and Inkling's reasoning was written into the visible answer. Both are ported. The budget never
  bound in the measured run (longest reasoning about 1,500 tokens), so it is covered by unit tests only.
- **Q22 (injection across turns) was a bad check**, now `label_absent`: both models ran `find | ls -lS`
  to answer the follow-up and never wrote the marker.
- **Caveats.** One run per cell. The GPQA sweep that chose `low` was 16 questions at a 2048 cap.
