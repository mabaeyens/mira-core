# Model Comparison: MacBook Pro M5 (32GB RAM) Performance Guide

## Current Verdict (2026-06-07)

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

> **Hardware**: MacBook Pro 14-inch, M5 (2025), 32GB LPDDR5X RAM, 1TB SSD, macOS 26.4.1, Ollama 0.24.0 / llama.cpp b9260
> **Last Updated**: May 24, 2026 (historical benchmarks below) — current verdict above updated 2026-05-30

> Benchmark results measured and timed directly via the Ollama API, llama-server, and mlx-lm on this hardware. All test runs executed locally.

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
| OS | macOS 26.4.1 (Darwin 25.5.0) |
| Ollama | 0.24.0 (ggml-Metal engine) |

---

## Quick Reference

### Recommendation Matrix

| Option | Sustained t/s | Warm TTFT | Memory | Quality (SWE) | Verdict |
|--------|--------------|-----------|--------|---------------|---------|
| **Qwen3.6-35B-A3B-4bit (mlx-lm)** | **~52–55 t/s** | **300–450ms** | ⚠️ 18GB | **58.7%** | **⭐ Current Mira default** |
| **qwen3.6:35b-mlx** (Ollama) | **~43 t/s** | **~60–245ms** | ⚠️ 21GB | **58.7%** | Quality leader (Ollama; benched 2026-05-25) |
| **gemma-4-26b-a4b-it-4bit (mlx-lm)** | **~36 t/s** | **250–505ms** | ✅ 17GB | 42.3% | Manual fallback (demoted 2026-05-31) |
| gemma4:26b-mlx (Ollama) | ~39 t/s | ~970–1,200ms | ✅ 17GB | 42.3% | Superseded by mlx-lm (2026-05-30) |
| Qwen3.6-35B-A3B (llama.cpp) | ~29 t/s | ~1s | ⚠️ 23GB | 58.7% | Quality-first alternative |
| gemma4:26b Q4_K_M (Ollama) | ~39 t/s | ~290–420ms warm / 31s cold | ✅ 17GB | 42.3% | Superseded by MLX |
| qwen3.6:latest Q4_K_M (Ollama) | ~1.5–2 t/s | 14s | ❌ 24GB | — | Avoid |
| oMLX | — | — | ❌ OOM | — | Dead end — do not revisit |

**Current Mira config**: mlx-lm 0.31.3 + `Qwen3.6-35B-A3B-4bit` (port 8080). Ollama remains available as an optional inference backend (switchable via the in-app model browser); Gemma4 available as a manual fallback for workflows that need its 75% MLX Community leaderboard score.

### Backends Evaluated

| Backend | Status | Notes |
|---------|--------|-------|
| Ollama 0.24.0 | 🔄 Optional fallback | Switchable via in-app model browser |
| mlx-lm 0.31.3 | ✅ Default | Thinking controlled per-request via `chat_template_kwargs`; warmup on startup |
| llama.cpp b9260 | 🔧 Optional | Works but manual setup; no GPU sharing with Ollama |
| oMLX v0.3.9 | ❌ Dead end | OOM crash on 32GB M5 |

---

## gemma4:26b

### Ollama: Q4_K_M vs MLX (May 23, 2026)

Head-to-head via Ollama API (`stream:false`, `num_predict:512`, `temperature:0.1`).

| Prompt | Model | t/s | TTFT | Wall time | Tokens |
|--------|-------|-----|------|-----------|--------|
| Short factual | gemma4:26b | 38.8 | 31,597ms ¹ | 34.8s | 115 |
| Short factual | gemma4:26b-mlx | 38.3 | 2,643ms | 8.4s | 99 |
| Reasoning | gemma4:26b | 39.3 | 418ms | 27.8s | 512 |
| Reasoning | gemma4:26b-mlx | 39.3 | 1,208ms | 17.6s | 512 |
| Code generation | gemma4:26b | 40.5 | 289ms | 19.2s | 512 |
| Code generation | gemma4:26b-mlx | 39.4 | 969ms | 17.4s | 512 |
| Long output | gemma4:26b | 41.1 | 317ms | 19.2s | 512 |
| Long output | gemma4:26b-mlx | 38.5 | 1,051ms | 17.6s | 512 |

¹ Cold-load penalty — 17GB weights paged from disk on first query.

| Model | Avg t/s | Avg TTFT | Notes |
|-------|---------|----------|-------|
| gemma4:26b | 39.9 | 8,155ms | Brutal cold TTFT; warm queries fast |
| **gemma4:26b-mlx** | **38.9** | **1,468ms** | Consistent TTFT; 10–20% faster wall time |

- t/s is essentially identical (~39); MLX does not improve sustained throughput
- MLX eliminates the cold-load penalty (31s → 2.6s on first query)
- For a chat UI, MLX is meaningfully better — first word appears much sooner

### Thinking mode (gemma4 via Ollama)

| Prompt | Thinking | Total time | Eval tokens | Think chars |
|--------|----------|-----------|-------------|-------------|
| Simple (2+2) | Disabled | 2.79s | — | — |
| Simple (2+2) | Enabled | 12.39s | — | — |
| Coding (CSV reader) | Disabled | ~5s | 1,282 | — |
| Coding (CSV reader) | Enabled | 43.79s | 1,282 | 1,995 |
| Complex (caching layer) | Enabled | 91.1s | 2,125 | — |

### Optimized Modelfile + environment

**`models/gemma4-optimized.modelfile`** pins three parameters per-model (portable, not env-var dependent):

| Parameter | Value | Effect |
|-----------|-------|--------|
| `num_ctx` | `65536` | Explicit 64K context. Overrides global env var if they diverge. |
| `num_keep` | `768` | Pins system prompt in KV cache — saves ~200ms per turn. |
| `llama.cpp.flash_attn` | `true` | Flash attention kernel pinned at model level. |

```bash
ollama create gemma4-ultra -f ~/Documents/Projects/mira-core/models/gemma4-optimized.modelfile
ollama run gemma4-ultra "hello" --verbose
```

See [Ollama environment (Performance Optimization Tips)](#performance-optimization-tips) for global env vars.

---

## Qwen3.6:35B-A3B

All Qwen3.6 models are MoE (Mixture of Experts) — only ~3B parameters active per token despite 35B total. This gives it fast TTFT and competitive t/s despite larger weight size than gemma4.

| Active params | Total params | SWE-Bench Verified | Context |
|--------------|-------------|-------------------|---------|
| ~3B | 35.1B | 58.7% | 256K |

### Ollama: 35b-mlx (May 24, 2026)

**Model**: `qwen3.6:35b-mlx` — NVFP4 quantization, 21GB, MLX-optimized for Apple Silicon.

| Prompt | Thinking | TTFT | Eval tok | Eval t/s | Wall time | Think chars |
|--------|----------|------|----------|----------|-----------|-------------|
| Simple (2+2) | disabled | 4,479ms ¹ | 7 | 42.2 | 9.3s | — |
| Simple (2+2) | enabled | 59ms | 199 | 42.4 | 4.8s | 645 |
| Coding (CSV reader) | disabled | 245ms | 143 | 43.3 | 3.6s | — |
| Coding (CSV reader) | enabled | 67ms | 512 ² | 43.3 | 12.0s | 2,038 |
| Complex (caching layer) | enabled | 201ms | 512 ² | 43.2 | 12.1s | 2,046 |

¹ Cold-load penalty (21GB weights paged from disk).
² Hit `num_predict:512` cap — full response truncated.

**vs gemma4:26b-mlx:**
- Sustained t/s: **+11% faster** (43 vs 39)
- Cold TTFT: **1.7× slower** (4,479ms vs 2,643ms)
- Warm TTFT: **4–20× faster** (60–245ms vs 969–1,208ms) — MoE active-param advantage at prefill
- Model size: **4GB larger** (21GB vs 17GB; ~8GB headroom vs ~13GB)
- Quality: **+38% SWE-bench** (58.7% vs 42.3%)

**Memory fit**: 21GB weights + ~2GB KV cache (64K, q8_0) + ~1GB overhead ≈ **24GB total** (~8GB headroom).

### Ollama: Q4_K_M (May 2026) — baseline

`qwen3.6:latest` (Q4_K_M, 24GB). **Avoid** — extremely slow vs gemma4.

| Prompt | Thinking | Total time | vs gemma4:26b |
|--------|----------|-----------|----------------|
| Simple (2+2) | Disabled | 64.47s | **23× SLOWER** |
| Simple (2+2) | Enabled | 30.72s | **11× SLOWER** |
| Coding (CSV reader) | Enabled | >180s (timeout) | >4× SLOWER |
| Complex (caching layer) | Enabled | >180s (timeout) | >2× SLOWER |

Cause: 24GB at Q4_K_M exhausts the Metal budget, forcing partial CPU offload. NVFP4 (35b-mlx, 21GB) stays fully on GPU.

### mlx-lm 0.31.3 — thinking mode resolved (2026-05-30, updated 2026-05-31)

> **Update 2026-05-31 (supersedes 2026-05-30):** Qwen3.6-35B-A3B-4bit is now the default model. Thinking is controlled per-request via `extra_body={"chat_template_kwargs": {"enable_thinking": True}}` — no server restart needed to toggle. Warm TTFT: 307ms (short), 315ms (medium), 355ms (long). Thinking overhead on mlx-lm: ≤14ms (noise). `_warmup_model()` on startup eliminates the 29–34s model-switch cold start. Benchmark run 2026-05-31: see `/tmp/mira_benchmark_report_2026-05-31.md`.

mlx-lm was tested as a direct Python alternative after oMLX crashed. Raw throughput is excellent but thinking mode could not be disabled at the time of initial testing.

| Model | Weights | t/s | TTFT (initial test) | Current status |
|-------|---------|-----|------|---------|
| Qwen3.6-35B-A3B-4bit | 16GB | **44–55** | 28–62s ⚠️ (was thinking on) | **✅ Default — thinking per-request** |
| gemma-4-26b-a4b-it-4bit | 15.6GB | **36–38** | 250–505ms ✅ | Manual fallback |
| gemma-4-26b-a4b-it-UD-MLX-4bit (unsloth) | 15.6GB | **32** | 11–68s ⚠️ | Not benched on current setup |

Resolved: `--chat-template-args '{"enable_thinking": false}'` suppresses thinking at the server level (used for Gemma4). For Qwen3, per-request control is preferred. Models cached at `~/.cache/huggingface/hub/`.

Detailed per-prompt results (budget = 4000 tokens):

| Task | Think tok | Content tok | TTFT | Overall t/s |
|------|-----------|-------------|------|------------|
| Fibonacci fn | 1,588 | 74 | 37s | 44.5 |
| FastAPI endpoint | 1,167 | 85 | 28s | 44.0 |
| Code review | 1,896 | 804 | 62s | 43.3 |

### llama.cpp b9260 (May 21, 2026)

Tested because llama.cpp b9260 correctly compiles Metal shaders for the M5 (no tensor API errors), unlike Ollama 0.24.0.

**Model**: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` — `UD-Q4_K_XL.gguf` (22.9GB)

| Prompt | Tokens | Time | Speed |
|--------|--------|------|-------|
| Simple (2+2) | 8 | 0.56s | ~14.2 t/s ¹ |
| Coding (CSV reader) | 144 | 4.82s | **~29.9 t/s** |
| Medium (TCP vs UDP) | 182 | 6.69s | **~27.2 t/s** |

¹ Short response — prompt eval overhead dominates.

MTP decode (`--spec-type draft-mtp --spec-draft-n-max 4`) showed no meaningful speedup (~+4% on coding, −10% on medium). Use standard decode.

**Run command** (stop Ollama first to free Metal budget):

```bash
ollama stop gemma4:26b-mlx

llama-server \
  --model ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --alias qwen3.6-35b-a3b \
  --n-gpu-layers 999 \
  --ctx-size 16384 \
  --parallel 1 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --port 8080
```

---

## Qwen3.6:27B

The 27B model is a **dense** architecture (all 27B params active per token). Bandwidth ceiling: 153.6 GB/s ÷ 17.5GB ≈ 8.8 t/s theoretical max. This makes it fundamentally slower than the 35B MoE on bandwidth-bound hardware.

### llama.cpp b9260 (May 21, 2026)

**Model**: `Qwen3.6-27B Q4_K_M` (17.5GB, dense)

| Prompt | Tokens | Time | Speed |
|--------|--------|------|-------|
| Simple (2+2) | 9 | 2.51s | ~3.6 t/s ¹ |
| Coding (CSV reader) | 174 | 28.19s | **~6.2 t/s** |
| Medium (TCP vs UDP) | 168 | 27.36s | **~6.1 t/s** |

¹ Short response — prompt eval overhead dominates.

~6 t/s actual vs 8.8 t/s theoretical is consistent. M5 Neural Accelerators being active in llama.cpp didn't help — the bottleneck is memory bandwidth, not compute. **Not viable for Mira.**

---

## Final Comparison

| Model | Server | Sustained t/s | TTFT (warm) | vs gemma4:26b-mlx |
|-------|--------|--------------|-------------|-------------------|
| **Qwen3.6-35B-A3B-4bit** | **mlx-lm 0.31.3** | **~52–55 t/s** | **300–450ms** | **⭐ Current default; thinking per-request ✅** |
| **qwen3.6:35b-mlx** | Ollama 0.24.0 | **~43 t/s** | **~60–245ms** | +38% quality; 1.8× slower wall time on agentic tasks |
| **gemma-4-26b-a4b-it-4bit** | **mlx-lm 0.31.3** | **~36 t/s** | **250–505ms** | Manual fallback (demoted 2026-05-31) |
| **gemma4:26b-mlx** | Ollama 0.24.0 | **~39 t/s** | **~970–1,200ms** | Superseded by mlx-lm (2026-05-30) |
| gemma4:26b Q4_K_M | Ollama 0.24.0 | **~39 t/s** | ~290–420ms warm / 31s cold | Superseded by MLX |
| Qwen3.6-35B-A3B UD-Q4_K_XL | llama-server b9260 | **~29 t/s** | ~1s | 1.3× slower |
| Qwen3.6-27B Q4_K_M (dense) | llama-server b9260 | ~6 t/s | ~1s | 6× slower |

---

## Backend Notes

### oMLX v0.3.9 — OOM dead end (May 23, 2026)

oMLX was tested as an MLX inference backend. Loading Qwen3.6-35B-A3B (~23GB MLX format) plus oMLX's Python/MLX runtime (~3–4GB) exceeded the 32GB memory budget and crashed the machine. Consistent with a history of instability on base M5 32GB. oMLX has been uninstalled. **Do not revisit on this hardware.**

### llama.cpp b9260 — M5 Neural Accelerators

llama.cpp b9260 successfully compiles Metal shaders for M5 (no tensor API errors), unlike Ollama 0.24.0:

```
# Ollama 0.24.0 — tensor API not active
ggml_metal_device_init: has tensor = false

# llama.cpp b9260 — no errors, Metal VRAM budget: 25,559 MiB (~25.5 GB)
```

The M5 GPU has per-core Neural Accelerators (Apple advertises 4× AI speedup over M4). Despite llama.cpp activating them, the bottleneck for large MoE models on 32GB is memory bandwidth — not compute. Watch for an Ollama update fixing `has tensor = false`; when it lands, all Ollama-served models should see a speed increase.

**Memory constraint**: Metal budget is 25.5GB. Running Ollama + llama-server simultaneously causes OOM. Stop Ollama before starting llama-server.

---

## Reference

### Qwen3.6 Variants Available on Ollama (May 2026)

| Tag | Size | Quantization | Architecture | Context | Best For |
|-----|------|--------------|--------------|---------|----------|
| `qwen3.6:35b-mlx` | 21 GB | NVFP4 | Qwen3.5 MoE | 256K | General + coding (recommended) |
| `qwen3.6:35b-a3b-coding-nvfp4` | 22 GB | NVFP4 | Qwen3.5 MoE | 256K | Coding tasks |
| `qwen3.6:27b-coding-nvfp4` | 20 GB | NVFP4 | Qwen3.5 MoE | 256K | Coding tasks |
| `qwen3.6:35b-a3b` | 24 GB | Q4_K_M | Qwen3.5 MoE | 256K | — |
| `qwen3.6:27b` | 17 GB | Q4_K_M | Qwen3.5 MoE | 256K | — |
| `qwen3.6:latest` | 24 GB | Q4_K_M | Qwen3.5 MoE | 256K | — |

### Hardware Fit on 32GB RAM

| Model | Size | Headroom | Memory Pressure |
|-------|------|---------|------------------|
| `gemma4:26b-mlx` | 17 GB | **~13 GB** | ✅ Low |
| `qwen3.6:27b-coding-nvfp4` | 20 GB | **~10 GB** | ✅ Low-Medium |
| `qwen3.6:35b-mlx` | 21 GB | **~8 GB** | ⚠️ Medium |
| `qwen3.6:35b-a3b-coding-nvfp4` | 22 GB | **~8 GB** | ⚠️ Medium |
| `qwen3.6:latest` | 24 GB | **~4 GB** | ❌ High (partial CPU offload) |

### SWE Benchmark

| Model | SWE-Bench Verified | HumanEval | MBPP | Average |
|-------|--------------------|-----------|------|---------|
| gemma4:26b | 42.3% | 74.1% | 58.2% | 58.2% |
| qwen3.6:27b | **54.1%** | **82.4%** | **68.9%** | **68.5%** |
| qwen3.6:35b-a3b | **58.7%** | **84.2%** | **71.3%** | **71.4%** |

### Quantization Reference

| Quantization | Bits/Weight | Size Reduction | Best For |
|--------------|-------------|----------------|----------|
| FP16 | 16 | None | Maximum quality |
| Q4_K_M | 4–6 (mixed) | ~50% | General use (Apple Silicon GGUF) |
| Q8_0 | 8 | ~25% | KV cache |
| NVFP4 | 4 | ~50% | Qwen3.6 MLX models on Apple Silicon |
| MXFP8 | 8 | ~25% | AMD/Intel GPUs |

For M5 MacBook: Q4_K_M is optimal for GGUF models; NVFP4 is the format used by Qwen3.6 MLX models.

---

## Bench History

| Date | Key finding |
|------|-------------|
| 2026-05-24 | gemma4:26b-mlx vs Q4_K_M via Ollama — MLX eliminates cold-load penalty; t/s identical |
| 2026-05-25 | qwen3.6:35b-mlx vs gemma4:26b-mlx via Ollama — qwen3.6 +38% quality but 1.8× slower wall time |
| 2026-05-29 | omlx 0.3.12 viability test — no advantage over mlx-lm on gemma4; 15–30× TTFT regression on qwen3.6 |
| 2026-05-30 | mlx-lm promoted to default — 4–6× wall time improvement over Ollama on agentic tasks; OptiQ and unsloth rejected |
| 2026-05-31 | Latency matrix: Qwen3.6 on mlx-lm wins (307ms warm TTFT, ≤14ms thinking overhead); Ollama cache saves 4.6× on warm prefix; Qwen3 becomes default |
| 2026-06-06 | omlx 0.4.1 vs dFlash: omlx 4–10× faster TTFT (963ms–4.7s vs 5.4–29.6s); dFlash OOM-safe above 18K KV |
| 2026-06-07 | TTFT shootout: omlx 0.4.1 (0ms warm) vs ollama 0.30.6 MLX (90ms) vs dFlash (~48s); omlx becomes default |

---

### Memory Usage Breakdown

#### gemma4:26b-mlx (Q4_K_M, 17GB weights)

```
Model Weights:     17.0 GB
KV Cache (64K):   ~1.0 GB (with q8_0 quantization)
Other Overhead:    ~1.0 GB
----------------------------
Total:            ~19.0 GB   Headroom: ~13.0 GB ✅
```

#### qwen3.6:35b-mlx (NVFP4, 21GB weights)

```
Model Weights:     21.0 GB
KV Cache (64K):   ~2.0 GB (with q8_0 quantization)
Other Overhead:    ~1.0 GB
----------------------------
Total:            ~24.0 GB   Headroom: ~8.0 GB ⚠️
```

#### qwen3.6:27b-coding-nvfp4 (NVFP4, 20GB weights)

```
Model Weights:     20.0 GB
KV Cache (64K):   ~2.0 GB (with q8_0 quantization)  ← actual config (OLLAMA_CONTEXT_LENGTH=65536)
Other Overhead:    ~1.0 GB
----------------------------
Total:            ~23.0 GB   Headroom: ~9.0 GB ⚠️
```

> **Note:** The 256K in the Qwen3.6 reference table is the model's *native* maximum context. The actual running context is 64K, capped by `OLLAMA_CONTEXT_LENGTH=65536` in `~/.zprofile`. Raising it to 128K would push the KV cache to ~4GB (~5GB headroom — risky); 256K native would require ~8GB just for the KV cache and cause OOM. **Keep at 64K.**

### Performance Optimization Tips

#### Ollama environment (`~/.zprofile`)

```zsh
export OLLAMA_CONTEXT_LENGTH=65536   # 64k context window — without this Ollama defaults to 4k–8k
export OLLAMA_FLASH_ATTENTION=1      # reduces KV cache memory ~40%; confirmed active in log
export OLLAMA_NUM_PARALLEL=1         # single-user app; avoids doubling KV cache cost
export OLLAMA_KV_CACHE_TYPE=q8_0    # halves KV cache vs f16; negligible quality loss at 64k
export OLLAMA_MAX_LOADED_MODELS=1   # prevents accidental dual-load when switching models
```

Reload: `source ~/.zprofile`, then restart Ollama (`killall ollama && open -a Ollama`).

| Setting | Effect |
|---------|--------|
| `OLLAMA_CONTEXT_LENGTH=65536` | Sets KV cache to 64k; must match `context_window` in config |
| `OLLAMA_FLASH_ATTENTION=1` | Flash attention kernel — confirmed active in server log |
| `OLLAMA_NUM_PARALLEL=1` | Prevents a second KV cache; 32GB leaves no room for two |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | Frees ~1–2 GB at 64k context — negligible perplexity impact |
| `OLLAMA_MAX_LOADED_MODELS=1` | Safety net: prevents a second model from lingering in VRAM during model switches |

**Server log confirmation:**

```
GPULayers:31[ID:0 Layers:31(0..30)]   ← all layers on Metal GPU ✓
FlashAttention:Enabled                 ← active ✓
KvSize:65536                           ← correct ✓
KvCacheType:q8_0                       ← active ✓
Parallel:1                             ← correct ✓
```
