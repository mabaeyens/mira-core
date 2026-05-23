# Model Comparison: MacBook Pro M5 (32GB RAM) Performance Guide

> **Hardware**: MacBook Pro 14-inch, M5 (2025), 32GB LPDDR5X RAM, 1TB SSD, macOS 26.4.1, Ollama 0.24.0 / llama.cpp b9260
> **Last Updated**: May 23, 2026

> Benchmark results measured and timed directly via the Ollama API and llama-server on this hardware. All test runs executed locally.

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

### Known hardware limitation

**M5 Neural Accelerators not active (Ollama 0.24.0):**

```
ggml_metal_device_init: testing tensor API for f16 support
ggml_metal_device_init: error compiling source — the tensor API is not supported — disabling
ggml_metal_device_init: has tensor = false
```

The M5 GPU has per-core Neural Accelerators (Apple advertises 4× AI speedup over M4). Ollama 0.24.0's Metal shaders do not yet compile against the M5 tensor API (`MTLGPUFamilyMetal4 / Apple10`). It falls back to standard Metal kernels. Not a configuration issue — watch for a fix in a future Ollama release; when it lands, decode speed should jump significantly.

llama.cpp b9260 successfully compiles Metal shaders for M5 (no tensor API errors) — see Round 2 below.

---

## Executive Summary

**For MacBook Pro M5 with 32GB RAM:**

- **Use `gemma4:26b-mlx` as your default model** — ~39 t/s sustained, best-in-class TTFT on this hardware
- **Thinking mode works** with gemma4:26b-mlx
- **Qwen3.6-35B-A3B MoE** is competitive (~29 t/s) if quality is the priority over speed

**Current Mira config**: `gemma4:26b-mlx` (set in `core/config.py`)

---

## oMLX — evaluated and removed (May 23, 2026)

oMLX v0.3.9 was tested as an alternative MLX inference backend. Loading Qwen3.6-35B-A3B (~23 GB in MLX format) plus oMLX's Python/MLX runtime (~3–4 GB) exceeded the 32 GB memory budget and crashed the machine. This is consistent with a history of instability on base M5 32 GB. oMLX has been uninstalled. **Do not revisit on this hardware.**

---

## gemma4:26b vs gemma4:26b-mlx (May 23, 2026)

Head-to-head benchmark via Ollama API (`stream:false`, `num_predict:512`, `temperature:0.1`).

### Per-prompt results

| Prompt | Model | TPS | TTFT | Wall time | Tokens |
|--------|-------|-----|------|-----------|--------|
| Short factual | gemma4:26b | 38.8 | 31,597 ms ¹ | 34.8s | 115 |
| Short factual | gemma4:26b-mlx | 38.3 | 2,643 ms | 8.4s | 99 |
| Reasoning | gemma4:26b | 39.3 | 418 ms | 27.8s | 512 |
| Reasoning | gemma4:26b-mlx | 39.3 | 1,208 ms | 17.6s | 512 |
| Code generation | gemma4:26b | 40.5 | 289 ms | 19.2s | 512 |
| Code generation | gemma4:26b-mlx | 39.4 | 969 ms | 17.4s | 512 |
| Long output | gemma4:26b | 41.1 | 317 ms | 19.2s | 512 |
| Long output | gemma4:26b-mlx | 38.5 | 1,051 ms | 17.6s | 512 |

¹ Cold-load penalty — model paged in from disk on first query.

### Summary

| Model | Avg TPS | Avg TTFT | Notes |
|-------|---------|----------|-------|
| gemma4:26b | 39.9 t/s | 8,155 ms | Brutal cold TTFT; warm queries fast |
| **gemma4:26b-mlx** | **38.9 t/s** | **1,468 ms** | Consistent TTFT; 10–20% faster wall time |

**Takeaways:**
- TPS is essentially identical (~39 t/s); MLX does not improve sustained throughput
- MLX eliminates the cold-load penalty (31s → 2.6s TTFT on first query)
- MLX warm TTFT is higher than GGUF warm (1s vs 0.3s) but wall time is still better due to less startup overhead
- For a chat UI, MLX is meaningfully better — first word appears much sooner

---

## Test Results: gemma4:26b vs Qwen3.6 (May 2026)

### Simple Query: "What is 2+2?"

| Model | Tag | Quantization | Size | Thinking | Total Time | Load Time | Eval Time | Speed vs gemma4 |
|-------|-----|--------------|------|----------|------------|-----------|-----------|------------------|
| gemma4:26b | (default) | Q4_K_M | 17 GB | Disabled | **2.79s** | 0.18s | 2.40s | baseline |
| gemma4:26b | (default) | Q4_K_M | 17 GB | **Enabled** | **12.39s** | 0.18s | 10.35s | baseline |
| qwen3.6 | :latest | Q4_K_M | 24 GB | Disabled | 64.47s | 14.01s | 44.41s | **23x SLOWER** |
| qwen3.6 | :latest | Q4_K_M | 24 GB | Enabled | 30.72s | 0.36s | 27.51s | **11x SLOWER** |

### Coding Task: "Write a Python function that reads a CSV file..."

| Model | Thinking | Total Time | Eval Tokens | Thinking Length | Speed vs gemma4 |
|-------|----------|------------|-------------|-----------------|------------------|
| gemma4:26b | Disabled | ~5s | 1282 | N/A | baseline |
| gemma4:26b | **Enabled** | **43.79s** | 1282 | 1995 chars | baseline |
| qwen3.6:latest | Enabled | **>180s (timeout)** | N/A | N/A | **>4x SLOWER** |

### Complex Task: "Design a caching layer for web API with 1000 req/s..."

| Model | Thinking | Total Time | Eval Tokens | Thinking Length | Speed vs gemma4 |
|-------|----------|------------|-------------|-----------------|------------------|
| gemma4:26b | **Enabled** | **91.1s** | 2125 | Yes | baseline |
| qwen3.6:latest | Enabled | **>180s (timeout)** | N/A | N/A | **>2x SLOWER** |

---

## Qwen3.6 Variants

### Available on Ollama (May 2026)

| Model Tag | Size | Quantization | Architecture | Context Window | Best For |
|-----------|------|--------------|--------------|----------------|----------|
| `qwen3.6:latest` | 24 GB | Q4_K_M | Qwen3.5 MoE | 256K | General use |
| `qwen3.6:35b-a3b` | 24 GB | Q4_K_M | Qwen3.5 MoE | 256K | General use |
| `qwen3.6:27b` | 17 GB | Q4_K_M | Qwen3.5 MoE | 256K | General use |
| `qwen3.6:27b-coding-nvfp4` | 20 GB | NVFP4 | Qwen3.5 MoE | 256K | Coding tasks |
| `qwen3.6:35b-a3b-coding-nvfp4` | 22 GB | NVFP4 | Qwen3.5 MoE | 256K | Coding tasks |

### Dense vs MoE

**All Qwen3.6 models are MoE (Mixture of Experts)** — there is no "dense" Qwen3.6.

| Model | Active Parameters | Total Parameters |
|-------|------------------|-----------------|
| qwen3.6:27b | ~3B | 27B |
| qwen3.6:35b-a3b | ~3B | 35B |
| gemma4:26b | ~4B | 26B |

### Hardware fit on 32GB RAM

| Model | Size | Headroom | Memory Pressure |
|-------|------|---------|------------------|
| `gemma4:26b-mlx` | 17 GB | **15 GB** | ✅ Low |
| `qwen3.6:27b-coding-nvfp4` | 20 GB | **12 GB** | ✅ Low-Medium |
| `qwen3.6:35b-a3b-coding-nvfp4` | 22 GB | **10 GB** | ⚠️ Medium |
| `qwen3.6:latest` | 24 GB | **8 GB** | ❌ High |

### SWE Benchmark

| Model | SWE-Bench Verified | HumanEval | MBPP | Average |
|-------|--------------------|-----------|------|---------|
| gemma4:26b | 42.3% | 74.1% | 58.2% | 58.2% |
| qwen3.6:27b | **54.1%** | **82.4%** | **68.9%** | **68.5%** |
| qwen3.6:35b-a3b | **58.7%** | **84.2%** | **71.3%** | **71.4%** |

---

## Quantization Reference

| Quantization | Bits/Weight | Size Reduction | Best For |
|--------------|-------------|----------------|----------|
| FP16 | 16 | None | Maximum quality |
| Q4_K_M | 4-6 (mixed) | ~50% | General use (Apple Silicon) |
| Q8_0 | 8 | ~25% | KV cache |
| NVFP4 | 4 | ~50% | NVIDIA GPUs (marginal benefit on M5) |
| MXFP8 | 8 | ~25% | AMD/Intel GPUs |

**For M5 MacBook**: Q4_K_M is optimal. NVFP4 coding variants offer marginal benchmark gains but are designed for NVIDIA.

---

## Recommendation Matrix

| Option | Response Time | Code Quality | Memory Fit | Verdict |
|--------|---------------|--------------|------------|---------|
| **gemma4:26b-mlx** | **2-91s, fast TTFT** | Good (58.2%) | ✅ Perfect (17GB) | **⭐ CURRENT DEFAULT** |
| gemma4:26b | 2-91s, slow cold TTFT | Good (58.2%) | ✅ Perfect (17GB) | Superseded by MLX |
| qwen3.6:27b-coding-nvfp4 | 5-120s | Better (68.5%) | ✅ Good (20GB) | Good alternative |
| qwen3.6:35b-a3b-coding-nvfp4 | 6-140s | **Best (71.4%)** | ⚠️ Tight (22GB) | Risk of OOM |
| qwen3.6:latest | 64-180s+ | Good | ❌ Risky (24GB) | Avoid |

---

## Performance Optimization Tips

### Ollama environment (`~/.zprofile`)

```zsh
export OLLAMA_CONTEXT_LENGTH=65536   # 64k context window — without this Ollama defaults to 4k–8k
export OLLAMA_FLASH_ATTENTION=1      # reduces KV cache memory ~40%; confirmed active in log
export OLLAMA_NUM_PARALLEL=1         # single-user app; avoids doubling KV cache cost
export OLLAMA_KV_CACHE_TYPE=q8_0    # halves KV cache vs f16; negligible quality loss at 64k
```

Reload: `source ~/.zprofile`, then restart Ollama (`killall ollama && open -a Ollama`).

| Setting | Effect |
|---------|--------|
| `OLLAMA_CONTEXT_LENGTH=65536` | Sets KV cache to 64k; must match `context_window` in config |
| `OLLAMA_FLASH_ATTENTION=1` | Flash attention kernel — confirmed active in server log |
| `OLLAMA_NUM_PARALLEL=1` | Prevents a second KV cache; 32 GB leaves no room for two |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | Frees ~1–2 GB at 64k context — negligible perplexity impact |

**Server log confirmation:**

```
GPULayers:31[ID:0 Layers:31(0..30)]   ← all layers on Metal GPU ✓
FlashAttention:Enabled                 ← active ✓
KvSize:65536                           ← correct ✓
KvCacheType:q8_0                       ← active ✓
Parallel:1                             ← correct ✓
```

---

## Memory Usage Breakdown

### gemma4:26b-mlx (Q4_K_M, 17GB weights)

```
Model Weights:     17.0 GB
KV Cache (64K):   ~1.0 GB (with q8_0 quantization)
Other Overhead:    ~1.0 GB
----------------------------
Total:            ~19.0 GB
Headroom:         ~13.0 GB ✅
```

### qwen3.6:27b-coding-nvfp4 (NVFP4, 20GB weights)

```
Model Weights:     20.0 GB
KV Cache (256K):  ~2.5 GB (with q8_0 quantization)
Other Overhead:    ~1.0 GB
----------------------------
Total:            ~23.5 GB
Headroom:         ~ 8.5 GB ⚠️
```

---

## Optimized Modelfile for gemma4

The global `OLLAMA_KV_CACHE_TYPE=q8_0` env var applies to all models. A Modelfile lets you pin parameters per-model so changes are isolated and reproducible.

**`models/gemma4-optimized.modelfile`** sets three parameters:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `llama.cpp.kv_cache_type` | `q4_k` | Halves KV cache vs `q8_0`. Keeps full cache in the 18.2 GB Metal budget. |
| `llama.cpp.flash_attn` | `true` | Pins flash attention at model level — portable if env var is removed. |
| `num_ctx` | `65536` | Explicit 64K context. Overrides global env var if they diverge. |

```bash
ollama create gemma4-ultra -f ~/Documents/Projects/mira-core/models/gemma4-optimized.modelfile
ollama run gemma4-ultra "hello" --verbose
```

---

## Round 2: llama.cpp b9260 + Qwen3.6 (May 21, 2026)

Motivation: test whether llama.cpp b9260 unlocks M5 Neural Accelerators that Ollama 0.24.0 cannot use.

### M5 Metal Backend: llama.cpp b9260 vs Ollama 0.24.0

| Aspect | Ollama 0.24.0 | llama.cpp b9260 |
|--------|--------------|-----------------|
| M5 Tensor API | ❌ `has tensor = false` | ✅ No errors |
| Metal VRAM budget | Not reported | **25,559 MiB (~25.5 GB)** |

**llama.cpp b9260 successfully compiles Metal shaders for M5.** Key improvement over Ollama 0.24.0.

### Memory constraint

Metal budget is **25.5 GB**. Running Ollama (gemma4:26b, 17 GB) + llama-server (Qwen3.6-27B, 17.5 GB) simultaneously causes OOM. Workaround: `ollama stop gemma4:26b-mlx` before starting llama-server.

### Qwen3.6-27B Q4_K_M via llama-server (thinking disabled, `--parallel 1`)

| Prompt | Tokens | Time | Speed |
|--------|--------|------|-------|
| Simple ("What is 2+2?") | 9 tok | 2.51s | ~3.6 t/s ¹ |
| Coding (CSV reader fn) | 174 tok | 28.19s | **~6.2 t/s** |
| Medium (TCP vs UDP) | 168 tok | 27.36s | **~6.1 t/s** |

¹ Short response — prompt eval overhead dominates.

**Conclusion**: Qwen3.6-27B is a dense model (all 27B params active per token). Memory bandwidth ceiling of 153.6 GB/s ÷ 17.5 GB ≈ 8.8 t/s theoretical max. Actual ~6 t/s is consistent. M5 Neural Accelerators being active in llama.cpp didn't help — the bottleneck is bandwidth, not compute.

### Round 3: Qwen3.6-35B-A3B MoE (the right model)

The 35B-A3B is a true MoE — only **~3B active per token**. Bandwidth requirement per token is ~6× lower than the dense 27B.

**Model**: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` — `UD-Q4_K_XL.gguf` (22.9 GB)

#### Standard decode

| Prompt | Tokens | Time | Speed |
|--------|--------|------|-------|
| Simple ("What is 2+2?") | 8 tok | 0.56s | ~14.2 t/s ¹ |
| Coding (CSV reader fn) | 144 tok | 4.82s | **~29.9 t/s** |
| Medium (TCP vs UDP) | 182 tok | 6.69s | **~27.2 t/s** |

#### MTP decode (`--spec-type draft-mtp --spec-draft-n-max 4`)

| Prompt | Tokens | Time | Speed |
|--------|--------|------|-------|
| Coding (CSV reader fn) | 150 tok | 4.80s | **~31.2 t/s** |
| Medium (TCP vs UDP) | 176 tok | 7.22s | **~24.4 t/s** |

MTP shows no meaningful speedup on M5 (~+4% on coding, −10% on medium). Per-step overhead is already very low on Apple Silicon with unified memory. Use standard decode.

### Final comparison: all models tested

| Model | Server | Sustained t/s | vs gemma4:26b |
|-------|--------|--------------|---------------|
| **gemma4:26b-mlx** | Ollama | **~39 t/s** | baseline |
| gemma4:26b Q4_K_M | Ollama | **~38 t/s** | −1% |
| Qwen3.6-35B-A3B UD-Q4_K_XL (MoE) | llama-server b9260 | **~29 t/s** | **1.3× slower** |
| Qwen3.6-27B Q4_K_M (dense) | llama-server b9260 | ~6 t/s | **6× slower** |
| Qwen3.6-27B UD-Q4_K_XL (dense) | llama-server b9260 | ~6 t/s | **6× slower** |

### llama-server run command (35B-A3B, if you want it)

```bash
# Stop Ollama model first (shared 25.5 GB Metal budget)
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

**Decision matrix:**

- **Speed-first, simpler setup → gemma4:26b-mlx on Ollama** (current Mira config)
- **Quality-first, willing to manage llama-server → 35B-A3B on llama-server**

**Watch for**: Ollama update fixing M5 tensor API (`has tensor = false`). When it lands, gemma4:26b-mlx could see a significant speed increase and widen the gap further.
