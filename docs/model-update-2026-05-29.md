# Model & Tooling Update — 2026-05-29

Scope: qwen3.6:35b-a3b and gemma4:e2b only. All other families dropped.

---

## 1. omlx v0.3.12 — memory guard fix

Released 2026-05-27. This is a patch to v0.3.11's new memory guard, which was too conservative on memory-tight Macs and started rejecting models that v0.3.10 had loaded fine (#1431). v0.3.12 tunes the dynamic ceiling to match how macOS actually handles memory pressure, and adds a **Custom tier** to pin the ceiling to a specific GB value.

**Verdict: skip.** omlx has been permanently abandoned for this machine (base M5 32GB — crashed on load). The memory guard fix doesn't change the underlying crash profile. No action needed.

---

## 2. Ollama v0.30.0-rc29 — major architectural change

Still pre-release (rc29, tagged 2026-05-28). The headline change is architectural: Ollama drops its GGML layer and moves to **llama.cpp directly**, with MLX used as the Apple Silicon inference backend.

What this means for Mira:
- MLX is now first-class in Ollama, not a bolted-on path. Performance and stability of `*-mlx` tags should improve over time.
- The M5 Metal tensor API blocker (`ggml_metal_device_init: has tensor = false`) may be resolved when llama.cpp's Metal shaders are updated for the M5 tensor API — worth re-checking once 0.30.0 goes stable.
- Still collecting feedback on performance regressions and memory utilization — **do not upgrade production Mira to rc29 yet.**

**Verdict: watch.** Upgrade to stable 0.30.0 when it ships. Re-run Q6–Q12 bench after upgrading to see if MLX decode speed changes.

---

## 3. gemma4:e2b — new small model, now with MLX tag

The Ollama gemma4 library updated **1 week ago** and now shows MLX tags for the small edge models:

| Tag | Size | Context | Backend | Notes |
|-----|------|---------|---------|-------|
| `gemma4:e2b` | 7.2 GB | 128K | llama.cpp | Standard quant |
| `gemma4:e2b-mlx` | 7.1 GB | 128K | MLX | New — 1 week ago |
| `gemma4:e4b` | 9.6 GB | 128K | llama.cpp | |
| `gemma4:e4b-mlx` | 9.6 GB | 128K | MLX | New — 1 week ago |

These are the "edge" variants (2B and 4B dense). The 2B is tiny — 7GB fits comfortably alongside the embedding model without any memory pressure.

**Benchmark scores from Google (full gemma4 family):**

| Metric | gemma4:e2b | gemma4:e4b | gemma4:26b |
|--------|-----------|-----------|-----------|
| MMLU Pro | 60.0% | 69.4% | 82.6% |
| AIME 2026 | 37.5% | 42.5% | 88.3% |
| LiveCodeBench v6 | 44.0% | 52.0% | 77.1% |
| GPQA Diamond | 43.4% | 58.6% | 82.3% |

The 2B is noticeably behind on reasoning tasks. It could still be useful as a fast fallback for lightweight/conversational queries where latency matters more than depth.

**Verdict: bench it.** Run Q1–Q9 against `gemma4:e2b-mlx` to establish a baseline. The MLX tag fits in memory, decode speed should be fast. Worth knowing where the floor is before committing to it as a fallback.

---

## 4. qwen3.6:35b-a3b — no new version, MLX is the tag to use

The qwen3.6 Ollama library shows **Updated 7 months ago** — no new model weights. The family is stable. The relevant tag is still `qwen3.6:35b-mlx` (NVFP4, 21 GB), which is what was benchmarked on 2026-05-25.

There is a new **Qwen3-30B-A3B** (`qwen3:30b`, updated as Qwen3-30B-A3B-Instruct-2507) in the qwen3 family — a separate 30B MoE model updated this month. Not to be confused with qwen3.6. It's smaller activated params and could be faster. Worth a look later if qwen3.6 is dropped, but not a priority now.

**Verdict: keep as is.** qwen3.6:35b-mlx scored 7/8 on Q6–Q9 and 4/4 on Q11–Q12. No new weights to pull. Current setup is correct.

---

## 5. Bench status recap (as of 2026-05-25)

Both models benchmarked: **gemma4:26b-mlx** and **qwen3.6:35b-mlx**, Q6–Q12.

| Metric | gemma4:26b-mlx | qwen3.6:35b-mlx |
|--------|---------------|----------------|
| Quality (Q6–Q9) | 7/8 | 7/8 |
| Quality (Q11–Q12) | 4/4 | 4/4 |
| Avg wall time | **48.4s** | 87.9s |
| Q12 call efficiency | extra probe call | exact 3 calls |

gemma4 is 1.8× faster; qwen3.6 is more precise with tool sequencing. Shared weakness: neither uses a single shell pipeline for Q6.

---

## 6. What's next

| Priority | Action |
|----------|--------|
| High | Bench `gemma4:e2b-mlx` on Q1–Q9 to establish baseline |
| Medium | Wait for Ollama 0.30.0 stable; re-run bench after upgrade |
| Low | Check `qwen3:30b` (2507) as a potential lighter alternative to qwen3.6 |
| Skip | omlx — do not revisit on base M5 32GB |
| Skip | gemma4:26b (non-mlx), all other families |
