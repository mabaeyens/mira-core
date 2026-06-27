# Inference tuning — 2026-06-27 (Apple M5, 32 GB, macOS 26.5.1, omlx 0.4.4)

Goal: squeeze more tok/s out of the two served models (Qwen3.6-35B-A3B, Gemma4-26B-A4B)
without changing the engine. Measured via omlx admin API + real `/v1/chat/completions`
streaming (token-accurate with `stream_options.include_usage`).

## TL;DR

- **Adopted:** `burst_decode_mode = aggressive` (server-wide) → **~+10% decode**
  (57.3 → 63.3 tok/s on omlx tg512; ~61 tok/s confirmed via real inference). Universal,
  lossless, zero extra memory, applied at runtime.
- **Rejected (measured net-negative or blocked):** DFlash, MTP, SpecPrefill — see below.
- The whole MLX stack inside omlx 0.4.4 is already latest (mlx 0.31.2, mlx-lm 0.31.3,
  dflash-mlx 0.1.9) and the **M5 Neural Accelerators are already active** (mlx 0.31 +
  macOS ≥26.2). No engine swap / version bump / Swift / Candle helps (same Metal kernels;
  Candle would regress — no M5-NA support).

## Why speculative decoding does NOT help this MoE

Qwen3.6-35B-A3B activates only ~3B params per token, so decode is already cheap and
bandwidth-bound. Speculative methods (draft + verify) add overhead that isn't repaid
unless draft acceptance is very high. Measured DFlash (z-lab draft, after a config shim
to make it load at all):

| Workload | Baseline tok/s | DFlash tok/s | Δ |
|---|---|---|---|
| Reasoning / chat | 60.8 | 45.5 | **−25%** |
| Code generation | 60.9 | 57.4 | **−6%** |
| Highly repetitive (best case) | 59.6 | 72.9 | **+22%** |

DFlash only wins on highly predictable output (rare for an assistant) and loses on the
reasoning/tool/chat traffic Mira actually serves. It also produced *different* output
(omlx `verify=adaptive` is not strictly lossless). Verdict: **keep DFlash off** (it stays
the demoted large-context fallback, consistent with prior benches).

## Per-directive results

### DFlash (z-lab/Qwen3.6-35B-A3B-DFlash) — rejected
- The draft uses a newer config schema (`rope_parameters`, nested `dflash_config`) than
  omlx's bundled `dflash-mlx 0.1.9`, which reads top-level `rope_theta`/`block_size`.
  Without a shim it silently fails (`DFlashDraftModelArgs.__init__() missing ... rope_theta
  and block_size`) and falls back to the normal engine (→ 0 effect, the trap).
- Shim applied (so it actually runs): added top-level `rope_theta=10000000`,
  `block_size=16` to the draft `config.json` in the HF cache snapshot + `~/.omlx/models`
  copy (backups at `config.json.orig`). Draft (~0.77 GB) downloaded to
  `~/.omlx/models/z-lab/Qwen3.6-35B-A3B-DFlash`.
- Even running correctly: net-negative (table above). Left **disabled**.

### MTP — skipped
- omlx confirms the current `Qwen3.6-35B-A3B-4bit` quant is `mtp_compatible: false`
  ("converted weights are missing mtp.* tensors; re-convert from HF preserving MTP").
- Same MoE architectural headwind as DFlash, **plus** it needs re-converting the full
  ~20 GB checkpoint. Not worth it. (`mlx-community/Qwen3.6-35B-A3B-MTP-4bit` is a
  drafter-only head, intended for the VLM MTP path, not native text MTP.)
- Gemma4: no MTP head at all (`mtp_compatible: false`).

### SpecPrefill — blocked (no compatible 3.6 scorer)
- Targets prefill/TTFT, needs a small standard LM scorer sharing the 3.6 tokenizer.
- The z-lab DFlash draft is architecturally incompatible as a scorer
  (`SpecPrefill: draft model load failed: Received 69 parameters not in model`).
- There is **no small dense Qwen3.6** model (series = 27B dense + 35B-A3B MoE only). The
  only fallback is a Qwen3.5 small model = version/vocab mismatch, for marginal value
  (Mira's warm KV cache already gives ~0 ms TTFT; cold long-prompt prefill is already
  ~879 tok/s). Left **disabled**.
- Long-prompt TTFT reference (6039-token prompt, no specprefill): cold 8.3 s, warm 2.8 s.

## Final omlx state (unchanged except burst)

- `global-settings.server.burst_decode_mode = aggressive`
- Qwen3.6-35B-A3B & gemma4-26b: `dflash/mtp/specprefill = off`, `turboquant_kv = on` (q4)

## Optional follow-ups (not done)

- Bump ollama client 0.30.10 → 0.30.11 (routine; ollama is not the active backend).
- If omlx ships a newer `dflash-mlx` that reads `rope_parameters`/`dflash_config`, the
  z-lab shim becomes unnecessary — but DFlash still won't help typical Mira decode.
- The z-lab draft + shimmed configs can be removed (omlx → delete model) since DFlash
  is off; kept for now in case of future experimentation.
