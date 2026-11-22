# omlx-ctl

> **STATUS: SUPPORTED, and the backup backend (re-verified 2026-11-22).** mira-mlx is
> the default; omlx is what you switch to when mira-mlx cannot serve something.
> It is also the only backend besides mira-mlx with real vision, and the only one
> with a model library of its own.

oMLX — multi-model OpenAI-compatible inference server for Apple Silicon.
Installed here: **0.7.0**. The flag table below was checked against `omlx serve --help` on
that version on 2026-11-22; recheck it after an upgrade.

Binary: `/Applications/oMLX.app/Contents/MacOS/omlx-cli`  
Symlink: `/Users/miguel/.local/bin/omlx`

## Crash history

- **0.3.8** — crashed on M5 base 32GB; app deleted May 2026
- **0.3.9** — crashed again; permanently abandoned at that point
- **0.3.12** — reinstalled May 2026; full Q1–Q13 bench completed without crashes
- **0.4.2** — added Gemma 4 support
- **0.4.3 / 0.4.4 / 0.5.1 / 0.7.0** — no crashes seen

## The constraint that still decides when to use it

oMLX holds all KV state in RAM. On 32GB with Qwen3.6 (~20GB weights), the KV
ceiling is about 18K tokens on a fresh session, and it drops to 4–8K after a long
generation. Large tool output can OOM it. The `OMLX_CONTEXT` value in
`backend_manager.py` is metadata-only and does not affect oMLX's internal memory
guard. This is the whole reason mira-mlx became the default: it sizes its own
budgets from the RAM it finds (`core/hardware.py`) and overflows the prompt cache
to disk instead of falling over.

So: omlx for interactive, short-answer sessions where warm TTFT (~0ms) matters
most, and restart oMLX.app between heavy workloads. mira-mlx for everything else.

## Historical benchmarks

Kept for the record. Both predate mira-mlx and one of the two backends compared
below no longer exists.

**2026-06-06, Qwen3.6-35B-A3B-4bit, omlx vs dflash** (dflash was retired
2026-08-01):

| Metric | omlx | dflash |
|--------|------|--------|
| TTFT | ~1s (4–10× faster) | ~10s (SSD prefix cache restore) |
| Throughput (long gen) | ~59 t/s | ~107 t/s (speculative decoding) |
| Max context before OOM | ~18K KV (fresh session) | 64K stable |
| Session stability | degrades; restart required after heavy use | stable indefinitely |
| Large tool output | OOM | works |

**2026-05-30, omlx vs mlx-lm:**

- **gemma4-26b**: viable, throughput identical to mlx-lm, wall time 3–4× worse (no-cache required to avoid OOM). No advantage over mlx-lm.
- **qwen3.6-35b**: not viable at the time. 15–30× TTFT regression vs mlx-lm (5–6s per warm query vs 194–380ms). MoE architecture handled less efficiently by the omlx engine of that era; later versions closed this and omlx served as Mira's default backend from June to July 2026.
- **Memory note**: hot cache (8GB) + gemma4 (15.26GB) exceeds the 23.2GB ceiling → OOM on large prompts. Must run `--no-cache --hot-cache-max-size 0` for a full benchmark suite.

## Architecture

omlx serves all models from a single `--model-dir`. Each subdirectory = one model. Model ID in API calls = directory name. LRU-based memory management; models are loaded/evicted as needed. Paged SSD cache available for KV state overflow.

## Model directory

`~/.omlx/models/` holds one subdirectory per model; the directory name is the model id in API
calls. It is **empty on this machine as of 2026-11-22** (the models were removed to free disk), so
the setups below must be redone before the `omlx-*` presets in `mira.yaml` will load. Each model is
either a real copy or a symlink into the HF cache (omlx sometimes downloads its own duplicate, so
prefer symlinks and log any download in `MODEL_REGISTRY.md`).

### Adding gemma4 (one-time setup)

```bash
ln -s ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/efbeee6e582ebfd06abc9d65e90839c4b5d2116b \
      ~/.omlx/models/gemma4-26b
```

### Adding an MTP model (one-time setup)

MTP (multi-token prediction) speculative decoding runs on **two** backends: mira-mlx's
own native MTP (`mira_mlx_mtp_enabled`, see [`multi-token-prediction.md`](multi-token-prediction.md))
and omlx, set up here. Only mlx-lm and vllm-mlx can't use it — mlx-lm strips the `mtp.*`
head when loading, and vllm-mlx caps at `effective_draft_tokens=1` for VLM-capable models
(Qwen3.x), so it gives no speedup. omlx runs the head for real — measured on M5: dense 27B
8.1→19.1 t/s (2.36×), 35B-A3B MoE 58→77.5 (1.34×), output distribution-preserving. This
section is the omlx route; the model prep below (baking the bf16 head back into a 4-bit
checkpoint) is the same idea mira-mlx uses via its `model-mtp.safetensors` sidecar.

A 4-bit checkpoint doesn't ship the head (the quant strips it), so bake it back:

1. Extract the **raw bf16** `mtp.*` tensors from the HF *source* repo (27B: `Qwen/Qwen3.8-27B`
   shard 18/18; 35B-A3B MoE: `Qwen/Qwen3.6-35B-A3B` shards 25–26, including the MoE experts).
   Keep them bf16 — omlx's `norm_repair` applies the RMSNorm offset itself.
2. Build `~/.omlx/models/<name>/`: symlink the 4-bit base's top-level files, then add the raw
   tensors as a **top-level** `model-mtp.safetensors` (omlx globs `model*.safetensors` and
   ignores the index; a subdir or non-`model*` name is silently skipped). `config.json` must
   keep `mtp_num_hidden_layers > 0` and a `model_type` starting `qwen3_5`.
3. In `~/.omlx/model_settings.json`, key by dir name:
   `{"mtp_enabled": true, "mtp_num_draft_tokens": 3}` (3 = max draft depth; an adaptive
   controller picks 1..3 per sequence). Confirm via the `MTP[...] accept=A/D` info log.

Dirs used by the `omlx-qwen38-27b-mtp` / `omlx-qwen36-moe-mtp` presets in `mira.yaml`:
`~/.omlx/models/Qwen3.8-27B-MTP` and `~/.omlx/models/Qwen3.6-35B-A3B-MTP` (recreate them as above).

## Start server

```bash
omlx serve --model-dir ~/.omlx/models --port 8080
```

Health check: `curl http://localhost:8080/v1/models`

## mira.yaml routing

```yaml
backend: omlx
model: gemma4-26b          # or: Qwen3.6-35B-A3B
host: http://localhost:8080
```

## Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model-dir` | `~/.omlx/models` | directory of model subdirs |
| `--port` | `8000` | use 8080 to match mlx-lm slot |
| `--paged-ssd-cache-dir` | off | SSD KV cache; set to a fast SSD path |
| `--hot-cache-max-size` | `0` (disabled) | in-memory prefix cache |
| `--max-concurrent-requests` | `8` | reduce if OOM |

## Stop server

`omlx stop` only stops the managed background server and does not kill a foreground `omlx serve`.
Find it with `pgrep -f 'omlx.cli serve'` and `kill <pid>`.
