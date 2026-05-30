# omlx-ctl

oMLX 0.3.12 — multi-model OpenAI-compatible inference server for Apple Silicon.

Binary: `/Applications/oMLX.app/Contents/MacOS/omlx-cli`  
Symlink: `/Users/miguel/.local/bin/omlx`

## Crash history

- **0.3.8** — crashed on M5 base 32GB; app deleted May 2026
- **0.3.9** — crashed again; permanently abandoned at that point
- **0.3.12** — reinstalled May 2026; full Q1–Q13 bench completed without crashes

## Benchmark verdict (2026-05-30)

- **gemma4-26b**: viable, throughput identical to mlx-lm, wall time 3–4× worse (no-cache required to avoid OOM). No advantage over mlx-lm.
- **qwen3.6-35b**: not viable. 15–30× TTFT regression vs mlx-lm (5–6s per warm query vs 194–380ms). MoE architecture handled less efficiently by omlx engine.
- **Memory note**: hot cache (8GB) + gemma4 (15.26GB) exceeds 23.2GB ceiling → OOM on large prompts. Must run `--no-cache --hot-cache-max-size 0` for full benchmark suite.

See `docs/bench-results-2026-05-30.md` — "omlx 0.3.12 Bench — Analysis" section.

## Architecture

omlx serves all models from a single `--model-dir`. Each subdirectory = one model. Model ID in API calls = directory name. LRU-based memory management; models are loaded/evicted as needed. Paged SSD cache available for KV state overflow.

## Model directory

```
~/.omlx/models/
├── Qwen3.6-35B-A3B/        → model_id: "Qwen3.6-35B-A3B"   (files present)
└── gemma4-26b/             → model_id: "gemma4-26b"         (symlink to HF cache)
```

### Adding gemma4 (one-time setup)

```bash
ln -s ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/efbeee6e582ebfd06abc9d65e90839c4b5d2116b \
      ~/.omlx/models/gemma4-26b
```

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

```bash
pkill -f omlx
```
