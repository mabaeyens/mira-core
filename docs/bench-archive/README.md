# Bench archive

Dated measurement records, kept as evidence. Each was true on the machine and build named inside
it; none is configuration advice for today.

**The current comparison point is `bench-results-2026-08-08.md`.** Compare a new run against
that one. Everything older is here so an old number can be re-examined rather than trusted from
memory, which is the whole reason these files were not deleted.

| File | Date | What it measured |
|---|---|---|
| `bench-results-2026-07-10.md` | 2026-07-10 | mira-mlx's first bench as default backend: Qwen3.6 vs Ministral-3-14B on throughput and the 13-question agentic suite. Contains the writeup of the three stacked tool-calling bugs found and fixed that day. |
| `bench-results-2026-07-18.md` | 2026-07-18 | KV quantization at 8 bits and MoE expert offload, measured on both models. The largest record here, and the baseline the 2026-08-01 run was checked against. |
| `bench-results-2026-07-20.md` | 2026-07-20 | Short follow-up on the offload coalescing ceiling. |
| `bench-results-2026-08-01.md` | 2026-08-01 | mlx 0.32.0 verified against the real model — 25/26 on the quality suite, no regression. dflash and ollama were retired the same day. |
| `bench-results-2026-08-08.md` | 2026-08-08 | **Current baseline.** Timings and tool-call behaviour for the agentic suite. |
| `bench-accuracy-2026-08-08.md` | 2026-08-08 | Accuracy scoring for the same runs. Named for the runs' date, not the write-up's. |
| `inference-tuning-2026-06-27.md` | 2026-06-27 | omlx 0.4.4-era tuning. Its conclusions still hold and are the reason none of these were revisited: speculative decoding does not help this MoE, and DFlash, MTP and SpecPrefill were each rejected with a stated reason. Its "final omlx state" section describes a backend that is now the backup, not the default. |

Accepted scores live in `docs/quality-baseline.md`, written by `scripts/bench_eval.py
--write-baseline`, not here.
