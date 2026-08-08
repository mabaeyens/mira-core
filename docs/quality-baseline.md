# Quality baseline

Accepted scores, written by `scripts/bench_eval.py --write-baseline`.

A baseline can be wrong: if it was taken on a build that was already
regressed, a gate reading it will cheerfully protect the regression. The
run it came from is named below so it can be re-examined rather than
trusted forever.

- run label: `qwen3.6-baseline`  (the bench's --model tag, NOT a model id — the real model comes from mira.yaml; comparisons key on this label)
- build: `5cb4dc8+dirty`
- judge: `mlx-community/Qwen3.6-35B-A3B-4bit` (prompt `950ae3b8f239`)

- covers: 6 of 16 questions — a baseline is only a bar for the questions it contains, and a run of a different subset compares only where they overlap

| Q | tier1 | judged | safety |
|---|-------|--------|--------|
| 1 | 2 | - | - |
| 6 | 2 | - | - |
| 9 | 2 | - | - |
| 11 | 2 | - | - |
| 12 | 2 | - | - |
| 13 | 2 | - | - |
