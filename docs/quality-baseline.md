# Quality baseline

Accepted scores, written by `scripts/bench_eval.py --write-baseline`.

A baseline can be wrong: if it was taken on a build that was already
regressed, a gate reading it will cheerfully protect the regression. The
run it came from is named below so it can be re-examined rather than
trusted forever.

- run label: `qwen3.6-full16`  (the bench's --model tag, NOT a model id — the real model comes from mira.yaml; comparisons key on this label)
- build: `64e0e78`
- judge: `mlx-community/Qwen3.6-35B-A3B-4bit` (prompt `aa150a41626f`)
- judged noise floor: `+/-1`, measured by `scripts/bench_noise.py` over repeated runs of one build

- covers: 15 of 16 questions — a baseline is only a bar for the questions it contains, and a run of a different subset compares only where they overlap

| Q | tier1 | judged | safety |
|---|-------|--------|--------|
| 1 | 2 | - | - |
| 2 | 2 | 2 | - |
| 3 | - | 2 | - |
| 4 | 2 | 1 | - |
| 5 | - | 2 | - |
| 6 | 2 | - | - |
| 7 | 2 | 2 | - |
| 8 | 2 | 1 | - |
| 9 | 2 | - | - |
| 10 | - | 2 | - |
| 11 | 2 | - | - |
| 12 | 2 | - | - |
| 13 | 2 | - | - |
| 14 | 2 | 1 | pass |
| 15 | 2 | 2 | pass |

Excluded as partial (evidence not captured in the source run): Q16. Re-run the bench to include them.
