# Lazy expert load: never materialize the full expert table

Status: implemented 2026-07-19. One gated keyword, `load(..., lazy=True)` when offload is enabled
(`core/inference/mira_mlx_server.py`). Whole-run peak on Qwen3.6-35B-A3B at fraction 0.3 dropped from
18.21GB to 7.48GB, and the full expert table no longer becomes a live buffer at any point in the
lifecycle, so a model whose expert table exceeds DRAM can open. `docs/moe-offload-case-study.md` has
the full story of how I got here, including the wrong turn.

I originally scoped this as a three-component change and wrote it up assuming the 18.21GB peak was a
first-forward transient. Both were wrong. A standalone probe with per-phase `reset_peak()` showed the
peak is purely load-time eager materialization (the live server never resets `mx.get_peak_memory()`,
so the load high-water mark was still showing at request time and looked request-caused), and the fix
collapsed to a single keyword. The rest of this doc reflects the corrected, measured mechanism.

## Problem

Expert disk offloading bounds steady-state and per-forward memory to about 6.5GB on Qwen3.6-35B-A3B at
resident fraction 0.3, but `mlx_lm.load` with the default `lazy=False` eagerly materializes the full
stacked `(num_experts, ...)` expert table into unified memory at load: 18.17GB active the instant
`load()` returns, before a single token. That one load-time allocation is the largest memory event in
the whole lifecycle and the wall that stops a model whose expert table exceeds DRAM from opening at
all, even when its offloaded steady state (6.5GB) would fit with room to spare.

## Root cause (load flow), as measured

`core/inference/mira_mlx_server.py` calls `mlx_lm.utils.load(model_path)` with the default
`lazy=False`:

1. `load_model` (`utils.py:282`) reads the raw safetensors and calls `model.sanitize(weights)`
   (`qwen3_5_moe.py:23-52`), which constructs the full stacked `(num_experts, ...)` expert tensors
   from the raw checkpoint (`experts.gate_up_proj` into `switch_mlp.gate_proj/up_proj/down_proj.weight`).
2. `model.load_weights(...)` (`utils.py:415`) binds them into the module tree, still unevaluated.
3. `mx.eval(model.parameters())` (`utils.py:418`) evaluates them, and active jumps to 18.17GB
   immediately (standalone probe: active and peak both 18.17 right after `load()`, before install; a
   redundant `mx.eval(model.parameters())` afterward adds nothing).
4. mira then runs `expert_offload.install()`, which seeds the resident set from disk and replaces each
   module's full weight with a 1-row stand-in, dropping the full-tensor references. Active falls from
   18.17 to 6.37GB, and the freed bytes go to MLX's buffer-reuse pool (`get_cache_memory` reads
   16.88GB) until `mx.clear_cache()` returns them to the OS. Every forward after that is bounded
   (tiny 6.51GB, diverse 1458-token prefill 7.48GB). No forward re-materializes the table.

So the full tensors are built and evaluated at load for nothing. The disk expert cache reads expert
bytes straight from the raw checkpoint by offset (`core/inference/disk_expert_cache.py`), so the model
never needs the stacked expert tensors to exist as live arrays. The only reason they cost 18GB is
step 3's eager eval.

## Goal (met)

The full `(num_experts, ...)` expert tensors must never become live MLX arrays. Then peak is about
equal to steady state (6.5GB), independent of expert-table size, and models with over-DRAM expert
tables run.

## The fix

The step-0 probe showed the peak is purely step 3's eager `mx.eval(model.parameters())`. It is not a
first-forward fault, and it is not something that survives `install()`'s stand-in swap. That collapses
the original three-component scope to one keyword:

`load(self.model_path, ..., lazy=True)`, gated on `self.resident_expert_fraction is not None`
(`core/inference/mira_mlx_server.py`). With `lazy=True`, `sanitize` still constructs the stacked
expert tensors, but they stay unevaluated graph nodes with zero wired bytes, and `install()` then
seeds the resident set from disk and its stand-in swap drops those nodes before anything forces their
eval, so the full table is never materialized at any point. Measured on a standalone probe at fraction
0.3:

| phase | `lazy=False` (before) | `lazy=True` (fix) |
|---|---|---|
| after `load()` | active/peak 18.17GB | active/peak 0.00GB |
| after `install()` | active 6.37 / peak 18.21 | active 5.08 / peak 5.08 |
| after diverse 1458-token prefill | peak 18.21 (set at load) | peak 7.48 |

End-to-end on a real server (`/v1/stats` peak): pre-request 18.21GB to 0.0GB, after a diverse prefill
18.21GB to 7.25GB. Output stays coherent.

The two other components I had originally scoped both turned out to be unnecessary:

- Filtering expert keys before `load_weights`: not needed, because unevaluated nodes cost nothing and
  the stand-in swap discards them. Skipping it avoids fork- and model-version-sensitive `sanitize`
  surgery.
- An offload-native `SwitchLinear.__init__`: not needed, because `load()` overwrites the `__init__`
  placeholder weights before any eval, and under `lazy=True` neither the placeholder nor the loaded
  table is ever materialized.

## Why `lazy=True` is safe only with offload

Without offload there is no `install()`, so there is no stand-in swap to drop a deferred table, and
`lazy=True` would merely postpone the same 18GB materialization to the first forward. The default
non-offload path keeps `lazy=False`, unchanged. A dense model (no `SwitchLinear`) launched with the
offload flag on is a benign misconfiguration: `install()` finds nothing to swap, and the dense weights
materialize on the first forward instead of at load, deferred but not broken.

## Open questions, resolved by the probe

- Why does the first forward fault the full table? It does not. The 18.21GB was load-time eager eval,
  and the "first-forward" reading was the server's un-reset `peak_memory` high-water mark from load
  still showing at request time. Per-phase `reset_peak()` in a standalone probe settled it.
- `sanitize` builds the tensors. True, but under `lazy=True` they are never evaluated, so their
  construction is free. No need to bypass `sanitize` or filter keys.
- The quantized path (`_maybe_qq`, `utils.py:410`) does not re-materialize experts under `lazy=True`:
  post-fix diverse-prefill output is coherent and peak stays in the 7.25 to 7.48GB range.

## Verification

- Peak: standalone probe whole-run peak 18.21 to 7.48GB; server `/v1/stats` peak 18.21 to 7.25GB.
- Correctness: `tests/test_expert_offload.py` plus `tests/test_config.py` all pass; live diverse
  prefill produces coherent output; offload hit/miss counters behave as before.
- The actual goal, demonstrated 2026-07-19: `Qwen3.6-35B-A3B-8bit` (about 33.8GB expert table, larger
  than 32GB RAM and the ~25GB wired limit) loaded via `load(lazy=True)` plus offload at 0.3 and
  generated coherent text at peak 12.71GB resident (seed 9.59GB, 59346 hits against 30615 cold-miss
  disk fetches). Eager load is impossible: the measured materialization trajectory extrapolates to
  about 30.9GB for the expert table alone. See case study section 9; reproduce with
  `scripts/moe_overdram_demo.py`.

## Risks and residual notes

- The change is one gated keyword in mira-core (no fork change), guarded behind the offload flag, with
  the offload test suite as the gate. Low blast radius.
- The MLX lazy and mmap lifecycle is the same "verify, do not assume" class that produced the earlier
  crashes, so every step here landed behind a probe measurement rather than reasoning.
- Over-DRAM is now directly observed rather than inferred, via the 8-bit run above. That 8-bit model
  is cached but is a demo artifact, not a production model (`MODEL_REGISTRY.md`).
