"""Opt-in MoE expert disk offloading (spec: specs/moe-expert-offload-02-runtime-cache.md).

Wires the fork's `SwitchLinear.enable_offload()`/`QuantizedSwitchLinear.enable_offload()`
(mlx_lm/models/switch_layers.py, mira-core-pin branch) to a DiskExpertCacheStore
reading straight from the model's own safetensors shards. Only every
`resident_expert_fraction` of each layer's experts stay resident in unified
memory; the rest are fetched on a cold miss and LRU-evicted.

Only touches SwitchLinear/QuantizedSwitchLinear instances — the always-active
`shared_expert` (qwen3_next.py) is a plain (Quantized)Linear, not per-expert
stacked, so it's never matched here and is never a candidate for eviction
(edge case (b) in the spec: it must never be offloaded).

Same discipline as expert_profiler.py: this only runs when explicitly
enabled, and a bug here can only make generation slower/raise, never change
which weights the model has — enable_offload() shrinks the resident tensor
by slicing the *same* eagerly-loaded weights already in memory, so cold
experts are byte-identical to what the eager loader would have kept
resident.
"""

import logging
from typing import Optional

from mlx_lm.utils import hf_repo_to_path

from core.inference.disk_expert_cache import DiskExpertCacheStore

logger = logging.getLogger(__name__)


def install(model, model_id: str, resident_expert_fraction: float) -> Optional[DiskExpertCacheStore]:
    """Enable disk-backed expert offloading on every SwitchLinear/
    QuantizedSwitchLinear module found in `model`.

    Returns the DiskExpertCacheStore (for /v1/stats hit-rate reporting), or
    None if the model has no such modules (e.g. a dense model like
    Ministral 3 14B) or `resident_expert_fraction` disables offloading
    entirely (>= 1.0, same as leaving it unset).
    """
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    if resident_expert_fraction >= 1.0:
        logger.info("expert offload: resident_expert_fraction >= 1.0, nothing to offload")
        return None

    model_path = hf_repo_to_path(model_id)
    store = DiskExpertCacheStore(model_path)

    n_found = 0
    n_enabled = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedSwitchLinear):
            attrs = ["weight", "scales", "biases"]
        elif isinstance(module, SwitchLinear):
            attrs = ["weight"]
        else:
            continue
        n_found += 1
        n_experts = module.num_experts
        resident_slots = max(1, round(n_experts * resident_expert_fraction))
        try:
            fetch_fn = store.reader_for(name, attrs)
        except KeyError:
            logger.warning("expert offload: could not resolve safetensors keys for %s, skipping", name)
            continue
        module.enable_offload(resident_slots, fetch_fn)
        n_enabled += 1

    if n_found == 0:
        logger.warning(
            "expert offload: no SwitchLinear/QuantizedSwitchLinear modules found on this "
            "model — offloading is a no-op (expected for dense models)"
        )
        return None
    logger.info(
        "expert offload: enabled on %d/%d switch-linear modules (resident_expert_fraction=%.2f)",
        n_enabled, n_found, resident_expert_fraction,
    )
    return store


def stats(model) -> dict:
    """Sum per-module hit/miss counters across every offload-enabled
    SwitchLinear/QuantizedSwitchLinear in `model` — hit tracking lives on
    each module (a hit never calls back into DiskExpertCacheStore), so this
    is the only place a whole-model total exists. Cheap enough to call from
    GET /v1/stats directly (not on any per-token hot path)."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    hits = misses = 0
    for _, module in model.named_modules():
        if not isinstance(module, (QuantizedSwitchLinear, SwitchLinear)):
            continue
        hits += getattr(module, "_offload_hits", 0)
        misses += getattr(module, "_offload_misses", 0)
    return {"hits": hits, "misses": misses}
