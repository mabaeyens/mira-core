"""Opt-in MoE expert-activation logging (docs/moe-offload-case-study.md).

This is pure instrumentation for the expert-offloading go/no-go decision — it
has no effect on generation output and must be explicitly enabled
(`--profile-experts` / `MIRA_MLX_PROFILE_EXPERTS=1`). It answers one question:
is expert activation skewed/correlated enough (per Eliseev & Mazur's
Mixtral-offloading paper and Alizadeh et al.'s "LLM in a Flash") to make a
resident-expert cache worthwhile, or is it close to uniform-random (in which
case offloading buys nothing)?

Mechanics: rather than hook each architecture's own SparseMoeBlock (Qwen3.6's
`Qwen3NextSparseMoeBlock` names its router `self.gate`/`self.top_k`; Gemma4's
router is a differently-shaped `self.proj`/config `top_k_experts` — the two
aren't attribute-compatible), this patches `mlx_lm.models.switch_layers`'s
`SwitchGLU`/`SwitchMLP` directly: both share the call signature
`__call__(self, x, indices)`, and `indices` (the already-selected expert ids)
is the one architecture-agnostic signal that's identical across every MoE
model mlx-lm ships, since SwitchGLU/SwitchMLP are the shared sparse-MoE
compute primitive underneath all of them. No gate/top-k math is duplicated —
this only ever *observes* an argument already computed by the real forward
pass, which is why a bug here can only corrupt logged data, never generation
output. The tradeoff: router weights (gate scores) aren't visible at this
interception point, only which experts were chosen — sufficient for spec 01's
concentration/overlap analysis; scoring/priority signal is deferred to spec 02
if offloading proceeds.

MLX's `nn.Module.__call__` is a dunder, so instance-level monkey-patching
(`instance.__call__ = ...`) is silently ignored by the `instance(x)` call
syntax — Python resolves special methods on the type, not the instance.
`install()` therefore patches the SwitchGLU/SwitchMLP *class* once (shared
across every layer/instance) and tags each instance with a layer index via a
plain attribute.
"""

import json
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


class ExpertProfiler:
    """Buffers routing-decision records in memory, flushed to JSONL async.

    Conversation attribution is best-effort: `active_job_ids_fn()` is called
    at record time and returns whatever the engine considers "active" right
    now, not a true per-token mapping into the batched forward tensor (mira-mlx
    doesn't track that boundary anywhere). This is exact when
    completion_batch_size == 1 (mira-mlx's default), which is the common case
    for a single-user desktop app; with true batched concurrency, an entry's
    active_job_ids may include jobs whose tokens weren't part of every
    logged row. Good enough for the aggregate skew/overlap analysis this spec
    exists to answer — not a claim of exact per-request attribution.
    """

    def __init__(
        self,
        log_path: Path,
        active_job_ids_fn: Callable[[], List[str]],
        flush_interval_s: float = 2.0,
    ):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_job_ids_fn = active_job_ids_fn
        self._buffer: "deque[dict]" = deque()
        self._lock = threading.Lock()
        self._call_counter = 0
        self._stop = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, args=(flush_interval_s,), name="expert-profiler-flush", daemon=True
        )
        self._flush_thread.start()
        logger.info("expert profiler enabled: logging to %s", self._log_path)

    def record(self, layer_idx: int, expert_ids: mx.array) -> None:
        # .tolist() forces synchronous eval on the caller's (engine) thread —
        # required, since mx arrays/streams are thread-local and must not
        # cross into the background flush thread still lazy.
        entry = {
            "ts": time.time(),
            "call_id": self._call_counter,
            "layer_idx": layer_idx,
            "expert_ids": expert_ids.tolist(),
            "active_job_ids": self._active_job_ids_fn(),
        }
        self._call_counter += 1
        with self._lock:
            self._buffer.append(entry)

    def _flush_loop(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            self._flush()
        self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            entries = list(self._buffer)
            self._buffer.clear()
        try:
            with open(self._log_path, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.exception("expert profiler: failed to flush %d entries", len(entries))

    def close(self) -> None:
        self._stop.set()
        self._flush_thread.join(timeout=5)
        self._flush()


_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


def _layer_idx_from_name(name: str, fallback_counter: int) -> int:
    """`named_modules()` traverses list-valued children (e.g. `model.layers`)
    in REVERSE order (confirmed empirically: layers.2, layers.1, layers.0) —
    a plain traversal-order counter silently reverses every layer's index.
    Parse the real index out of the dotted module path (e.g. "layers.14.mlp"
    -> 14) instead; fall back to the traversal counter only if the path has
    no such segment (logged once as a warning by the caller)."""
    m = _LAYER_IDX_RE.search(name)
    return int(m.group(1)) if m else fallback_counter


def _patch_switch_class(cls: type, profiler: ExpertProfiler) -> None:
    if getattr(cls, "_mira_profiler_patched", False):
        return
    original_call = cls.__call__

    def patched_call(self, x, indices, *args, **kwargs):
        y = original_call(self, x, indices, *args, **kwargs)
        try:
            profiler.record(self._mira_profiler_layer_idx, indices)
        except Exception:  # noqa: BLE001 - never let logging break generation
            logger.debug("expert profiler: failed to record layer", exc_info=True)
        return y

    cls.__call__ = patched_call
    cls._mira_profiler_patched = True


def install(model, profiler: ExpertProfiler) -> bool:
    """Patch every SwitchGLU/SwitchMLP instance found in `model` to log
    routing decisions via `profiler`.

    Returns False (no-op, warns) if the model has no such modules — e.g. a
    dense model like Ministral 3 14B, mira-mlx's own non-flagship default.
    """
    from mlx_lm.models.switch_layers import SwitchGLU, SwitchMLP

    fallback_counter = 0
    used_fallback = False
    patched_classes = set()
    n_found = 0
    for name, module in model.named_modules():
        if not isinstance(module, (SwitchGLU, SwitchMLP)):
            continue
        n_found += 1
        layer_idx = _layer_idx_from_name(name, fallback_counter)
        if layer_idx == fallback_counter and "layers." not in name:
            used_fallback = True
        fallback_counter += 1
        module._mira_profiler_layer_idx = layer_idx
        cls = type(module)
        if cls not in patched_classes:
            _patch_switch_class(cls, profiler)
            patched_classes.add(cls)
    if n_found == 0:
        logger.warning(
            "expert profiler: no SwitchGLU/SwitchMLP modules found on this model — "
            "profiling is a no-op (expected for dense models; expert offloading only "
            "applies to MoE architectures)"
        )
        return False
    if used_fallback:
        logger.warning(
            "expert profiler: could not parse a layer index from module path for at least "
            "one MoE block — falling back to traversal-order numbering, which may not match "
            "physical layer order for this architecture"
        )
    logger.info("expert profiler: patched %d MoE layers", n_found)
    return True
