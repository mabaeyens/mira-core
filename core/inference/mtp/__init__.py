"""Native MTP (Multi-Token Prediction) for mira-mlx — Qwen3.5/3.6 MoE.

Mira-owned, self-authored self-speculative decoding. The *mechanism* (an MTP
head that drafts token t+2 from the backbone's pre-norm hidden fused with the
sampled main token's embedding, verified in one backbone forward) is understood
from DeepSeek-V3, mlx-lm PR #990, and omlx's Apache-2.0 ``patches/mlx_lm_mtp``
reference — see ``docs``/README Acknowledgements. This code is original.

This package is scoped, on purpose, to what mira actually runs: the Qwen3.6 MoE
default (``model_type == "qwen3_5_moe"``), **width-1** decode (mira forces
single-sequence decode on the M5; omlx's MTP is width-1 too). It does NOT try to
support batched MTP.

Layout:
- ``qwen3_mtp``  — the model-side head + the class patches that attach it.
- ``sidecar``    — locating / assembling the bf16 ``model-mtp.safetensors`` head
  so no external app or hand-built ``~/.omlx/models`` dir is required.

The head is attached by an idempotent patch of ``mlx_lm.models.qwen3_5`` applied
*before* ``mlx_lm.load`` — a subclass can't be used there because ``load``
instantiates the model class from mlx-lm's ``model_type`` registry, so nothing
would construct a mira subclass. (The forthcoming decode loop is a different
story: it is a class mira instantiates itself and *is* a subclass.)

Activation is a two-step handshake so a stock, MTP-off build is byte-identical to
today: ``apply()`` installs the patches, ``set_active(True)`` tells the patched
``TextModel.__init__`` to actually build the head. With the flag off,
``hasattr(model, "mtp")`` is False and ``sanitize`` strips ``mtp.*``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Module-global activation + depth. Read by the patched TextModel.__init__ at
# construction time (inside mlx_lm.load), which is why they can't be passed as
# arguments. Set them via set_active()/set_depth() before loading.
_ACTIVE: bool = False
_DEPTH: int = 3  # max chained draft tokens per verify cycle (Qwen3.6 default)

_DEPTH_MIN, _DEPTH_MAX = 1, 8


def set_active(active: bool) -> None:
    global _ACTIVE
    _ACTIVE = bool(active)


def is_active() -> bool:
    return _ACTIVE


def set_depth(depth: int) -> None:
    """Set the max chained draft depth, clamped to [1, 8]."""
    global _DEPTH
    _DEPTH = max(_DEPTH_MIN, min(_DEPTH_MAX, int(depth)))


def get_depth() -> int:
    return _DEPTH


def apply() -> bool:
    """Install the Qwen3.5/3.6 MTP model patches. Idempotent; safe to call more
    than once. Returns True if the patch is in place (or already was), False if
    mlx-lm's qwen3_5 module isn't importable."""
    from . import qwen3_mtp

    return qwen3_mtp.apply()
