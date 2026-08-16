"""Model-side MTP head for Qwen3.5/3.6 (dense + MoE), attached to mlx-lm.

Original mira code. The mechanism is understood from mlx-lm PR #990 and omlx's
Apache-2.0 ``patches/mlx_lm_mtp`` reference; this is a fresh implementation scoped
to what mira runs (Qwen3.6 MoE, width-1). See package docstring for why this is a
patch of ``mlx_lm.models.qwen3_5`` rather than a subclass.

Scope of THIS slice (spec §5.1 head + §5.2 sidecar/sanitize):
- The MTP head (``MTPHead`` / ``MTPBlock``) and its attachment.
- Backbone forward exposing the **pre-norm** hidden the head fuses.
- ``sanitize`` that KEEPS ``mtp.*`` and applies the RMSNorm +1 convention shift
  (the silent accept-rate killer if missed), plus MoE-expert unfusing for the
  head layers.
- ``mtp_forward`` / ``make_mtp_cache`` so a decode loop can drive the head.

Deliberately DEFERRED to the decode-loop slice (§5.3): the ``n_confirmed`` split
of ``GatedDeltaNet`` and ``mtp_partial_rollback`` (linear-layer KV rollback on a
rejected draft). Nothing here drives ``n_confirmed > 0`` yet.

Attribute names inside the head are fixed by the checkpoint tensor keys
(``mtp.fc.weight``, ``mtp.layers.N.*``, ``mtp.norm.weight``,
``mtp.pre_fc_norm_hidden.weight`` …); the class names are mira's.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MARKER = "_mira_mtp_owned"  # stamped on functions/classes we install


def _ours(cls: Any, attr: str) -> bool:
    """True iff cls.<attr> is a method we installed (checks this class only, not
    inherited), so a re-apply after some other patch clobbered it re-establishes
    ownership instead of chaining wraps."""
    return getattr(cls.__dict__.get(attr), _MARKER, False)


def apply() -> bool:
    """Idempotently install the Qwen3.5/3.6 MTP patches. Returns False if mlx-lm's
    qwen3_5 module is not importable."""
    try:
        from mlx_lm.models import qwen3_5 as q35
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_5 not importable; MTP patch skipped")
        return False

    # If upstream mlx-lm ever ships native MTP (PR #990 merged), stand down.
    if hasattr(q35.TextModel, "mtp_forward") and not _ours(q35.TextModel, "mtp_forward"):
        logger.info("mlx-lm already provides Qwen3.5/3.6 MTP; mira patch stands down")
        return True

    _patch_args(q35)
    _register_head_classes(q35)
    _patch_inner_text_model(q35)
    _patch_text_model(q35)
    _patch_outer_model(q35)
    _patch_moe_sanitize()
    logger.info("mira native MTP: Qwen3.5/3.6 model patch applied")
    return True


# --------------------------------------------------------------------------- #
# TextModelArgs — surface mtp_num_hidden_layers (BaseModelArgs.from_dict drops   #
# unknown keys, so a plain load would discard it).                              #
# --------------------------------------------------------------------------- #

def _patch_args(q35: Any) -> None:
    args_cls = q35.TextModelArgs
    if getattr(args_cls, "_mira_mtp_args_patched", False):
        return
    original = args_cls.from_dict.__func__  # unwrap classmethod

    def from_dict(cls, params):
        inst = original(cls, params)
        inst.mtp_num_hidden_layers = int(params.get("mtp_num_hidden_layers", 0) or 0)
        return inst

    args_cls.from_dict = classmethod(from_dict)
    args_cls._mira_mtp_args_patched = True


# --------------------------------------------------------------------------- #
# The head: MTPBlock (one full-attention transformer layer) + MTPHead.          #
# --------------------------------------------------------------------------- #

def _register_head_classes(q35: Any) -> None:
    if hasattr(q35, "_MiraMTPHead"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    Attention = q35.Attention
    SparseMoeBlock = q35.SparseMoeBlock
    MLP = q35.MLP
    create_attention_mask = q35.create_attention_mask

    class MTPBlock(nn.Module):
        """A single transformer layer inside the MTP head. Always full-attention
        (the head does not use the backbone's linear/GatedDeltaNet path); MoE when
        the backbone is MoE. Attribute names match the ``mtp.layers.N.*`` keys."""

        def __init__(self, args):
            super().__init__()
            self.self_attn = Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            if args.num_experts > 0:
                self.mlp = SparseMoeBlock(args)
            else:
                self.mlp = MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPHead(nn.Module):
        """Multi-Token Prediction head. Predicts token t+2 by fusing the backbone
        pre-norm hidden at t with the embedding of the sampled main token t+1,
        then running ``mtp_num_hidden_layers`` full-attention blocks. Attribute
        names match ``mtp.pre_fc_norm_*`` / ``mtp.fc`` / ``mtp.layers.N`` /
        ``mtp.norm``."""

        def __init__(self, args):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [MTPBlock(args) for _ in range(args.mtp_num_hidden_layers)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            e = self.pre_fc_norm_embedding(embed_tokens(next_token_ids))
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [None] * len(self.layers)
            mask = create_attention_mask(fused, cache[0] if cache else None)
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask, c)
            return self.norm(fused)

    q35._MiraMTPBlock = MTPBlock
    q35._MiraMTPHead = MTPHead


# --------------------------------------------------------------------------- #
# Inner Qwen3_5TextModel — return PRE-norm hidden (the head fuses it; the outer  #
# TextModel applies model.norm to make logits).                                 #
# --------------------------------------------------------------------------- #

def _patch_inner_text_model(q35: Any) -> None:
    cls = q35.Qwen3_5TextModel
    if _ours(cls, "__call__"):
        return

    create_attention_mask = q35.create_attention_mask
    create_ssm_mask = q35.create_ssm_mask

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            hidden = input_embeddings
        else:
            hidden = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(hidden, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[self.ssm_idx])
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=c)
        return hidden  # pre-norm; caller applies self.norm

    setattr(__call__, _MARKER, True)
    cls.__call__ = __call__


# --------------------------------------------------------------------------- #
# TextModel — attach head, expose pre/post-norm, mtp_forward, make_mtp_cache,    #
# sanitize (keep mtp.* + norm shift).                                           #
# --------------------------------------------------------------------------- #

def _patch_text_model(q35: Any) -> None:
    cls = q35.TextModel
    init_wrapped = getattr(cls, "_mira_mtp_init_wrapped", False)
    if init_wrapped and _ours(cls, "__call__"):
        return

    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__

    def __init__(self, args):
        original_init(self, args)
        from . import is_active

        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        if n_mtp > 0 and is_active():
            self.mtp = q35._MiraMTPHead(args)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
    ):
        hidden = self.model(inputs, cache, input_embeddings=input_embeddings)
        normed = self.model.norm(hidden)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        if return_hidden:
            return logits, hidden  # hidden is pre-norm — what the head fuses
        return logits

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, logits_keep: int = 0):
        """Run the MTP head and project to vocab logits. ``logits_keep`` limits the
        lm_head projection to the last N positions (0 = all) — the large vocab
        makes skipping unused rows worthwhile."""
        head_out = self.mtp(hidden_states, next_token_ids, self.model.embed_tokens, mtp_cache)
        src = head_out
        if logits_keep and src.shape[1] > logits_keep:
            src = src[:, -logits_keep:, :]
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(src)
        return self.lm_head(src)

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    setattr(__call__, _MARKER, True)
    setattr(mtp_forward, _MARKER, True)
    setattr(make_mtp_cache, _MARKER, True)
    setattr(_sanitize_text_model, _MARKER, True)
    if not init_wrapped:
        cls.__init__ = __init__
        cls._mira_mtp_init_wrapped = True
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = _sanitize_text_model


def _sanitize_text_model(self, weights):
    """Replace stock ``TextModel.sanitize`` (which unconditionally strips
    ``mtp.*``). Keep the head weights when a head is attached, and apply the
    RMSNorm +1 convention shift — including to the head's own norms, decided
    PER-KEY from each weight's magnitude, because a head sidecar can carry raw-HF
    (mean ~0) and already-MLX (mean ~1) norm conventions mixed together. Getting
    this wrong doesn't error; it silently collapses the head to flat logits and
    drives draft acceptance to ~0%."""
    import mlx.core as mx

    # Backbone norms shift only on a raw-HF checkpoint, detected via un-transposed
    # conv1d. An already-MLX base (mira's 4-bit default) → no backbone shift.
    backbone_shift = any(
        "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1
        for k, v in weights.items()
    )

    has_head = hasattr(self, "mtp")
    if not has_head:
        weights = {k: v for k, v in weights.items() if "mtp." not in k}
    elif not any("mtp." in k for k in weights):
        raise ValueError(
            "MTP is enabled but the weights carry no mtp.* tensors. The public "
            "4-bit quant strips the MTP head; provide the bf16 model-mtp.safetensors "
            "sidecar (see core/inference/mtp/sidecar.py) or disable mtp_enabled."
        )

    if self.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    backbone_norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    head_norm_suffixes = (
        ".pre_fc_norm_hidden.weight",
        ".pre_fc_norm_embedding.weight",
        "mtp.norm.weight",
    )

    def _is_raw_hf(v) -> bool:
        # Raw-HF RMSNorm gammas center near 0; MLX-converted near 1.
        try:
            return float(mx.mean(v.astype(mx.float32)).item()) < 0.5
        except Exception:
            return backbone_shift

    out = {}
    for k, v in weights.items():
        if "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1:
            v = v.moveaxis(2, 1)
        if getattr(v, "ndim", 0) == 1:
            if "mtp." in k and (
                any(k.endswith(s) for s in head_norm_suffixes)
                or any(k.endswith(s) for s in backbone_norm_suffixes)
            ):
                # Per-key decision for every norm inside the head.
                if backbone_shift or _is_raw_hf(v):
                    v = v + 1.0
            elif backbone_shift and any(k.endswith(s) for s in backbone_norm_suffixes):
                v = v + 1.0
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Outer Model (qwen3_5.Model) — pass through return_hidden / mtp_forward /       #
# make_mtp_cache to the language model.                                          #
# --------------------------------------------------------------------------- #

def _patch_outer_model(q35: Any) -> None:
    cls = q35.Model
    if _ours(cls, "__call__"):
        return

    def __call__(self, inputs, cache=None, input_embeddings=None, return_hidden: bool = False):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings, return_hidden=return_hidden
        )

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, logits_keep: int = 0):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache, logits_keep=logits_keep
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    for fn in (__call__, mtp_forward, make_mtp_cache):
        setattr(fn, _MARKER, True)
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache


# --------------------------------------------------------------------------- #
# qwen3_5_moe.Model.sanitize — also unfuse the MTP head's MoE experts (the head  #
# ships fused gate_up_proj like the backbone), then delegate to TextModel.       #
# --------------------------------------------------------------------------- #

def _patch_moe_sanitize() -> None:
    try:
        from mlx_lm.models import qwen3_5_moe as moe
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_5_moe not importable; MoE MTP sanitize skipped")
        return
    cls = moe.Model
    if _ours(cls, "sanitize"):
        return

    def _unfuse(weights, prefix):
        gate_up_key = f"{prefix}.experts.gate_up_proj"
        if gate_up_key not in weights:
            return
        gate_up = weights.pop(gate_up_key)
        mid = gate_up.shape[-2] // 2
        weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
        weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
            f"{prefix}.experts.down_proj"
        )

    def sanitize(self, weights):
        new_weights = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            new_weights[key] = value

        # Backbone MoE experts (unchanged from stock).
        for l in range(self.language_model.args.num_hidden_layers):
            _unfuse(new_weights, f"language_model.model.layers.{l}.mlp")

        # MTP head MoE experts (bf16 sidecar; same fused layout as the backbone).
        mtp_num = int(getattr(self.language_model.args, "mtp_num_hidden_layers", 0) or 0)
        for l in range(mtp_num):
            _unfuse(new_weights, f"language_model.mtp.layers.{l}.mlp")

        return self.language_model.sanitize(new_weights)

    setattr(sanitize, _MARKER, True)
    cls.sanitize = sanitize
