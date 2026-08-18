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
- The ``n_confirmed`` split of ``GatedDeltaNet`` + ``mtp_partial_rollback`` so a
  rejected draft rolls the hybrid cache back cheaply: full-attention ``KVCache``
  layers ``trim`` the rejected positions; linear ``GatedDeltaNet`` layers restore
  the pre-forward ``(conv, ssm)`` state and replay ONLY the recurrence over the
  accepted slice from the stashed projected inputs — no second backbone forward.
  With ``n_confirmed == 0`` (every non-MTP forward) ``GatedDeltaNet`` is
  behaviourally identical to stock, so the default path is untouched.

Attribute names inside the head are fixed by the checkpoint tensor keys
(``mtp.fc.weight``, ``mtp.layers.N.*``, ``mtp.norm.weight``,
``mtp.pre_fc_norm_hidden.weight`` …); the class names are mira's.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MARKER = "_mira_mtp_owned"  # stamped on functions/classes we install


def _ours(cls: Any, attr: str) -> bool:
    """True iff cls.<attr> is a method we installed (checks this class only, not
    inherited), so a re-apply after some other patch clobbered it re-establishes
    ownership instead of chaining wraps."""
    return getattr(cls.__dict__.get(attr), _MARKER, False)


# MoE expert-gather sort threshold. mlx-lm hardcodes ``do_sort = indices.size >= 64``
# in SwitchGLU, which leaves the single-stream MTP verify batch UNSORTED: at the
# depth-3 default it routes 4 tokens x top_k=8 = 32 indices (< 64), so an expert
# two verified tokens share is read from memory twice instead of once. Sorting
# coalesces same-expert rows into a single weight read. Measured 2026-08-18:
# sorting the depth-3 verify batch is bit-identical to stock and ~20% faster per
# MoE forward (held steady across 32 cold-streamed layers), projecting ~+12%
# decode. We lower the threshold to 16 so verify batches (M=2..7, size 16..56)
# sort while single-token stock decode (M=1, size 8) stays unsorted — sorting 8
# no-reuse experts there is pure overhead. Set MIRA_MLX_MTP_MOE_SORT_THRESHOLD=64
# to restore stock behavior (the A/B "off" arm).
_MOE_SORT_THRESHOLD = int(os.environ.get("MIRA_MLX_MTP_MOE_SORT_THRESHOLD", "16") or "16")


def set_moe_sort_threshold(n: int) -> None:
    """Set the routed-index count at/above which the MoE expert gather is sorted
    (coalesced). Read live by the patched SwitchGLU on every forward, so a paired
    A/B can flip it in one process with no reload."""
    global _MOE_SORT_THRESHOLD
    _MOE_SORT_THRESHOLD = max(1, int(n))


def get_moe_sort_threshold() -> int:
    return _MOE_SORT_THRESHOLD


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
    _patch_gated_delta_net(q35)
    _patch_decoder_layer(q35)
    _patch_inner_text_model(q35)
    _patch_text_model(q35)
    _patch_outer_model(q35)
    _patch_moe_sanitize()
    _patch_switchglu_sort_threshold()
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
# GatedDeltaNet — n_confirmed-aware forward. On a verify forward (0 < n_confirmed #
# < S) it stashes the pre-forward (conv, ssm) state + projected (qkv, a, b) on    #
# the cache so a rejected draft can replay the recurrence over the accepted slice #
# without a second backbone forward. With n_confirmed == 0 it is behaviourally    #
# identical to stock, so every non-MTP forward is untouched.                      #
# --------------------------------------------------------------------------- #

def _patch_gated_delta_net(q35: Any) -> None:
    cls = q35.GatedDeltaNet
    if _ours(cls, "__call__"):
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.nn.layers.distributed import sum_gradients
    from mlx_lm.models.gated_delta import gated_delta_update

    def _process_chunk(self, qkv_chunk, a_chunk, b_chunk, conv_state, ssm_state,
                       ssm_mask=None, lengths=None):
        """The GatedDeltaNet recurrence over one token chunk, factored out so the
        rollback can replay just the accepted prefix. Mirrors stock's conv +
        gated_delta_update, returning (out, new_conv_state, new_ssm_state)."""
        B, S_chunk = qkv_chunk.shape[:2]
        conv_in = mx.concatenate([conv_state, qkv_chunk], axis=1)
        n_keep = self.conv_kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, S_chunk)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_conv_state = mx.take_along_axis(conv_in, positions, axis=1)
        else:
            new_conv_state = mx.contiguous(conv_in[:, -n_keep:])
        conv_out = nn.silu(self.conv1d(conv_in))
        q, k, v = [
            t.reshape(B, S_chunk, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, new_ssm_state = gated_delta_update(
            q, k, v, a_chunk, b_chunk, self.A_log, self.dt_bias, ssm_state,
            ssm_mask, use_kernel=not self.training,
        )
        return out, new_conv_state, new_ssm_state

    def __call__(self, inputs, mask=None, cache=None, n_confirmed: int = 0):
        B, S, _ = inputs.shape
        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        ssm_state = cache[1] if cache else None
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        if n_confirmed > 0 and n_confirmed < S and cache is not None:
            # MTP verify: process the whole window unsplit, but stash the
            # pre-forward state + projected inputs so a rejection can replay the
            # accepted prefix through _process_chunk (see mtp_partial_rollback).
            out, conv_f, ssm_f = self._process_chunk(qkv, a, b, conv_state, ssm_state, mask)
            cache.rollback_state = (conv_state, ssm_state)
            cache._mira_mtp_stash = (qkv, a, b)
        else:
            lengths = cache.lengths if cache is not None else None
            out, conv_f, ssm_f = self._process_chunk(
                qkv, a, b, conv_state, ssm_state, mask, lengths=lengths
            )

        if cache is not None:
            cache[0] = conv_f
            cache[1] = ssm_f
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out

    setattr(__call__, _MARKER, True)
    cls._process_chunk = _process_chunk
    cls.__call__ = __call__


# --------------------------------------------------------------------------- #
# DecoderLayer — pass n_confirmed to the linear (GatedDeltaNet) sublayer only.   #
# --------------------------------------------------------------------------- #

def _patch_decoder_layer(q35: Any) -> None:
    cls = q35.DecoderLayer
    if _ours(cls, "__call__"):
        return

    def __call__(self, x, mask=None, cache=None, n_confirmed: int = 0):
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache, n_confirmed=n_confirmed)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))

    setattr(__call__, _MARKER, True)
    cls.__call__ = __call__


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

    def __call__(self, inputs, cache=None, input_embeddings=None, n_confirmed: int = 0):
        if input_embeddings is not None:
            hidden = input_embeddings
        else:
            hidden = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(hidden, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[self.ssm_idx])
        for layer, c in zip(self.layers, cache):
            if layer.is_linear:
                hidden = layer(hidden, mask=ssm_mask, cache=c, n_confirmed=n_confirmed)
            else:
                hidden = layer(hidden, mask=fa_mask, cache=c)
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
        n_confirmed: int = 0,
    ):
        hidden = self.model(
            inputs, cache, input_embeddings=input_embeddings, n_confirmed=n_confirmed
        )
        normed = self.model.norm(hidden)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        if return_hidden:
            return logits, hidden  # hidden is pre-norm — what the head fuses
        return logits

    def mtp_forward(
        self, hidden_states, next_token_ids, mtp_cache,
        return_hidden: bool = False, logits_keep: int = 0, draft_vocab: int = 0,
    ):
        """Run the MTP head and project to vocab logits. ``logits_keep`` limits the
        lm_head projection to the last N positions (0 = all) — the large vocab
        makes skipping unused rows worthwhile. ``return_hidden`` also returns the
        head's post-norm hidden so depth-k drafting chains on the head's own
        output.

        ``draft_vocab`` (0 = full) restricts the projection to the first ``draft_vocab``
        vocab rows. This is a DRAFT-ONLY speedup: the full-vocab backbone verify decides
        every emitted token, so a reduced-vocab draft cannot change the output — it can
        only lower the accept rate if a true next token's id lands beyond ``draft_vocab``
        (measured negligible when it covers the generated-token range). The full-vocab
        lm_head is ~46% of a draft forward on Qwen3.8-27B; a 32k-row projection cuts that
        ~6x. Never pass ``draft_vocab`` on the verify forward — only on head drafts."""
        import mlx.core as mx

        head_out = self.mtp(hidden_states, next_token_ids, self.model.embed_tokens, mtp_cache)
        src = head_out
        if logits_keep and src.shape[1] > logits_keep:
            src = src[:, -logits_keep:, :]
        proj = self.model.embed_tokens if self.args.tie_word_embeddings else self.lm_head
        if draft_vocab and draft_vocab < self.args.vocab_size:
            logits = mx.quantized_matmul(
                src, proj.weight[:draft_vocab], scales=proj.scales[:draft_vocab],
                biases=proj.biases[:draft_vocab], transpose=True,
                group_size=proj.group_size, bits=proj.bits,
            )
        elif self.args.tie_word_embeddings:
            logits = proj.as_linear(src)
        else:
            logits = proj(src)
        if return_hidden:
            return logits, head_out
        return logits

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Roll the backbone cache back to ``accepted`` drafts after a verify
        forward over ``[confirmed, d1..dk]`` (num_drafts = k). Full-attention KV
        layers trim ``k - accepted`` positions; linear layers restore the stashed
        pre-forward ``(conv, ssm)`` and replay the recurrence over the kept prefix
        (confirmed + accepted drafts). Returns False if any layer lacks the state
        to roll back, so the caller can fall back to a plain step."""
        layers = self.model.layers
        if len(cache) != len(layers):
            return False
        trim_n = num_drafts - accepted
        if trim_n <= 0:
            return True  # full accept — nothing rejected to roll back
        keep = 1 + accepted  # confirmed token + accepted drafts
        # Preflight: every layer must be rollback-capable before we mutate any.
        for layer, c in zip(layers, cache):
            if getattr(layer, "is_linear", False):
                if getattr(c, "rollback_state", None) is None or getattr(
                    c, "_mira_mtp_stash", None
                ) is None:
                    return False
            elif not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                return False
        for layer, c in zip(layers, cache):
            if getattr(layer, "is_linear", False):
                conv_0, ssm_0 = c.rollback_state
                qkv_s, a_s, b_s = c._mira_mtp_stash
                _, conv_m, ssm_m = layer.linear_attn._process_chunk(
                    qkv_s[:, :keep], a_s[:, :keep], b_s[:, :keep], conv_0, ssm_0, None
                )
                c[0] = conv_m
                c[1] = ssm_m
                c.rollback_state = None
                c._mira_mtp_stash = None
            else:
                c.trim(trim_n)
        return True

    for fn in (__call__, mtp_forward, make_mtp_cache, mtp_partial_rollback, _sanitize_text_model):
        setattr(fn, _MARKER, True)
    if not init_wrapped:
        cls.__init__ = __init__
        cls._mira_mtp_init_wrapped = True
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls.sanitize = _sanitize_text_model


def _sanitize_text_model(self, weights):
    """Replace stock ``TextModel.sanitize`` (which unconditionally strips
    ``mtp.*``). Keep the head weights when a head is attached, and apply the
    RMSNorm +1 convention shift — including to the head's own norms. A head sidecar
    carries ONE convention, so the shift is decided ONCE from the head norms whose
    raw-HF gamma reliably centers near 0 (``pre_fc_*`` + the head layer norms) and
    applied uniformly to EVERY head norm — including ``q_norm``/``k_norm``/``mtp.norm``,
    whose raw-HF gammas center ABOVE 0.5 and so cannot be judged by a per-key mean
    test. Getting this wrong doesn't error; it silently leaves those norms -1 off and
    bleeds draft acceptance (worst on the deep drafts, where the head's attention
    compounds), or collapses the head to flat logits entirely (~0% accept)."""
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
    # Every RMSNorm inside the head that follows the +1 convention. q_norm / k_norm /
    # mtp.norm are here on purpose: their raw-HF gammas center ABOVE 0.5 (measured
    # ~0.79 / ~0.78 / ~1.25 on the Qwen3.8-27B sidecar), so a per-key mean<0.5 test
    # misclassifies them as already-MLX and leaves them -1 off — omlx documents this
    # exact failure (qwen35_model.py: ~14pp of draft acceptance, prose 62.8%->76.3%).
    head_norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".pre_fc_norm_hidden.weight",
        ".pre_fc_norm_embedding.weight",
        "mtp.norm.weight",
    )
    # Decide the head-norm convention ONCE, from the head norms whose raw-HF gamma
    # DOES reliably center near 0 (pre_fc_* + the head layer norms), then apply it to
    # every head norm uniformly — instead of judging q_norm/k_norm/mtp.norm per-key.
    reliable_head_suffixes = (
        ".pre_fc_norm_hidden.weight",
        ".pre_fc_norm_embedding.weight",
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
    )

    def _mean(v) -> float:
        return float(mx.mean(v.astype(mx.float32)).item())

    reliable_means = []
    for k, v in weights.items():
        if ("mtp." in k and getattr(v, "ndim", 0) == 1
                and any(k.endswith(s) for s in reliable_head_suffixes)):
            try:
                reliable_means.append(_mean(v))
            except Exception:
                pass
    # Raw-HF head sidecar (reliable norms near 0) → shift ALL head norms; already-MLX
    # head sidecar (reliable norms near 1) → shift none. Fall back to the backbone
    # signal only when no reliable head norm is present.
    head_shift = (
        (sum(reliable_means) / len(reliable_means) < 0.5)
        if reliable_means else backbone_shift
    )

    out = {}
    for k, v in weights.items():
        if "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1:
            v = v.moveaxis(2, 1)
        if getattr(v, "ndim", 0) == 1:
            if "mtp." in k and any(k.endswith(s) for s in head_norm_suffixes):
                if head_shift:
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

    def __call__(
        self, inputs, cache=None, input_embeddings=None,
        return_hidden: bool = False, n_confirmed: int = 0,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings,
            return_hidden=return_hidden, n_confirmed=n_confirmed,
        )

    def mtp_forward(
        self, hidden_states, next_token_ids, mtp_cache,
        return_hidden: bool = False, logits_keep: int = 0, draft_vocab: int = 0,
    ):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache,
            return_hidden=return_hidden, logits_keep=logits_keep, draft_vocab=draft_vocab,
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        return self.language_model.mtp_partial_rollback(cache, accepted, num_drafts)

    for fn in (__call__, mtp_forward, make_mtp_cache, mtp_partial_rollback):
        setattr(fn, _MARKER, True)
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_partial_rollback = mtp_partial_rollback


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


def _patch_switchglu_sort_threshold() -> None:
    """Lower SwitchGLU's expert-gather sort threshold from mlx-lm's hardcoded 64
    (see ``_MOE_SORT_THRESHOLD``) so the single-stream MTP verify batch coalesces
    its expert reads. The output is bit-identical to stock — this is a pure
    bandwidth win — so it is safe to install unconditionally: for the batch sizes
    that occur without MTP (M=1 decode, size 8; large prefill) the decision is
    unchanged, only the M=2..7 verify batch flips from unsorted to sorted."""
    try:
        from mlx_lm.models import switch_layers as sl
    except ImportError:
        logger.debug("mlx_lm.models.switch_layers not importable; sort-threshold patch skipped")
        return
    cls = sl.SwitchGLU
    if _ours(cls, "__call__"):
        return

    import mlx.core as mx

    def __call__(self, x, indices):
        # Verbatim mlx-lm SwitchGLU.__call__, except do_sort uses the mira-tunable
        # threshold (read live) instead of the hardcoded 64.
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= _MOE_SORT_THRESHOLD
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = sl._gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )
        if do_sort:
            x = sl._scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    setattr(__call__, _MARKER, True)
    cls.__call__ = __call__
    logger.info(
        "mira native MTP: SwitchGLU expert-gather sort threshold patch installed "
        "(default %d, was 64)", _MOE_SORT_THRESHOLD
    )
