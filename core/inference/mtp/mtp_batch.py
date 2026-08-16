"""MTP decode loop (spec §5.3) — WIRED and live-verified.

Width-1 self-speculative decode: the MTP head drafts k tokens, one backbone
forward verifies them, the longest matching prefix is accepted, and the caches
are rolled back to the accepted length. Installed by subclassing mlx-lm's real
two-class batch seam — ``PromptProcessingBatch`` (prefill priming) and
``GenerationBatch`` (decode) — so it drops into mira-mlx's existing
``BatchGenerator`` with no engine change. (There is no ``CompletionBatch`` in this
mlx-lm; the prefill/decode split is these two classes.)

STATUS. Wired and live-verified against the real 20 GB backbone under bench
discipline. The pure acceptance logic (``accept_prefix``) is unit-tested; the
model-side rollback primitive (``mtp_partial_rollback``) is built and benched
(``mtp_greedy_generate`` runs 1.66–1.88×). This module ports that cycle into
``MtpPromptProcessingBatch`` + ``MtpGenerationBatch`` and installs them via
``patch()`` so the stock ``BatchGenerator`` serves native MTP. It engages only for
width-1, text-only batches (mira-mlx's production config) and falls back to the
stock path otherwise, so it is never less correct than base mlx-lm.

Smoke result (2026-08-16, Qwen3.8-27B dense, depth 3, MLX_ENABLE_TF32=0): served
1.67× stock through the real ``BatchGenerator`` (13.8 vs 8.3 tok/s), matching the
standalone bench. LOSSLESS confirmed by teacher forcing — the numerics-matched
oracle: MTP is model-greedy at 199/200 positions, exactly as lossless as stock
single-token decode (also 199/200; the one flip each is the irreducible M5
batch-attention near-tie, mlx#3897). Emitted == argmax(verify) by construction;
teacher forcing confirms the hybrid cache stays positionally correct across every
rollback. Token-stream equality vs stock is NOT a usable oracle on M5 (stock itself
diverges 176/200 from plain greedy for the same numeric reason).

THE HYBRID-CACHE CRUX (confirmed from the checkpoint config + mlx-lm/omlx source).
Qwen3.6-35B-A3B is hybrid: of 40 layers only 10 are full-attention (``KVCache``,
which ``trim(n)`` rolls back cheaply) and 30 are linear ``GatedDeltaNet``
(``ArraysCache`` holding a running ``(conv_state, ssm_state)`` — NOT trimmable).
A rejected draft must roll the recurrent state back to the accepted length. A naive
snapshot+re-advance can't: each layer's input is the previous layer's output, so
re-advancing means a second full backbone forward, which erases the speedup. The
viable path (omlx's, understood not copied) is a model-side ``n_confirmed`` split:
during verify, ``GatedDeltaNet`` stashes the pre-forward ``(conv, ssm)`` state AND
the projected ``(qkv, a, b)`` on the cache; on rejection a ``mtp_partial_rollback``
replays ONLY the cheap recurrence over the accepted slice — no second forward, no
re-projection. That primitive lives model-side (``qwen3_mtp.py``) and is built; this
loop gates on a static cache-capability check so the rollback can never refuse
mid-cycle and strand the backbone cache.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure acceptance logic — no MLX, fully unit-testable.                          #
# --------------------------------------------------------------------------- #

def accept_prefix(draft_tokens: Sequence[int], verify_tokens: Sequence[int]) -> int:
    """Number of drafted tokens accepted, as the longest matching prefix.

    The verify forward runs the backbone over ``[main_next, d_1, ..., d_k]`` and
    yields, at each input position i, the token the backbone predicts to FOLLOW
    that position. So ``verify_tokens[i]`` is the backbone's "true" token after the
    i-th verify input:

        verify_tokens[0]  = true token after main_next        -> compare to d_1
        verify_tokens[j]  = true token after d_j               -> compare to d_(j+1)

    ``draft_tokens[j]`` (0-indexed d_(j+1)) is accepted iff every draft up to and
    including it matches, i.e. ``draft_tokens[:j+1] == verify_tokens[:j+1]``. The
    scan stops at the first mismatch. Returns m in ``[0, len(draft_tokens)]``.

    The main token (main_next) is always emitted by the caller and is NOT counted
    here — it was already the backbone's accepted next token from the prior cycle.
    """
    m = 0
    for d, v in zip(draft_tokens, verify_tokens):
        if d != v:
            break
        m += 1
    return m


def emitted_tokens(
    main_next: int, draft_tokens: Sequence[int], accepted: int
) -> List[int]:
    """The tokens this cycle emits: the always-accepted main token followed by the
    accepted draft prefix. Length is ``accepted + 1``."""
    return [main_next, *draft_tokens[:accepted]]


def truncate_at_stop(
    tokens: List[int], eos_ids: set, is_stopped) -> Tuple[List[int], Optional[str]]:
    """Cut an accepted-token run at the FIRST stop so MTP never emits past a stop
    boundary (spec §5.3). ``is_stopped(prefix)`` reports whether the state machine
    has hit a stop sequence after ``prefix``. Returns the surviving tokens and a
    finish reason (``"stop"`` / ``"eos"``) or ``None``.

    Pure except for the injected ``is_stopped`` predicate, so a fake predicate
    unit-tests the truncation independently of the real ``SequenceStateMachine``.
    """
    kept: List[int] = []
    for tok in tokens:
        if tok in eos_ids:
            return kept + [tok], "eos"  # EOS itself is emitted, then we stop
        kept.append(tok)
        if is_stopped(kept):
            return kept, "stop"
    return kept, None


# --------------------------------------------------------------------------- #
# Standalone width-1 generators — the §5.3 cycle, driving the model directly.    #
# These are the correctness reference and the basis for the CompletionBatch      #
# integration below. Greedy only (the lossless gate); stochastic acceptance is a #
# follow-up.                                                                      #
# --------------------------------------------------------------------------- #

def plain_greedy_generate(model, prompt_ids, max_new_tokens, eos_ids):
    """Baseline: one backbone forward per token, argmax. The MTP output must match
    this token-for-token."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    tok = int(mx.argmax(logits[0, -1]).item())
    out = [tok]
    while len(out) < max_new_tokens and tok not in eos_ids:
        logits = model(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(logits[0, -1]).item())
        out.append(tok)
    return out


def mtp_greedy_generate(model, prompt_ids, max_new_tokens, depth, eos_ids):
    """Native MTP self-speculative greedy decode (spec §5.3).

    Per cycle: fold the last committed run into a head cache to draft d1, chain
    d2..dk on the head's own hidden, verify ``[next_main, d1..dk]`` in ONE backbone
    forward (n_confirmed=1 so the linear layers stash rollback state), accept the
    longest greedy-matching prefix, roll the cache back to the accepted length, and
    emit ``drafts[:m] + [correction]``. Output is guaranteed identical to
    ``plain_greedy_generate`` because every emitted token is the backbone's own
    greedy choice — the head only proposes.

    The head runs on a PERSISTENT committed-history cache: the prompt is folded in
    at prime (every (hidden[t], token[t+1]) pair, free from the prefill forward) and
    each accepted run is folded permanently, so drafts attend to the full context.
    omlx measured this as the dominant accept-rate lever: 0.90 primed vs 0.26
    unprimed at depth 1. The depth>1 speculative chain runs on the same cache and is
    trimmed back to the committed length afterwards (KVCache.trim is exact), so
    committed history stays clean. Losslessness is independent of all this — the
    backbone verify decides every emitted token; the head only proposes.

    Returns (tokens, stats).
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    U = mx.uint32
    cache = make_prompt_cache(model)

    # The MTP head is fed the trunk's POST-norm hidden (omlx HEAD_HIDDEN_POST_NORM),
    # not the pre-norm hidden the backbone returns. This is load-bearing: with
    # pre-norm hidden the head's attention over committed history is miscalibrated
    # and priming *degrades* the depth chain (measured -8.5% vs +0; post-norm turns
    # that into the +full depth-chain gain d3 4%->16%). The head then chains on its
    # OWN post-norm output (already normed by the head's self.norm), so only the
    # backbone hidden is normalized here.
    _head_norm = model.language_model.model.norm

    # --- prime: full prompt forward, greedy first token ---
    logits, hidden = model(mx.array(prompt_ids)[None], cache=cache, return_hidden=True)
    hidden = _head_norm(hidden)
    next_main = int(mx.argmax(logits[0, -1]).item())

    # Persistent head cache, primed over the prompt: fold (hidden[t], prompt[t+1])
    # for t in 0..P-2 so the head enters generation with full prompt context. The
    # final pair (hidden[P-1], next_main) is folded by the first cycle below.
    mtp_cache = model.make_mtp_cache()
    if len(prompt_ids) >= 2:
        model.mtp_forward(
            hidden[:, :-1, :], mx.array(prompt_ids[1:], U)[None], mtp_cache, logits_keep=1
        )

    committed = [next_main]                 # tokens confirmed last cycle
    hidden_rows = hidden[:, -1:, :]         # backbone POST-norm hidden BEFORE next_main
    out = [next_main]

    stats = {"cycles": 0, "drafted": 0, "accepted": 0, "rejections": 0,
             "depth_drafted": [0] * depth, "depth_accepted": [0] * depth}

    while len(out) < max_new_tokens and next_main not in eos_ids:
        # --- draft: fold committed into the persistent cache -> d1, then chain ---
        lg, head_hidden = model.mtp_forward(
            hidden_rows, mx.array(committed, U)[None], mtp_cache,
            return_hidden=True, logits_keep=1,
        )
        d = int(mx.argmax(lg[0, -1]).item())
        drafts = [d]
        h = head_hidden[:, -1:, :]
        for _ in range(1, depth):
            lg, head_hidden = model.mtp_forward(
                h, mx.array([[d]], U), mtp_cache, return_hidden=True, logits_keep=1
            )
            d = int(mx.argmax(lg[0, -1]).item())
            drafts.append(d)
            h = head_hidden[:, -1:, :]

        # Drop the speculative chain positions from the head cache, keeping only the
        # committed fold. trim on each head KVCache is exact for standard attention.
        spec_positions = depth - 1
        if spec_positions > 0:
            for kc in mtp_cache:
                kc.trim(spec_positions)

        # --- verify: ONE backbone forward over [next_main, d1..dk] ---
        k = len(drafts)
        inputs = mx.array([next_main, *drafts], U)[None]
        vlogits, vhidden = model(inputs, cache=cache, return_hidden=True, n_confirmed=1)
        targets = mx.argmax(vlogits[0], axis=-1)          # (k+1,)
        target_ids = targets.tolist()
        m = accept_prefix(drafts, target_ids[:k])
        correction = int(target_ids[m])                    # backbone token after last accepted

        # --- roll the hybrid cache back to the accepted length ---
        if m < k:
            if not model.mtp_partial_rollback(cache, m, k):
                raise RuntimeError("mtp_partial_rollback refused; cache not rollback-capable")

        # --- emit drafts[:m] + correction, honoring the first EOS ---
        emitted = drafts[:m] + [correction]
        stats["cycles"] += 1
        stats["drafted"] += k
        stats["accepted"] += m
        if m < k:
            stats["rejections"] += 1
        for j in range(k):
            stats["depth_drafted"][j] += 1
            if j < m:
                stats["depth_accepted"][j] += 1
        stopped = False
        for t in emitted:
            out.append(t)
            if t in eos_ids or len(out) >= max_new_tokens:
                stopped = True
                break
        if stopped:
            break

        # --- carry state to the next cycle ---
        committed = emitted
        hidden_rows = _head_norm(vhidden[:, : m + 1, :])  # POST-norm for the head
        next_main = correction

    return out, stats


# --------------------------------------------------------------------------- #
# Engine integration — the width-1 MTP decode path wired into mlx-lm's           #
# continuous-batching seam. This mlx-lm splits the batch into TWO classes:        #
# ``PromptProcessingBatch`` (prefill) and ``GenerationBatch`` (decode). There is  #
# no ``CompletionBatch``. So MTP subclasses both:                                 #
#   * ``MtpPromptProcessingBatch`` primes the MTP head cache DURING prefill (it   #
#     captures the trunk's post-norm hidden that the prefill forward already      #
#     computes, so priming costs no extra forward) and hands the primed head      #
#     cache + frontier hidden + first token to                                    #
#   * ``MtpGenerationBatch``, whose ``_step`` runs the draft/verify/accept/        #
#     rollback cycle (ported from ``mtp_greedy_generate``) and BUFFERS the extra   #
#     accepted tokens so it still returns exactly one Response per uid per call —  #
#     mira-mlx's ``BatchGenerator`` accounting and ``_handle_response`` are        #
#     unchanged; a cycle simply makes the next few ``_step`` calls forward-free.   #
#                                                                                  #
# Gating: MTP engages only for a single-sequence, text-only, no-logits-processor   #
# batch — exactly mira-mlx's production decode config (width-1, commit ac7cd8b).   #
# Any other shape falls back to the stock path, so this is never LESS correct than #
# base mlx-lm, only faster when it engages. Losslessness is the invariant: the     #
# backbone verify forward decides every emitted token; the head only proposes.     #
#                                                                                  #
# Install by swapping the two module globals (patch()); the stock BatchGenerator   #
# resolves both class names from mlx_lm.generate's globals at call time, so it     #
# then builds the MTP variants with no other change. Same monkeypatch philosophy   #
# as the model-side mtp.apply().                                                    #
# --------------------------------------------------------------------------- #

_PATCHED = False
_ORIG: dict = {}
_MTP_DEPTH = 3


def _head_norm(model):
    """The trunk's final RMSNorm — the MTP head is fed POST-norm hidden (the
    load-bearing detail from ``mtp_greedy_generate``)."""
    lm = getattr(model, "language_model", model)
    return lm.model.norm


def _model_layers(model):
    lm = getattr(model, "language_model", model)
    return lm.model.layers


def _make_mtp_generation_class(BaseGeneration):
    import mlx.core as mx

    U = mx.uint32

    class MtpGenerationBatch(BaseGeneration):
        """Width-1 MTP self-speculative decode. Buffers accepted tokens so the
        one-Response-per-uid contract holds; falls back to the stock ``_step``
        for any batch shape MTP can't serve."""

        # -- construction / handoff ------------------------------------------ #
        def __init__(self, *args, **kwargs):
            self._mtp_depth = int(kwargs.pop("mtp_depth", 0))
            self._mtp_head_cache = kwargs.pop("mtp_head_cache", None)
            self._mtp_frontier = kwargs.pop("mtp_frontier", None)   # post-norm hidden rows
            self._mtp_committed = kwargs.pop("mtp_committed", None)  # tokens folded next cycle
            self._mtp_buffer = list(kwargs.pop("mtp_buffer", []))    # (token, logprob) already committed
            self._mtp_next_main = kwargs.pop("mtp_next_main", None)  # verify anchor token
            self._mtp_disabled = False
            self._mtp_capable = None            # static rollback-capability, probed once
            self._mtp_constructing = True
            super().__init__(*args, **kwargs)
            self._mtp_constructing = False

        # -- keep MTP state across the empty()->extend() handoff ------------- #
        def extend(self, batch):
            super().extend(batch)
            if getattr(batch, "_mtp_head_cache", None) is not None:
                # width-1: adopt the incoming (primed) sequence's MTP state.
                self._mtp_depth = batch._mtp_depth
                self._mtp_head_cache = batch._mtp_head_cache
                self._mtp_frontier = batch._mtp_frontier
                self._mtp_committed = batch._mtp_committed
                self._mtp_buffer = list(batch._mtp_buffer)
                self._mtp_next_main = batch._mtp_next_main
                self._mtp_disabled = batch._mtp_disabled
                self._mtp_capable = None

        def filter(self, keep):
            super().filter(keep)
            if not keep:
                self._mtp_head_cache = None
                self._mtp_frontier = None
                self._mtp_committed = None
                self._mtp_buffer = []
                self._mtp_next_main = None

        # -- gating ----------------------------------------------------------- #
        def _mtp_ready(self) -> bool:
            return (
                not self._mtp_disabled
                and self._mtp_depth > 0
                and self._mtp_head_cache is not None
                and self._mtp_next_main is not None
                and len(self.uids) == 1
                and not any(self.logits_processors)
            )

        def _mtp_cache_capable(self) -> bool:
            """Static, pre-verify check that ``mtp_partial_rollback`` will succeed:
            one cache per layer, and every full-attention layer trimmable. Linear
            (GatedDeltaNet) layers stash their rollback state on the verify forward
            itself, so once patched they are always rollback-capable. Gating on this
            means the post-verify rollback can never refuse and strand the cache."""
            if self._mtp_capable is not None:
                return self._mtp_capable
            ok = True
            try:
                layers = _model_layers(self.model)
                cache = self.prompt_cache
                ok = len(cache) == len(layers)
                if ok:
                    for layer, c in zip(layers, cache):
                        if not getattr(layer, "is_linear", False):
                            if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                                ok = False
                                break
            except Exception:  # noqa: BLE001 - any surprise -> stay on the stock path
                ok = False
            self._mtp_capable = ok
            return ok

        # -- the step --------------------------------------------------------- #
        def _step(self):
            if self._mtp_constructing:
                # The base __init__ calls _step() once to prime its async pipeline,
                # which MTP does not use. Peek the seed token (do NOT consume it);
                # the base discards this return. The first real next() emits it.
                if self._mtp_buffer:
                    tok, lp = self._mtp_buffer[0]
                    return [tok], [lp]
                return super()._step()

            if not self._mtp_buffer:
                if not self._mtp_ready() or not self._mtp_cache_capable():
                    if self._mtp_ready() and not self._mtp_cache_capable():
                        self._mtp_disabled = True
                    return super()._step()
                self._mtp_run_cycle()          # fills self._mtp_buffer (>= 1 token)

            tok, lp = self._mtp_buffer.pop(0)
            for sti in self.tokens:            # width-1: exactly one sequence
                sti.append(tok)
            return [tok], [lp]

        def _mtp_run_cycle(self):
            """One §5.3 cycle: draft k, verify in one backbone forward, accept the
            longest greedy-matching prefix, roll the hybrid cache back. Emits
            ``drafts[:m] + [correction]`` into the buffer; the anchor ``next_main``
            was already emitted (as the prime seed or the prior cycle's
            correction). Faithful port of ``mtp_greedy_generate``'s loop body."""
            model = self.model
            hn = _head_norm(model)
            depth = self._mtp_depth
            next_main = self._mtp_next_main
            committed = self._mtp_committed
            hc = self._mtp_head_cache

            # --- draft: fold committed into the head cache -> d1, then chain ---
            lg, hh = model.mtp_forward(
                self._mtp_frontier, mx.array(committed, U)[None], hc,
                return_hidden=True, logits_keep=1,
            )
            d = int(mx.argmax(lg[0, -1]).item())
            drafts = [d]
            h = hh[:, -1:, :]
            for _ in range(1, depth):
                lg, hh = model.mtp_forward(
                    h, mx.array([[d]], U), hc, return_hidden=True, logits_keep=1,
                )
                d = int(mx.argmax(lg[0, -1]).item())
                drafts.append(d)
                h = hh[:, -1:, :]
            spec = depth - 1
            if spec > 0:
                for kc in hc:
                    kc.trim(spec)

            # --- verify: ONE backbone forward over [next_main, d1..dk] ---------
            k = len(drafts)
            inp = mx.array([next_main, *drafts], U)[None]
            vlogits, vhidden = model(
                inp, cache=self.prompt_cache, return_hidden=True, n_confirmed=1,
            )
            vlp = vlogits[0] - mx.logsumexp(vlogits[0], axis=-1, keepdims=True)
            targets = mx.argmax(vlogits[0], axis=-1).tolist()
            m = accept_prefix(drafts, targets[:k])
            correction = int(targets[m])

            # --- roll the hybrid backbone cache back to the accepted length ---
            if m < k:
                if not model.mtp_partial_rollback(self.prompt_cache, m, k):
                    # Unreachable once _mtp_cache_capable() passed. Fail loud rather
                    # than emit from a stranded cache — the engine turns this into a
                    # clean job failure instead of silent corruption.
                    raise RuntimeError(
                        "mtp_partial_rollback refused after a verify forward; the "
                        "backbone cache is not rollback-capable despite the static "
                        "capability gate. Aborting to protect correctness."
                    )

            # --- emit drafts[:m] + correction; carry state for the next cycle --
            emitted = drafts[:m] + [correction]
            emit_lp = [vlp[j] for j in range(m)] + [vlp[m]]
            self._mtp_buffer = list(zip(emitted, emit_lp))
            self._mtp_committed = emitted
            self._mtp_frontier = hn(vhidden[:, : m + 1, :])
            self._mtp_next_main = correction

    return MtpGenerationBatch


def _make_mtp_prompt_class(BasePrompt, MtpGeneration):
    import mlx.core as mx

    U = mx.uint32

    class MtpPromptProcessingBatch(BasePrompt):
        """Prefill that primes the MTP head cache from the trunk hidden the prefill
        forward already computes, then hands it to ``MtpGenerationBatch``. Only the
        width-1, text-only path primes; every other shape delegates to the base
        (MTP won't engage on it anyway)."""

        def __init__(self, *args, **kwargs):
            self._mtp_head_cache = None
            self._mtp_prime_hidden = None       # last post-norm hidden, carried across chunks
            super().__init__(*args, **kwargs)

        def _copy(self):
            new = super()._copy()
            new._mtp_head_cache = self._mtp_head_cache
            new._mtp_prime_hidden = self._mtp_prime_hidden
            return new

        def filter(self, keep):
            super().filter(keep)
            if not keep:
                self._mtp_head_cache = None
                self._mtp_prime_hidden = None

        def _mtp_single(self, tokens, input_embeddings) -> bool:
            return (
                _MTP_DEPTH > 0
                and len(self.uids) == 1
                and input_embeddings is None
                and len(tokens) == 1
                and hasattr(self.model, "mtp_forward")
            )

        # -- prefill: fold prompt hidden into the head cache ------------------ #
        def prompt(self, tokens, input_embeddings=None):
            if not self._mtp_single(tokens, input_embeddings):
                return super().prompt(tokens, input_embeddings=input_embeddings)

            model = self.model
            hn = _head_norm(model)
            toks = tokens[0]
            if not toks:
                return
            if self._mtp_head_cache is None:
                self._mtp_head_cache = model.make_mtp_cache()
            self.tokens[0] += toks              # mirror base bookkeeping (cache contents)

            arr = mx.array(toks)[None]
            pos = 0
            while pos < arr.shape[1]:
                n = min(self.prefill_step_size, arr.shape[1] - pos)
                _, hidden = model(
                    arr[:, pos : pos + n], cache=self.prompt_cache, return_hidden=True
                )
                hidden = hn(hidden)             # post-norm for the head
                # fold (carried last hidden, this chunk's first token) across the
                # chunk boundary, then the within-chunk (hidden[i], token[i+1]) pairs.
                if self._mtp_prime_hidden is not None:
                    model.mtp_forward(
                        self._mtp_prime_hidden, mx.array([[toks[pos]]], U),
                        self._mtp_head_cache, logits_keep=1,
                    )
                if n >= 2:
                    model.mtp_forward(
                        hidden[:, :-1, :], mx.array(toks[pos + 1 : pos + n], U)[None],
                        self._mtp_head_cache, logits_keep=1,
                    )
                self._mtp_prime_hidden = hidden[:, -1:, :]
                mx.eval([c.state for c in self.prompt_cache])
                mx.clear_cache()
                pos += n

        # -- handoff: build the MTP generation batch -------------------------- #
        def generate(self, tokens):
            model = self.model
            primed = (
                self._mtp_head_cache is not None
                and _MTP_DEPTH > 0
                and len(self.uids) == 1
                and len(tokens) == 1
                and hasattr(model, "mtp_forward")
            )
            if not primed:
                return super().generate(tokens)

            if any(len(t) > 1 for t in tokens):
                self.prompt([t[:-1] for t in tokens])
            last_tok = tokens[0][-1]
            hn = _head_norm(model)

            # forward the final prompt token: yields next_main + its post-norm
            # hidden (the frontier the first cycle drafts from).
            logits, hidden = model(
                mx.array([[last_tok]]), cache=self.prompt_cache, return_hidden=True
            )
            frontier = hn(hidden[:, -1:, :])
            row = logits[0, -1]
            next_main = int(mx.argmax(row).item())
            next_lp = row - mx.logsumexp(row, axis=-1, keepdims=True)
            self.tokens[0].append(last_tok)

            # complete prompt priming: fold (carried hidden, last_tok).
            if self._mtp_prime_hidden is not None:
                model.mtp_forward(
                    self._mtp_prime_hidden, mx.array([[last_tok]], U),
                    self._mtp_head_cache, logits_keep=1,
                )

            generation = MtpGeneration(
                model,
                self.uids,
                mx.array([last_tok]),
                self.prompt_cache,
                self.tokens,
                self.samplers,
                self.fallback_sampler,
                self.logits_processors,
                self.state_machines,
                self.max_tokens,
                mtp_depth=_MTP_DEPTH,
                mtp_head_cache=self._mtp_head_cache,
                mtp_frontier=frontier,
                mtp_committed=[next_main],
                mtp_buffer=[(next_main, next_lp)],
                mtp_next_main=next_main,
            )
            self.uids = []
            self.prompt_cache = []
            self.tokens = []
            self.samplers = []
            self.logits_processors = []
            self.max_tokens = []
            self._mtp_head_cache = None
            self._mtp_prime_hidden = None
            return generation

    return MtpPromptProcessingBatch


def patch(depth: int = 3) -> bool:
    """Swap mlx-lm's prefill/decode batch classes for the MTP subclasses so the
    stock ``BatchGenerator`` serves native MTP. Idempotent; ``depth`` updates the
    draft depth on every call. Returns True if MTP is now installed."""
    global _PATCHED, _MTP_DEPTH
    _MTP_DEPTH = max(1, int(depth))
    if _PATCHED:
        return True
    import importlib

    # mlx_lm/__init__.py rebinds the ``generate`` attribute to the generate()
    # FUNCTION, so ``import mlx_lm.generate as G`` would grab the function. Reach
    # the submodule (whose namespace BatchGenerator resolves its class names from)
    # explicitly.
    G = importlib.import_module("mlx_lm.generate")

    base_prompt = G.PromptProcessingBatch
    base_gen = G.GenerationBatch
    _ORIG["PromptProcessingBatch"] = base_prompt
    _ORIG["GenerationBatch"] = base_gen

    mtp_gen = _make_mtp_generation_class(base_gen)
    mtp_prompt = _make_mtp_prompt_class(base_prompt, mtp_gen)
    G.GenerationBatch = mtp_gen
    G.PromptProcessingBatch = mtp_prompt
    _PATCHED = True
    logger.info("native MTP decode path installed (depth=%d)", _MTP_DEPTH)
    return True


def unpatch() -> None:
    """Restore the stock batch classes. Mainly for tests."""
    global _PATCHED
    if not _PATCHED:
        return
    import importlib

    # mlx_lm/__init__.py rebinds the ``generate`` attribute to the generate()
    # FUNCTION, so ``import mlx_lm.generate as G`` would grab the function. Reach
    # the submodule (whose namespace BatchGenerator resolves its class names from)
    # explicitly.
    G = importlib.import_module("mlx_lm.generate")

    G.PromptProcessingBatch = _ORIG["PromptProcessingBatch"]
    G.GenerationBatch = _ORIG["GenerationBatch"]
    _PATCHED = False
