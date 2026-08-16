"""MTP decode loop (spec §5.3) — DRAFT, not yet live-verified.

Width-1 self-speculative decode: the MTP head drafts k tokens, one backbone
forward verifies them, the longest matching prefix is accepted, and the caches
are rolled back to the accepted length. A subclass of mlx-lm's ``CompletionBatch``
so it drops into mira-mlx's existing ``BatchGenerator`` seam.

STATUS. The pure acceptance logic (``accept_prefix``) is complete and unit-tested.
The orchestration (``MtpCompletionBatch``) is a grounded scaffold: every mlx-lm API
it calls is confirmed to exist, but it CANNOT run until the model-side rollback
primitive it depends on is built, and it can only be correctness-verified against
the real 20 GB backbone under bench discipline. Those spots are marked
``# VERIFY-LIVE`` / raise ``NotImplementedError``. Do not wire this into the engine
until the live draft/verify smoke test passes.

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
re-projection. That primitive lives model-side (extending ``qwen3_mtp.py``); it is
the blocking prerequisite for this loop and is not built yet.
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
# Orchestration — grounded scaffold, NOT yet runnable (see module docstring).   #
# --------------------------------------------------------------------------- #

def _load_completion_batch_base():
    """Import mlx-lm's CompletionBatch lazily so this module imports without mlx
    present (the pure logic above stays usable / testable)."""
    from mlx_lm.generate import CompletionBatch

    return CompletionBatch


def make_mtp_completion_batch_class():
    """Build the ``MtpCompletionBatch`` subclass. Deferred behind a factory so the
    ``CompletionBatch`` base is imported only when MTP is actually used."""
    CompletionBatch = _load_completion_batch_base()

    class MtpCompletionBatch(CompletionBatch):
        """Width-1 MTP self-speculative decode. Falls back to the stock single-token
        ``_step`` whenever MTP can't apply (width != 1, no head, a logits processor
        that would desync speculative positions, etc.), so it is never less correct
        than the base — only faster when it engages."""

        def __init__(self, *args, mtp_max_draft: int = 3, **kwargs):
            self._mtp_max_draft = max(1, int(mtp_max_draft))
            self._mtp_cache = None          # per-head KVCache list, built on first use
            self._mtp_last_hidden = None    # backbone pre-norm hidden at the frontier
            super().__init__(*args, **kwargs)

        # -- gating ------------------------------------------------------------ #
        def _mtp_eligible(self) -> bool:
            """Only width-1, head present, no per-token logits processors (they
            would have to be replayed per speculative position — deferred), and
            greedy or a sampler we can apply position-by-position."""
            if len(self.uids) != 1:
                return False
            if not hasattr(self.model, "mtp_forward"):
                return False
            if any(self.logits_processors):
                return False  # TODO(§5.3): replay processors across drafted positions
            return True

        # -- the cycle --------------------------------------------------------- #
        def _step(self):
            if not self._mtp_eligible():
                return super()._step()          # exact stock behavior
            return self._mtp_step()

        def _mtp_step(self):
            # This is the draft/verify/accept/rollback cycle. Every call below is a
            # confirmed API; the body is the DRAFT to be exercised live.
            raise NotImplementedError(
                "MTP _mtp_step is a reviewed scaffold; the draft/verify/rollback body "
                "and its model-side mtp_partial_rollback prerequisite are built and "
                "verified in the live backbone session (spec §5.3). Outline:\n"
                "  1. draft k via model.mtp_forward(self._mtp_last_hidden, main_next,\n"
                "     self._mtp_cache), chaining the head's own hidden per depth.\n"
                "  2. verify: logits, hidden = model(mx.array([[main_next, *drafts]]),\n"
                "     cache=self.prompt_cache, return_hidden=True)   # ONE forward,\n"
                "     with n_confirmed=1 so GatedDeltaNet stashes rollback state.\n"
                "  3. verify_tokens = per-position sample(logits); m = accept_prefix(\n"
                "     drafts, verify_tokens); toks = emitted_tokens(main_next, drafts, m).\n"
                "  4. rollback: model.mtp_partial_rollback(self.prompt_cache, m + 1)\n"
                "     (trims the 10 KVCache layers by k-m, replays the 30 linear\n"
                "     layers' recurrence over the accepted slice).\n"
                "  5. self._mtp_last_hidden = hidden[:, m:m+1, :]; next main = "
                "verify_tokens[m].\n"
                "  6. return the LIST toks (contract change from the base's single "
                "token)."
            )

        # -- response fan-out -------------------------------------------------- #
        def next(self):
            # The base emits exactly one Response; MTP emits up to k+1. Until
            # _mtp_step lands, defer wholesale to the base so importing/constructing
            # this class is harmless.
            return super().next()

    return MtpCompletionBatch
