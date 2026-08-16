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
