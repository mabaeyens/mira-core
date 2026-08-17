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
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Accept-rate accumulator (spec §2: "accept-rate stats in /v1/stats").          #
# Process-global, written once per verify cycle on the engine's single decode   #
# thread and read cross-thread by GET /v1/stats. The per-call stats dict the     #
# standalone bench loop returns is scoped to one generate(); this is the         #
# lifetime counter the served path never had. One lock acquire per cycle (not    #
# per token) keeps the multi-field read self-consistent at negligible cost.      #
# --------------------------------------------------------------------------- #

_STATS_LOCK = threading.Lock()
_STATS: dict = {
    "cycles": 0,
    "drafted": 0,       # total draft tokens proposed across all cycles
    "accepted": 0,      # total drafts the backbone confirmed
    "rejections": 0,    # cycles that accepted fewer than they drafted
    "emitted": 0,       # tokens emitted (accepted drafts + the guaranteed correction)
    "depth_drafted": [],   # index j -> times a draft at position j was proposed
    "depth_accepted": [],  # index j -> times a draft at position j was accepted
    "depth_chosen": [],    # index d -> times the controller chose depth d (0 == park)
    "handbacks": 0,        # sequences that globally handed back to the stock decoder
}


def _record_handback() -> None:
    """Count one global hand-back (a sequence dropped MTP for stock decode)."""
    with _STATS_LOCK:
        _STATS["handbacks"] += 1


def _record_cycle(drafts: Sequence[int], accepted: int, depth: int) -> None:
    """Fold one verify cycle's accept counts into the process-global stats.
    ``drafts`` is what the head proposed this cycle, ``accepted`` the longest
    matching prefix the backbone confirmed."""
    k = len(drafts)
    with _STATS_LOCK:
        _STATS["cycles"] += 1
        _STATS["drafted"] += k
        _STATS["accepted"] += accepted
        _STATS["emitted"] += accepted + 1        # +1 for the always-emitted correction
        if accepted < k:
            _STATS["rejections"] += 1
        dd, da = _STATS["depth_drafted"], _STATS["depth_accepted"]
        if len(dd) < depth:
            dd.extend([0] * (depth - len(dd)))
            da.extend([0] * (depth - len(da)))
        for j in range(k):
            dd[j] += 1
            if j < accepted:
                da[j] += 1
        dc = _STATS["depth_chosen"]        # controller-chosen depth histogram
        if len(dc) <= depth:
            dc.extend([0] * (depth + 1 - len(dc)))
        dc[depth] += 1


def stats() -> dict:
    """Lifetime MTP accept-rate snapshot for GET /v1/stats. Returns zeroed fields
    (accept_rate None) if MTP has not run this process."""
    with _STATS_LOCK:
        cycles = _STATS["cycles"]
        drafted = _STATS["drafted"]
        accepted = _STATS["accepted"]
        rejections = _STATS["rejections"]
        emitted = _STATS["emitted"]
        depth_drafted = list(_STATS["depth_drafted"])
        depth_accepted = list(_STATS["depth_accepted"])
        depth_chosen = list(_STATS["depth_chosen"])
        handbacks = _STATS["handbacks"]
    return {
        "cycles": cycles,
        # Fraction of drafted tokens the backbone confirmed — the headline number.
        # omlx reports ~0.88 on the MoE, ~0.83 on the dense 27B.
        "accept_rate": round(accepted / drafted, 3) if drafted else None,
        # 1 + accepted/cycle: the decode speedup this run actually earned, since
        # each cycle emits one guaranteed token plus its accepted drafts.
        "tokens_per_cycle": round(emitted / cycles, 3) if cycles else None,
        "drafted": drafted,
        "accepted": accepted,
        "rejections": rejections,
        # Per-position accept rate: index 0 is the nearest draft (d1), the last is
        # the deepest. It falls off with depth; where it nears 0 is the useful k.
        "depth_accept_rate": [
            round(a / d, 3) if d else None
            for a, d in zip(depth_accepted, depth_drafted)
        ],
        # How often the adaptive controller chose each depth; index 0 is the park
        # (plain decode). A healthy MoE run settles mostly at 1-2; all-0 means it
        # parked (speculation never paid) or handed back to the stock decoder.
        "depth_chosen": depth_chosen,
        # Sequences that globally handed back to stock decode (park-dominated, MTP
        # a net loss). Nonzero on the compute-bound MoE, ~0 on dense.
        "handbacks": handbacks,
    }


def reset_stats() -> None:
    """Zero the accumulator. For test isolation only; the server never resets."""
    with _STATS_LOCK:
        _STATS.update(cycles=0, drafted=0, accepted=0, rejections=0, emitted=0, handbacks=0)
        _STATS["depth_drafted"] = []
        _STATS["depth_accepted"] = []
        _STATS["depth_chosen"] = []


# --------------------------------------------------------------------------- #
# Adaptive draft-depth controller (spec: native-mtp-depth-controller).          #
# Pure host-side bookkeeping — no MLX, fully unit-testable. One per width-1      #
# sequence; picks the draft depth per cycle to maximize expected tokens per      #
# wall-clock second, parks at depth 0 when speculation doesn't pay, and hands    #
# the sequence back to the stock decoder after a sustained park.                 #
# --------------------------------------------------------------------------- #

class _DepthController:
    """Chooses the draft depth for each MTP cycle.

    Ports omlx's ``_DepthController`` as pure host bookkeeping. Each cycle picks a
    depth ``d`` in ``0..max_depth`` maximizing ``score(d) = E(d) / t[d]``:

      * ``E(d) = 1 + p0 + p0·p1 + … + p0···p(d-1)`` — expected committed tokens,
        the guaranteed correction (the ``1``) plus each fully-accepted prefix.
      * ``p[j]`` — EMA of the conditional accept at chain position ``j``
        (``P(draft j accepted | draft j-1 accepted)``).
      * ``t[d]`` — EMA of measured wall-clock per cycle at chosen depth ``d``.

    Depth 0 is the **park**: no speculation, one plain decode. It wins when even a
    depth-1 draft costs more wall-clock than the token it would save — the
    compute-bound-MoE case the old fixed-depth path got wrong. A 1-in-``PROBE_EVERY``
    probe re-measures a neighbouring depth so stale EMAs recover when content shifts
    (prose↔code).

    Global hand-back (v2): the park is cheaper than a losing draft but not free (it
    still folds ``committed`` into the head cache and pays the per-cycle sync).
    Detection is THROUGHPUT-based, not park-fraction: over the last ``HANDBACK_WINDOW``
    cycles, if MTP's realized tokens/sec (Σemitted / Σcost) is below what parking
    every cycle would yield — the park rate ``1/t[0]``, uplifted by ``HANDBACK_MARGIN``
    to approximate the stock rate (stock is park minus the head-fold tax hand-back
    sheds) — :meth:`should_handback` fires and the caller drops the whole sequence to
    the stock decoder. Park-fraction was rejected: resident (production) it is
    anti-correlated with loss (a 0.79× prose run parked 60%, a 0.92× structured run
    parked 76%); only offload inflated the park share. Realized-throughput vs park
    rate is offload-independent and catches both park-dominated and drafting-but-
    losing sequences.

    Lossless by construction: depth only sets how many tokens a cycle *proposes*,
    never which token commits — the emit is always ``argmax(verify)``.
    """

    ALPHA = 0.2               # EMA weight on each new observation
    PROBE_EVERY = 16          # explore a neighbour depth once every N post-warmup cycles
    SEED_ACCEPT = 0.6         # optimistic conditional-accept seed so the warmup drafts
    HANDBACK_WINDOW = 32      # cycles of realized-throughput history for the hand-back test
    HANDBACK_MARGIN = 0.15    # park->stock uplift (the head-fold tax hand-back sheds)

    def __init__(self, max_depth: int):
        self.max_depth = max(1, int(max_depth))
        self._p = [self.SEED_ACCEPT] * self.max_depth        # conditional accept EMA
        self._t: List[Optional[float]] = [None] * (self.max_depth + 1)  # cost EMA per depth
        # Warmup sweep: measure every depth once (deepest first, park last) before
        # trusting any score, so t[d] exists for all d before the first real choose().
        self._warmup = list(range(self.max_depth, -1, -1))
        self._cycles = 0
        # rolling (emitted_tokens, cost) for the last HANDBACK_WINDOW cycles
        self._win: List[tuple] = []

    # -- selection ----------------------------------------------------------- #
    def _expected_tokens(self, d: int) -> float:
        e = 1.0            # the always-emitted correction
        run = 1.0
        for j in range(d):
            run *= self._p[j]
            e += run
        return e

    def _score(self, d: int) -> float:
        t = self._t[d]
        if t is None or t <= 0.0:
            return float("inf")     # unmeasured depth: force it to be tried
        return self._expected_tokens(d) / t

    def choose(self) -> int:
        if self._warmup:
            return self._warmup[0]
        best = max(range(self.max_depth + 1), key=self._score)
        # Bounded exploration: every PROBE_EVERY-th cycle, nudge to a neighbour so
        # its EMAs stay fresh across content shifts. Cheap stand-in for omlx's
        # duty-bounded staleness probing.
        if self.PROBE_EVERY and self._cycles % self.PROBE_EVERY == self.PROBE_EVERY - 1:
            probe = best + 1 if best < self.max_depth else best - 1
            best = min(self.max_depth, max(0, probe))
        return best

    # -- observation --------------------------------------------------------- #
    def observe(self, depth: int, cost: float, k: int, accepted: int) -> None:
        """Fold one cycle's outcome back into the EMAs. ``depth`` is what
        :meth:`choose` returned, ``cost`` the measured wall-clock, ``k`` the drafts
        actually proposed (``==depth`` for a spec cycle, 0 for a park), ``accepted``
        the longest matching prefix."""
        self._cycles += 1
        if self._warmup and self._warmup[0] == depth:
            self._warmup.pop(0)
        # cost EMA for the depth we ran
        prev = self._t[depth]
        self._t[depth] = cost if prev is None else (1 - self.ALPHA) * prev + self.ALPHA * cost
        # conditional-accept EMA: positions 0..accepted-1 were hits (each conditional
        # on the previous being accepted, which held); position `accepted` (if it was
        # actually drafted) was the miss. Deeper positions weren't tested this cycle.
        for j in range(k):
            hit = 1.0 if j < accepted else 0.0
            self._p[j] = (1 - self.ALPHA) * self._p[j] + self.ALPHA * hit
            if j >= accepted:
                break               # first miss ends the observed conditional chain
        # rolling realized throughput for global hand-back: emitted == accepted+1
        # for every cycle (a park emits its 1 correction, a spec emits accepted+1).
        self._win.append((accepted + 1, cost))
        if len(self._win) > self.HANDBACK_WINDOW:
            self._win.pop(0)

    def should_handback(self) -> bool:
        """True once MTP is realizing fewer tokens/sec than parking every cycle
        would — the honest, offload-independent signal that MTP is a net loss here.

        Over the last full ``HANDBACK_WINDOW`` cycles (past warmup), compare the
        realized rate ``Σemitted / Σcost`` against the park rate ``1/t[0]`` uplifted
        by ``HANDBACK_MARGIN`` (park costs a head-fold more than a pure stock decode,
        so stock ≈ park × (1+margin); handing back sheds that tax). If realized is
        below that, drop the sequence to the stock decoder — graceful degradation,
        not a speedup. Park-fraction was rejected: resident it is anti-correlated
        with loss (0.79× prose parked 60%, 0.92× structured parked 76%)."""
        if self._warmup or len(self._win) < self.HANDBACK_WINDOW:
            return False
        t_park = self._t[0]
        if t_park is None or t_park <= 0:
            return False
        tot_tok = sum(e for e, _ in self._win)
        tot_cost = sum(c for _, c in self._win)
        if tot_cost <= 0:
            return False
        realized_rate = tot_tok / tot_cost
        park_rate = 1.0 / t_park
        return realized_rate < park_rate * (1.0 + self.HANDBACK_MARGIN)


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


def _proc_list(logits_processors):
    """The single sequence's processor list (width-1), or [] when absent."""
    if not logits_processors:
        return []
    inner = logits_processors[0]
    return list(inner) if inner else []


def _seq_mtp_eligible(samplers, logits_processors) -> bool:
    """Whether native MTP may serve this sequence losslessly.

    Two requirements, both set by mira-mlx at admission:
      * greedy sampling — MTP's greedy accept (draft == argmax(verify)) matches
        stock decode only when stock is argmax too (``sampler.mtp_greedy``).
      * every logits processor is replayable — either pure (``mtp_pure``, re-run
        per verify position) or a recognized stateful processor exposing the
        ``mtp_observe``/``mtp_would_bind`` hooks (advanced over committed tokens,
        handed back to stock before it would change a logit). An unrecognized
        processor keeps MTP off the sequence rather than risk a silent mismatch.
    """
    s = samplers or []
    if not (s and getattr(s[0], "mtp_greedy", False)):
        return False
    for p in _proc_list(logits_processors):
        if getattr(p, "mtp_pure", False):
            continue
        if hasattr(p, "mtp_observe") and hasattr(p, "mtp_would_bind"):
            continue
        return False
    return True


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
            self._depth_ctl = None              # adaptive depth controller, built at extend()
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
                # Fresh controller per sequence: the EMAs must start clean, not carry
                # a previous sequence's content statistics. Mirrors adopting a fresh
                # head cache above.
                self._depth_ctl = (
                    _DepthController(self._mtp_depth) if self._mtp_depth > 0 else None
                )

        def filter(self, keep):
            super().filter(keep)
            if not keep:
                self._mtp_head_cache = None
                self._mtp_frontier = None
                self._mtp_committed = None
                self._mtp_buffer = []
                self._mtp_next_main = None
                self._depth_ctl = None

        # -- gating ----------------------------------------------------------- #
        def _mtp_ready(self) -> bool:
            return (
                not self._mtp_disabled
                and self._mtp_depth > 0
                and self._mtp_head_cache is not None
                and self._mtp_next_main is not None
                and len(self.uids) == 1
                and _seq_mtp_eligible(self.samplers, self.logits_processors)
            )

        def _mtp_pure_procs(self):
            return [p for p in _proc_list(self.logits_processors)
                    if getattr(p, "mtp_pure", False)]

        def _mtp_stateful_procs(self):
            return [p for p in _proc_list(self.logits_processors)
                    if not getattr(p, "mtp_pure", False)]

        def _mtp_resolve_greedy(self, vrows, drafts):
            """Greedy verify with the pure logits processors replayed, so the
            committed tokens equal stock greedy-with-penalties decode.

            ``vrows`` is the (k+1, V) backbone logits for [next_main, *drafts].
            For each position j the pure processors are applied over the true
            context (``self.tokens[0]`` + the drafts accepted so far — the exact
            history stock would have seen), then the token is argmax(processed).
            A draft is accepted iff it equals that processed argmax; the scan
            stops at the first mismatch (its argmax is the correction) or runs to
            the bonus position. Returns ``(m_accepted, committed, logprobs)``
            where ``committed == emitted`` and ``m`` counts accepted drafts."""
            pure = self._mtp_pure_procs()
            k = len(drafts)
            ctx = list(self.tokens[0])          # ends with next_main
            emitted, emit_lp = [], []
            for j in range(k + 1):
                row = vrows[j:j + 1]            # (1, V)
                if pure:
                    ctx_arr = mx.array(ctx, U) if ctx else mx.array([], U)
                    for proc in pure:
                        row = proc(ctx_arr, row)
                lp = (row - mx.logsumexp(row, axis=-1, keepdims=True))[0]
                tgt = int(mx.argmax(row[0]).item())
                emitted.append(tgt)
                emit_lp.append(lp)
                if j < k and drafts[j] == tgt:
                    ctx.append(tgt)
                    continue
                return j, emitted, emit_lp      # j drafts accepted; emitted is committed
            return k, emitted, emit_lp          # all drafts accepted + bonus token

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
                    # Dropping to stock decode. If MTP was ever primed (head cache
                    # live, anchor known), the base async pipeline was never
                    # advanced past the prompt — the constructing _step() peeked
                    # the buffer and skipped super()._step(), so _next_tokens still
                    # points at the last prompt token and _next_logprobs is []. A
                    # bare super()._step() would then emit fewer logprobs than
                    # tokens and raise IndexError in GenerationBatch.next. Re-prime
                    # stock from the anchor exactly once (handback zeroes the head
                    # cache, so this can't re-fire), then step.
                    if (self._mtp_head_cache is not None
                            and self._mtp_next_main is not None):
                        self._mtp_handback()
                    return super()._step()
                # Global hand-back (v2): if the controller has parked almost every
                # recent cycle, MTP is a net loss on this sequence. Re-prime the base
                # async pipeline from the last emitted token and drop to stock decode
                # for good. Checked only here, at buffer-empty, so _mtp_next_main is
                # provably the last emitted token and the KV cache is at accepted len.
                if self._depth_ctl is not None and self._depth_ctl.should_handback():
                    self._mtp_handback()
                    return super()._step()
                # A stateful processor (thinking budget) about to force a token
                # within this cycle's reach can't be replayed by the argmax
                # verify. Hand the sequence to stock, which applies it exactly.
                # Horizon = max draft depth + 1 (a cycle emits at most depth+1).
                if any(p.mtp_would_bind(self._mtp_depth + 1)
                       for p in self._mtp_stateful_procs()):
                    self._mtp_handback()
                    return super()._step()
                self._mtp_run_cycle()          # fills self._mtp_buffer (>= 1 token)

            tok, lp = self._mtp_buffer.pop(0)
            for sti in self.tokens:            # width-1: exactly one sequence
                sti.append(tok)
            return [tok], [lp]

        def _mtp_run_cycle(self):
            """One cycle at an adaptively chosen depth. The controller picks the
            draft depth per cycle from its accept/cost EMAs; depth 0 parks (plain
            decode, no speculation) when speculating would cost more wall-clock than
            the token it saves, and a sustained park hands the sequence back to the
            stock decoder. Both paths emit ``argmax(verify)`` — depth changes only
            how many tokens a cycle proposes, never which commits, so losslessness
            is unchanged."""
            hn = _head_norm(self.model)
            next_main = self._mtp_next_main
            committed = self._mtp_committed
            hc = self._mtp_head_cache

            depth = (self._mtp_depth if self._depth_ctl is None
                     else self._depth_ctl.choose())

            t0 = time.perf_counter()
            if depth <= 0:
                k, m = self._mtp_park_cycle(hn, next_main, committed, hc)
            else:
                k, m = self._mtp_spec_cycle(hn, depth, next_main, committed, hc)
            elapsed = time.perf_counter() - t0

            if self._depth_ctl is not None:
                self._depth_ctl.observe(depth, elapsed, k, m)
                # should_handback() is acted on in _step() at the next buffer-empty
                # boundary (where _mtp_next_main is provably the last emitted token),
                # not here mid-cycle — see _mtp_handback().

            # Advance any stateful processor (thinking budget) over the tokens this
            # cycle committed, so its counter still binds at the right absolute token
            # if a later stock step runs. would_bind() gated out the forcing case, so
            # observe() only ever sees below-budget no-ops here.
            stateful = self._mtp_stateful_procs()
            if stateful and self._mtp_buffer:
                committed_toks = [t for t, _ in self._mtp_buffer]
                full = list(self.tokens[0]) + committed_toks
                for p in stateful:
                    p.mtp_observe(full, len(committed_toks))

        def _mtp_handback(self):
            """Global hand-back: re-prime the base ``GenerationBatch`` async pipeline
            and disable MTP for the rest of the sequence.

            MTP and the stock decoder disagree on the last emitted token by exactly
            one cache slot: MTP has emitted ``_mtp_next_main`` (from the buffer) but
            left it OUT of the backbone KV cache (it is the anchor a spec cycle would
            forward next); the stock ``_step`` expects the last emitted token already
            cached, with ``_next_tokens`` holding the *next* token to forward+emit.
            Bridge that: forward ``_mtp_next_main`` once (caching it, matching the
            stock invariant) and seed ``_next_tokens`` / ``_next_logprobs`` with that
            forward's prediction — emitting nothing. The first ``super()._step()``
            then forwards+emits ``next_main + 1``: no token duplicated, none skipped.
            ``self.tokens`` already holds ``next_main`` (MTP appended it on emit) and
            this manual forward must NOT append again — it is a re-prime, not an emit.
            """
            inp = mx.array([self._mtp_next_main], U)[None]
            logits = self.model(inp, cache=self.prompt_cache)   # plain decode; caches next_main
            logits = logits[0, -1]
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            self._next_tokens = mx.argmax(logits, keepdims=True)  # the NEXT token to emit
            self._next_logprobs = [lp]
            mx.async_eval(self._next_tokens, self._next_logprobs)
            self._mtp_head_cache = None          # MTP is done on this sequence; free the head KV
            self._mtp_disabled = True
            _record_handback()

        def _mtp_spec_cycle(self, hn, depth, next_main, committed, hc):
            """§5.3 speculative cycle: draft ``depth``, verify in one backbone
            forward, accept the longest greedy-matching prefix, roll the hybrid
            cache back. Emits ``drafts[:m] + [correction]``; the anchor ``next_main``
            was already emitted. Faithful port of ``mtp_greedy_generate``'s loop
            body. Returns ``(k, m)``."""
            model = self.model
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
            # Greedy verify with the pure logits processors replayed per position
            # (repetition penalty &c.), so the accepted prefix and the correction
            # match stock greedy-with-penalties decode. emitted == drafts[:m] +
            # [correction]; with no processors this reduces to the old argmax path.
            m, emitted, emit_lp = self._mtp_resolve_greedy(vlogits[0], drafts)
            _record_cycle(drafts, m, depth)

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

            # --- carry state for the next cycle (len(emitted) == m + 1) --------
            self._mtp_buffer = list(zip(emitted, emit_lp))
            self._mtp_committed = emitted
            self._mtp_frontier = hn(vhidden[:, : m + 1, :])
            self._mtp_next_main = emitted[-1]
            return k, m

        def _mtp_park_cycle(self, hn, next_main, committed, hc):
            """Depth-0 park: no speculation. This is the ``m=0`` special case of a
            spec cycle — with zero drafts, the correction is by definition the true
            token following ``next_main``, so the carried state stays on the exact
            invariant a spec cycle maintains and re-entry to speculation is seamless.

            Still folds the pending ``committed`` tokens into the head cache (one
            head forward, no draft chain, nothing to trim) so it advances in lockstep
            with the backbone — this is the park's only cost above a plain decode.
            Then a single 1-token backbone forward over the anchor. Returns ``(0, 0)``."""
            model = self.model
            # keep the head cache coherent for re-entry: fold committed, no chain.
            # This fold is a separate graph from the backbone decode below, so it
            # evaluates lazily (forced when the next spec cycle reads hc, or dropped
            # if we hand back). Its cost therefore lands in the adjacent cycle's
            # timing; because parks cluster in the same depth-0 bucket the t[0] EMA
            # still converges in steady state, and hand-back caps any lazy backlog.
            model.mtp_forward(
                self._mtp_frontier, mx.array(committed, U)[None], hc, logits_keep=1,
            )
            # plain 1-token decode over the anchor (no drafts -> no rollback).
            inp = mx.array([next_main], U)[None]
            vlogits, vhidden = model(
                inp, cache=self.prompt_cache, return_hidden=True, n_confirmed=1,
            )
            # Greedy verify with the pure processors replayed (drafts empty -> the
            # single position after the anchor); matches stock greedy-with-penalties.
            _m, emitted, emit_lp = self._mtp_resolve_greedy(vlogits[0], [])
            _record_cycle([], 0, 0)

            self._mtp_buffer = [(emitted[0], emit_lp[0])]
            self._mtp_committed = [emitted[0]]
            self._mtp_frontier = hn(vhidden[:, :1, :])
            self._mtp_next_main = emitted[0]
            return 0, 0

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
                # Only prime when MTP can actually serve this sequence: greedy
                # sampling, and every logits processor replayable (pure penalties,
                # or a recognized stateful budget). Otherwise don't build a head
                # cache the generate() handoff would only throw away.
                and _seq_mtp_eligible(self.samplers, self.logits_processors)
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
                # See _mtp_single: only serve sequences MTP can decode losslessly
                # (greedy + replayable processors). Otherwise hand off to the stock
                # generation batch, which primes its own async pipeline.
                and _seq_mtp_eligible(self.samplers, self.logits_processors)
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
