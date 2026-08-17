"""Regression: MTP must drop to stock decode without crashing.

mira-mlx always attaches at least one logits processor per sequence
(`_build_logits_processors` never returns an empty list — a repetition
penalty, a thinking budget, or a passthrough backstop). MtpGenerationBatch
can't losslessly serve such a sequence (its verify commits a plain argmax),
so `_mtp_ready()` is False and every decode step must fall back to the stock
`super()._step()`.

The bug this guards: the generate() handoff primed the MtpGeneration with a
non-empty buffer, so the base __init__'s priming `_step()` hit the
`_mtp_constructing` short-circuit and never ran `super()._step()`. The stock
async pipeline stayed unprimed (`_next_logprobs == []`), and the first stock
fallback returned fewer logprobs than tokens -> IndexError in
GenerationBatch.next(). Emitted one token, then took the engine down.

Two fixes are asserted here:
  * generate()/_mtp_single don't prime MTP when a processor is present.
  * a buffer-empty fallback re-primes the stock pipeline (handback) first,
    so `super()._step()` can never return length-mismatched tokens/logprobs.
"""
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.generate")

from mlx_lm.generate import GenerationBatch  # noqa: E402
from core.inference.mtp import mtp_batch  # noqa: E402

VOCAB = 8


class _StubModel:
    """Just enough model for GenerationBatch._step and _mtp_handback: a single
    forward that returns deterministic per-position logits. Cache is ignored."""

    def __call__(self, inputs, cache=None, return_hidden=False, **kwargs):
        seq = inputs.shape[1]
        # token id t -> favour (t + 1) % VOCAB, so decode makes progress.
        rows = []
        for j in range(seq):
            t = int(inputs[0, j].item())
            row = [0.0] * VOCAB
            row[(t + 1) % VOCAB] = 5.0
            rows.append(row)
        logits = mx.array([rows])  # (1, seq, VOCAB)
        if return_hidden:
            return logits, mx.zeros((1, seq, 4))
        return logits


class _NoStop:
    def make_state(self):
        return None

    def match(self, state, token):
        # never matches a stop sequence -> finish_reason stays None, so
        # next() takes the branch that indexes logprobs[i] (the crash site).
        return state, None, "running"


def _make_batch(logits_processors):
    Mtp = mtp_batch._make_mtp_generation_class(GenerationBatch)
    lp = mx.zeros((VOCAB,))
    return Mtp(
        _StubModel(),
        uids=[1],
        inputs=mx.array([2]),
        prompt_cache=[],
        tokens=[[2]],
        samplers=[None],
        fallback_sampler=lambda l: mx.argmax(l, axis=-1),
        logits_processors=logits_processors,
        state_machines=[_NoStop()],
        max_tokens=[100],
        mtp_depth=3,
        mtp_head_cache=object(),   # "primed": non-None so the bug path is live
        mtp_frontier=None,
        mtp_committed=[5],
        mtp_buffer=[(5, lp)],      # non-empty -> construction short-circuits
        mtp_next_main=5,
    )


def test_processor_bearing_sequence_falls_back_without_indexerror():
    # One passthrough processor: MTP is never ready, so step 2 must fall back.
    batch = _make_batch([[lambda ctx, logits: logits]])

    # Step 1 drains the one primed buffer token.
    r1 = batch.next()
    assert len(r1) == 1

    # Step 2 is the regression: buffer empty, not MTP-ready -> stock fallback.
    # Before the fix this raised IndexError (logprobs[] shorter than tokens).
    r2 = batch.next()
    assert len(r2) == 1
    assert r2[0].logprobs is not None

    # A third step confirms the pipeline is now cleanly on stock decode.
    r3 = batch.next()
    assert len(r3) == 1


def test_handback_disables_mtp_and_frees_head_cache():
    batch = _make_batch([[lambda ctx, logits: logits]])
    batch.next()   # drain buffer
    batch.next()   # triggers the handback re-prime
    assert batch._mtp_disabled is True
    assert batch._mtp_head_cache is None


# --- eligibility gate ------------------------------------------------------

def test_seq_mtp_eligible_requires_greedy_and_replayable():
    greedy = lambda l: l
    greedy.mtp_greedy = True
    not_greedy = lambda l: l
    pure = lambda t, l: l
    pure.mtp_pure = True
    unknown = lambda t, l: l  # neither pure nor a recognized stateful processor

    class _Stateful:
        def mtp_observe(self, *a):
            pass

        def mtp_would_bind(self, *a):
            return False

    assert mtp_batch._seq_mtp_eligible([greedy], [[pure]])
    assert mtp_batch._seq_mtp_eligible([greedy], [[pure, _Stateful()]])
    assert not mtp_batch._seq_mtp_eligible([not_greedy], [[pure]])   # stochastic
    assert not mtp_batch._seq_mtp_eligible([greedy], [[unknown]])    # unknown proc
    assert not mtp_batch._seq_mtp_eligible([], [[]])                 # no sampler


# --- processor-aware verify (losslessness core) ----------------------------

def _greedy_sampler():
    s = lambda l: mx.argmax(l, axis=-1)
    s.mtp_greedy = True
    return s


def _make_eligible_batch(tokens_ctx, processors):
    Mtp = mtp_batch._make_mtp_generation_class(GenerationBatch)
    lp = mx.zeros((VOCAB,))
    return Mtp(
        _StubModel(),
        uids=[1],
        inputs=mx.array([tokens_ctx[-1]]),
        prompt_cache=[],
        tokens=[list(tokens_ctx)],
        samplers=[_greedy_sampler()],
        fallback_sampler=lambda l: mx.argmax(l, axis=-1),
        logits_processors=[processors],
        state_machines=[_NoStop()],
        max_tokens=[100],
        mtp_depth=3,
        mtp_head_cache=object(),
        mtp_frontier=None,
        mtp_committed=[tokens_ctx[-1]],
        mtp_buffer=[(tokens_ctx[-1], lp)],
        mtp_next_main=tokens_ctx[-1],
    )


def test_resolve_greedy_no_processor_is_plain_argmax_accept():
    batch = _make_eligible_batch([2], processors=[])
    V = VOCAB
    row0 = [0.0] * V; row0[2] = 3.0; row0[3] = 2.5   # raw argmax -> 2
    row1 = [0.0] * V; row1[5] = 1.0                   # after accepting -> 5
    vrows = mx.array([row0, row1])
    m, emitted, _ = batch._mtp_resolve_greedy(vrows, drafts=[2])
    assert m == 1                 # draft 2 accepted (matches raw argmax)
    assert emitted == [2, 5]      # accepted draft + bonus token


def test_resolve_greedy_applies_real_repetition_penalty():
    # Repetition penalty on context token 2 lowers logit[2] below logit[3],
    # so the committed token flips from 2 (raw) to 3 (processed) and a draft of
    # 2 is rejected. This is the property that makes MTP lossless vs stock decode.
    from mlx_lm.sample_utils import make_repetition_penalty
    rp = make_repetition_penalty(1.5, context_size=64)
    rp.mtp_pure = True
    batch = _make_eligible_batch([2], processors=[rp])
    V = VOCAB
    row0 = [0.0] * V; row0[2] = 3.0; row0[3] = 2.5   # raw argmax 2; penalised -> 3
    row1 = [0.0] * V; row1[5] = 1.0
    vrows = mx.array([row0, row1])

    m, emitted, _ = batch._mtp_resolve_greedy(vrows, drafts=[2])
    assert m == 0                 # draft 2 rejected: processed argmax is 3, not 2
    assert emitted == [3]         # correction is the penalised argmax

    # Control: same logits, no processor -> the draft is accepted.
    ctrl = _make_eligible_batch([2], processors=[])
    m2, emitted2, _ = ctrl._mtp_resolve_greedy(vrows, drafts=[2])
    assert m2 == 1 and emitted2 == [2, 5]
