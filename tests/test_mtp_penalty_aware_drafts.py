"""Penalty-aware drafts (spec: native-mtp-penalty-aware-drafts).

The draft chain runs the head argmax through the SAME pure logits processors
verify applies at the matching position, so a draft predicts the penalized
target it will be judged against instead of the raw-head token verify would
reject. This is the accept-rate lever that flips native MoE MTP net-positive.

Two properties, both on the ``_mtp_draft_token`` helper directly:
  * no processors -> plain argmax, byte-identical to the raw-head path;
  * a real repetition penalty over the draft context shifts the drafted token
    to the penalized argmax, i.e. it now matches what verify commits.
"""
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.generate")

from mlx_lm.generate import GenerationBatch  # noqa: E402

# Reuse the eligible-batch scaffolding and vocab from the fallback suite.
from tests.test_mtp_batch_fallback import _make_eligible_batch, VOCAB  # noqa: E402


def test_draft_token_no_processor_is_plain_argmax():
    # With no pure processors the helper must be exactly argmax — drafts stay
    # byte-identical to the pre-change raw-head path.
    batch = _make_eligible_batch([2], processors=[])
    V = VOCAB
    row = [0.0] * V
    row[2] = 3.0
    row[3] = 2.5                       # raw argmax -> 2
    logits_row = mx.array([row])       # (1, V)
    assert batch._mtp_draft_token(logits_row, ctx=[2]) == 2


def test_draft_token_applies_repetition_penalty_aligning_with_verify():
    # Draft context contains token 2. A rep penalty of 1.5 pushes logit[2] below
    # logit[3], so the penalty-aware draft is 3 — exactly the token verify's own
    # penalized argmax commits. Without the fix the draft would be 2 (raw head),
    # which verify then rejects. This is the accept-rate recovery.
    from mlx_lm.sample_utils import make_repetition_penalty
    rp = make_repetition_penalty(1.5, context_size=64)
    rp.mtp_pure = True
    V = VOCAB

    def fresh_row():
        # The mlx rep penalty scatters into the logits array in place, so each
        # call needs its own copy (production feeds fresh logits every cycle).
        row = [0.0] * V
        row[2] = 3.0
        row[3] = 2.5                   # raw argmax 2; penalised over ctx [2] -> 3
        return mx.array([row])         # (1, V)

    # Penalty-aware draft matches the penalized target.
    batch = _make_eligible_batch([2], processors=[rp])
    assert batch._mtp_draft_token(fresh_row(), ctx=[2]) == 3

    # Control: same logits, no processor -> raw argmax 2 (the token verify would
    # have rejected). Proves the shift is the penalty, not the logits.
    ctrl = _make_eligible_batch([2], processors=[])
    assert ctrl._mtp_draft_token(fresh_row(), ctx=[2]) == 2


def test_draft_token_context_grows_with_chain():
    # As the chain extends, the growing context can flip later drafts too: with
    # both 2 and 5 penalized, the second-position draft avoids both.
    from mlx_lm.sample_utils import make_repetition_penalty
    rp = make_repetition_penalty(1.5, context_size=64)
    rp.mtp_pure = True
    batch = _make_eligible_batch([2], processors=[rp])
    V = VOCAB
    row = [0.0] * V
    row[2] = 3.0
    row[5] = 2.8
    row[7] = 2.6                       # raw argmax 2; ctx [2,5] penalised -> 7
    logits_row = mx.array([row])       # (1, V)
    assert batch._mtp_draft_token(logits_row, ctx=[2, 5]) == 7
