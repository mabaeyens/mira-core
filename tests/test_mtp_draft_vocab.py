"""Reduced-vocab draft projection wiring (config knob -> patch -> module).

The draft head projects only the first N of the 248320 vocab rows when drafting
(``mtp_forward(draft_vocab=N)``), cutting the lm_head cost of each draft. Lossless
by construction: the full-vocab backbone verify decides every emitted token, so a
narrow N can only lower accept if a true token id lands beyond N.

These lock the plumbing, not the kernel: patch() overrides the module default only
when given a positive value (so the env-only bench path is untouched), and 0 is a
no-op that leaves the env default in place.
"""
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.generate")

from core.inference.mtp import mtp_batch  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_draft_vocab():
    # patch() mutates the module global; snapshot and restore so tests don't leak.
    saved = mtp_batch._DRAFT_VOCAB
    yield
    mtp_batch._DRAFT_VOCAB = saved


def test_patch_sets_draft_vocab_when_positive():
    mtp_batch._DRAFT_VOCAB = 0
    mtp_batch.patch(depth=3, draft_vocab=131072)
    assert mtp_batch._DRAFT_VOCAB == 131072


def test_patch_zero_draft_vocab_leaves_env_default_untouched():
    # 0 (the CLI default when no engine arg is passed) must NOT clobber whatever
    # the env-derived module default was — the bench path relies on the env value.
    mtp_batch._DRAFT_VOCAB = 65536
    mtp_batch.patch(depth=3, draft_vocab=0)
    assert mtp_batch._DRAFT_VOCAB == 65536


def test_patch_draft_vocab_default_is_zero():
    # Calling patch() with no draft_vocab arg is the pre-feature call and must be
    # a no-op on the width (depth-only callers keep working unchanged).
    mtp_batch._DRAFT_VOCAB = 32768
    mtp_batch.patch(depth=2)
    assert mtp_batch._DRAFT_VOCAB == 32768
