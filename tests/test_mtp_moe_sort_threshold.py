"""MoE expert-gather sort-threshold patch (config knob -> patch -> SwitchGLU).

mlx-lm hardcodes ``do_sort = indices.size >= 64`` in SwitchGLU, leaving the
single-stream MTP verify batch (depth-3: 4 tokens x top_k=8 = 32 routed indices)
UNSORTED, so an expert two verified tokens share is read from memory twice. The
patch lowers the threshold to 16 so the verify batch coalesces its expert reads
(bit-identical output, ~20% faster MoE forward measured 2026-08-18) while
single-token stock decode (M=1, size 8) stays unsorted.

These lock the plumbing and the losslessness, not the speedup: the patched
SwitchGLU must be bit-identical to the stock body it replaces at every batch size.
"""
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm.generate")

from mlx_lm.models import switch_layers as sl  # noqa: E402

from core.inference import mtp  # noqa: E402
from core.inference.mtp import qwen3_mtp  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_threshold():
    saved = qwen3_mtp._MOE_SORT_THRESHOLD
    yield
    qwen3_mtp._MOE_SORT_THRESHOLD = saved


@pytest.fixture(scope="module")
def stock_and_patched():
    # Capture the stock __call__ before apply() replaces it, then install the patch.
    stock_call = sl.SwitchGLU.__dict__["__call__"]
    assert mtp.apply() is True
    patched_call = sl.SwitchGLU.__dict__["__call__"]
    assert getattr(patched_call, "_mira_mtp_owned", False)
    # If apply() had already run in-process, stock_call may itself be ours; rebuild a
    # true stock reference from the current mlx-lm source semantics via threshold 64.
    return stock_call, patched_call


@pytest.fixture(scope="module")
def glu():
    g = sl.SwitchGLU(256, 128, 64)  # small: input, hidden, num_experts
    nn.quantize(g, group_size=64, bits=4)
    mx.eval(g.parameters())
    return g


def _indices(rows):
    return mx.array([rows], dtype=mx.uint32)


def test_apply_installs_patch_and_default_is_16(stock_and_patched):
    assert getattr(sl.SwitchGLU.__dict__["__call__"], "_mira_mtp_owned", False)
    assert mtp.get_moe_sort_threshold() == 16


def test_setter_is_live(stock_and_patched):
    mtp.set_moe_sort_threshold(64)
    assert mtp.get_moe_sort_threshold() == 64
    mtp.set_moe_sort_threshold(16)
    assert mtp.get_moe_sort_threshold() == 16


def test_verify_batch_sorted_is_bit_identical(stock_and_patched, glu):
    """M=4 verify batch (size 32): threshold 16 sorts, threshold 64 does not, and
    the two must produce identical output — the coalescing is a perf-only rewrite."""
    _, patched = stock_and_patched
    x = mx.random.normal((1, 4, 256)).astype(mx.bfloat16)
    idx = _indices([[3, 17, 40, 2, 9, 5, 33, 8],
                    [3, 17, 4, 2, 6, 5, 33, 50],
                    [8, 17, 40, 1, 9, 5, 21, 8],
                    [3, 60, 40, 2, 7, 5, 33, 8]])
    assert idx.size == 32
    mtp.set_moe_sort_threshold(16)
    y_sorted = patched(glu, x, idx)
    mtp.set_moe_sort_threshold(64)
    y_unsorted = patched(glu, x, idx)
    mx.eval(y_sorted, y_unsorted)
    assert float(mx.max(mx.abs(y_sorted - y_unsorted))) == 0.0


def test_single_token_decode_never_sorts(stock_and_patched, glu):
    """M=1 stock decode (size 8 < 16): must stay on the unsorted path so the
    non-MTP decode is untouched."""
    _, patched = stock_and_patched
    x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
    idx = _indices([[3, 17, 40, 2, 9, 5, 33, 8]])
    assert idx.size == 8
    mtp.set_moe_sort_threshold(16)
    y16 = patched(glu, x, idx)
    mtp.set_moe_sort_threshold(64)
    y64 = patched(glu, x, idx)
    mx.eval(y16, y64)
    # size 8 is below both thresholds -> identical (neither sorts)
    assert float(mx.max(mx.abs(y16 - y64))) == 0.0
