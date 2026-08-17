"""ThinkingBudget's native-MTP replay hooks must match its own __call__.

Native MTP does not call ThinkingBudget speculatively (that would corrupt its
per-token counter). Instead it advances the counter over committed tokens with
mtp_observe(), and hands the sequence back to stock decode before the budget
would force a token (mtp_would_bind()). For MTP to stay lossless, observe() must
leave ThinkingBudget in exactly the state that one __call__ per token would.
"""
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from core.inference.mira_mlx_server import ThinkingBudget  # noqa: E402

START = (100,)
END = (200,)


def _run_via_call(stream, budget, preopened=False):
    tb = ThinkingBudget(budget=budget, think_start=START, think_end=END,
                        preopened=preopened)
    history = [9, 9]  # a bit of prompt
    logits = mx.zeros((1, 8))
    for t in stream:
        history.append(t)
        tb(mx.array(history), logits)
    return tb, history


def _run_via_observe(stream, budget, preopened=False):
    tb = ThinkingBudget(budget=budget, think_start=START, think_end=END,
                        preopened=preopened)
    full = [9, 9] + list(stream)
    tb.mtp_observe(full, len(stream))
    return tb


def test_observe_matches_call_across_a_reasoning_block():
    # open (100), three thinking tokens, close (200), then answer tokens.
    stream = [100, 1, 2, 3, 200, 7, 8]
    a, _ = _run_via_call(stream, budget=1000)
    b = _run_via_observe(stream, budget=1000)
    assert (a.count, a.started, a.closed) == (b.count, b.started, b.closed)
    assert b.started and b.closed
    # answer tokens after the close are not counted.
    assert b.count == 4


def test_observe_matches_call_preopened_no_explicit_start():
    # Qwen3-style: block already open at token 0, no think_start emitted.
    stream = [1, 2, 3, 4, 200, 5]
    a, _ = _run_via_call(stream, budget=1000, preopened=True)
    b = _run_via_observe(stream, budget=1000, preopened=True)
    assert (a.count, a.started, a.closed) == (b.count, b.started, b.closed)
    assert b.closed and b.count == 4


def test_observe_never_counts_before_the_block_opens():
    stream = [5, 6, 7]  # no start marker, not preopened
    b = _run_via_observe(stream, budget=1000)
    assert not b.started and b.count == 0


def test_would_bind_only_when_forcing_is_within_reach():
    tb = ThinkingBudget(budget=5, think_start=START, think_end=END, preopened=True)
    tb.count = 1
    assert not tb.mtp_would_bind(3)   # 1 + 3 = 4 < 5
    tb.count = 3
    assert tb.mtp_would_bind(3)       # 3 + 3 = 6 >= 5

    closed = ThinkingBudget(budget=5, think_start=START, think_end=END,
                            preopened=True)
    closed.closed = True
    assert not closed.mtp_would_bind(9)   # permanently inert once closed

    unstarted = ThinkingBudget(budget=5, think_start=START, think_end=END,
                               preopened=False)
    assert not unstarted.mtp_would_bind(9)  # can't reach budget before it starts
