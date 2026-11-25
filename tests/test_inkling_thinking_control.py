"""Inkling 2 gets the same thinking controls as Qwen: the per-turn toggle and the budget.

Inkling has no <think>/</think>. A reasoning message opens with
<|content_thinking|> and ends with <|end_message|>, which is also how answers and
tool calls end, and the prompt itself ends with one.
"""
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from core.inference.mira_mlx_server import ThinkingBudget  # noqa: E402
from core.orchestrator import _is_inkling, _sends_thinking_kwargs  # noqa: E402

THINK, END, OTHER, ANSWER = 8, 10, 5, 4
VOCAB = 32


def _logits():
    return mx.zeros((1, VOCAB))


def _budget(budget):
    return ThinkingBudget(budget, think_start=(THINK,), think_end=(END,),
                          preopened=False, end_requires_start=True)


def test_prompt_ending_in_end_message_does_not_disarm_the_budget():
    b = _budget(3)
    # History ends with the prompt's closing <|end_message|>, before any reasoning.
    b([OTHER, END], _logits())
    assert not b.closed
    assert not b.started


def test_budget_forces_end_message_after_the_limit():
    b = _budget(3)
    history = [OTHER, END]
    history.append(THINK)
    b(history, _logits())
    assert b.started
    forced_logits = None
    for _ in range(5):
        history.append(OTHER)
        out = b(history, _logits())
        if b.forced:
            forced_logits = out
            break
    assert forced_logits is not None
    assert int(mx.argmax(forced_logits.reshape(-1)).item()) == END
    assert b.closed


def test_a_natural_close_inside_the_budget_disarms_it():
    b = _budget(50)
    history = [END, THINK, OTHER, OTHER]
    for i in range(1, len(history) + 1):
        b(history[:i], _logits())
    b(history + [END], _logits())
    assert b.closed
    assert b.forced == 0


def test_an_answer_with_no_reasoning_never_arms_it():
    b = _budget(2)
    history = [END]
    for _ in range(10):
        history.append(ANSWER)
        b(history, _logits())
    assert not b.started
    assert b.forced == 0


def test_default_still_closes_on_the_end_token_without_a_start():
    """Qwen keeps its behaviour: a pre-closed block disarms the budget."""
    b = ThinkingBudget(3, think_start=(THINK,), think_end=(END,), preopened=False)
    b([OTHER, END], _logits())
    assert b.closed


def test_orchestrator_gate_covers_inkling_and_qwen_only():
    assert _is_inkling("mira-mlx", "/x/models--kiteteam--Inkling-2-Mini-q4l-MLX/snap")
    assert not _is_inkling("ollama", "Inkling-2-Mini")
    assert _sends_thinking_kwargs("mira-mlx", "Inkling-2-Mini-q4l-MLX")
    assert _sends_thinking_kwargs("mira-mlx", "Qwen3.6-35B-A3B-4bit")
    assert not _sends_thinking_kwargs("mira-mlx", "gemma-4-26b-it")
