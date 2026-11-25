"""Inkling's quoted <think> and </think> are text, not markers.

Found 2026-11-25 on the live-with bench: asked about core/thinking_stripper.py, the
model wrote the literal tags in its reasoning and answer. A plain "</think>" closer
ended the reasoning early (the rest leaked into the answer) and a quoted "<think>"
opened a block that swallowed the answer.
"""
from core.inference.inkling_stream import CLOSE_MARK, THINK_CLOSE  # import-guard: lazy-mlx
from core.thinking_stripper import ThinkingStripper


def _run(chunks, **kw):
    s = ThinkingStripper(preopened=True, inkling=True, **kw)
    out = []
    for c in chunks:
        out += list(s.feed(c))
    out += list(s.drain())
    think = "".join(e["content"] for e in out if e["type"] == "thinking")
    text = "".join(e["content"] for e in out if e["type"] == "token")
    return think, text


def test_a_quoted_closer_in_reasoning_does_not_end_it():
    think, text = _run(["check marker </think> ", "then more" + THINK_CLOSE, "42"])
    assert "</think>" in think and "then more" in think
    assert text.strip() == "42"


def test_a_quoted_opener_in_the_answer_is_kept():
    think, text = _run(["plan" + THINK_CLOSE, 'it returns "<think>" and 4'])
    assert think == "plan"
    assert text.strip() == 'it returns "<think>" and 4'


def test_the_sentinel_split_across_chunks_still_closes():
    think, text = _run(["plan" + CLOSE_MARK[:2], CLOSE_MARK[2:] + "\n\nanswer"])
    assert think == "plan"
    assert text.strip() == "answer"


def test_default_stripper_is_unchanged():
    s = ThinkingStripper(preopened=True)
    out = list(s.feed("plan</think>ans")) + list(s.drain())
    assert "".join(e["content"] for e in out if e["type"] == "token") == "ans"
