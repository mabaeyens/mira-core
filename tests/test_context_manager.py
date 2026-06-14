"""Unit tests for context-window bookkeeping (compression planning + token math)."""
import pytest

from core import context_manager as ctxmgr


# -- thinking_tokens -------------------------------------------------------

def test_thinking_tokens_zero():
    assert ctxmgr.thinking_tokens(0) == 0


def test_thinking_tokens_rounds():
    assert ctxmgr.thinking_tokens(35) == 10          # 35 / 3.5
    assert ctxmgr.thinking_tokens(7) == 2            # round(2.0)
    assert ctxmgr.thinking_tokens(5) == 1            # round(1.43)


# -- context_pct -----------------------------------------------------------

def test_context_pct_zero_window_or_prompt():
    assert ctxmgr.context_pct(100, 0) == 0
    assert ctxmgr.context_pct(0, 10000) == 0


def test_context_pct_half_and_full():
    assert ctxmgr.context_pct(5000, 10000) == 50
    assert ctxmgr.context_pct(10000, 10000) == 100


def test_context_pct_clamped_at_100():
    assert ctxmgr.context_pct(25000, 10000) == 100


def test_context_pct_rounds_small_fraction_to_zero():
    assert ctxmgr.context_pct(1, 10000) == 0


# -- plan_compression ------------------------------------------------------

def _conv(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n)]


def test_plan_none_when_not_enough_messages():
    assert ctxmgr.plan_compression(_conv(6), keep_recent=6) is None
    assert ctxmgr.plan_compression(_conv(3), keep_recent=6) is None


def test_plan_splits_compress_and_keep():
    history = _conv(10)
    to_compress, to_keep = ctxmgr.plan_compression(history, keep_recent=6)
    assert [m["content"] for m in to_compress] == ["m0", "m1", "m2", "m3"]
    assert [m["content"] for m in to_keep] == ["m4", "m5", "m6", "m7", "m8", "m9"]


def test_plan_ignores_system_messages_in_count():
    history = [{"role": "system", "content": "sys"}] + _conv(7)
    plan = ctxmgr.plan_compression(history, keep_recent=6)
    assert plan is not None
    to_compress, to_keep = plan
    assert len(to_compress) == 1          # only the oldest non-system message
    assert len(to_keep) == 6
    assert all(m["role"] != "system" for m in to_compress + to_keep)


# -- build_summary_prompt --------------------------------------------------

def test_build_summary_prompt_includes_uppercased_roles():
    prompt = ctxmgr.build_summary_prompt([{"role": "user", "content": "hello"}])
    assert len(prompt) == 1 and prompt[0]["role"] == "user"
    assert "USER: hello" in prompt[0]["content"]
    assert "Summarize this conversation excerpt" in prompt[0]["content"]


def test_build_summary_prompt_truncates_long_messages():
    long_content = "x" * 5000
    prompt = ctxmgr.build_summary_prompt([{"role": "user", "content": long_content}])
    # 2000-char cap per message keeps the summarization prompt bounded.
    assert ("x" * 2000) in prompt[0]["content"]
    assert ("x" * 2001) not in prompt[0]["content"]


# -- rebuild_history -------------------------------------------------------

def test_rebuild_history_preserves_system_and_keeps_recent():
    history = [{"role": "system", "content": "sys"}] + _conv(4)
    to_keep = history[3:]  # last 2
    rebuilt = ctxmgr.rebuild_history(history, "SUMMARY", to_keep)
    assert rebuilt[0] == {"role": "system", "content": "sys"}
    assert rebuilt[1]["content"] == "[Earlier conversation summary]\nSUMMARY"
    assert rebuilt[2] == {"role": "assistant", "content": "Understood, I have the context."}
    assert rebuilt[3:] == to_keep


def test_plan_then_rebuild_round_trip_keeps_recent_verbatim():
    history = [{"role": "system", "content": "sys"}] + _conv(10)
    to_compress, to_keep = ctxmgr.plan_compression(history, keep_recent=6)
    rebuilt = ctxmgr.rebuild_history(history, "S", to_keep)
    assert rebuilt[-6:] == to_keep
