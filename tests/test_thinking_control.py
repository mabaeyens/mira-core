"""Regression tests for per-turn thinking control on the OpenAI-compatible
backends (mira-mlx / mlx-lm / omlx / vllm-mlx).

Bug: omlx is the default backend, but it was missing from the Qwen3
thinking-control branch in `_call_llm`. As a result `enable_thinking` was
either omitted (thinking_enabled=True) or sent at the wrong nesting level
(thinking_enabled=False), so Qwen3 fell back to its chat-template default
(thinking ON) and the per-turn toggle never took effect.
"""

import types

import pytest
from unittest.mock import MagicMock, patch

from core.orchestrator import ChatOrchestrator, _uses_qwen_thinking_template


@pytest.fixture
def orchestrator():
    return ChatOrchestrator(verbose=False)


def _capture_create_kwargs(orchestrator, *, backend, model, thinking_enabled):
    """Drive `_call_llm` with a mocked OpenAI client and return the kwargs
    passed to chat.completions.create()."""
    orchestrator.backend = backend
    orchestrator.model = model
    orchestrator._oai = MagicMock()
    with patch("core.orchestrator.bc.normalize_oai_stream", side_effect=lambda s: s), \
         patch("core.orchestrator.bc.normalize_messages_for_oai", side_effect=lambda m: m):
        orchestrator._call_llm([{"role": "user", "content": "hi"}],
                               thinking_enabled=thinking_enabled)
    return orchestrator._oai.chat.completions.create.call_args.kwargs


@pytest.mark.parametrize("backend", ["omlx", "mlx-lm", "vllm-mlx"])
def test_qwen3_off_nests_enable_thinking_under_chat_template_kwargs(orchestrator, backend):
    kwargs = _capture_create_kwargs(
        orchestrator, backend=backend, model="Qwen3.6-35B-A3B", thinking_enabled=False
    )
    ckwargs = kwargs["extra_body"]["chat_template_kwargs"]
    assert ckwargs["enable_thinking"] is False
    # The misplaced top-level key must NOT be present — omlx ignores it.
    assert "enable_thinking" not in kwargs["extra_body"] or \
        kwargs["extra_body"].get("enable_thinking") is None or \
        isinstance(kwargs["extra_body"].get("chat_template_kwargs"), dict)


@pytest.mark.parametrize("backend", ["omlx", "mlx-lm", "vllm-mlx"])
def test_qwen3_on_nests_enable_thinking_under_chat_template_kwargs(orchestrator, backend):
    kwargs = _capture_create_kwargs(
        orchestrator, backend=backend, model="Qwen3.6-35B-A3B", thinking_enabled=True
    )
    ckwargs = kwargs["extra_body"]["chat_template_kwargs"]
    assert ckwargs["enable_thinking"] is True


def test_non_qwen_model_does_not_set_chat_template_kwargs(orchestrator):
    """gemma-style models have no enable_thinking template variable; the off
    path still passes the top-level flag, the on path passes nothing."""
    off = _capture_create_kwargs(
        orchestrator, backend="omlx", model="gemma-4-26b-it", thinking_enabled=False
    )
    assert off["extra_body"] == {"enable_thinking": False}

    on = _capture_create_kwargs(
        orchestrator, backend="omlx", model="gemma-4-26b-it", thinking_enabled=True
    )
    assert "extra_body" not in on


# -- response side: the same template that honours enable_thinking also
# pre-opens `<think>` in the prompt, so the stripper has to be told ------------

def _chunk(content="", thinking=None, done=False):
    msg = types.SimpleNamespace(content=content, tool_calls=None, thinking=thinking)
    return types.SimpleNamespace(message=msg, done=done)


def _stream_events(orchestrator, *, backend, model, thinking_enabled, chunks):
    orchestrator.backend = backend
    orchestrator.model = model
    with patch.object(orchestrator, "_call_llm", return_value=iter(chunks)):
        return list(orchestrator._stream_llm_with_thinking(thinking_enabled))


@pytest.mark.parametrize("backend", ["mira-mlx", "omlx", "mlx-lm", "vllm-mlx"])
def test_qwen3_thinking_on_does_not_leak_reasoning_into_the_answer(orchestrator, backend):
    """The regression: Qwen3's template puts `<think>\\n` in the prompt, so the
    model emits only the closing tag. Before the fix the whole reasoning stream
    was served as the answer, with a stray `</think>` in it."""
    events = _stream_events(
        orchestrator, backend=backend, model="Qwen3.6-35B-A3B", thinking_enabled=True,
        chunks=[_chunk("The user is asking about X."), _chunk("</think>"),
                _chunk("The answer is 42."), _chunk(done=True)],
    )
    visible = "".join(e["content"] for e in events if e.get("type") == "token")
    thinking = "".join(e["content"] for e in events if e.get("type") == "thinking")
    done = [e for e in events if e.get("type") == "llm_done"][0]

    assert visible == "The answer is 42."
    assert "</think>" not in visible
    assert thinking == "The user is asking about X."
    assert done["full_content"] == "The answer is 42."
    assert done["thinking_chars"] == len("The user is asking about X.")


def test_qwen3_thinking_off_streams_the_answer_untouched(orchestrator):
    """Thinking off means the template emits a pre-CLOSED empty block, so the
    output carries no tags and must not be treated as pre-opened."""
    events = _stream_events(
        orchestrator, backend="mira-mlx", model="Qwen3.6-35B-A3B", thinking_enabled=False,
        chunks=[_chunk("Here is "), _chunk("the answer."), _chunk(done=True)],
    )
    visible = "".join(e["content"] for e in events if e.get("type") == "token")
    assert visible == "Here is the answer."
    assert [e for e in events if e.get("type") == "llm_done"][0]["thinking_chars"] == 0


def test_out_of_band_reasoning_backend_keeps_its_answer(orchestrator):
    """A backend that sends reasoning in its own delta puts only the answer in
    `content`; treating that as a pre-opened block would hide the whole turn."""
    events = _stream_events(
        orchestrator, backend="omlx", model="Qwen3.6-35B-A3B", thinking_enabled=True,
        chunks=[_chunk(thinking="reasoning out of band"), _chunk("The answer is 42."),
                _chunk(done=True)],
    )
    visible = "".join(e["content"] for e in events if e.get("type") == "token")
    assert visible == "The answer is 42."
    assert [e for e in events if e.get("type") == "llm_done"][0]["full_content"] == "The answer is 42."


def test_template_predicate_shared_by_request_and_response_sides():
    assert _uses_qwen_thinking_template("mira-mlx", "mlx-community/Qwen3.6-35B-A3B-4bit")
    assert not _uses_qwen_thinking_template("mira-mlx", "gemma-4-26b-it")
    assert not _uses_qwen_thinking_template("unknown-backend", "Qwen3.6-35B-A3B")


# -- utility calls (title, summary, forced JSON) bypass _call_llm, so they carry
# their own thinking suppression. Without it a title turn reasons unbounded -----

def _sync_response(text="{\"title\": \"x\"}"):
    msg = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_utility_call_suppresses_thinking_on_qwen(orchestrator):
    """Measured 2026-08-13: a title turn on Qwen3.6 reasoned 1651 tokens (~29s)
    because it went straight to the model with the template's thinking-on
    default and no budget. _llm_chat_sync must send enable_thinking=False."""
    orchestrator.backend = "mira-mlx"
    orchestrator.model = "Qwen3.6-35B-A3B"
    orchestrator._oai = MagicMock()
    orchestrator._oai.chat.completions.create.return_value = _sync_response()
    orchestrator._llm_chat_sync([{"role": "user", "content": "name this"}])
    kwargs = orchestrator._oai.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_utility_call_carries_budget_backstop_on_qwen(orchestrator):
    """The budget rides along as a backstop for the turns Qwen opens a block
    anyway, matching the answer path."""
    from core.orchestrator import MAX_THINKING_TOKENS
    orchestrator.backend = "mira-mlx"
    orchestrator.model = "Qwen3.6-35B-A3B"
    extra = orchestrator._thinking_off_extra()
    if MAX_THINKING_TOKENS > 0:
        assert extra["chat_template_kwargs"]["thinking_budget"] == MAX_THINKING_TOKENS
    else:
        assert "thinking_budget" not in extra["chat_template_kwargs"]


def test_utility_call_thinking_off_on_non_qwen(orchestrator):
    orchestrator.backend = "omlx"
    orchestrator.model = "gemma-4-26b-it"
    assert orchestrator._thinking_off_extra() == {"enable_thinking": False}


def test_generate_title_goes_through_thinking_suppression(orchestrator):
    orchestrator.backend = "mira-mlx"
    orchestrator.model = "Qwen3.6-35B-A3B"
    orchestrator._oai = MagicMock()
    orchestrator._oai.chat.completions.create.return_value = _sync_response(
        "{\"title\": \"Quantization Basics\"}"
    )
    title = orchestrator.generate_title("Explain 4-bit quantization")
    kwargs = orchestrator._oai.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert title == "Quantization Basics"
