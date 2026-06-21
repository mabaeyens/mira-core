"""Regression tests for per-turn thinking control on the OpenAI-compatible
backends (mlx-lm / dflash / omlx).

Bug: omlx is the default backend, but it was missing from the Qwen3
thinking-control branch in `_call_llm`. As a result `enable_thinking` was
either omitted (thinking_enabled=True) or sent at the wrong nesting level
(thinking_enabled=False), so Qwen3 fell back to its chat-template default
(thinking ON) and the per-turn toggle never took effect.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.orchestrator import ChatOrchestrator


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
         patch("core.orchestrator.bc.normalize_messages_for_oai", side_effect=lambda m: m), \
         patch("core.orchestrator.restart_dflash_if_dead", return_value=None):
        orchestrator._call_llm([{"role": "user", "content": "hi"}],
                               thinking_enabled=thinking_enabled)
    return orchestrator._oai.chat.completions.create.call_args.kwargs


@pytest.mark.parametrize("backend", ["omlx", "mlx-lm", "dflash"])
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


@pytest.mark.parametrize("backend", ["omlx", "mlx-lm", "dflash"])
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
