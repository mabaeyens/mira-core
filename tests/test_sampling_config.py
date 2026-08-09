"""Sampling parameters reach the backend, and top_k does not clobber thinking.

Context: until 2026-08-09 Mira sent no sampling parameters at all, so
mira_mlx_server's own `temperature`/`top_p` defaults of 0.0 applied and every
reply was greedy-decoded. Nothing chose that; it was the absence of a choice.
Qwen3.6 ships generation_config.json asking for temperature 1.0 / top_k 20 /
top_p 0.95. Measured 2026-08-09: greedy itself was fine, but temperature 1.0
with top_k/top_p left at 0 degenerated badly — which is precisely why top_k has
to work, not just temperature.

The specific hazard these tests pin down: `top_k` is not an OpenAI-API
parameter, so it has to ride in `extra_body` — which is exactly where the
per-turn thinking config already lives. Assigning instead of merging there would
silently disable the thinking toggle, which is a bug that has bitten this file
before (see test_thinking_control.py).
"""

from unittest.mock import MagicMock, patch

import pytest

from core.orchestrator import ChatOrchestrator


@pytest.fixture
def orchestrator():
    return ChatOrchestrator(verbose=False)


def _capture(orchestrator, *, backend="mira-mlx", model="Qwen3.6-35B-A3B",
             thinking_enabled=True):
    orchestrator.backend = backend
    orchestrator.model = model
    orchestrator._oai = MagicMock()
    with patch("core.orchestrator.bc.normalize_oai_stream", side_effect=lambda s: s), \
         patch("core.orchestrator.bc.normalize_messages_for_oai", side_effect=lambda m: m):
        orchestrator._call_llm([{"role": "user", "content": "hi"}],
                               thinking_enabled=thinking_enabled)
    return orchestrator._oai.chat.completions.create.call_args.kwargs


def test_temperature_and_top_p_are_always_sent(orchestrator):
    """They must be sent explicitly. Omitting them is what produced silent
    greedy decoding — the server defaults both to 0.0 when they are absent."""
    kwargs = _capture(orchestrator)
    assert "temperature" in kwargs
    assert "top_p" in kwargs


def test_top_k_is_omitted_when_zero(orchestrator):
    with patch("core.orchestrator.TOP_K", 0):
        kwargs = _capture(orchestrator)
    assert "top_k" not in kwargs.get("extra_body", {})


def test_top_k_rides_in_extra_body_without_clobbering_thinking(orchestrator):
    """The regression that matters: extra_body must be MERGED, not assigned."""
    with patch("core.orchestrator.TOP_K", 20):
        kwargs = _capture(orchestrator, thinking_enabled=True)
    extra_body = kwargs["extra_body"]
    assert extra_body["top_k"] == 20
    # Thinking config must survive alongside it.
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is True


def test_top_k_does_not_clobber_thinking_when_disabled(orchestrator):
    with patch("core.orchestrator.TOP_K", 20):
        kwargs = _capture(orchestrator, thinking_enabled=False)
    extra_body = kwargs["extra_body"]
    assert extra_body["top_k"] == 20
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is False


def test_configured_values_are_the_ones_sent(orchestrator):
    with patch("core.orchestrator.TEMPERATURE", 1.0), \
         patch("core.orchestrator.TOP_P", 0.95):
        kwargs = _capture(orchestrator)
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.95


def test_defaults_preserve_pre_2026_08_09_greedy_behaviour():
    """This change must not alter a single reply until someone edits mira.yaml."""
    from core import config
    assert config.TEMPERATURE == 0.0
    assert config.TOP_P == 0.0
    assert config.TOP_K == 0
