"""The repetition penalties and the per-request seed.

Two knobs that had no wiring at all. `make_logits_processors()` was called with
no arguments in `mira_mlx_server`, so every penalty mlx-lm offers was off and no
configuration could turn one on; and nothing ever seeded the sampler, so two
identical requests came back byte-identical even at temperature 1.0 and
regenerating a reply handed the user back the reply they had just rejected.

The bar these tests hold, taken from `specs/generation-runaway-guard.md` §4: an
install that configures nothing must put the same bytes on the wire and build
the same processor list as before this existed. Every assertion about the unset
case is therefore about *absence*, which is the half that a knob's own tests
usually skip.
"""
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mlx.core")  # mlx is macOS-only, absent on Linux CI

import mlx.core as mx

from core.inference.mira_mlx_server import (
    ChatJob,
    _build_logits_processors,
    _penalties_from_body,
)
from core.orchestrator import ChatOrchestrator


# ── what the request body asks for ────────────────────────────────────────────


def test_ordinary_body_asks_for_no_penalties():
    assert _penalties_from_body({"messages": [], "temperature": 0.0}) == {}


def test_zero_is_off_not_a_setting():
    """0 means "no penalty" in mlx-lm's own reading, and a 0 that arrived as a
    key would still build a processor here if we only checked for None."""
    body = {"repetition_penalty": 0, "presence_penalty": 0.0,
            "frequency_penalty": None}
    assert _penalties_from_body(body) == {}


def test_a_configured_penalty_carries_its_context_size():
    body = {"repetition_penalty": 1.1, "repetition_context_size": 256}
    assert _penalties_from_body(body) == {
        "repetition_penalty": 1.1, "repetition_context_size": 256,
    }


def test_context_size_alone_is_not_carried():
    """Passing a size without a penalty changes nothing in mlx-lm (it defaults
    to 20 either way), so carrying it would only make the log read as though the
    request were configured."""
    assert _penalties_from_body({"repetition_context_size": 256}) == {}


def test_penalties_are_independent():
    body = {"presence_penalty": 0.5, "presence_context_size": 64,
            "frequency_penalty": 0.2}
    assert _penalties_from_body(body) == {
        "presence_penalty": 0.5, "presence_context_size": 64,
        "frequency_penalty": 0.2,
    }


# ── what reaches the processor list ───────────────────────────────────────────


def _processors(penalties=None, thinking_budget=0):
    return _build_logits_processors(
        thinking_budget=thinking_budget,
        think_start=(151667,),
        think_end=(151668,),
        prompt_tokens=[1, 2, 3],
        enable_thinking=True,
        penalties=penalties,
    )


def test_unset_penalties_build_the_same_list_as_before():
    """One entry, and it is the no-op. Not an empty list: mlx-lm's
    PromptProcessingBatch.extend turns a falsy per-sequence list into None when
    the batch mixes sequences with and without processors, and _step() then
    iterates None and kills the engine thread."""
    assert len(_processors()) == 1
    assert _processors()[0].__name__ == "_passthrough_processor"


def test_a_penalty_actually_changes_a_repeated_token_s_logit():
    """Behavioural, in the style of the top_k=1-is-argmax check. Asserting that
    the list got longer would pass just as well against a processor that does
    nothing."""
    processors = _processors(penalties={"repetition_penalty": 2.0})
    penalty = [p for p in processors if p.__name__ != "_passthrough_processor"]
    assert len(penalty) == 1

    tokens = mx.array([7, 7, 7])
    logits = mx.ones((1, 16))
    # Snapshot first: mlx-lm's processor assigns into the array it was handed
    # (`logits[:, tokens] = ...`) and returns that same array, so comparing the
    # result against `logits` afterwards compares it against itself.
    before = logits.tolist()[0]
    out = penalty[0](tokens, logits).tolist()[0]

    assert out[7] < before[7], "repeated token was not penalised"
    assert out[3] == before[3], "an unrelated token moved"


def test_penalties_and_the_thinking_budget_coexist():
    processors = _processors(penalties={"repetition_penalty": 1.1},
                             thinking_budget=2048)
    names = [type(p).__name__ if not hasattr(p, "__name__") else p.__name__
             for p in processors]
    assert "ThinkingBudget" in names
    assert len(processors) == 2, "the no-op should be dropped once real ones exist"


# ── the seed ──────────────────────────────────────────────────────────────────


def test_seed_defaults_to_none_meaning_draw_one():
    job = ChatJob(messages=[], tools=None, stream=False, max_tokens=16,
                  temperature=0.0, top_p=0.0)
    assert job.seed is None
    assert job.penalties is None


def test_seeding_is_what_makes_two_identical_draws_differ():
    """The mechanism the fix relies on, pinned without an engine: mlx's sampler
    draws from the global RNG, so the same state gives the same token sequence
    and a different state gives a different one. If this ever stops holding, a
    per-request seed cannot fix regenerate and the whole approach is wrong."""
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=1.0)
    logits = mx.log(mx.ones((1, 64)) / 64)  # uniform: every token equally likely

    def draw(seed, n=24):
        mx.random.seed(seed)
        return [int(sampler(logits)[0]) for _ in range(n)]

    assert draw(11) == draw(11), "same seed must reproduce"
    assert draw(11) != draw(12), "different seeds must diverge"


def test_temperature_zero_ignores_the_seed_entirely():
    """Why the engine only seeds when temperature > 0: at 0 the sampler is
    argmax, so seeding would perturb the RNG that concurrent sequences draw from
    while changing nothing about this request."""
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    logits = mx.array([[0.1, 0.9, 0.3, 0.2]])

    mx.random.seed(1)
    first = int(sampler(logits)[0])
    mx.random.seed(999)
    assert int(sampler(logits)[0]) == first == 1


# ── what the orchestrator puts on the wire ────────────────────────────────────


def _sent(**config):
    orch = ChatOrchestrator(verbose=False)
    orch.backend, orch.model = "mira-mlx", "Qwen3.6-35B-A3B"
    orch._oai = MagicMock()
    patches = [patch(f"core.orchestrator.{k}", v) for k, v in config.items()]
    with patch("core.orchestrator.bc.normalize_oai_stream", side_effect=lambda s: s), \
         patch("core.orchestrator.bc.normalize_messages_for_oai", side_effect=lambda m: m):
        for p in patches:
            p.start()
        try:
            orch._call_llm([{"role": "user", "content": "hi"}])
        finally:
            for p in patches:
                p.stop()
    return orch._oai.chat.completions.create.call_args.kwargs


def test_nothing_is_sent_when_nothing_is_configured():
    """mira-mlx is not the only backend behind this client and none of these are
    OpenAI-schema keys, so an unconfigured install must not start sending
    them."""
    body = _sent().get("extra_body", {})
    for key in ("repetition_penalty", "presence_penalty", "frequency_penalty",
                "seed", "repetition_context_size"):
        assert key not in body


def test_a_configured_penalty_reaches_extra_body_with_its_size():
    body = _sent(REPETITION_PENALTY=1.1, REPETITION_CONTEXT_SIZE=256)["extra_body"]
    assert body["repetition_penalty"] == 1.1
    assert body["repetition_context_size"] == 256


def test_penalties_do_not_clobber_the_thinking_toggle():
    """The bug this file's neighbour was written for: extra_body already carries
    chat_template_kwargs, and assigning instead of merging silently turns
    thinking off."""
    body = _sent(REPETITION_PENALTY=1.1, SEED=7)["extra_body"]
    assert body["chat_template_kwargs"]["enable_thinking"] is True
    assert body["seed"] == 7


def test_seed_zero_is_a_seed_not_an_absence():
    """0 is falsy and is a perfectly good seed; the None check has to be
    explicit or pinning a run to seed 0 would silently draw a random one."""
    assert _sent(SEED=0)["extra_body"]["seed"] == 0
