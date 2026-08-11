"""What happens when a generation is cut off at the token cap.

Both failure modes here were found by real conversations, not by reasoning about
the code: two batches of multi-turn exchanges on 2026-08-11 put 13 of 51 turns
into one of these states, and every single token-cap hit produced one of them.
No unit test caught either, because every component was behaving as designed.
"""
from collections import Counter

import pytest

from core.orchestrator import _degenerate_run
from core.thinking_stripper import ThinkingStripper


def _events(stripper, raw):
    return list(stripper.feed(raw))


# ── the reply that is really the reasoning ────────────────────────────────────


def test_unclosed_preopened_block_is_still_reclassified_when_the_model_stopped():
    """The existing behaviour, which must not regress.

    A model that *chose* not to close the block was never reasoning: the template
    promised a close tag, so its absence means the tags misled us. Publishing the
    text is right here, and swallowing it would save an empty assistant turn.
    """
    s = ThinkingStripper(preopened=True)
    _events(s, "Here is the answer.")
    out = list(s.drain())

    assert s.full_content == "Here is the answer."
    assert [e["type"] for e in out] == ["token"]


def test_unclosed_preopened_block_is_not_the_answer_when_cut_off():
    """The fix. Truncated mid-thought, the block is unclosed because generation
    stopped, not because the text was an answer all along."""
    s = ThinkingStripper(preopened=True)
    _events(s, "The user wants me to trace Example 1 concretely, row by row")
    s.truncated()
    out = list(s.drain())

    assert s.full_content == "", "chain of thought was published as the answer"
    assert all(e["type"] == "thinking" for e in out)


def test_truncated_reasoning_is_not_emitted_twice():
    """The signature of the live bug: identical reply and thinking lengths.

    The same characters went out once on the thinking channel while streaming and
    again as the answer from drain(). Four turns in the corpus matched exactly --
    13,299 = 13,299, 19,247 = 19,247, 14,025 = 14,025.
    """
    s = ThinkingStripper(preopened=True)
    thinking = "".join(f"step {i} of the reasoning. " for i in range(200))
    streamed = "".join(e["content"] for e in _events(s, thinking)
                       if e["type"] == "thinking")
    s.truncated()
    drained = "".join(e["content"] for e in s.drain() if e["type"] == "thinking")

    assert len(s.full_content) != len(streamed + drained)
    assert s.full_content == ""


def test_truncation_does_not_affect_a_properly_closed_block():
    """Cut off *after* the reasoning closed: the partial answer is real and must
    survive. Only the pre-opened, never-closed path changes."""
    s = ThinkingStripper(preopened=True)
    _events(s, "reasoning about it</think>The answer begins and then stop")
    s.truncated()
    list(s.drain())

    assert s.full_content == "The answer begins and then stop"


# ── the reply that is one character, four thousand times ──────────────────────


def test_the_exact_reply_that_poisoned_a_conversation():
    # Verbatim shape from corpus-qlik-data-modelling, 2026-08-11: five turns in
    # a row of exactly 4096 '!', a full max_tokens of one repeated token. The
    # last four made no tool calls at all -- once saved, the reply came back in
    # the history and produced itself again.
    found = _degenerate_run("!" * 4096)
    assert found is not None
    assert found[0] == "!"
    assert found[1] == 1.0


@pytest.mark.parametrize("reply", ["ha" * 3000, "…" * 500, "?" * 900])
def test_other_single_character_loops_are_caught(reply):
    assert _degenerate_run(reply) is not None, f"missed a loop: {reply[:40]!r}"


def test_a_repeated_phrase_is_deliberately_not_caught():
    """Scope marker, not an oversight.

    A loop that repeats a whole sentence has ordinary letter frequencies and this
    guard cannot see it. Catching it needs n-gram repetition detection, which can
    fire on legitimate output -- a table of identical values, a list of repeated
    statuses -- and no phrase loop has ever been observed from Mira. The measured
    failure was one character, 4096 times. If a phrase loop shows up in a real
    conversation, this test is the place to come back to.
    """
    assert _degenerate_run("\n".join(["I cannot answer that."] * 400)) is None


def test_a_real_ascii_diagram_is_not_degenerate():
    """Regression on the guard itself.

    The first version of this rule scored two genuine diagrams as broken --
    box-drawing characters legitimately dominate them. One was among the better
    answers in the corpus. A guard that eats good replies is worse than no guard.
    """
    diagram = (
        "Here's the flow:\n\n```\n"
        "┌─────────────┐      ┌──────────────┐      ┌───────────┐\n"
        "│  Source     │─────▶│ Delta LOAD   │─────▶│  QVD      │\n"
        "└─────────────┘      └──────────────┘      └───────────┘\n"
        "```\n\n"
        "The delta step reads only rows newer than the stored high water mark, "
        "then concatenates them onto the existing QVD before storing it again.\n"
    )
    assert _degenerate_run(diagram) is None


@pytest.mark.parametrize("reply", [
    "yes",
    "42",
    "",
    "No, that is not correct: Aggr() builds a virtual table whose granularity "
    "is independent of the chart's, which is why the dimension list matters.",
])
def test_healthy_replies_survive(reply):
    assert _degenerate_run(reply) is None


def test_short_repetitive_replies_are_left_alone():
    """A terse real answer can look repetitive. The guard only fires once a reply
    is long enough that repetition cannot be an accident."""
    assert _degenerate_run("!!!") is None
    assert _degenerate_run("!" * 199) is None


def test_threshold_is_below_what_a_real_answer_reaches():
    """Guards the constant itself: prose must sit well clear of the line, or the
    next long answer with heavy punctuation gets discarded."""
    prose = ("Set analysis is evaluated once per aggregation, before the chart "
             "dimension loop runs. That is the part people get wrong. ") * 20
    body = [c for c in prose if c not in " \t\n"]
    share = max(Counter(body).values()) / len(body)
    assert share < 0.2, f"real prose is {share:.0%} one character; the guard is at 40%"
