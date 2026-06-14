"""Unit tests for the streaming ThinkingStripper.

These cover the part of the orchestrator most prone to off-by-one bugs: the
dual-pass <think>/Gemma-channel state machine and its handling of markers that
straddle chunk boundaries. The stripper is pure, so it can be exercised directly
without the LLM, network, or orchestrator.
"""
import pytest

from core.thinking_stripper import ThinkingStripper, _partial_marker_tail


def _run(chunks):
    """Feed ``chunks`` (then drain) and return (visible_text, thinking_text, stripper)."""
    s = ThinkingStripper()
    visible, thinking = [], []
    for c in chunks:
        for ev in s.feed(c):
            (visible if ev["type"] == "token" else thinking).append(ev["content"])
    for ev in s.drain():
        (visible if ev["type"] == "token" else thinking).append(ev["content"])
    return "".join(visible), "".join(thinking), s


# -- _partial_marker_tail --------------------------------------------------

def test_partial_tail_none_when_no_prefix():
    assert _partial_marker_tail("hello world", "<think>") == 0


def test_partial_tail_detects_marker_start():
    assert _partial_marker_tail("answer <th", "<think>") == 3  # "<th"


def test_partial_tail_never_matches_full_marker():
    # A complete marker is not a "tail to hold" — only proper prefixes count.
    assert _partial_marker_tail("<think>", "<think>") == 0


def test_partial_tail_longest_across_markers():
    assert _partial_marker_tail("x<chan", "<think>", "<channel|>") == 5  # "<chan"


# -- plain passthrough -----------------------------------------------------

def test_plain_text_passes_through():
    visible, thinking, s = _run(["Hello, ", "world!"])
    assert visible == "Hello, world!"
    assert thinking == ""
    assert s.full_content == "Hello, world!"
    assert s.thinking_chars == 0


# -- Qwen <think> ----------------------------------------------------------

def test_think_block_stripped():
    visible, thinking, s = _run(["<think>reasoning here</think>The answer."])
    assert visible == "The answer."
    assert thinking == "reasoning here"
    assert s.thinking_chars == len("reasoning here")


def test_think_block_split_across_chunks():
    visible, thinking, _ = _run(["<thi", "nk>hid", "den</thi", "nk>shown"])
    assert visible == "shown"
    assert thinking == "hidden"


def test_text_before_and_after_think():
    visible, thinking, _ = _run(["before <think>mid</think> after"])
    assert visible == "before  after"
    assert thinking == "mid"


def test_multiple_think_blocks():
    visible, thinking, _ = _run(["a<think>x</think>b<think>y</think>c"])
    assert visible == "abc"
    assert thinking == "xy"


def test_unterminated_think_drained_as_thinking():
    # Stream ends mid-thought: the buffered remainder is flushed as thinking.
    visible, thinking, _ = _run(["<think>never closed"])
    assert visible == ""
    assert thinking == "never closed"


# -- Gemma channel ---------------------------------------------------------

def test_gemma_channel_stripped():
    visible, thinking, _ = _run(["<|channel>thought\nreasoning<channel|>visible"])
    assert visible == "visible"
    assert thinking == "reasoning"


def test_gemma_channel_split_across_chunks():
    visible, thinking, _ = _run(["pre<|channel>thou", "ght\nhid<chan", "nel|>post"])
    assert visible == "prepost"
    assert thinking == "hid"


def test_unterminated_gemma_drained_as_thinking():
    visible, thinking, _ = _run(["<|channel>thought\nopen ended"])
    assert visible == ""
    assert thinking == "open ended"


# -- robustness ------------------------------------------------------------

def test_char_by_char_matches_single_chunk():
    text = "intro <think>secret reasoning</think> body <|channel>thought\nmore<channel|> end"
    whole_v, whole_t, _ = _run([text])
    char_v, char_t, _ = _run(list(text))
    assert whole_v == char_v
    assert whole_t == char_t
    assert whole_v == "intro  body  end"
    assert whole_t == "secret reasoningmore"


def test_angle_bracket_that_is_not_a_marker_is_emitted():
    visible, thinking, _ = _run(["1 < 2 and 3 > 2"])
    assert visible == "1 < 2 and 3 > 2"
    assert thinking == ""


def test_full_content_and_thinking_chars_track_totals():
    _, _, s = _run(["<think>abc</think>XY", "Z"])
    assert s.full_content == "XYZ"
    assert s.thinking_chars == 3
