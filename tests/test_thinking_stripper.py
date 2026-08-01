"""Unit tests for the streaming ThinkingStripper.

These cover the part of the orchestrator most prone to off-by-one bugs: the
dual-pass <think>/Gemma-channel state machine and its handling of markers that
straddle chunk boundaries. The stripper is pure, so it can be exercised directly
without the LLM, network, or orchestrator.
"""
import pytest

from core.thinking_stripper import ThinkingStripper, _partial_marker_tail


def _run(chunks, preopened=False):
    """Feed ``chunks`` (then drain) and return (visible_text, thinking_text, stripper)."""
    s = ThinkingStripper(preopened=preopened)
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


# -- Qwen pre-opened <think> (template put the opening tag in the PROMPT) ---
#
# Qwen3's chat template appends a bare "<think>\n" to the prompt whenever
# thinking is enabled, so the model's output starts inside the block and only
# ever emits the closing tag. Every test above feeds an explicit opening tag,
# which is the case that never occurs with this model.

def test_preopened_block_is_stripped():
    visible, thinking, s = _run(
        ["The user is asking ", "about Kalman filters.", "</think>", "\n\nA Kalman filter."],
        preopened=True,
    )
    assert visible == "\n\nA Kalman filter."
    assert thinking == "The user is asking about Kalman filters."
    assert s.thinking_chars == len("The user is asking about Kalman filters.")
    assert s.full_content == "\n\nA Kalman filter."


def test_preopened_never_leaks_the_closing_tag():
    # find("<think>") does not match "</think>", so without pre-open handling
    # the stray closing tag rides through as visible answer text.
    visible, _, s = _run(["reasoning", "</think>", "answer"], preopened=True)
    assert "</think>" not in visible
    assert "</think>" not in s.full_content


def test_preopened_close_tag_split_across_chunks():
    visible, thinking, _ = _run(["hidden</thi", "nk>shown"], preopened=True)
    assert visible == "shown"
    assert thinking == "hidden"


def test_preopened_without_close_tag_is_reclassified_as_answer():
    # The template promises the model closes the block. If it never does, the
    # turn must not vanish into thinking and save an empty assistant message.
    visible, _, s = _run(["Here is the answer, ", "no close tag at all."], preopened=True)
    assert visible == "Here is the answer, no close tag at all."
    assert s.full_content == "Here is the answer, no close tag at all."
    assert s.thinking_chars == 0


def test_saw_reasoning_disarms_preopen():
    # Backends that split reasoning into their own delta send only the answer
    # through `content`; assuming a pre-opened block there would hide it.
    s = ThinkingStripper(preopened=True)
    s.saw_reasoning()
    visible = "".join(ev["content"] for ev in s.feed("The answer.") if ev["type"] == "token")
    visible += "".join(ev["content"] for ev in s.drain() if ev["type"] == "token")
    assert visible == "The answer."
    assert s.thinking_chars == 0


def test_preopened_off_by_default_keeps_old_behaviour():
    visible, thinking, _ = _run(["plain answer, no tags"])
    assert visible == "plain answer, no tags"
    assert thinking == ""


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
