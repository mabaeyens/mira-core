"""Inkling 2 message framing: what the client sees, and what it must not."""

from core.inference.inkling_stream import (
    THINK_CLOSE,
    InklingStreamFilter,
    is_inkling_tokenizer,
)

VOCAB = {
    "<|message_model|>": 1,
    "<|content_text|>": 2,
    "<|content_thinking|>": 3,
    "<|end_message|>": 4,
    "<|content_invoke_tool_json|>": 5,
    "<|content_model_end_sampling|>": 6,
    "<|content_xml|>": 7,
}
MM, TXT, THINK, END, TOOL, EOS = 1, 2, 3, 4, 5, 6


class FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def run(tokens):
    """Feed (token, segment) pairs; return the visible text."""
    f = InklingStreamFilter(FakeTokenizer(VOCAB))
    return "".join(f.feed(t, s) for t, s in tokens)


def test_detects_inkling_vocab_only():
    assert is_inkling_tokenizer(FakeTokenizer(VOCAB))
    assert not is_inkling_tokenizer(FakeTokenizer({"<think>": 9, "</think>": 10}))


def test_reasoning_then_answer():
    out = run([
        (MM, "<|message_model|>"), (THINK, "<|content_thinking|>"),
        (100, "Let me "), (101, "think."), (END, "<|end_message|>"),
        (MM, "<|message_model|>"), (TXT, "<|content_text|>"),
        (102, "pong"), (END, "<|end_message|>"), (EOS, "<|content_model_end_sampling|>"),
    ])
    assert out == "Let me think." + THINK_CLOSE + "pong"


def test_framing_tokens_never_leak():
    out = run([
        (MM, "<|message_model|>"), (TXT, "<|content_text|>"), (100, "hi"),
        (END, "<|end_message|>"), (EOS, "<|content_model_end_sampling|>"),
    ])
    assert "<|" not in out
    assert out == "hi"


def test_tool_header_name_is_hidden():
    # <|message_model|>get_weather<|content_invoke_tool_json|>{...}<|end_message|>
    out = run([
        (MM, "<|message_model|>"), (200, "get_weather"), (TOOL, "<|content_invoke_tool_json|>"),
        (201, '{"name":"get_weather"}'), (END, "<|end_message|>"),
    ])
    assert out == ""


def test_text_after_a_tool_call_still_shows():
    out = run([
        (MM, "<|message_model|>"), (200, "get_weather"), (TOOL, "<|content_invoke_tool_json|>"),
        (201, "{}"), (END, "<|end_message|>"),
        (MM, "<|message_model|>"), (TXT, "<|content_text|>"), (102, "done"), (END, "<|end_message|>"),
    ])
    assert out == "done"


def test_second_reasoning_block_closes_again():
    out = run([
        (MM, "<|message_model|>"), (THINK, "<|content_thinking|>"), (100, "a"), (END, "<|end_message|>"),
        (MM, "<|message_model|>"), (TXT, "<|content_text|>"), (102, "x"), (END, "<|end_message|>"),
        (MM, "<|message_model|>"), (THINK, "<|content_thinking|>"), (100, "b"), (END, "<|end_message|>"),
    ])
    assert out == "a" + THINK_CLOSE + "x" + "b" + THINK_CLOSE


def test_text_between_messages_is_dropped():
    # A stray token after <|end_message|> and before the next header is not content.
    out = run([(TXT, "<|content_text|>"), (100, "ok"), (END, "<|end_message|>"), (103, "stray")])
    assert out == "ok"


def test_mode_tracks_for_trailing_flush():
    f = InklingStreamFilter(FakeTokenizer(VOCAB))
    f.feed(MM, "")
    f.feed(TXT, "")
    assert f.mode == "text"
    f.feed(END, "")
    assert f.mode == "idle"
