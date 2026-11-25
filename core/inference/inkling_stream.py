"""Inkling 2's message framing, mapped onto what Mira's clients already parse.

Inkling does not use ``<think>...</think>``. Every assistant turn is a run of
messages, each one a header and a body:

    <|message_model|><|content_thinking|>reasoning<|end_message|>
    <|message_model|><|content_text|>answer<|end_message|>
    <|message_model|>get_weather<|content_invoke_tool_json|>{...}<|end_message|>

Decoded as-is, the framing tokens (and the function name in a tool header) would
land in the visible message. ``InklingStreamFilter`` drops all of that and turns
the end of a reasoning message into ``</think>``. It does not emit an opener,
which matches how Qwen3.6 reaches the clients today (its template pre-opens the
block, so only the closer is ever generated).

Tool-call bodies are not handled here: the engine buffers them separately from
the state machine (see ``mlx_lm.tool_parsers.inkling``).
"""

from __future__ import annotations

# Inkling's reasoning can quote "</think>" and "<think>" verbatim (it is asked about
# Mira's own stripper, among other things). A plain closer would end the block early
# and a quoted opener would swallow the answer, so the real closer carries invisible
# separators the model never writes, and ThinkingStripper(inkling=True) matches only
# that.
CLOSE_MARK = "\u2063</think>\u2063"
THINK_CLOSE = CLOSE_MARK + "\n\n"

MESSAGE = "<|message_model|>"
TEXT = ("<|content_text|>", "<|content_xml|>")
THINKING = "<|content_thinking|>"
TOOL = ("<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>")
END = "<|end_message|>"
# Everything that is framing and never visible, content markers included.
_FRAMING = (
    MESSAGE,
    THINKING,
    END,
    "<|content_model_end_sampling|>",
    "<|content_tool_error|>",
    "<|begin_of_text|>",
    "<|endoftext|>",
    *TEXT,
    *TOOL,
)


def is_inkling_tokenizer(tokenizer) -> bool:
    """True when the vocabulary carries Inkling's message-framing tokens."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:  # noqa: BLE001 - a tokenizer without a vocab is not Inkling
        return False
    return MESSAGE in vocab and THINKING in vocab and END in vocab


class InklingStreamFilter:
    """Per-request filter. ``feed(token, segment)`` returns the visible text."""

    def __init__(self, tokenizer) -> None:
        vocab = tokenizer.get_vocab()
        self._message = vocab[MESSAGE]
        self._thinking = vocab[THINKING]
        self._end = vocab[END]
        self._text = {vocab[t] for t in TEXT if t in vocab}
        self._tool = {vocab[t] for t in TOOL if t in vocab}
        self._framing = {vocab[t] for t in _FRAMING if t in vocab}
        # "idle": between messages. "header": after <|message_model|>, before the
        # content marker (a tool call's function name lives here). "text" and
        # "think" carry visible/reasoning text. "tool" is buffered elsewhere.
        self.mode = "idle"

    def feed(self, token: int, segment: str) -> str:
        if token == self._message:
            self.mode = "header"
            return ""
        if token == self._thinking:
            self.mode = "think"
            return ""
        if token in self._text:
            self.mode = "text"
            return ""
        if token in self._tool:
            self.mode = "tool"
            return ""
        if token == self._end:
            closing_think = self.mode == "think"
            self.mode = "idle"
            return THINK_CLOSE if closing_think else ""
        if token in self._framing:
            return ""
        if self.mode in ("think", "text"):
            return segment
        return ""
