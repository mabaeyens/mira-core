"""Streaming thinking-token stripper.

Removes model "thinking" content from a token stream so the UI never sees
reasoning text mixed into the answer. Two formats are handled in sequence:

  * Pass 1 — Qwen3-style ``<think>...</think>`` blocks.
  * Pass 2 — Gemma 4 ``<|channel>thought\\n...<channel|>`` blocks (mlx-lm raw
    stream format).

:class:`ThinkingStripper` is a state machine fed one raw token at a time via
:meth:`feed`, with :meth:`drain` flushing buffered partials at end of stream.
It holds no reference to the orchestrator, network, or model, so the dual-pass
buffering — the part most prone to off-by-one bugs across chunk boundaries — is
directly unit-testable.
"""

from typing import Iterator, Dict

# Gemma 4 thinking token boundaries (mlx-lm raw stream format).
_GEMMA_THINK_OPEN = "<|channel>thought\n"
_GEMMA_THINK_CLOSE = "<channel|>"


def _partial_marker_tail(buf: str, *markers: str) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of any
    marker. Such a tail might be the start of a marker that spans into the next
    stream chunk, so it must be held back rather than emitted. Returns 0 when no
    suffix could begin a marker."""
    best = 0
    for marker in markers:
        for n in range(min(len(buf), len(marker) - 1), 0, -1):
            if buf[-n:] == marker[:n]:
                best = max(best, n)
                break
    return best


class ThinkingStripper:
    """Strips ``<think>`` and Gemma channel-thought blocks from a token stream.

    Feed raw content tokens via :meth:`feed`; both methods yield typed events::

        {"type": "thinking", "content": str}  # reasoning text (hidden answer)
        {"type": "token",    "content": str}  # visible answer text

    ``thinking_chars`` and ``full_content`` accumulate the totals across the
    whole stream and are read by the caller once the stream completes.
    """

    def __init__(self) -> None:
        self._think_buf = ""
        self._in_thinking = False
        self._gemma_buf = ""
        self._in_gemma_thinking = False
        self.thinking_chars = 0
        self.full_content = ""

    def feed(self, raw_token: str) -> Iterator[Dict]:
        """Process one raw content token, yielding any complete events."""
        self._think_buf += raw_token
        yield from self._pass1_strip_think()
        yield from self._pass2_strip_gemma()

    def drain(self) -> Iterator[Dict]:
        """Flush buffered content at end of stream (no partial holds remain)."""
        if self._think_buf:
            if self._in_thinking:
                self.thinking_chars += len(self._think_buf)
                yield {"type": "thinking", "content": self._think_buf}
            else:
                self._gemma_buf += self._think_buf
            self._think_buf = ""
        if self._gemma_buf:
            if self._in_gemma_thinking:
                self.thinking_chars += len(self._gemma_buf)
                yield {"type": "thinking", "content": self._gemma_buf}
            else:
                self.full_content += self._gemma_buf
                yield {"type": "token", "content": self._gemma_buf}
            self._gemma_buf = ""

    # -- internals -----------------------------------------------------------

    def _pass1_strip_think(self) -> Iterator[Dict]:
        """Strip Qwen3 ``<think>...</think>`` blocks; non-thinking text flows to
        ``_gemma_buf`` for pass 2."""
        while self._think_buf:
            if self._in_thinking:
                close = self._think_buf.find("</think>")
                if close == -1:
                    hold = _partial_marker_tail(self._think_buf, "</think>")
                    emit = self._think_buf[:len(self._think_buf) - hold] if hold else self._think_buf
                    if emit:
                        self.thinking_chars += len(emit)
                        yield {"type": "thinking", "content": emit}
                    self._think_buf = self._think_buf[len(self._think_buf) - hold:] if hold else ""
                    break
                thinking_fragment = self._think_buf[:close]
                if thinking_fragment:
                    self.thinking_chars += len(thinking_fragment)
                    yield {"type": "thinking", "content": thinking_fragment}
                self._in_thinking = False
                self._think_buf = self._think_buf[close + len("</think>"):]
            else:
                open_tag = self._think_buf.find("<think>")
                if open_tag == -1:
                    hold = _partial_marker_tail(self._think_buf, "<think>")
                    self._gemma_buf += self._think_buf[:len(self._think_buf) - hold] if hold else self._think_buf
                    self._think_buf = self._think_buf[len(self._think_buf) - hold:] if hold else ""
                    break
                if open_tag > 0:
                    self._gemma_buf += self._think_buf[:open_tag]
                    self._think_buf = self._think_buf[open_tag:]
                else:
                    self._in_thinking = True
                    self._think_buf = self._think_buf[len("<think>"):]

    def _pass2_strip_gemma(self) -> Iterator[Dict]:
        """Strip Gemma 4 ``<|channel>thought\\n...<channel|>`` blocks; emit the
        remainder as visible tokens."""
        while self._gemma_buf:
            if self._in_gemma_thinking:
                close = self._gemma_buf.find(_GEMMA_THINK_CLOSE)
                if close == -1:
                    hold = _partial_marker_tail(self._gemma_buf, _GEMMA_THINK_CLOSE)
                    emit = self._gemma_buf[:len(self._gemma_buf) - hold] if hold else self._gemma_buf
                    if emit:
                        self.thinking_chars += len(emit)
                        yield {"type": "thinking", "content": emit}
                    self._gemma_buf = self._gemma_buf[len(self._gemma_buf) - hold:] if hold else ""
                    break
                thinking_fragment = self._gemma_buf[:close]
                if thinking_fragment:
                    self.thinking_chars += len(thinking_fragment)
                    yield {"type": "thinking", "content": thinking_fragment}
                self._in_gemma_thinking = False
                self._gemma_buf = self._gemma_buf[close + len(_GEMMA_THINK_CLOSE):]
            else:
                open_tag = self._gemma_buf.find(_GEMMA_THINK_OPEN)
                if open_tag == -1:
                    hold = _partial_marker_tail(self._gemma_buf, _GEMMA_THINK_OPEN)
                    emit = self._gemma_buf[:len(self._gemma_buf) - hold] if hold else self._gemma_buf
                    if emit:
                        self.full_content += emit
                        yield {"type": "token", "content": emit}
                    self._gemma_buf = self._gemma_buf[len(self._gemma_buf) - hold:] if hold else ""
                    break
                if open_tag > 0:
                    regular = self._gemma_buf[:open_tag]
                    self.full_content += regular
                    yield {"type": "token", "content": regular}
                    self._gemma_buf = self._gemma_buf[open_tag:]
                else:
                    self._in_gemma_thinking = True
                    self._gemma_buf = self._gemma_buf[len(_GEMMA_THINK_OPEN):]
