"""Control-token strings in message text stay plain text. See ControlGuard."""
import re


class ControlGuard:
    """Keeps control-token strings in message text as plain text.

    The tokenizer parses the strings it knows (turn markers, think tags, tool-call
    tags) as single control tokens wherever they appear. Message text is therefore
    rewritten to private placeholders before the chat template runs, and `encode`
    turns each placeholder back into the ordinary pieces of the original string
    (never the control token), so the model reads the tag as text, unchanged.
    The strings come from the tokenizer, so every model family is covered.
    """

    _PLACEHOLDER = re.compile("\ue000(\\d+)\ue001")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        try:
            decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
            found = {getattr(t, "content", str(t)) for t in decoder.values()}
        except Exception:  # noqa: BLE001 - hardening must never break a request
            found = set()
        self.strings = sorted((x for x in found if len(x) >= 3 and x[0] in "<["),
                              key=len, reverse=True)
        self._index = {x: i for i, x in enumerate(self.strings)}
        self._pattern = (re.compile("|".join(re.escape(x) for x in self.strings))
                         if self.strings else None)

    def scrub(self, obj, skip=("role", "type")):
        """Neutralize every string inside a message or tool schema (any nesting)."""
        if isinstance(obj, str):
            return self.neutralize(obj)
        if isinstance(obj, dict):
            return {k: (v if k in skip else self.scrub(v, skip)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.scrub(v, skip) for v in obj]
        return obj

    def neutralize(self, text: str) -> str:
        if not self._pattern or not text:
            return text
        return self._pattern.sub(lambda m: f"\ue000{self._index[m.group(0)]}\ue001", text)

    def encode(self, text: str) -> list:
        enc = lambda t: self.tokenizer.encode(t, add_special_tokens=False)  # noqa: E731
        if not self._pattern or "\ue000" not in text:
            return enc(text)
        ids, pos = [], 0
        for m in self._PLACEHOLDER.finditer(text):
            ids += enc(text[pos:m.start()]) if m.start() > pos else []
            lit = self.strings[int(m.group(1))]
            # Split after the first character so the tokenizer cannot match the whole string.
            ids += enc(lit[0]) + enc(lit[1:])
            pos = m.end()
        if pos < len(text):
            ids += enc(text[pos:])
        return ids
