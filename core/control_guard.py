"""Control-token strings in message text stay plain text. See ControlGuard."""
import re

_PH_OPEN, _PH_CLOSE, _BREAK = "", "", ""


class ControlGuard:
    """Keeps control-token strings in message text as plain text.

    The tokenizer parses the strings it knows (turn markers, think tags, tool-call
    tags) as single control tokens wherever they appear. Text is therefore rewritten
    before the chat template runs, and `encode` turns the rewrite back into the
    ordinary pieces of the original text (never the control token), so the model
    reads a quoted tag as text, unchanged. The strings come from the tokenizer, so
    every model family is covered.

    Three marks, all private-use characters, any of which already in input text is
    replaced first so text can never forge or crash them:
      placeholder  a whole control string
      break        "encode the text on each side separately", placed where a text
                   fragment ends in the start of a control string (or starts with
                   its end), because the template can join the fragment to its own
                   text and complete it after the fact
    """

    _MARK = re.compile(f"{_PH_OPEN}(\\d+){_PH_CLOSE}|{_BREAK}")
    _INPUT_MARKS = re.compile(f"[{_PH_OPEN}{_PH_CLOSE}{_BREAK}]")

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
        self._prefixes = {s[:k] for s in self.strings for k in range(1, len(s))}
        self._suffixes = {s[-k:] for s in self.strings for k in range(2, len(s))}
        self._max_prefix = max((len(s) for s in self.strings), default=0)

    def scrub(self, obj, skip=("role",)):
        """Neutralize every string, and every dict key, inside a message or tool schema."""
        if isinstance(obj, str):
            return self.neutralize(obj)
        if isinstance(obj, dict):
            return {(self.neutralize(k) if isinstance(k, str) else k):
                    (v if k in skip else self.scrub(v, skip)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.scrub(v, skip) for v in obj]
        return obj

    def neutralize(self, text: str) -> str:
        if not self._pattern or not text:
            return text
        text = self._INPUT_MARKS.sub("�", text)
        text = self._pattern.sub(lambda m: f"{_PH_OPEN}{self._index[m.group(0)]}{_PH_CLOSE}", text)
        # A tail that starts a control string could be completed by the template's own
        # text after this fragment; a head that ends one could complete the template's
        # text before it.
        if not text.endswith(_PH_CLOSE):
            for k in range(min(self._max_prefix, len(text)), 0, -1):
                if text[-k:] in self._prefixes:
                    text += _BREAK
                    break
        for k in range(min(self._max_prefix, len(text)), 1, -1):
            if text[:k] in self._suffixes:
                text = _BREAK + text
                break
        return text

    def encode(self, text: str) -> list:
        enc = lambda t: self.tokenizer.encode(t, add_special_tokens=False) if t else []  # noqa: E731
        if not self._pattern or (_PH_OPEN not in text and _BREAK not in text):
            return enc(text)
        ids, pos = [], 0
        for m in self._MARK.finditer(text):
            seg = text[pos:m.start()]
            pos = m.end()
            if m.group(1) is None:  # break
                ids += enc(seg)
                continue
            idx = int(m.group(1))
            if idx >= len(self.strings):
                ids += enc(seg + m.group(0))  # not ours: plain text
                continue
            lit = self.strings[idx]
            # Keep a single leading space attached to the first character, as the
            # tokenizer would have had it, then split so the whole string cannot match.
            if seg.endswith(" "):
                ids += enc(seg[:-1]) + enc(" " + lit[0])
            else:
                ids += enc(seg) + enc(lit[0])
            ids += enc(lit[1:])
        ids += enc(text[pos:])
        return ids
