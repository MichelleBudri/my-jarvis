"""Split streamed LLM text into sentences that can be spoken one by one."""

from __future__ import annotations

import re
import unicodedata

_BOUNDARY = re.compile(r"[.!?…]+[\"')\]]*\s+|\n+")
_LAST_WORD = re.compile(r"(\w+)$")
_CLAUSE = re.compile(r"[,;:]\s+")
_ABBREVIATIONS = {
    "sr",
    "sra",
    "srta",
    "dr",
    "dra",
    "prof",
    "profa",
    "av",
    "etc",
    "mr",
    "mrs",
    "ms",
    "st",
    "vs",
    "jr",
}
_MARKUP = re.compile(r"[*_#`~>|]+")


def clean_for_speech(text: str) -> str:
    """Drop markdown symbols and emoji that a TTS voice would read aloud."""
    text = _MARKUP.sub("", text)
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Cs"))
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    def __init__(
        self, min_chars: int = 20, first_min_chars: int = 6, first_clause_chars: int = 30
    ) -> None:
        self.min_chars = min_chars
        # A long first sentence may be cut at its first comma so the voice starts sooner.
        self.first_clause_chars = first_clause_chars
        # The first sentence goes out as soon as possible: it sets the perceived latency.
        self.first_min_chars = first_min_chars
        self._emitted = False
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        start = 0
        for m in _BOUNDARY.finditer(self._buf):
            if m.group().startswith("."):
                word = _LAST_WORD.search(self._buf[start : m.start()])
                if word and word.group(1).lower() in _ABBREVIATIONS:
                    continue
            sentence = self._buf[start : m.end()].strip()
            # Very short fragments sound choppy on their own; merge with the next one.
            limit = self.min_chars if self._emitted else self.first_min_chars
            if len(sentence) < limit:
                continue
            out.append(sentence)
            self._emitted = True
            start = m.end()
        self._buf = self._buf[start:]
        if not self._emitted and not out:
            for m in _CLAUSE.finditer(self._buf):
                if m.start() >= self.first_clause_chars:
                    out.append(self._buf[: m.end()].strip())
                    self._buf = self._buf[m.end() :]
                    self._emitted = True
                    break
        return out

    def flush(self) -> str | None:
        rest, self._buf = self._buf.strip(), ""
        return rest or None
