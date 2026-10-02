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
# "1. ", "2) ", "- " opening a line: news summaries come back as lists despite the prompt.
_LIST_MARKER = re.compile(r"^[ \t]*(?:\d{1,2}[.)]|[-•])[ \t]+", re.MULTILINE)

# Units after a number, spelled out. The prompt asks for this, but tool data
# ("89%", "14°C") often leaks through, and the voice would misread or skip the symbol.
_UNITS = {
    "pt": (
        (r"\s*°\s*C\b", " graus"),
        (r"\s*[°º]", " graus"),
        (r"\s*%", " por cento"),
        (r"\s*km/h\b", " quilômetros por hora"),
        (r"\s*mm\b", " milímetros"),
    ),
    "en": (
        (r"\s*°\s*C\b", " degrees"),
        (r"\s*[°º]", " degrees"),
        (r"\s*%", " per cent"),
        (r"\s*km/h\b", " kilometres an hour"),
        (r"\s*mm\b", " millimetres"),
    ),
}
_UNIT_RES = {
    lang: [(re.compile(r"(?<=\d)" + pattern), word) for pattern, word in rules]
    for lang, rules in _UNITS.items()
}


def spell_units(text: str, lang: str) -> str:
    for pattern, word in _UNIT_RES.get(lang, _UNIT_RES["en"]):
        text = pattern.sub(word, text)
    return text


def clean_for_speech(text: str, lang: str = "pt") -> str:
    """Drop markdown, list markers and emoji that a TTS voice would read aloud; spell out units."""
    text = spell_units(_MARKUP.sub("", _LIST_MARKER.sub("", text)), lang)
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
