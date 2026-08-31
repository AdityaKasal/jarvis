"""Turning a token stream into things worth speaking.

Two jobs, both about latency and both about the gap between text written for a
screen and text read aloud:

  * `SentenceChunker` cuts the model's output into sentences as they complete,
    so the first one reaches ElevenLabs while Claude is still writing the rest.
    Waiting for the full reply would add its whole generation time to the pause
    before Jarvis says anything.
  * `strip_for_speech` removes markdown. The system prompt asks for none, but
    "asked nicely" is not a guarantee, and a stray asterisk is read aloud.
"""

from __future__ import annotations

import re
from typing import Optional

# A period after one of these is an abbreviation, not the end of a sentence.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "approx", "no", "fig", "inc", "ltd", "co", "dept", "univ", "a.m",
    "p.m", "u.s", "u.k",
}

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s")
# Fall back to a clause boundary when a sentence runs long without punctuation,
# so a rambling reply still starts playing instead of buffering silently.
_CLAUSE_BREAK = re.compile(r"[,;:]\s")
SOFT_LIMIT = 180

_MARKDOWN = [
    (re.compile(r"```.*?```", re.S), " "),      # fenced code
    (re.compile(r"`([^`]*)`"), r"\1"),          # inline code
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),  # links and images
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),     # headings
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),          # bullets
    (re.compile(r"^\s*\d+\.\s+", re.M), ""),          # numbered lists
    (re.compile(r"^\s*>\s?", re.M), ""),              # block quotes
    (re.compile(r"(\*\*|__|\*|_|~~)"), ""),           # emphasis
]

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF️]"
)


def strip_for_speech(text: str) -> str:
    """Remove anything that would be mispronounced rather than spoken."""
    for pattern, replacement in _MARKDOWN:
        text = pattern.sub(replacement, text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _ends_on_abbreviation(chunk: str) -> bool:
    word = re.split(r"[\s(]", chunk.rstrip().rstrip(".!?\"')]"))[-1].lower()
    return word in ABBREVIATIONS


def _is_decimal_point(buffer: str, index: int) -> bool:
    """True for the period in "3.5" - a digit on each side."""
    before = buffer[index - 1] if index > 0 else ""
    after = buffer[index + 1] if index + 1 < len(buffer) else ""
    return before.isdigit() and after.isdigit()


class SentenceChunker:
    """Accumulate streamed text; emit complete sentences as they land."""

    def __init__(self, soft_limit: int = SOFT_LIMIT):
        self.soft_limit = soft_limit
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add streamed text. Returns however many sentences are now complete."""
        self._buffer += text
        out: list[str] = []
        while True:
            piece = self._take_one()
            if piece is None:
                return out
            out.append(piece)

    def flush(self) -> Optional[str]:
        """Emit whatever is left, complete or not. Call at end of turn."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest or None

    def _take_one(self) -> Optional[str]:
        for match in _SENTENCE_END.finditer(self._buffer):
            end = match.end()
            punct_at = match.start()
            if _is_decimal_point(self._buffer, punct_at):
                continue
            candidate = self._buffer[:end]
            if _ends_on_abbreviation(candidate):
                continue
            self._buffer = self._buffer[end:]
            return candidate.strip()

        if len(self._buffer) >= self.soft_limit:
            for match in _CLAUSE_BREAK.finditer(self._buffer):
                if match.end() < self.soft_limit // 2:
                    continue  # too early to be worth cutting
                piece = self._buffer[: match.end()].strip()
                self._buffer = self._buffer[match.end():]
                return piece
        return None
