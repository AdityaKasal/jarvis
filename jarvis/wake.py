"""Wake word matching, done on the transcript rather than on the audio.

A dedicated wake-word engine (Porcupine, openWakeWord) listens to raw audio and
is far cheaper per second, because it never runs a transcription model. It also
costs a heavy dependency, a model file, and in Porcupine's case a third API
key - on a project whose whole point is that transcription is local and free.

Every utterance already goes through whisper here, so matching on the text it
produces adds nothing to the pipeline. The saving is not CPU - whisper still
runs on the television - it is that nothing reaches Claude, which is where the
money and the absurd answers were going.

Matching is fuzzy on purpose. Speech recognition renders "Jarvis" as "Javis",
"Jarvis,", "jarvis." and worse, and a wake word that only responds to perfect
transcription is a wake word that does not work.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Optional

log = logging.getLogger("jarvis.wake")

# How closely a spoken word must match. 0.78 accepts "javis" and "jarvis," but
# rejects "harvest" and "service", both of which whisper produces from ordinary
# speech and neither of which should wake anything.
SIMILARITY = 0.78

# How far into an utterance the wake word may appear. People say "um, Jarvis,
# what time is it"; nobody says twenty words and then the wake word.
MAX_OFFSET = 3

_PUNCT = re.compile(r"[^\w\s']")


def normalise(word: str) -> str:
    return _PUNCT.sub("", word).strip().lower()


def similar(a: str, b: str) -> bool:
    if a == b:
        return True
    # Length alone rules most pairs out, and is far cheaper than the matcher.
    if abs(len(a) - len(b)) > 3:
        return False
    return SequenceMatcher(None, a, b).ratio() >= SIMILARITY


@dataclass
class WakeWord:
    """Decides whether an utterance was addressed to Jarvis."""

    phrases: list[list[str]]
    enabled: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WakeWord":
        wake = config.get("wake", {}) or {}
        raw = wake.get("words") or ["jarvis"]
        phrases = [[normalise(w) for w in phrase.split() if normalise(w)]
                   for phrase in raw]
        phrases = [p for p in phrases if p]
        # Longest first, so "hey jarvis" is consumed whole rather than leaving
        # a stray "hey" at the front of the question.
        phrases.sort(key=len, reverse=True)
        return cls(phrases=phrases, enabled=bool(wake.get("enabled", True)))

    def find(self, text: str) -> Optional[str]:
        """Return what was said *after* the wake word, or None if absent.

        An empty string is a meaningful result: the wake word was heard on its
        own, so Jarvis should wake up and wait rather than answer nothing.
        """
        if not self.enabled:
            return text

        words = text.split()
        tokens = [normalise(w) for w in words]

        for start in range(min(MAX_OFFSET, len(tokens))):
            for phrase in self.phrases:
                end = start + len(phrase)
                if end > len(tokens):
                    continue
                if all(similar(tokens[start + i], phrase[i])
                       for i in range(len(phrase))):
                    remainder = " ".join(words[end:]).strip()
                    log.debug("woken by %r at %d, remainder %r",
                              " ".join(phrase), start, remainder)
                    # Drop a leading comma left by "Jarvis, what time is it".
                    return remainder.lstrip(",.:;- ").strip()
        return None
