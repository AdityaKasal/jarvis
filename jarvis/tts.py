"""Text to speech with ElevenLabs."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings

from jarvis.config import require_env

log = logging.getLogger("jarvis.tts")

# Raw signed 16-bit PCM at the same rate whisper wants, which means no decoding
# step and no ffmpeg. Asking for mp3 here would add a decode to every sentence.
OUTPUT_FORMAT = "pcm_16000"


class Voice:
    """One configured ElevenLabs voice, streamed a sentence at a time."""

    def __init__(self, config: dict[str, Any]):
        tts = config.get("tts", {}) or {}
        # Explicit, not ElevenLabs(): the SDK reads ELEVENLABS_API_KEY into a
        # *default argument*, which is evaluated when the module is imported.
        # Import this module before .env is loaded and the client silently
        # authenticates as nobody, which the API reports as a 401 that looks
        # like a bad key rather than a bad import order.
        self.client = ElevenLabs(api_key=require_env("ELEVENLABS_API_KEY"))
        self.voice_id = tts.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        self.model_id = tts.get("model_id", "eleven_flash_v2_5")
        self.settings = VoiceSettings(
            stability=tts.get("stability", 0.5),
            similarity_boost=tts.get("similarity_boost", 0.75),
            speed=tts.get("speed", 1.0),
        )

    def stream(self, text: str) -> Iterator[bytes]:
        """Yield PCM for one sentence, starting before synthesis finishes."""
        started = time.monotonic()
        first = True
        for chunk in self.client.text_to_speech.stream(
            self.voice_id,
            text=text,
            model_id=self.model_id,
            output_format=OUTPUT_FORMAT,
            voice_settings=self.settings,
        ):
            if first:
                log.debug("first audio byte in %.0fms for %r",
                          (time.monotonic() - started) * 1000, text[:40])
                first = False
            yield chunk

    def voices(self) -> list[tuple[str, str]]:
        """(voice_id, name) for everything in the account."""
        page = self.client.voices.search(page_size=100)
        return [(v.voice_id, v.name) for v in page.voices]
