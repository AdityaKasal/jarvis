"""Speech to text with faster-whisper."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

log = logging.getLogger("jarvis.stt")

# Whisper was trained on subtitle data, and on silence or noise it falls back to
# what subtitle files are full of. These come back with confident-looking scores,
# so the logprob check below does not catch them - they have to be named.
HALLUCINATIONS = {
    "you", "you.", "thank you.", "thank you", "thanks for watching!",
    "thanks for watching.", "bye.", "bye", "so", ".", "!", "?",
    "subscribe!", "please subscribe.", "[blank_audio]", "(silence)",
    "transcription by castingwords",
}


class Transcriber:
    """faster-whisper, loaded once on first use.

    Model load costs a few seconds and a few hundred MB, so it is deferred:
    `python run.py devices` and `run.py chat` should not pay for it.
    """

    def __init__(self, config: dict[str, Any], sample_rate: int = 16000):
        stt = config.get("stt", {}) or {}
        self.model_name = stt.get("model", "small.en")
        self.device = stt.get("device", "cpu")
        self.compute_type = stt.get("compute_type", "int8")
        self.language = stt.get("language", "en")
        self.beam_size = stt.get("beam_size", 1)
        self.sample_rate = sample_rate
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy; import on demand

            log.info("loading whisper %r (%s/%s)...",
                     self.model_name, self.device, self.compute_type)
            started = time.monotonic()
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            log.info("whisper ready in %.1fs", time.monotonic() - started)
        return self._model

    def warm_up(self) -> None:
        """Load the model and run one tiny inference.

        The first transcription is slower than every later one. Paying that
        during startup is much better than paying it in the silence after the
        user's first sentence.
        """
        silence = np.zeros(self.sample_rate // 2, dtype=np.float32)
        self.transcribe(silence)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe one utterance. Returns "" when there is nothing to hear."""
        if audio.size == 0:
            return ""
        audio = np.asarray(audio, dtype=np.float32)

        started = time.monotonic()
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            # Our own VAD already found the turn boundaries; running whisper's
            # too can trim real speech off a quiet ending.
            vad_filter=False,
            # Each utterance is transcribed independently. Conditioning on the
            # previous one is what makes whisper get stuck repeating a phrase
            # for the rest of a conversation.
            condition_on_previous_text=False,
        )

        kept: list[str] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if seg.no_speech_prob > 0.6:
                log.debug("dropped (no_speech %.2f): %r", seg.no_speech_prob, text)
                continue
            if seg.avg_logprob < -1.0:
                log.debug("dropped (logprob %.2f): %r", seg.avg_logprob, text)
                continue
            if text.lower() in HALLUCINATIONS:
                log.debug("dropped (known hallucination): %r", text)
                continue
            kept.append(text)

        result = " ".join(kept).strip()
        log.debug("transcribed %.1fs of audio in %.2fs: %r",
                  audio.size / self.sample_rate, time.monotonic() - started, result)
        return result
