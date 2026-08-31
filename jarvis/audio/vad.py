"""Voice activity detection - deciding when a spoken turn starts and ends.

Whisper has no notion of turn-taking; it transcribes whatever buffer you hand
it. Something has to decide where that buffer begins and ends, and this is it.

Frames are scored by RMS against a noise floor measured at the start of every
listen, so the thresholds in config.yaml are ratios rather than absolute levels
- the same numbers work at a quiet desk and in a coffee shop.

This is deliberately not webrtcvad or silero-vad. Both are better at telling
speech from non-speech *noise*, and both cost either a C extension build or a
torch download. On a close-talking mic, energy is enough.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def rms(frame: np.ndarray) -> float:
    """Root-mean-square level of a float32 frame, in the 0..1 range."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


@dataclass
class VadConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    start_ratio: float = 3.0
    stop_ratio: float = 1.8
    absolute_floor: float = 0.004
    min_speech_ms: int = 250
    # A dip this short inside the onset does not reset it. Speech is full of
    # gaps - stops, plosives, the seam between words - and requiring the whole
    # min_speech_ms to be *unbroken* means a clipped sentence never triggers at
    # all. Measured on real speech, the longest unbroken run above threshold
    # can be barely over 250ms even when half of all frames are above it.
    onset_grace_ms: int = 100
    silence_hangover_ms: int = 800
    preroll_ms: int = 300
    calibration_ms: int = 400
    max_utterance_s: float = 30.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VadConfig":
        audio = cfg.get("audio", {}) or {}
        vad = cfg.get("vad", {}) or {}
        return cls(
            sample_rate=audio.get("sample_rate", 16000),
            frame_ms=audio.get("frame_ms", 20),
            **{k: v for k, v in vad.items() if k in cls.__dataclass_fields__},
        )

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    def _frames(self, ms: float) -> int:
        return max(1, int(round(ms / self.frame_ms)))

    @property
    def min_speech_frames(self) -> int:
        return self._frames(self.min_speech_ms)

    @property
    def onset_grace_frames(self) -> int:
        return self._frames(self.onset_grace_ms)

    @property
    def hangover_frames(self) -> int:
        return self._frames(self.silence_hangover_ms)

    @property
    def preroll_frames(self) -> int:
        return self._frames(self.preroll_ms)

    @property
    def calibration_frames(self) -> int:
        return self._frames(self.calibration_ms)


class Vad:
    """Streaming turn detector. Feed it frames; it tells you when a turn ends.

    Usage:
        vad = Vad(cfg)
        for frame in mic.frames():
            event = vad.feed(frame)
            if event == "ended":
                audio = vad.audio()
                break
    """

    def __init__(self, config: VadConfig):
        self.config = config
        self._preroll: deque[np.ndarray] = deque(maxlen=config.preroll_frames)
        self.reset()

    def reset(self) -> None:
        self.noise_floor = 0.0
        self.speaking = False
        self.finished = False
        self._calibration: list[float] = []
        self._speech_run = 0
        self._onset_gap = 0
        self._silence_run = 0
        self._frames: list[np.ndarray] = []
        self._preroll.clear()

    @property
    def calibrated(self) -> bool:
        return len(self._calibration) >= self.config.calibration_frames

    @property
    def duration_s(self) -> float:
        return len(self._frames) * self.config.frame_ms / 1000

    def audio(self) -> np.ndarray:
        """The captured utterance, pre-roll included."""
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    def feed(self, frame: np.ndarray) -> Optional[str]:
        """Consume one frame. Returns "started", "ended", or None.

        Terminal: once a turn has ended, further frames are ignored until
        `reset`. Callers that stop at the first "ended" never notice, but a
        detector that reports the same turn ending forever is lying about its
        state, and the barge-in watcher reads that state from another thread.
        """
        if self.finished:
            return None

        level = rms(frame)

        if not self.calibrated:
            self._calibration.append(level)
            self._preroll.append(frame)
            if self.calibrated:
                # Median, not mean: one cough during calibration should not
                # deafen us for the rest of the turn.
                self.noise_floor = float(np.median(self._calibration))
            return None

        cfg = self.config
        start_level = max(self.noise_floor * cfg.start_ratio, cfg.absolute_floor)
        # Once latched onto speech we hold on more easily than we grabbed on;
        # a single threshold would chop the turn at every unvoiced consonant.
        stop_level = max(self.noise_floor * cfg.stop_ratio, cfg.absolute_floor * 0.75)

        if not self.speaking:
            self._preroll.append(frame)
            if level >= start_level:
                self._speech_run += 1
                self._onset_gap = 0
                if self._speech_run >= cfg.min_speech_frames:
                    self.speaking = True
                    self._onset_gap = 0
                    self._silence_run = 0
                    # Seed from the pre-roll ring so the first syllable - which
                    # happened before we crossed the threshold - is not clipped.
                    self._frames = list(self._preroll)
                    return "started"
            elif self._speech_run:
                # Mid-onset dip: hold the count through a short gap, drop it
                # once the gap is long enough that this was not one utterance.
                self._onset_gap += 1
                if self._onset_gap > cfg.onset_grace_frames:
                    self._speech_run = 0
                    self._onset_gap = 0
            else:
                # Track slow drift (fan spinning up, room filling) so a long
                # wait does not end with a stale, over-sensitive floor.
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * level
            return None

        self._frames.append(frame)
        if level >= stop_level:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= cfg.hangover_frames:
                return self._end()

        if self.duration_s >= cfg.max_utterance_s:
            return self._end()
        return None

    def _end(self) -> str:
        self.speaking = False
        self.finished = True
        return "ended"
