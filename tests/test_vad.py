"""VAD turn detection, driven by synthetic audio."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.audio.vad import Vad, VadConfig, rms

CONFIG = VadConfig(
    sample_rate=16000, frame_ms=20, start_ratio=3.0, stop_ratio=1.8,
    absolute_floor=0.004, min_speech_ms=100, silence_hangover_ms=200,
    preroll_ms=100, calibration_ms=200, max_utterance_s=5.0,
)

RNG = np.random.default_rng(0)


def noise(level: float = 0.002) -> np.ndarray:
    return (RNG.standard_normal(CONFIG.frame_samples) * level).astype(np.float32)


def speech(level: float = 0.08) -> np.ndarray:
    return (RNG.standard_normal(CONFIG.frame_samples) * level).astype(np.float32)


def feed(vad: Vad, frames: list[np.ndarray]) -> list[tuple[int, str]]:
    events = []
    for i, frame in enumerate(frames):
        event = vad.feed(frame)
        if event:
            events.append((i, event))
    return events


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(320, dtype=np.float32)) == 0.0
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_calibration_sets_a_noise_floor():
    vad = Vad(CONFIG)
    for _ in range(CONFIG.calibration_frames):
        vad.feed(noise())
    assert vad.calibrated
    assert 0 < vad.noise_floor < 0.01


def test_detects_a_turn_and_ends_on_silence():
    vad = Vad(CONFIG)
    frames = ([noise()] * CONFIG.calibration_frames
              + [noise()] * 5
              + [speech()] * 25
              + [noise()] * (CONFIG.hangover_frames + 2))

    events = feed(vad, frames)
    kinds = [kind for _, kind in events]
    assert kinds == ["started", "ended"]

    # The turn must contain the speech, and must start before it: the pre-roll
    # is the whole reason the first syllable survives.
    audio = vad.audio()
    assert audio.size > 25 * CONFIG.frame_samples


def test_ignores_a_single_loud_click():
    vad = Vad(CONFIG)
    frames = ([noise()] * CONFIG.calibration_frames
              + [speech(0.5)]          # one frame: a keyboard tap, not a word
              + [noise()] * 20)
    assert feed(vad, frames) == []
    assert not vad.speaking


def test_stays_latched_through_a_short_pause():
    """A gap shorter than the hangover is a pause, not the end of a turn."""
    vad = Vad(CONFIG)
    pause = CONFIG.hangover_frames - 1
    frames = ([noise()] * CONFIG.calibration_frames
              + [speech()] * 10
              + [noise()] * pause
              + [speech()] * 10
              + [noise()] * (CONFIG.hangover_frames + 2))

    kinds = [kind for _, kind in feed(vad, frames)]
    assert kinds == ["started", "ended"]


def test_loud_room_needs_a_louder_voice():
    """Thresholds are ratios, so a noisy room raises the bar rather than
    triggering on the noise itself."""
    vad = Vad(CONFIG)
    for _ in range(CONFIG.calibration_frames):
        vad.feed(noise(0.02))
    assert feed(vad, [noise(0.02)] * 30) == []
    assert feed(vad, [speech(0.3)] * 10)[0][1] == "started"


def test_gives_up_on_an_endless_utterance():
    vad = Vad(CONFIG)
    frames = ([noise()] * CONFIG.calibration_frames
              + [speech()] * int(CONFIG.max_utterance_s * 1000 / CONFIG.frame_ms + 10))
    kinds = [kind for _, kind in feed(vad, frames)]
    assert kinds == ["started", "ended"]
    assert vad.duration_s == pytest.approx(CONFIG.max_utterance_s, abs=0.1)


def test_reset_clears_state_between_turns():
    vad = Vad(CONFIG)
    feed(vad, [noise()] * CONFIG.calibration_frames + [speech()] * 10)
    assert vad.speaking
    vad.reset()
    assert not vad.speaking
    assert not vad.calibrated
    assert vad.audio().size == 0
