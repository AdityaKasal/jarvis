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


def test_reset_clears_the_turn_but_keeps_the_noise_floor():
    vad = Vad(CONFIG)
    feed(vad, [noise()] * CONFIG.calibration_frames + [speech()] * 10)
    assert vad.speaking
    floor = vad.noise_floor

    vad.reset()
    assert not vad.speaking
    assert vad.audio().size == 0
    # Still calibrated: re-measuring every turn would spend the first
    # calibration_ms of the next turn treating speech as silence.
    assert vad.calibrated
    assert vad.noise_floor == floor


def test_recalibrate_measures_the_room_again():
    vad = Vad(CONFIG)
    feed(vad, [noise()] * CONFIG.calibration_frames)
    assert vad.calibrated
    vad.recalibrate()
    assert not vad.calibrated
    assert vad.noise_floor == 0.0


def test_a_second_turn_does_not_lose_its_opening_word():
    """The bug this guards: reset() used to clear the calibration too, so the
    first calibration_ms of every turn after the first was swallowed as noise
    floor. With a wake word at the front of the sentence, the swallowed part
    is the only part that decides whether Jarvis answers at all."""
    vad = Vad(CONFIG)
    feed(vad, [noise()] * CONFIG.calibration_frames + [speech()] * 20
         + [noise()] * (CONFIG.hangover_frames + 2))

    vad.reset()
    # Second turn starts talking immediately - no silence to calibrate on.
    events = feed(vad, [speech()] * 20)
    assert [kind for _, kind in events] == ["started"]
    # And the opening frames are in the captured audio, not eaten by calibration.
    assert vad.duration_s > 0.2


def test_onset_survives_the_gaps_between_syllables():
    """Real speech does not stay above threshold for 250ms unbroken.

    Measured on actual audio, the longest unbroken run can be barely over
    min_speech_ms even when half of all frames are above threshold. Requiring
    an unbroken run means a clipped sentence never triggers at all.
    """
    vad = Vad(CONFIG)
    # Two-frame dips, the length of a stop consonant, all the way through.
    staccato = []
    for _ in range(12):
        staccato += [speech()] * 3 + [noise()] * 2

    events = feed(vad, [noise()] * CONFIG.calibration_frames + staccato)
    assert [kind for _, kind in events] == ["started"]


def test_a_long_gap_still_abandons_the_onset():
    """The grace period is for syllables, not for a cough and then silence."""
    grace = CONFIG.onset_grace_frames
    vad = Vad(CONFIG)
    frames = ([noise()] * CONFIG.calibration_frames
              + [speech()] * 3            # short of min_speech_frames
              + [noise()] * (grace + 3)   # too long to be a syllable gap
              + [speech()] * 3)           # count restarts, never reaches it
    assert feed(vad, frames) == []
