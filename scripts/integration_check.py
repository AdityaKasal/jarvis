#!/usr/bin/env python3
"""End-to-end check of the wake word and the follow-up window.

Not part of `pytest`: this one spends real money. It runs the genuine loop -
the VAD, whisper, the wake word, Claude, ElevenLabs - and replaces only the
microphone, feeding it audio directly instead of through the room. That is the
point: an acoustic test needs a quiet room, and rooms have televisions in them.

    python scripts/integration_check.py

Needs macOS `say` to synthesise the spoken prompts. On Linux, substitute
espeak; on Windows, any tool that writes a wav.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run
from jarvis.assistant import Assistant, VoiceSession
from jarvis.config import load_config

SR, FRAME = 16000, 320  # 20ms frames


class FakeMicrophone:
    """Yields room tone until handed something to say.

    Real time, not as fast as possible: the follow-up window is measured in
    wall-clock seconds, so a mic that raced ahead would not test it.
    """

    def __init__(self, floor: float = 0.0012):
        self._speech: queue.Queue[np.ndarray] = queue.Queue()
        self._pending: list[np.ndarray] = []
        self._rng = np.random.default_rng(7)
        self._floor = floor
        self.level = 0.0

    def __enter__(self): return self
    def __exit__(self, *exc): pass
    def open(self): return self
    def close(self): pass
    def drain(self) -> int: return 0

    def say(self, wav: Path, gain: float = 0.06) -> None:
        audio = run.read_wav(wav, SR)
        peak = float(np.percentile(np.abs(audio), 99)) or 1.0
        audio = (audio / peak * gain).astype(np.float32)
        # Trailing silence, so the hangover fires and the turn actually ends.
        self._speech.put(np.concatenate([audio, np.zeros(int(SR * 1.2), np.float32)]))

    def frames(self, timeout: float = 1.0):
        while True:
            if not self._pending:
                try:
                    block = self._speech.get_nowait()
                    self._pending = [block[i:i + FRAME]
                                     for i in range(0, len(block) - FRAME, FRAME)]
                except queue.Empty:
                    pass
            frame = (self._pending.pop(0) if self._pending
                     else (self._rng.standard_normal(FRAME) * self._floor
                           ).astype(np.float32))
            self.level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            time.sleep(FRAME / SR)
            yield frame


def synthesise(text: str, into: Path) -> Path:
    aiff, wav = into.with_suffix(".aiff"), into.with_suffix(".wav")
    subprocess.run(["say", "-r", "160", "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1",
                    str(aiff), str(wav)], check=True, capture_output=True)
    return wav


def main() -> int:
    config = load_config()
    config["assistant"]["greeting"] = ""      # start asleep, without preamble
    session = VoiceSession(Assistant(config), config, console=False)
    mic = FakeMicrophone()
    session.mic = mic

    heard: list[dict] = []

    def watch():
        with session.events.subscribe() as sub:
            for event in sub.stream(timeout=1.0):
                if event is None:
                    continue
                data = event.to_dict()
                if data["type"] in ("turn", "ignored"):
                    heard.append(data)
                    print(f"    {data.get('role', 'IGNORED')}: {data['text'][:80]}",
                          flush=True)

    threading.Thread(target=watch, daemon=True).start()
    threading.Thread(target=session.run, daemon=True).start()

    def replies() -> int:
        return sum(1 for h in heard if h["type"] == "turn"
                   and h.get("role") == "assistant")

    def ignored() -> int:
        return sum(1 for h in heard if h["type"] == "ignored")

    def until(predicate, seconds: float) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.25)
        return False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clips = {
            "wake": synthesise("Jarvis, what is the capital of France?", tmp / "1"),
            "follow": synthesise("And what is its population?", tmp / "2"),
            "late": synthesise("What about the capital of Spain?", tmp / "3"),
        }

        until(lambda: session.state == "asleep", 90)
        # Let calibration finish on room tone. In a real room nobody starts
        # talking in the same millisecond the microphone opens.
        time.sleep(2.0)
        print(f"[0] asleep, floor {session.vad.noise_floor:.4f}")

        failures = []

        print("\n[1] wake word + question -> should answer")
        before = replies()
        time.sleep(1.0)
        mic.say(clips["wake"])
        if not until(lambda: replies() > before, 90):
            failures.append("wake word did not produce an answer")
        until(lambda: session.state in ("listening", "asleep"), 30)
        if not session.awake:
            failures.append("answering did not open the follow-up window")

        print("\n[2] follow-up, no wake word, inside the window -> should answer")
        before, before_ignored = replies(), ignored()
        time.sleep(1.0)
        mic.say(clips["follow"])
        until(lambda: replies() > before or ignored() > before_ignored, 90)
        if replies() == before:
            failures.append("follow-up inside the window was not answered")
        until(lambda: session.state in ("listening", "asleep"), 30)

        print("\n[3] waiting out the window...")
        if not until(lambda: not session.awake, 70):
            failures.append("the follow-up window never expired")

        print("\n[4] no wake word, after expiry -> should be ignored")
        before, before_ignored = replies(), ignored()
        time.sleep(1.0)
        mic.say(clips["late"])
        until(lambda: replies() > before or ignored() > before_ignored, 90)
        time.sleep(2)
        if replies() > before:
            failures.append("answered an unaddressed question after expiry")
        if ignored() == before_ignored:
            failures.append("unaddressed speech after expiry was not reported")

    session.stop()
    time.sleep(1)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("  All four checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
