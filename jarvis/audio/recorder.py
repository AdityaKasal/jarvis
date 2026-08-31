"""Microphone capture as a stream of fixed-size frames."""

from __future__ import annotations

import logging
import queue
from typing import Any, Callable, Iterator, Optional

import numpy as np
import sounddevice as sd

from jarvis.audio.vad import Vad, rms

log = logging.getLogger("jarvis.audio")


class Microphone:
    """A live input stream, exposed as an iterator of float32 mono frames.

    PortAudio calls us on its own high-priority thread and will not wait for
    us, so the callback does nothing but hand the frame to a queue. Anything
    slower - VAD, transcription, an API call - happens on the consumer side.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        device: Optional[int | str] = None,
        max_queued_frames: int = 500,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queued_frames)
        self._stream: Optional[sd.InputStream] = None
        self._last_level = 0.0
        self._overflows = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Input overflow means we did not drain the queue fast enough; the
            # audio it reports is already gone, so only the count is useful.
            self._overflows += 1
        frame = indata[:, 0].copy()
        self._last_level = rms(frame)
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def open(self) -> "Microphone":
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._overflows:
            log.debug("microphone dropped %d block(s) to overflow", self._overflows)

    def __enter__(self) -> "Microphone":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def level(self) -> float:
        """RMS of the most recent frame. Read from any thread (barge-in uses it)."""
        return self._last_level

    def drain(self) -> int:
        """Throw away everything buffered. Returns how many frames were dropped.

        Call this before listening: while Jarvis was speaking, the mic kept
        recording, and without a drain the next turn begins by transcribing
        several seconds of Jarvis's own voice.
        """
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped

    def frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Yield frames as they arrive. Yields None-free; blocks between frames."""
        while True:
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                if self._stream is None:
                    return
                continue


def listen(
    mic: Microphone,
    vad: Vad,
    on_speech_start: Optional[Callable[[], Any]] = None,
    timeout_s: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Block until one spoken turn completes; return it as float32 audio.

    Takes a live `Vad` rather than a config so the caller keeps hold of the
    measured noise floor afterwards - barge-in sets its threshold from it.

    Returns None if `timeout_s` elapses with nobody speaking - the caller uses
    that to notice an empty room rather than hanging on the mic forever.
    """
    vad.reset()
    mic.drain()

    waited = 0.0
    frame_s = vad.config.frame_ms / 1000

    for frame in mic.frames():
        event = vad.feed(frame)
        if event == "started":
            log.debug("speech detected (floor %.4f)", vad.noise_floor)
            if on_speech_start is not None:
                on_speech_start()
        elif event == "ended":
            return vad.audio()

        if not vad.speaking:
            waited += frame_s
            if timeout_s is not None and waited >= timeout_s:
                return None
    return None
