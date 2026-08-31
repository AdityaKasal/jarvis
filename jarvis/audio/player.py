"""Speaker playback for the PCM stream ElevenLabs sends back."""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, Optional

import sounddevice as sd

log = logging.getLogger("jarvis.audio")


class Speaker:
    """Interruptible playback of signed 16-bit little-endian mono PCM.

    Writes are blocking and chunked. Blocking is what paces playback to real
    time for free; chunking is what bounds how long an interrupt takes to be
    noticed - a whole ElevenLabs response written in one call could not be
    stopped partway through.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[int | str] = None,
        chunk_ms: int = 40,
    ):
        self.sample_rate = sample_rate
        self.device = device
        # 2 bytes per sample, one channel.
        self._chunk_bytes = max(2, int(sample_rate * chunk_ms / 1000) * 2)
        self._stream: Optional[sd.RawOutputStream] = None
        self._interrupted = threading.Event()

    def open(self) -> "Speaker":
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            device=self.device,
            channels=1,
            dtype="int16",
        )
        self._stream.start()
        return self

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "Speaker":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def interrupt(self) -> None:
        """Ask playback to stop. Safe to call from another thread."""
        self._interrupted.set()

    def reset(self) -> None:
        self._interrupted.clear()

    def play(self, chunks: Iterable[bytes]) -> bool:
        """Play a PCM byte stream. Returns False if it was interrupted.

        Chunk boundaries from the network have nothing to do with frame sizes,
        so bytes are re-packed into fixed writes before they reach PortAudio.
        """
        if self._stream is None:
            raise RuntimeError("Speaker is not open")

        buf = bytearray()
        for chunk in chunks:
            if self._interrupted.is_set():
                self._abort()
                return False
            buf.extend(chunk)
            while len(buf) >= self._chunk_bytes:
                if self._interrupted.is_set():
                    self._abort()
                    return False
                self._stream.write(bytes(buf[: self._chunk_bytes]))
                del buf[: self._chunk_bytes]

        if buf and not self._interrupted.is_set():
            # Pad the tail to a whole number of samples; a stray odd byte would
            # desynchronise every sample after it.
            if len(buf) % 2:
                buf.append(0)
            self._stream.write(bytes(buf))
        return not self._interrupted.is_set()

    def drain(self) -> None:
        """Wait for audio already handed to the device to finish playing.

        `write` returns once PortAudio has *accepted* the samples, not once the
        speaker has played them. Without this the loop starts listening while
        the last word is still in the air, and hears itself.
        """
        if self._stream is None:
            return
        time.sleep(float(self._stream.latency) + 0.05)

    def _abort(self) -> None:
        """Drop buffered audio so an interrupt is heard immediately."""
        if self._stream is None:
            return
        self._stream.abort()
        self._stream.start()
