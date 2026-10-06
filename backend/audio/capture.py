"""Microphone input as an async stream of 16 kHz int16 frames."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from backend.audio.vad import FRAME_SAMPLES, SAMPLE_RATE

log = logging.getLogger(__name__)


class Microphone:
    def __init__(self, device: str | int | None = None, max_frames: int = 256) -> None:
        self.device = device
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=max_frames)
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status) -> None:  # audio thread
            if status:
                log.debug("input status: %s", status)
            loop.call_soon_threadsafe(self._put, indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def _put(self, frame: np.ndarray) -> None:
        if self._queue.full():
            self._queue.get_nowait()  # drop the oldest frame rather than block audio
        self._queue.put_nowait(frame)

    def flush(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            yield await self._queue.get()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SilenceWatch:
    """Notices a microphone that only delivers zeros: macOS does that, with no error, when
    the app running Jarvis was denied microphone access (common when launched at login)."""

    def __init__(self, frames: int = 94) -> None:  # 94 frames of 32 ms = 3 s
        self.frames = frames
        self.seen = 0
        self.done = False

    def __call__(self, frame: np.ndarray) -> bool:
        """True once, when the first `frames` frames were all digital silence."""
        if self.done:
            return False
        if frame.any():
            self.done = True
            return False
        self.seen += 1
        if self.seen >= self.frames:
            self.done = True
            return True
        return False
