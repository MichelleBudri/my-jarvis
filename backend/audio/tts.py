"""Text-to-speech: Piper voices (default) or the macOS `say` command."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np

OnStart = Callable[[], None]


class Speaker(Protocol):
    @property
    def speaking(self) -> bool: ...

    async def warmup(self) -> None: ...

    async def speak(self, sentences: AsyncIterator[str], on_start: OnStart | None = None) -> bool:
        """Speak sentences as they arrive. Returns False if interrupted."""
        ...

    def interrupt(self) -> None: ...


class PiperSpeaker:
    def __init__(
        self,
        voice_path: Path,
        device: str | int | None = None,
        length_scale: float | None = None,
    ) -> None:
        self.voice_path = voice_path
        self.device = device
        self.length_scale = length_scale
        self._voice = None
        self._synth_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._stop = threading.Event()
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def _load(self) -> None:
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(self.voice_path)

    def _synth(self, text: str) -> np.ndarray:
        from piper import SynthesisConfig

        self._load()
        cfg = SynthesisConfig(length_scale=self.length_scale)
        chunks = [c.audio_int16_array for c in self._voice.synthesize(text, syn_config=cfg)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    def _play(self, audio_q: queue.Queue, on_start: OnStart | None) -> bool:
        import sounddevice as sd

        rate = self._voice.config.sample_rate
        block = rate // 20  # 50 ms: small enough for instant interruption
        started = False
        with sd.OutputStream(samplerate=rate, channels=1, dtype="int16", device=self.device) as out:
            while (audio := audio_q.get()) is not None:
                if not started and on_start:
                    on_start()
                    started = True
                for i in range(0, len(audio), block):
                    if self._stop.is_set():
                        out.abort()
                        return False
                    out.write(audio[i : i + block])
        return True

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._synth_pool, self._synth, "ok")

    async def speak(self, sentences: AsyncIterator[str], on_start: OnStart | None = None) -> bool:
        loop = asyncio.get_running_loop()
        self._stop.clear()
        self._speaking = True
        audio_q: queue.Queue = queue.Queue()
        notify = (lambda: loop.call_soon_threadsafe(on_start)) if on_start else None
        await loop.run_in_executor(self._synth_pool, self._load)
        player = loop.run_in_executor(None, self._play, audio_q, notify)
        try:
            # Synthesize sentence N+1 while sentence N is playing.
            async for sentence in sentences:
                if self._stop.is_set():
                    break
                audio = await loop.run_in_executor(self._synth_pool, self._synth, sentence)
                audio_q.put(audio)
        finally:
            audio_q.put(None)
            completed = await player
            self._speaking = False
        return completed and not self._stop.is_set()

    def interrupt(self) -> None:
        self._stop.set()


class SaySpeaker:
    """Fallback using macOS `say`: no downloads, but less natural and not streamed."""

    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice
        self._proc: asyncio.subprocess.Process | None = None
        self._stop = asyncio.Event()
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    async def warmup(self) -> None:
        return None

    async def speak(self, sentences: AsyncIterator[str], on_start: OnStart | None = None) -> bool:
        self._stop.clear()
        self._speaking = True
        voice_args = ["-v", self.voice] if self.voice else []
        try:
            async for sentence in sentences:
                if self._stop.is_set():
                    break
                if on_start:
                    on_start()
                    on_start = None
                self._proc = await asyncio.create_subprocess_exec("say", *voice_args, sentence)
                await self._proc.wait()
        finally:
            self._proc = None
            self._speaking = False
        return not self._stop.is_set()

    def interrupt(self) -> None:
        self._stop.set()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
