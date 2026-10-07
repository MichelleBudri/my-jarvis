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
OnLevel = Callable[[float], None]
OnAudio = Callable[[np.ndarray], None]
OnSentence = Callable[[str, float], None]  # text, seconds of speech; on the event loop

SAY_S_PER_CHAR = 0.065  # `say` gives no duration: about this long per character


class Speaker(Protocol):
    # Each sentence as it starts to play, so the HUD can write it in step with the voice.
    on_sentence: OnSentence | None

    @property
    def speaking(self) -> bool: ...

    @property
    def interrupted(self) -> bool:
        """True after `interrupt()`, until the next `speak()`."""
        ...

    async def warmup(self) -> None: ...

    async def speak(self, sentences: AsyncIterator[str], on_start: OnStart | None = None) -> bool:
        """Speak sentences as they arrive. Returns False if interrupted."""
        ...

    async def prepare(self, sentences: list[str]) -> None:
        """Synthesize ahead of time, so speaking them later starts and flows at once."""
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
        self._ready: dict[str, np.ndarray] = {}  # sentences synthesized by `prepare`
        self._stop = threading.Event()
        self._speaking = False
        self.on_level: OnLevel | None = None  # loudness of each block played, for the HUD
        # Each block played, at 16 kHz, as it goes to the speakers: the echo canceller's
        # reference. Called from the audio thread.
        self.on_audio: OnAudio | None = None
        self.on_sentence: OnSentence | None = None

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def interrupted(self) -> bool:
        return self._stop.is_set()

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

    def _play(
        self,
        audio_q: queue.Queue,
        on_start: OnStart | None,
        on_level: OnLevel | None,
        on_sentence: OnSentence | None = None,
    ) -> bool:
        import sounddevice as sd

        from backend.audio.aec import resample
        from backend.audio.level import level
        from backend.audio.vad import SAMPLE_RATE

        rate = self._voice.config.sample_rate
        block = rate // 20  # 50 ms: small enough for instant interruption
        started = False
        on_audio = self.on_audio
        with sd.OutputStream(samplerate=rate, channels=1, dtype="int16", device=self.device) as out:
            while (item := audio_q.get()) is not None:
                sentence, audio = item
                if not started and on_start:
                    on_start()
                    started = True
                if on_sentence:
                    on_sentence(sentence, len(audio) / rate)
                # Resampled per sentence, not per block: no filter edges every 50 ms.
                ref = resample(audio, rate) if on_audio else None
                for i in range(0, len(audio), block):
                    if self._stop.is_set():
                        out.abort()
                        return False
                    if on_level:
                        on_level(level(audio[i : i + block]))
                    if ref is not None:
                        on_audio(ref[i * SAMPLE_RATE // rate : (i + block) * SAMPLE_RATE // rate])
                    out.write(audio[i : i + block])
        if on_level:
            on_level(0.0)
        return True

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._synth_pool, self._synth, "ok")

    async def prepare(self, sentences: list[str]) -> None:
        loop = asyncio.get_running_loop()
        for sentence in sentences:
            if sentence not in self._ready:
                audio = await loop.run_in_executor(self._synth_pool, self._synth, sentence)
                self._ready[sentence] = audio

    async def speak(self, sentences: AsyncIterator[str], on_start: OnStart | None = None) -> bool:
        loop = asyncio.get_running_loop()
        self._stop.clear()
        self._speaking = True
        audio_q: queue.Queue = queue.Queue()
        notify = (lambda: loop.call_soon_threadsafe(on_start)) if on_start else None
        report = None
        if (on_level := self.on_level) is not None:
            report = lambda v: loop.call_soon_threadsafe(on_level, v)  # noqa: E731
        said = None
        if (on_sentence := self.on_sentence) is not None:
            said = lambda text, s: loop.call_soon_threadsafe(on_sentence, text, s)  # noqa: E731
        await loop.run_in_executor(self._synth_pool, self._load)
        player = loop.run_in_executor(None, self._play, audio_q, notify, report, said)
        try:
            # Synthesize sentence N+1 while sentence N is playing.
            async for sentence in sentences:
                if self._stop.is_set():
                    break
                audio = self._ready.pop(sentence, None)
                if audio is None:
                    audio = await loop.run_in_executor(self._synth_pool, self._synth, sentence)
                audio_q.put((sentence, audio))
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
        self.on_sentence: OnSentence | None = None

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def interrupted(self) -> bool:
        return self._stop.is_set()

    async def warmup(self) -> None:
        return None

    async def prepare(self, sentences: list[str]) -> None:
        return None  # `say` cannot synthesize ahead

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
                if self.on_sentence:
                    self.on_sentence(sentence, len(sentence) * SAY_S_PER_CHAR)
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
