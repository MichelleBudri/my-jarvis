"""Speech-to-text with Whisper on Apple Silicon (mlx-whisper)."""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Phrases Whisper tends to invent on silence or noise (learned from video subtitles).
_HALLUCINATIONS = {
    "obrigado",
    "obrigada",
    "tchau",
    "legendas pela comunidade amaraorg",
    "legenda adriana zanotto",
    "inscrevase no canal",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "you",
    "bye",
    "subtitles by the amaraorg community",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def is_hallucination(text: str) -> bool:
    norm = _normalize(text)
    return not norm or norm in _HALLUCINATIONS


def echoes_hint(text: str, hint: str | None) -> bool:
    """True when Whisper just repeated (most of) its prompt instead of transcribing."""
    if not hint:
        return False
    t, h = _normalize(text), _normalize(hint)
    return t == h or (len(t) >= 0.6 * len(h) and t in h)


_WAKE_PREFIXES = ("", "hey", "hei", "ei", "ey", "oi", "olá", "ola", "ok", "okay", "hi", "hello")


def is_wake_phrase(text: str, name: str = "Jarvis") -> bool:
    """True when the user only called the assistant ("Hey Jarvis.") without a request."""
    norm = _normalize(text)
    names = {"jarvis", _normalize(name)}
    return any(norm == f"{p} {n}".strip() for p in _WAKE_PREFIXES for n in names)


class WhisperSTT:
    def __init__(self, model: str, language: str, hint: str | None = None) -> None:
        self.model = model
        self.language = language
        self.hint = hint
        self._path: str | None = None
        self.last_raw = ""  # last transcript before filtering, for diagnostics
        # MLX state is not thread-safe: run every call on the same worker thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    def _model_path(self) -> str:
        """Use the local copy when present, so startup needs no network at all."""
        if self._path is None:
            from huggingface_hub import snapshot_download

            try:
                self._path = snapshot_download(self.model, local_files_only=True)
            except Exception:  # noqa: BLE001
                self._path = snapshot_download(self.model)
        return self._path

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path(),
            language=self.language,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.hint,
            verbose=None,
        )
        return str(result.get("text", "")).strip()

    async def _run(self, audio: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe_sync, audio)

    async def warmup(self) -> None:
        await self._run(np.zeros(16_000, dtype=np.float32))

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz int16 audio; returns "" for silence or known hallucinations."""
        text = await self._run(audio.astype(np.float32) / 32768.0)
        self.last_raw = text
        if is_hallucination(text) or echoes_hint(text, self.hint):
            return ""
        return text
