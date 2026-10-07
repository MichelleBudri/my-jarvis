"""Speech-to-text with Whisper on Apple Silicon (mlx-whisper)."""

from __future__ import annotations

import asyncio
import gc
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


# Hallucinations that are also real farewells: right after a reply, take them at face value.
_FAREWELLS = {"obrigado", "obrigada", "tchau", "thank you", "bye"}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def is_hallucination(text: str) -> bool:
    norm = _normalize(text)
    return not norm or norm in _HALLUCINATIONS


def is_farewell(text: str) -> bool:
    return _normalize(text) in _FAREWELLS


# The whole utterance must be a sign-off: "thanks, you can rest" yes, "you can rest after
# telling me the weather" no. qwen3 often answered these without calling go_to_sleep.
_SIGN_OFF_FILLER = (
    r"(?:ok|okay|certo|ta bom|tá bom|está bom|esta bom|tudo bem|tudo certo|beleza|perfeito"
    r"|ótimo|otimo|então|entao|muito|obrigad[oa]|valeu|thanks|thank you|great|{name})"
)
_SIGN_OFF = (
    r"pode (?:ir )?(?:descansar|dormir)|vai descansar|vá descansar|pode parar de ouvir"
    r"|(?:é|e|era) (?:só|so) isso|por enquanto (?:é|e) (?:só|so)|(?:é|e) tudo"
    r"|até (?:mais|logo|amanhã|amanha|depois)|tchau"
    r"|that'?s all|that is all|go to sleep|you can (?:rest|sleep)|goodbye|bye"
)


def is_sign_off(text: str, name: str = "Jarvis") -> bool:
    """True when the user is only ending the conversation ("Obrigada, pode descansar.")."""
    names = "|".join(re.escape(n) for n in {"jarvis", _normalize(name)})
    filler = _SIGN_OFF_FILLER.format(name=names)
    tail = r"(?:por (?:enquanto|hoje|agora)|for now|for today)"
    pattern = rf"(?:{filler}\s+)*(?:{_SIGN_OFF})(?:\s+(?:{filler}|{tail}))*"
    return re.fullmatch(pattern, re.sub(r"\s+", " ", _normalize(text))) is not None


_THANKS = re.compile(r"\b(?:obrigad[oa]|valeu|agradeço|agradeco|thanks|thank you|cheers)\b")


def only_thanks(text: str, name: str = "Jarvis") -> bool:
    """True when the whole utterance is a thank-you ("Muito obrigada, Jarvis.")."""
    names = "|".join(re.escape(n) for n in {"jarvis", _normalize(name)})
    words = rf"(?:{_SIGN_OFF_FILLER.format(name=names)}|agradeço|agradeco|cheers|senhor|sir)"
    norm = re.sub(r"\s+", " ", _normalize(text))
    return thanks(text) and re.fullmatch(rf"{words}(?: {words})*", norm) is not None


def thanks(text: str) -> bool:
    """True when the person thanked ("Obrigada, pode descansar")."""
    return _THANKS.search(_normalize(text)) is not None


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
    def __init__(
        self, model: str, language: str, hint: str | None = None, cache_mb: int | None = None
    ) -> None:
        self.model = model
        self.language = language
        self.hint = hint
        self.cache_mb = cache_mb  # cap on the GPU buffers MLX keeps for reuse; None = no cap
        self._capped = False
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

        if self.cache_mb is not None and not self._capped:
            import mlx.core as mx

            mx.set_cache_limit(self.cache_mb * 1024**2)
            self._capped = True

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

    def _unload_sync(self) -> None:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
        gc.collect()
        mx.clear_cache()

    async def unload(self) -> None:
        """Free the model (~500 MB); the next transcription loads it again (~0.3 s)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._unload_sync)

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz int16 audio; returns "" for silence or known hallucinations."""
        text = await self._run(audio.astype(np.float32) / 32768.0)
        self.last_raw = text
        if is_hallucination(text) or echoes_hint(text, self.hint):
            return ""
        return text
