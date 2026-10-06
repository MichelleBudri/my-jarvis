"""Acoustic echo cancellation: removes Jarvis's own voice from the microphone.

WebRTC's AEC3 (through the `livekit` package, which runs it locally) gets every block the
speakers play as a reference and subtracts its echo from the microphone, so the voice
activity detector hears the user and not the speakers: barge-in without headphones.
"""

from __future__ import annotations

import logging
import threading
from math import gcd

import numpy as np

from backend.audio.vad import FRAME_SAMPLES, SAMPLE_RATE

log = logging.getLogger(__name__)

CHUNK = SAMPLE_RATE // 100  # WebRTC processes 10 ms at a time
MAX_REFERENCE_S = 2.0  # more queued than this means playback and capture drifted apart


def resample(audio: np.ndarray, rate: int, to: int = SAMPLE_RATE) -> np.ndarray:
    """int16 audio at `rate` → `to`, filtered (a polyphase low-pass keeps it alias-free)."""
    if rate == to or len(audio) == 0:
        return audio.astype(np.int16, copy=False)
    from scipy.signal import resample_poly

    g = gcd(rate, to)
    out = resample_poly(audio.astype(np.float32), to // g, rate // g)
    return np.clip(out, -32768, 32767).astype(np.int16)


class EchoCanceller:
    """Feed it what is played (`play`, from any thread) and every microphone frame
    (`process`, in order); it returns the cleaned frames, same size, ~10 ms later."""

    def __init__(self, noise_suppression: bool = False) -> None:
        from livekit import rtc

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=True,
            auto_gain_control=False,  # the wake word and VAD thresholds assume raw levels
        )
        self._lock = threading.Lock()
        self._reference: list[np.ndarray] = []  # 16 kHz, waiting to be matched with the mic
        self._queued = 0
        self._mic = np.zeros(0, dtype=np.int16)
        self._out = np.zeros(0, dtype=np.int16)

    def play(self, audio: np.ndarray) -> None:
        """16 kHz int16 audio that is going to the speakers right now."""
        if len(audio) == 0:
            return
        with self._lock:
            self._reference.append(audio.astype(np.int16, copy=False))
            self._queued += len(audio)
            limit = int(MAX_REFERENCE_S * SAMPLE_RATE)
            while self._queued > limit and self._reference:
                self._queued -= len(self._reference.pop(0))

    def _take_reference(self) -> np.ndarray:
        """The next 10 ms of what was played, or silence."""
        out = np.zeros(CHUNK, dtype=np.int16)
        filled = 0
        with self._lock:
            while filled < CHUNK and self._reference:
                head = self._reference[0]
                n = min(CHUNK - filled, len(head))
                out[filled : filled + n] = head[:n]
                filled += n
                if n == len(head):
                    self._reference.pop(0)
                else:
                    self._reference[0] = head[n:]
            self._queued -= filled
        return out

    def _frame(self, audio: np.ndarray):
        return self._rtc.AudioFrame(audio.tobytes(), SAMPLE_RATE, 1, len(audio))

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        """Cleaned microphone frames of FRAME_SAMPLES (zero or one per call, usually)."""
        self._mic = np.concatenate([self._mic, frame])
        cleaned = []
        while len(self._mic) >= CHUNK:
            chunk, self._mic = self._mic[:CHUNK], self._mic[CHUNK:]
            # Reference first: AEC3 expects each capture chunk after the render it echoes.
            self._apm.process_reverse_stream(self._frame(self._take_reference()))
            mic = self._frame(chunk)
            self._apm.process_stream(mic)  # in place
            cleaned.append(np.frombuffer(mic.data, dtype=np.int16).copy())
        if cleaned:
            self._out = np.concatenate([self._out, *cleaned])
        frames = []
        while len(self._out) >= FRAME_SAMPLES:
            frames.append(self._out[:FRAME_SAMPLES])
            self._out = self._out[FRAME_SAMPLES:]
        return frames

    def close(self) -> None:
        """Free the native module now: left to the garbage collector at interpreter exit,
        livekit fails an assertion after its library is gone and prints a traceback."""
        handle = getattr(self._apm, "_ffi_handle", None)
        if handle is not None:
            handle.dispose()


def load_echo_canceller() -> EchoCanceller | None:
    try:
        return EchoCanceller()
    except Exception as exc:  # noqa: BLE001 - optional: Jarvis works without it
        log.warning("Echo cancellation off: %s", exc)
        return None
