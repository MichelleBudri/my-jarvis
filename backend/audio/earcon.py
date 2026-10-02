"""Short tones that tell the user Jarvis woke up or went back to sleep."""

from __future__ import annotations

import numpy as np

RATE = 24_000


def tones(freqs: list[float], note_ms: int = 70, volume: float = 0.15) -> np.ndarray:
    """Consecutive sine notes with short fades, so they don't click."""
    n = RATE * note_ms // 1000
    t = np.arange(n) / RATE
    fade = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (RATE * 0.008))
    notes = [np.sin(2 * np.pi * f * t) * fade for f in freqs]
    return (np.concatenate(notes) * volume * 32767).astype(np.int16)


WAKE = tones([660, 880])
SLEEP = tones([880, 587])


def play(audio: np.ndarray, device: str | int | None = None) -> None:
    """Non-blocking playback."""
    import sounddevice as sd

    sd.play(audio, RATE, device=device)
