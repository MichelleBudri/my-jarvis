"""Loudness of an audio block on a 0-1 scale, for the HUD."""

from __future__ import annotations

import numpy as np

FLOOR_DB = -60.0  # silence
RANGE_DB = 50.0  # -10 dBFS and up is full scale: speech peaks sit near the top


def level(samples: np.ndarray) -> float:
    """RMS of int16 samples in decibels, mapped to 0 (silence) .. 1 (loud)."""
    if samples.size == 0:
        return 0.0
    x = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))
    db = 20.0 * np.log10(rms + 1e-9)
    return round(min(1.0, max(0.0, (db - FLOOR_DB) / RANGE_DB)), 3)
