"""Wake word detection with openWakeWord ("Hey Jarvis")."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

CHUNK_SAMPLES = 1280  # openWakeWord scores 80 ms of 16 kHz audio at a time


class WakeModel(Protocol):
    def predict(self, x: np.ndarray) -> dict[str, float]: ...

    def reset(self) -> None: ...


class WakeWordDetector:
    """Regroups 32 ms microphone frames into the 80 ms chunks openWakeWord expects."""

    def __init__(self, model: WakeModel, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold
        self._pending = np.zeros(0, dtype=np.int16)

    def scores(self, frame: np.ndarray) -> list[float]:
        """Feed one frame; returns a score for each complete 80 ms chunk (usually 0 or 1)."""
        self._pending = np.concatenate((self._pending, frame.astype(np.int16, copy=False)))
        out = []
        while len(self._pending) >= CHUNK_SAMPLES:
            chunk, self._pending = (
                self._pending[:CHUNK_SAMPLES],
                self._pending[CHUNK_SAMPLES:],
            )
            out.append(max(self.model.predict(chunk).values(), default=0.0))
        return out

    def detect(self, frame: np.ndarray) -> float | None:
        """Returns the score when the wake word is heard, else None."""
        best = max(self.scores(frame), default=0.0)
        if best < self.threshold:
            return None
        self.reset()  # start clean, so one "Hey Jarvis" never fires twice
        return best

    def reset(self) -> None:
        self.model.reset()
        self._pending = np.zeros(0, dtype=np.int16)


def load_wake_model(model_path: Path, models_dir: Path) -> WakeModel:
    from openwakeword.model import Model

    melspec = models_dir / "melspectrogram.onnx"
    embedding = models_dir / "embedding_model.onnx"
    for path in (model_path, melspec, embedding):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run ./scripts/setup.sh")
    return Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        melspec_model_path=str(melspec),
        embedding_model_path=str(embedding),
    )
