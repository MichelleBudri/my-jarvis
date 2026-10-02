"""Voice activity detection and utterance segmentation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from backend.config import VADConfig

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # Silero VAD's fixed window at 16 kHz
FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE


def frames_for(ms: float) -> int:
    return max(1, round(ms / FRAME_MS))


@dataclass
class SegmentEvent:
    started: bool = False
    utterance: np.ndarray | None = None


class UtteranceSegmenter:
    """Turns a stream of (frame, speech probability) into complete utterances."""

    def __init__(self, cfg: VADConfig) -> None:
        self.threshold = cfg.threshold
        # Hysteresis: once speaking, a slightly lower probability still counts as speech.
        self.release = max(0.0, cfg.threshold - 0.15)
        self.start_frames = frames_for(cfg.start_ms)
        self.min_speech_frames = frames_for(cfg.min_speech_ms)
        self.silence_frames = frames_for(cfg.silence_ms)
        self.max_frames = frames_for(cfg.max_utterance_s * 1000)
        self._preroll: deque[np.ndarray] = deque(
            maxlen=frames_for(cfg.preroll_ms) + self.start_frames
        )
        self.reset()

    def reset(self) -> None:
        self._preroll.clear()
        self._frames: list[np.ndarray] = []
        self.in_speech = False
        self._run = 0
        self._speech = 0
        self._silence = 0

    def process(self, frame: np.ndarray, prob: float) -> SegmentEvent:
        if not self.in_speech:
            self._preroll.append(frame)
            self._run = self._run + 1 if prob >= self.threshold else 0
            if self._run >= self.start_frames:
                self.in_speech = True
                self._frames = list(self._preroll)
                self._preroll.clear()
                self._speech, self._silence = self._run, 0
                return SegmentEvent(started=True)
            return SegmentEvent()

        self._frames.append(frame)
        if prob >= self.release:
            self._silence = 0
            if prob >= self.threshold:
                self._speech += 1
        else:
            self._silence += 1

        if self._silence >= self.silence_frames or len(self._frames) >= self.max_frames:
            frames, speech, silence = self._frames, self._speech, self._silence
            self.reset()
            if speech < self.min_speech_frames:
                return SegmentEvent()
            keep = len(frames) - max(0, silence - 3)  # trim most of the trailing silence
            return SegmentEvent(utterance=np.concatenate(frames[:keep]))
        return SegmentEvent()


class SileroVAD:
    def __init__(self) -> None:
        from pysilero_vad import SileroVoiceActivityDetector

        self._vad = SileroVoiceActivityDetector()

    def __call__(self, frame: np.ndarray) -> float:
        return float(self._vad(frame.astype(np.int16, copy=False).tobytes()))

    def reset(self) -> None:
        self._vad.reset()
