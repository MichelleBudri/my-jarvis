import asyncio

import numpy as np
import pytest

from backend.audio import earcon
from backend.audio.vad import UtteranceSegmenter
from backend.audio.wakeword import WakeWordDetector
from backend.config import Settings, VADConfig
from backend.state import State
from backend.voice import VoiceLoop

FRAME = np.zeros(512, dtype=np.int16)
FRAME_S = 0.032


class FakeWakeModel:
    """Scores 0.9 on the chunk right after `wake_soon()` is called."""

    def __init__(self):
        self.armed = False

    def wake_soon(self):
        self.armed = True

    def predict(self, x):
        score, self.armed = (0.9 if self.armed else 0.0), False
        return {"hey_jarvis": score}

    def reset(self):
        pass


class FakeVAD:
    def __init__(self):
        self.prob = 0.0

    def __call__(self, frame):
        return self.prob

    def reset(self):
        pass


class FakeSTT:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    async def transcribe(self, audio):
        self.calls += 1
        return self.text


class Harness:
    def __init__(self, text="", wake=True):
        self.s = Settings()
        self.now = 0.0
        self.vad = FakeVAD()
        self.wake_model = FakeWakeModel()
        self.stt = FakeSTT(text)
        self.chimes = []
        self.loop = VoiceLoop(
            self.s,
            conv=None,
            stt=self.stt,
            speaker=None,
            vad=self.vad,
            segmenter=UtteranceSegmenter(VADConfig()),
            wake=WakeWordDetector(self.wake_model, 0.5) if wake else None,
            chime=self.chimes.append,
            clock=lambda: self.now,
        )

    async def frames(self, n, prob=0.0):
        self.vad.prob = prob
        for _ in range(n):
            self.now += FRAME_S
            self.loop.on_frame(FRAME)
            await asyncio.sleep(0)

    @property
    def state(self):
        return self.loop.state.state


def test_sleeps_until_wake_word_then_times_out():
    async def scenario():
        h = Harness()
        assert h.state is State.SLEEPING
        await h.frames(30, prob=0.9)  # speech without the wake word is ignored
        assert h.state is State.SLEEPING and h.stt.calls == 0

        h.wake_model.wake_soon()
        await h.frames(3)
        assert h.state is State.LISTENING
        assert h.chimes == [earcon.WAKE]

        await h.frames(round(h.s.wakeword.listen_timeout_s / FRAME_S) + 1)
        assert h.state is State.SLEEPING
        assert h.chimes[-1] is earcon.SLEEP

    asyncio.run(scenario())


def test_request_then_follow_up_window():
    async def scenario():
        h = Harness(text="Jarvis.")  # only the name: no LLM call, keeps listening
        h.wake_model.wake_soon()
        await h.frames(3)
        await h.frames(20, prob=0.9)
        await h.frames(30)  # 700 ms of silence ends the utterance
        assert h.stt.calls == 1
        await asyncio.sleep(0)
        assert h.state is State.LISTENING
        follow_up = h.s.wakeword.follow_up_s
        assert h.loop.listen_deadline == pytest.approx(h.loop.mute_until + follow_up)

        # Speech that started before the deadline is not cut off by it.
        deadline = h.loop.listen_deadline
        h.now = deadline - 0.2
        await h.frames(20, prob=0.9)
        assert h.now > deadline
        assert h.state is State.LISTENING
        await h.frames(30)
        await asyncio.sleep(0)
        assert h.stt.calls == 2
        await h.frames(round(h.s.wakeword.follow_up_s / FRAME_S) + 20)
        assert h.state is State.SLEEPING

    asyncio.run(scenario())


def test_without_wake_word_always_listens():
    async def scenario():
        h = Harness(text="", wake=False)
        assert h.state is State.LISTENING
        await h.frames(20, prob=0.9)
        await h.frames(30)
        await asyncio.sleep(0)
        await h.frames(1000)
        assert h.stt.calls == 1
        assert h.state is State.LISTENING
        assert h.loop.listen_deadline is None and h.chimes == []

    asyncio.run(scenario())
