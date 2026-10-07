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


class FakeSpeaker:
    def __init__(self):
        self.said: list[str] = []
        self.on_sentence = None
        self.interrupted = False

    async def speak(self, sentences, on_start=None):
        async for sentence in sentences:
            if on_start:
                on_start()
            if self.on_sentence:
                self.on_sentence(sentence, 0.5)
            self.said.append(sentence)
        return True

    def interrupt(self):
        pass


class FakeConversation:
    """Replies with fixed text; `on_reply` runs mid-reply, like a tool call would."""

    def __init__(self, text="Até logo.", on_reply=None):
        self.s = Settings()
        self.text, self.on_reply = text, on_reply
        self.notes: list[str] = []
        self.last_stats = type(
            "Stats", (), {"first_token_s": None, "tokens_per_second": None, "tools": []}
        )()

    async def reply(self, user_text):
        if self.on_reply:
            self.on_reply()
        yield self.text

    def add_assistant_note(self, text):
        self.notes.append(text)

    def add_exchange(self, user_text, answer):
        self.notes.append((user_text, answer))


def test_announcement_wakes_up_speaks_and_listens_for_follow_up():
    async def scenario():
        h = Harness()
        h.loop.speaker = FakeSpeaker()
        h.loop.conv = FakeConversation()
        h.loop.announce("Com licença, o timer de 5 minutos terminou.")
        assert h.state is State.SPEAKING
        await h.loop.task
        await asyncio.sleep(0)
        assert h.loop.speaker.said == ["Com licença, o timer de 5 minutos terminou."]
        assert h.loop.conv.notes == h.loop.speaker.said  # kept for follow-up questions
        assert h.chimes == [earcon.ALERT]
        assert h.state is State.LISTENING and h.loop.listen_deadline is not None

    asyncio.run(scenario())


def test_announcement_waits_while_the_user_speaks():
    async def scenario():
        h = Harness()
        h.loop.speaker = FakeSpeaker()
        h.loop.conv = FakeConversation()
        h.wake_model.wake_soon()
        await h.frames(3)
        await h.frames(10, prob=0.9)  # user is mid-sentence
        h.loop.announce("Timer!")
        assert h.state is State.LISTENING and h.loop.pending == ["Timer!"]

    asyncio.run(scenario())


def test_go_to_sleep_after_the_farewell():
    async def scenario():
        h = Harness(text="Preciso sair agora.")  # not a sign-off: the model calls the tool
        h.loop.speaker = FakeSpeaker()
        h.loop.conv = FakeConversation("Às suas ordens.", on_reply=h.loop.request_sleep)
        h.wake_model.wake_soon()
        await h.frames(3)
        await h.frames(20, prob=0.9)
        await h.frames(30)
        await h.loop.task
        await asyncio.sleep(0)
        assert h.loop.speaker.said == ["Às suas ordens."]
        assert h.state is State.SLEEPING and h.chimes[-1] is earcon.SLEEP

    asyncio.run(scenario())


def test_filtered_thank_you_while_awake_goes_to_sleep():
    async def scenario():
        h = Harness(text="")  # Whisper's "Obrigada." is dropped as a likely hallucination
        h.stt.last_raw = " Obrigada."
        h.wake_model.wake_soon()
        await h.frames(3)
        await h.frames(20, prob=0.9)
        await h.frames(30)
        await h.loop.task
        await asyncio.sleep(0)
        assert h.state is State.SLEEPING and h.chimes[-1] is earcon.SLEEP

    asyncio.run(scenario())


def test_sign_off_sleeps_even_if_the_model_skips_the_tool():
    async def scenario():
        h = Harness(text="Obrigada, pode descansar.")
        h.loop.speaker = FakeSpeaker()
        h.loop.conv = FakeConversation("Pode descansar, senhora. Disponha.")  # never asked
        h.wake_model.wake_soon()
        await h.frames(3)
        await h.frames(20, prob=0.9)
        await h.frames(30)
        await h.loop.task
        await asyncio.sleep(0)
        # Answered by code: thanked, so "Por nada"; no echo of the sign-off.
        assert h.loop.speaker.said == ["Por nada. Estarei por aqui."]
        assert h.loop.conv.notes == [("Obrigada, pode descansar.", "Por nada. Estarei por aqui.")]
        assert h.state is State.SLEEPING and h.chimes[-1] is earcon.SLEEP

    asyncio.run(scenario())


def test_sign_off_without_thanks_gets_no_por_nada():
    async def scenario():
        h = Harness(text="Então, está bom. Pode descansar.")
        h.loop.speaker = FakeSpeaker()
        h.loop.conv = FakeConversation("unused")
        h.wake_model.wake_soon()
        await h.frames(3)
        await h.frames(20, prob=0.9)
        await h.frames(30)
        await h.loop.task
        await asyncio.sleep(0)
        assert h.loop.speaker.said == ["Pois não. Estarei por aqui."]
        assert h.state is State.SLEEPING

    asyncio.run(scenario())


def test_hud_gets_each_sentence_as_it_is_spoken_then_where_it_ended():
    from backend.hud.bus import HudBus
    from backend.voice import end_reply, write_as_spoken

    bus = HudBus()
    got = []
    bus.publish = got.append
    speaker = FakeSpeaker()
    speaker.on_sentence = write_as_spoken(bus)

    async def sentences():
        yield "Bom dia."
        yield "Lá fora estão 22 graus."

    asyncio.run(speaker.speak(sentences()))
    end_reply(bus, speaker)
    assert got == [
        {"type": "reply", "delta": "Bom dia. ", "duration_s": 0.5},
        {"type": "reply", "delta": "Lá fora estão 22 graus. ", "duration_s": 0.5},
        {"type": "reply_end", "cut": False},
    ]
    speaker.interrupted = True
    end_reply(bus, speaker)
    assert got[-1] == {"type": "reply_end", "cut": True}


def test_a_lone_thank_you_gets_por_nada_by_code_and_keeps_listening():
    async def scenario():
        h = Harness(text="Muito obrigada, Jarvis.")
        h.loop.speaker = FakeSpeaker()
        h.loop.conv = FakeConversation("Por nada, senhora. Disponha.")
        h.wake_model.wake_soon()
        await h.frames(3)
        await h.frames(20, prob=0.9)
        await h.frames(30)
        await h.loop.task
        await asyncio.sleep(0)
        assert h.loop.speaker.said == ["Por nada."]
        assert h.state is State.LISTENING  # a thank-you is not a goodbye

    asyncio.run(scenario())


def test_only_thanks_needs_the_whole_utterance():
    from backend.audio.stt import only_thanks, thanks

    assert only_thanks("Obrigada, Jarvis.") and only_thanks("Valeu!")
    assert not only_thanks("Obrigada, e que horas são?")
    assert not only_thanks("Obrigada pela ajuda com o timer")
    assert thanks("Muito obrigado, pode descansar") and not thanks("Pode descansar.")
