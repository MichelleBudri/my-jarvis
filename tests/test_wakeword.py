import numpy as np
import pytest

from backend.audio import earcon
from backend.audio.stt import is_sign_off, is_wake_phrase
from backend.audio.wakeword import CHUNK_SAMPLES, WakeWordDetector
from backend.state import InvalidTransition, State, StateMachine

FRAME = np.zeros(512, dtype=np.int16)


class FakeModel:
    def __init__(self, scores=()):
        self.scores = list(scores)
        self.chunks = []
        self.resets = 0

    def predict(self, x):
        self.chunks.append(len(x))
        return {"hey_jarvis": self.scores.pop(0) if self.scores else 0.0}

    def reset(self):
        self.resets += 1


def test_detector_regroups_frames_into_80ms_chunks():
    model = FakeModel()
    det = WakeWordDetector(model)
    counts = [len(det.scores(FRAME)) for _ in range(5)]  # 5 × 512 = 2 × 1280
    assert counts == [0, 0, 1, 0, 1]
    assert model.chunks == [CHUNK_SAMPLES, CHUNK_SAMPLES]


def test_detect_fires_once_and_resets():
    model = FakeModel([0.1, 0.9])
    det = WakeWordDetector(model, threshold=0.5)
    results = [det.detect(FRAME) for _ in range(5)]
    assert results == [None, None, None, None, 0.9]
    assert model.resets == 1
    assert len(det._pending) == 0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Jarvis.", True),
        ("Hey, Jarvis!", True),
        ("Ei Jarvis", True),
        ("Olá, Jarvis.", True),
        ("Jarvis, que horas são?", False),
        ("Hey Jarvis, what time is it?", False),
        ("Friday", False),
    ],
)
def test_is_wake_phrase(text, expected):
    assert is_wake_phrase(text) is expected


def test_is_wake_phrase_with_custom_name():
    assert is_wake_phrase("Hey Friday.", "Friday")


def test_earcon_is_short_quiet_int16():
    assert earcon.WAKE.dtype == np.int16
    assert len(earcon.WAKE) / earcon.RATE < 0.2
    assert np.abs(earcon.WAKE).max() < 0.2 * 32767
    assert abs(int(earcon.WAKE[0])) < 100  # fades in, no click


def test_state_machine_notifies_and_validates():
    now = [0.0]
    sm = StateMachine(State.SLEEPING, clock=lambda: now[0])
    seen = []
    sm.add_listener(lambda old, new: seen.append((old, new)))

    now[0] = 2.0
    assert sm.to(State.LISTENING)
    assert not sm.to(State.LISTENING)
    now[0] = 3.5
    assert sm.elapsed == 1.5
    assert seen == [(State.SLEEPING, State.LISTENING)]
    assert sm.to(State.SPEAKING)  # unprompted announcement
    with pytest.raises(InvalidTransition):
        sm.to(State.THINKING)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Obrigada, pode descansar.", True),
        ("Pode dormir, Jarvis.", True),
        ("Ok, é só isso por enquanto.", True),
        ("Valeu, Jarvis, até mais!", True),
        ("Tchau.", True),
        ("That's all for now, thanks.", True),
        ("Pode descansar depois de me dizer o clima.", False),
        ("É só isso que tem de notícia?", False),
        ("Até que horas abre o mercado?", False),
        ("Obrigada.", False),  # handled by is_farewell, only when Whisper drops it
    ],
)
def test_is_sign_off(text, expected):
    assert is_sign_off(text) is expected
