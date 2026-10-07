import queue
import sys
import types

import numpy as np

from backend.audio import aec as aec_module
from backend.audio.aec import CHUNK, MAX_REFERENCE_S, EchoCanceller, resample
from backend.audio.tts import PiperSpeaker
from backend.audio.vad import FRAME_SAMPLES, SAMPLE_RATE


def test_resample_keeps_duration_and_type():
    audio = (np.sin(np.arange(22050) / 10) * 10000).astype(np.int16)
    out = resample(audio, 22050)
    assert out.dtype == np.int16 and len(out) == 16000
    assert resample(audio[:0], 22050).size == 0
    same = np.arange(10, dtype=np.int16)
    assert resample(same, SAMPLE_RATE) is same


def test_frames_keep_their_size_and_order():
    aec = EchoCanceller()
    out = []
    for _ in range(10):
        out += aec.process(np.zeros(FRAME_SAMPLES, np.int16))
    assert all(len(f) == FRAME_SAMPLES for f in out)
    # Everything comes back except what waits for a full 10 ms chunk or frame.
    assert 10 * FRAME_SAMPLES - len(out) * FRAME_SAMPLES < FRAME_SAMPLES + CHUNK
    aec.close()


def test_reference_queue_is_capped():
    aec = EchoCanceller()
    for _ in range(5):
        aec.play(np.ones(SAMPLE_RATE, np.int16))  # 5 s, never consumed
    assert aec._queued <= MAX_REFERENCE_S * SAMPLE_RATE
    aec.close()


def test_echo_is_removed():
    """Speech-like noise played, heard back 30 ms later through a short room response."""
    rng = np.random.default_rng(1)
    seconds = 4
    envelope = np.repeat(rng.uniform(0.2, 1, seconds * 10), SAMPLE_RATE // 10)
    played = (rng.standard_normal(seconds * SAMPLE_RATE) * 3000 * envelope).astype(np.int16)
    room = np.zeros(400)
    room[0], room[160], room[399] = 0.6, 0.2, 0.05
    delay = int(0.03 * SAMPLE_RATE)
    echo = np.convolve(played.astype(float), room)[: len(played)]
    mic = np.zeros(len(played))
    mic[delay:] = echo[:-delay]
    mic = mic.astype(np.int16)

    aec = EchoCanceller()
    block = SAMPLE_RATE // 20  # the speaker's 50 ms blocks
    cleaned, fed = [], 0
    for i in range(0, len(mic) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        while fed <= i + 2 * block:  # written ~100 ms before it is heard
            aec.play(played[fed : fed + block])
            fed += block
        cleaned += aec.process(mic[i : i + FRAME_SAMPLES])
    aec.close()
    out = np.concatenate(cleaned).astype(float)
    last = slice(-SAMPLE_RATE, None)  # after the filter converged
    erle = 10 * np.log10(np.mean(mic[last].astype(float) ** 2) / (np.mean(out[last] ** 2) + 1))
    assert erle > 15


def test_speaker_reports_what_it_plays_at_16k(monkeypatch):
    written = []

    class FakeStream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, block):
            written.append(len(block))

        def abort(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=FakeStream))
    speaker = PiperSpeaker(voice_path=None)
    speaker._voice = types.SimpleNamespace(config=types.SimpleNamespace(sample_rate=22050))
    heard = []
    speaker.on_audio = heard.append
    audio_q = queue.Queue()
    audio_q.put(("Olá.", np.zeros(22050, np.int16)))  # 1 s
    audio_q.put(None)
    sentences = []
    assert speaker._play(audio_q, None, None, lambda text, s: sentences.append((text, s)))
    assert sentences == [("Olá.", 1.0)]  # for the HUD to write it in step with the voice
    assert sum(written) == 22050
    assert len(heard) == len(written)  # one reference block per block played
    assert abs(sum(len(h) for h in heard) - SAMPLE_RATE) <= len(heard)


def test_echo_canceller_is_optional(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no native library")

    monkeypatch.setattr(aec_module, "EchoCanceller", broken)
    assert aec_module.load_echo_canceller() is None
