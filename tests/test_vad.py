import numpy as np

from backend.audio.stt import is_hallucination
from backend.audio.vad import FRAME_SAMPLES, UtteranceSegmenter, frames_for
from backend.config import VADConfig

CFG = VADConfig(threshold=0.5, start_ms=96, min_speech_ms=300, silence_ms=640, preroll_ms=96)


def run(segmenter, probs):
    events = []
    for i, p in enumerate(probs):
        frame = np.full(FRAME_SAMPLES, i, dtype=np.int16)
        ev = segmenter.process(frame, p)
        if ev.started or ev.utterance is not None:
            events.append((i, ev))
    return events


def test_detects_one_utterance_with_preroll():
    seg = UtteranceSegmenter(CFG)
    probs = [0.0] * 10 + [0.9] * 20 + [0.0] * 30
    events = run(seg, probs)
    assert [e.started for _, e in events] == [True, False]
    audio = events[1][1].utterance
    first_frame_index = audio[0]
    assert first_frame_index < 10  # pre-roll kept the onset of the word
    assert not seg.in_speech


def test_short_blip_is_ignored():
    seg = UtteranceSegmenter(CFG)
    events = run(seg, [0.9] * 4 + [0.0] * 40)
    assert all(e.utterance is None for _, e in events)


def test_hysteresis_keeps_quiet_syllables():
    seg = UtteranceSegmenter(CFG)
    probs = [0.9] * 15 + [0.4] * 25 + [0.9] * 5 + [0.0] * 30
    utterances = [e.utterance for _, e in run(seg, probs) if e.utterance is not None]
    assert len(utterances) == 1


def test_max_length_cuts_the_utterance():
    seg = UtteranceSegmenter(CFG.model_copy(update={"max_utterance_s": 1.0}))
    utterances = [e.utterance for _, e in run(seg, [0.9] * 100) if e.utterance is not None]
    assert utterances and len(utterances[0]) <= frames_for(1000) * FRAME_SAMPLES


def test_whisper_hallucinations_are_filtered():
    assert is_hallucination("Obrigado.")
    assert is_hallucination("Legendas pela comunidade Amara.org")
    assert is_hallucination(" ... ")
    assert not is_hallucination("Obrigado, Jarvis, pode desligar.")


def test_stt_drops_echoed_hint_but_keeps_short_phrases():
    import asyncio

    from backend.audio.stt import WhisperSTT

    stt = WhisperSTT("m", "pt", hint="Olá, Jarvis. Como está o tempo hoje? Quais são as notícias?")
    replies = iter(
        [
            "Olá, Jarvis. Como está o tempo hoje? Quais são as notícias?",
            "Olá, Jarvis.",
            "Como está o tempo hoje?",
            "Olá, Jarvis, que horas são?",
        ]
    )
    stt._transcribe_sync = lambda audio: next(replies)
    audio = np.zeros(16000, dtype=np.int16)
    results = [asyncio.run(stt.transcribe(audio)) for _ in range(4)]
    assert results == ["", "Olá, Jarvis.", "Como está o tempo hoje?", "Olá, Jarvis, que horas são?"]
    assert stt.last_raw == "Olá, Jarvis, que horas são?"
