"""Voice conversation loop: `uv run python -m backend voice`."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
from rich.console import Console

from backend.audio.sentences import SentenceSplitter, clean_for_speech
from backend.audio.tts import PiperSpeaker, SaySpeaker, Speaker
from backend.brain.conversation import Conversation
from backend.brain.llm import LLMError, OllamaClient
from backend.config import Settings
from backend.memory.store import MemoryStore
from backend.tools.geocode import resolve_location

console = Console(highlight=False)
ECHO_TAIL_S = 0.4  # ignore the mic briefly after speaking so the room echo dies down


def make_speaker(s: Settings) -> Speaker:
    if s.tts.engine == "say":
        return SaySpeaker(s.tts.say_voice)
    return PiperSpeaker(s.voice_path, s.audio.output_device, s.tts.length_scale)


async def reply_aloud(
    conv: Conversation,
    speaker: Speaker,
    user_text: str,
    min_chars: int,
    on_token: Callable[[str], None] | None = None,
    on_start: Callable[[], None] | None = None,
) -> bool:
    """Stream the LLM reply into the speaker sentence by sentence."""
    splitter = SentenceSplitter(min_chars)

    async def sentences() -> AsyncIterator[str]:
        async for token in conv.reply(user_text):
            if on_token:
                on_token(token)
            for sentence in splitter.feed(token):
                if spoken := clean_for_speech(sentence):
                    yield spoken
        if (rest := splitter.flush()) and (spoken := clean_for_speech(rest)):
            yield spoken

    return await speaker.speak(sentences(), on_start=on_start)


class VoiceLoop:
    def __init__(self, s: Settings, conv: Conversation, stt, speaker: Speaker, vad, segmenter):
        self.s = s
        self.conv = conv
        self.stt = stt
        self.speaker = speaker
        self.vad = vad
        self.segmenter = segmenter
        self.task: asyncio.Task | None = None
        self.mute_until = 0.0
        self._barged_in = False

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def _after_turn(self, _: asyncio.Task) -> None:
        if self._barged_in:
            # The user is mid-sentence: keep the utterance that interrupted us.
            self._barged_in = False
            return
        self.mute_until = time.monotonic() + ECHO_TAIL_S
        self.vad.reset()
        self.segmenter.reset()

    def on_frame(self, frame: np.ndarray) -> None:
        if self.busy and not self.s.audio.barge_in:
            return  # half duplex: don't listen while thinking or speaking
        if not self.busy and time.monotonic() < self.mute_until:
            return
        event = self.segmenter.process(frame, self.vad(frame))
        if event.started and self.busy:
            self._barged_in = True
            self.speaker.interrupt()
            self.task.cancel()
            console.print("\n[dim](interrupted)[/]")
        if event.utterance is not None:
            self.task = asyncio.create_task(self.handle(event.utterance, time.perf_counter()))
            self.task.add_done_callback(self._after_turn)

    async def handle(self, audio: np.ndarray, speech_end: float) -> None:
        s = self.s
        t0 = time.perf_counter()
        text = await self.stt.transcribe(audio)
        stt_s = time.perf_counter() - t0
        if not text:
            raw = getattr(self.stt, "last_raw", "")
            reason = f'ignored "{raw}"' if raw.strip() else "no speech recognized"
            console.print(f"[dim]({reason} · stt {stt_s:.2f}s)[/]")
            return
        console.print(f"\n[bold]{s.locale.you_label} ›[/] {text}")
        console.print(f"[bold cyan]{s.assistant_name} ›[/] ", end="")

        first_audio: float | None = None

        def on_start() -> None:
            nonlocal first_audio
            first_audio = time.perf_counter()

        try:
            await reply_aloud(
                self.conv,
                self.speaker,
                text,
                s.tts.sentence_min_chars,
                on_token=lambda t: console.print(t, end="", markup=False),
                on_start=on_start,
            )
        except LLMError as exc:
            console.print(f"\n[red]Model error: {exc}[/]")
            return
        st = self.conv.last_stats
        details = [f"stt {stt_s:.2f}s"]
        if st.first_token_s is not None:
            details.append(f"llm first word {st.first_token_s:.2f}s")
        if first_audio is not None:
            details.append(f"voice after {first_audio - speech_end:.2f}s")
        if st.tokens_per_second:
            details.append(f"{st.tokens_per_second:.0f} tok/s")
        console.print(f"\n[dim]{' · '.join(details)}[/]")


async def run_voice(s: Settings, resume: bool = False) -> int:
    from backend.audio.capture import Microphone
    from backend.audio.stt import WhisperSTT
    from backend.audio.vad import SileroVAD, UtteranceSegmenter

    await resolve_location(s)
    llm = OllamaClient(s.llm)
    store = MemoryStore(s.db_path)
    conv = Conversation(s, llm, store, store.last_conversation() if resume else None)
    stt = WhisperSTT(s.stt.model, s.stt_language, s.stt_hint)
    speaker = make_speaker(s)

    try:
        with console.status("Loading language model, Whisper and voice..."):
            await asyncio.gather(conv.prime(), stt.warmup(), speaker.warmup())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Startup failed: {exc}[/]")
        console.print("Run [bold]uv run python -m backend doctor[/] to diagnose.")
        await llm.aclose()
        store.close()
        return 1

    loop = VoiceLoop(s, conv, stt, speaker, SileroVAD(), UtteranceSegmenter(s.vad))
    mic = Microphone(s.audio.input_device)
    mic.start()
    mode = "barge-in on" if s.audio.barge_in else "half duplex"
    console.print(f"[bold cyan]{s.assistant_name}[/] listening · {s.llm.model} · {mode}")
    console.print("[dim]Speak naturally. Ctrl+C to quit.[/]")

    try:
        async for frame in mic.frames():
            loop.on_frame(frame)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        mic.stop()
        if loop.busy:
            speaker.interrupt()
            loop.task.cancel()
        await llm.aclose()
        store.close()

    addr = f", {s.owner_address}" if s.owner_address else ""
    console.print(f"\n[cyan]{s.assistant_name}:[/] {s.locale.farewell.format(addr=addr)}")
    return 0
