"""Voice conversation loop: `uv run python -m backend voice`."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
from rich.console import Console

from backend.audio import earcon
from backend.audio.level import level
from backend.audio.sentences import SentenceSplitter, clean_for_speech
from backend.audio.stt import is_farewell, is_sign_off, is_wake_phrase
from backend.audio.tts import PiperSpeaker, SaySpeaker, Speaker
from backend.audio.wakeword import WakeWordDetector
from backend.brain.conversation import Conversation
from backend.brain.llm import LLMError, OllamaClient
from backend.config import Settings
from backend.hud import HudBus
from backend.memory.store import MemoryStore
from backend.state import State, StateMachine
from backend.tools import Tool, build_registry
from backend.tools.geocode import resolve_location
from backend.tools.timers import TimerManager

console = Console(highlight=False)


def echo(text: str) -> None:
    """Print streamed reply text. soft_wrap: rich would wrap each chunk as if it started a
    line, turning a space into a newline mid-sentence ("São Bernardo do\nCampo")."""
    console.print(text, end="", markup=False, soft_wrap=True)


ECHO_TAIL_S = 0.4  # ignore the mic briefly after speaking so the room echo dies down


def show_tokens(bus: HudBus) -> Callable[[str], None]:
    """Reply text to the terminal and the HUD."""

    def on_token(text: str) -> None:
        echo(text)
        bus.publish({"type": "reply", "delta": text})

    return on_token


def make_speaker(s: Settings) -> Speaker:
    if s.tts.engine == "say":
        return SaySpeaker(s.tts.say_voice)
    return PiperSpeaker(s.voice_path, s.audio.output_device, s.tts.length_scale)


async def speak_tokens(
    speaker: Speaker,
    tokens: AsyncIterator[str],
    min_chars: int,
    lang: str,
    on_token: Callable[[str], None] | None = None,
    on_start: Callable[[], None] | None = None,
) -> bool:
    """Stream generated text into the speaker sentence by sentence."""
    splitter = SentenceSplitter(min_chars)

    async def sentences() -> AsyncIterator[str]:
        async for token in tokens:
            if on_token:
                on_token(token)
            for sentence in splitter.feed(token):
                if spoken := clean_for_speech(sentence, lang):
                    yield spoken
        if (rest := splitter.flush()) and (spoken := clean_for_speech(rest, lang)):
            yield spoken

    return await speaker.speak(sentences(), on_start=on_start)


async def reply_aloud(
    conv: Conversation,
    speaker: Speaker,
    user_text: str,
    min_chars: int,
    on_token: Callable[[str], None] | None = None,
    on_start: Callable[[], None] | None = None,
) -> bool:
    """Stream the LLM reply into the speaker sentence by sentence."""
    return await speak_tokens(
        speaker, conv.reply(user_text), min_chars, conv.s.locale.lang, on_token, on_start
    )


async def speak_briefing(
    s: Settings,
    speaker: Speaker,
    tokens: AsyncIterator[str],
    started_at: float,
    timings: dict[str, float],
    bus: HudBus | None = None,
) -> None:
    """Speak the activation briefing, then print when each part of the startup was ready."""
    bus = bus or HudBus()
    first_audio: float | None = None

    def on_start() -> None:
        nonlocal first_audio
        if first_audio is None:
            first_audio = time.perf_counter()
            bus.publish({"type": "state", "state": State.SPEAKING.value})

    console.print(f"\n[bold cyan]{s.assistant_name} ›[/] ", end="")
    try:
        await speak_tokens(
            speaker,
            tokens,
            s.tts.sentence_min_chars,
            s.locale.lang,
            on_token=show_tokens(bus),
            on_start=on_start,
        )
    finally:
        bus.publish({"type": "reply_end"})
    details = [f"{name} {at:.2f}s" for name, at in timings.items()]
    if first_audio is not None:
        details.insert(0, f"voice {first_audio - started_at:.2f}s after start")
    console.print(f"\n[dim]briefing · {' · '.join(details)}[/]")


def sleep_tool(request_sleep: Callable[[], None]) -> Tool:
    async def go_to_sleep() -> dict:
        request_sleep()
        return {"ok": True, "note": "reply with a brief farewell; you sleep after speaking"}

    return Tool(
        name="go_to_sleep",
        description=(
            "Stop listening until the wake word is said again. Call whenever the user ends "
            "the conversation: says goodbye or thanks you with nothing more to ask ('that's "
            "all', 'é só isso, obrigada', 'por enquanto é só'), or asks you to rest or stand by."
        ),
        handler=go_to_sleep,
        capability={
            "pt": "voltar a dormir quando a conversa terminar",
            "en": "go back to sleep when the conversation is over",
        },
    )


def tools_detail(st) -> str | None:
    if not st.tools:
        return None
    return f"tools {', '.join(st.tools)} {st.tools_s:.2f}s"


class VoiceLoop:
    """Routes microphone frames by state: wake word while sleeping, VAD while listening."""

    def __init__(
        self,
        s: Settings,
        conv: Conversation,
        stt,
        speaker: Speaker,
        vad,
        segmenter,
        wake: WakeWordDetector | None = None,
        chime: Callable[[np.ndarray], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        bus: HudBus | None = None,
    ):
        self.s = s
        self.conv = conv
        self.stt = stt
        self.speaker = speaker
        self.vad = vad
        self.segmenter = segmenter
        self.wake = wake
        self.chime = chime
        self.clock = clock
        self.state = StateMachine(State.SLEEPING if wake else State.LISTENING, clock)
        self.task: asyncio.Task | None = None
        self.mute_until = 0.0
        self.listen_deadline: float | None = None  # None = listen forever
        self._barged_in = False
        self._sleep_requested = False
        self.pending: list[str] = []  # announcements waiting for a quiet moment
        self.bus = bus or HudBus()
        self.bus.on_command = self.command
        self.state.add_listener(self._publish_state)
        self._publish_state(None, self.state.state)

    def _publish_state(self, _: State | None, new: State) -> None:
        self.bus.publish({"type": "state", "state": new.value})

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def _listen_for(self, seconds: float) -> None:
        self.listen_deadline = self.clock() + seconds if self.wake else None

    def _wake_up(self, score: float | None) -> None:
        self.vad.reset()
        self.segmenter.reset()
        self.state.to(State.LISTENING)
        self._listen_for(self.s.wakeword.listen_timeout_s)
        if self.chime:
            self.chime(earcon.WAKE)
        why = "HUD" if score is None else f"wake {score:.2f}"
        console.print(f"\n[cyan]●[/] [dim]listening ({why})[/]")

    def _go_to_sleep(self) -> None:
        self.state.to(State.SLEEPING)
        self.listen_deadline = None
        self.wake.reset()
        if self.chime:
            self.chime(earcon.SLEEP)
        console.print(f'[dim]○ sleeping · say "Hey {self.s.assistant_name}"[/]')

    def request_sleep(self) -> None:
        """Go to sleep once the current reply ends (the `go_to_sleep` tool)."""
        if self.wake:
            self._sleep_requested = True

    def command(self, name: str) -> None:
        """A HUD button: wake up as if called, or go to sleep, cutting any reply short."""
        if not self.wake:
            return
        if name == "wake" and self.state.state is State.SLEEPING:
            self._wake_up(None)
        elif name == "sleep" and self.state.state is not State.SLEEPING:
            if self.busy:
                self._sleep_requested = True
                self.speaker.interrupt()
                self.task.cancel()
            else:
                self._go_to_sleep()

    def announce(self, text: str) -> None:
        """Speak something unprompted as soon as Jarvis is not busy or being spoken to."""
        self.pending.append(text)
        self._maybe_announce()

    def _maybe_announce(self) -> None:
        if not self.pending or self.busy:
            return
        if self.state.state not in (State.SLEEPING, State.LISTENING) or self.segmenter.in_speech:
            return
        texts, self.pending = self.pending, []
        self.state.to(State.SPEAKING)
        self.task = asyncio.create_task(self._speak_announcement(texts))
        self.task.add_done_callback(self._after_turn)

    async def _speak_announcement(self, texts: list[str]) -> None:
        if self.chime:
            self.chime(earcon.ALERT)
            await asyncio.sleep(earcon.duration_s(earcon.ALERT) + 0.1)
        for text in texts:
            console.print(f"\n[bold cyan]{self.s.assistant_name} ›[/] {text}")
            self.conv.add_assistant_note(text)
            self.bus.publish({"type": "reply", "delta": f"{text} "})
        self.bus.publish({"type": "reply_end"})

        async def sentences() -> AsyncIterator[str]:
            for text in texts:
                yield text

        await self.speaker.speak(sentences())

    def brief(self, task: asyncio.Task) -> None:
        """Take the activation briefing, started before the loop existed, as the current turn."""
        self.state.to(State.SPEAKING)
        self.task = task
        task.add_done_callback(self._after_turn)

    def _after_turn(self, _: asyncio.Task) -> None:
        self.state.to(State.LISTENING)
        if self._sleep_requested:
            self._sleep_requested = False
            self._barged_in = False
            self._go_to_sleep()
            return
        if self._barged_in:
            # The user is mid-sentence: keep the utterance that interrupted us.
            self._barged_in = False
            return
        self.mute_until = self.clock() + ECHO_TAIL_S
        self._listen_for(ECHO_TAIL_S + self.s.wakeword.follow_up_s)
        self.vad.reset()
        self.segmenter.reset()

    def on_frame(self, frame: np.ndarray) -> None:
        if self.bus.clients:
            self.bus.publish({"type": "level", "mic": level(frame)})
        if self.pending:
            self._maybe_announce()
        if self.state.state is State.SLEEPING:
            if (score := self.wake.detect(frame)) is not None:
                self._wake_up(score)
            return
        # Not `busy`: the turn's done-callback may still be pending.
        working = self.state.state is not State.LISTENING
        if working and not self.s.audio.barge_in:
            return  # half duplex: don't listen while thinking or speaking
        if not working:
            now = self.clock()
            if now < self.mute_until:
                return
            expired = self.listen_deadline is not None and now >= self.listen_deadline
            if expired and not self.segmenter.in_speech:
                self._go_to_sleep()
                return
        event = self.segmenter.process(frame, self.vad(frame))
        if event.started and self.busy:
            self._barged_in = True
            self.speaker.interrupt()
            self.task.cancel()
            console.print("\n[dim](interrupted)[/]")
        if event.utterance is not None:
            self.state.to(State.THINKING)
            self.task = asyncio.create_task(self.handle(event.utterance, time.perf_counter()))
            self.task.add_done_callback(self._after_turn)

    async def handle(self, audio: np.ndarray, speech_end: float) -> None:
        s = self.s
        t0 = time.perf_counter()
        text = await self.stt.transcribe(audio)
        stt_s = time.perf_counter() - t0
        if text and is_wake_phrase(text, s.assistant_name):
            console.print(f'[dim](just my name: "{text}" · still listening)[/]')
            return
        if not text:
            raw = getattr(self.stt, "last_raw", "")
            if self.wake and is_farewell(raw):
                # Filtered as a Whisper hallucination, but while awake a lone "obrigada" is
                # nearly always a goodbye; if it was noise, we only sleep a little early.
                console.print(f'[dim](farewell "{raw.strip()}" · going to sleep)[/]')
                self.request_sleep()
                return
            reason = f'ignored "{raw}"' if raw.strip() else "no speech recognized"
            console.print(f"[dim]({reason} · stt {stt_s:.2f}s)[/]")
            return
        if is_sign_off(text, s.assistant_name):
            # Decided here, not by the model: qwen3 often replied "I'll rest now" without
            # calling go_to_sleep. The model still says the farewell; we sleep after it.
            self.request_sleep()
        console.print(f"\n[bold]{s.locale.you_label} ›[/] {text}")
        console.print(f"[bold cyan]{s.assistant_name} ›[/] ", end="")
        self.bus.publish({"type": "user", "text": text})

        first_audio: float | None = None

        def on_start() -> None:
            nonlocal first_audio
            first_audio = time.perf_counter()
            if self.state.state is State.THINKING:  # Piper reports from another thread, late
                self.state.to(State.SPEAKING)

        try:
            await reply_aloud(
                self.conv,
                self.speaker,
                text,
                s.tts.sentence_min_chars,
                on_token=show_tokens(self.bus),
                on_start=on_start,
            )
        except LLMError as exc:
            console.print(f"\n[red]Model error: {exc}[/]")
            return
        finally:
            self.bus.publish({"type": "reply_end"})
        st = self.conv.last_stats
        details = [f"stt {stt_s:.2f}s"]
        if st.first_token_s is not None:
            details.append(f"llm first word {st.first_token_s:.2f}s")
        if tools := tools_detail(st):
            details.append(tools)
        if first_audio is not None:
            details.append(f"voice after {first_audio - speech_end:.2f}s")
        if st.tokens_per_second:
            details.append(f"{st.tokens_per_second:.0f} tok/s")
        console.print(f"\n[dim]{' · '.join(details)}[/]")
        self.bus.publish(
            {
                "type": "stats",
                "stt_s": round(stt_s, 2),
                "first_token_s": st.first_token_s and round(st.first_token_s, 2),
                "voice_s": first_audio and round(first_audio - speech_end, 2),
                "tokens_per_second": st.tokens_per_second and round(st.tokens_per_second),
                "tools": st.tools,
            }
        )


async def run_voice(
    s: Settings,
    resume: bool = False,
    wake_word: bool | None = None,
    briefing: bool | None = None,
    started_at: float | None = None,
    hud: bool | None = None,
) -> int:
    from backend.audio.capture import Microphone
    from backend.audio.stt import WhisperSTT
    from backend.audio.vad import SileroVAD, UtteranceSegmenter
    from backend.audio.wakeword import load_wake_model
    from backend.brain import briefing as brief
    from backend.hud.feeds import start_feeds
    from backend.hud.server import HudServer, open_hud
    from backend.tools.news import NewsService
    from backend.tools.weather import WeatherService

    started_at = started_at or time.perf_counter()
    use_wake = s.wakeword.enabled if wake_word is None else wake_word
    use_briefing = s.briefing.enabled if briefing is None else briefing
    use_hud = s.hud.enabled if hud is None else hud

    bus = HudBus()
    bus.publish({"type": "state", "state": "booting"})
    server = HudServer(s, bus) if use_hud else None
    if server and not await server.start():
        server = None
    if server and s.hud.open_browser:
        background = [asyncio.create_task(open_hud(bus, server.url))]
    else:
        background = []

    await resolve_location(s)
    llm = OllamaClient(s.llm)
    store = MemoryStore(s.db_path)
    # The callbacks reach `loop`, created below, only after startup.
    timers = TimerManager(
        s,
        on_fire=lambda t: loop.announce(timers.announcement(t)),
        on_change=lambda: bus.publish({"type": "timers", "items": timers.snapshot()}),
    )
    extra = [sleep_tool(lambda: loop.request_sleep())] if use_wake else []
    # Shared with the briefing, so "more news?" right after it hits the cache.
    weather = WeatherService(s) if "weather" in s.tools.enabled else None
    news = NewsService(s) if "news" in s.tools.enabled else None
    tools = build_registry(s, timers, extra, weather, news)
    conv_id = store.last_conversation() if resume else None
    conv = Conversation(s, llm, store, conv_id, tools, warm_after_turn=True)
    conv.on_tool = lambda name: bus.publish({"type": "tool", "name": name})
    stt = WhisperSTT(s.stt.model, s.stt_language, s.stt_hint)
    speaker = make_speaker(s)
    if isinstance(speaker, PiperSpeaker):
        speaker.on_level = lambda v: bus.publish({"type": "level", "out": v})
    bus.publish(
        {
            "type": "hello",
            "name": s.assistant_name,
            "language": s.language,
            "model": s.llm.model,
            "owner": s.owner_address,
            "place": s.location.display_name,
            "tools": tools.names,
            "wake_word": use_wake,
        }
    )

    mode = "barge-in on" if s.audio.barge_in else "half duplex"
    console.print(f"[bold cyan]{s.assistant_name}[/] · {s.llm.model} · {mode}")
    console.print(f"[dim]Tools: {', '.join(tools.names) or 'none'}[/]")
    if use_wake:
        console.print(f'[dim]Say "Hey {s.assistant_name}" to wake me up. Ctrl+C to quit.[/]')
    else:
        console.print("[dim]Always listening (wake word off). Speak naturally. Ctrl+C to quit.[/]")
    if server:
        console.print(f"[dim]HUD: {server.url}[/]")

    timings: dict[str, float] = {}  # when each part was ready, since start

    async def ready(name: str, work):
        result = await work
        timings[name] = time.perf_counter() - started_at
        return result

    def load_wake() -> WakeWordDetector | None:
        if not use_wake:
            return None
        return WakeWordDetector(
            load_wake_model(s.wakeword_path, s.wakeword_dir), s.wakeword.threshold
        )

    # The briefing needs only the voice and its data: it starts while the language
    # model, Whisper and the wake word are still loading, and the model's part waits
    # for them inside Ollama.
    fetching: asyncio.Task | None = None
    if use_briefing:
        fetching = asyncio.create_task(ready("data", brief.fetch(conv, weather, news)))
    loading = asyncio.gather(
        ready("model", conv.prime()),
        ready("whisper", stt.warmup()),
        ready("wake word", asyncio.to_thread(load_wake)),
    )
    speaking: asyncio.Task | None = None
    try:
        await ready("voice", speaker.warmup())
        if use_briefing:
            tokens = brief.stream(conv, await fetching)
            speaking = asyncio.create_task(
                speak_briefing(s, speaker, tokens, started_at, timings, bus)
            )
            _, _, wake = await loading
        else:
            with console.status("Loading language model, Whisper and voice..."):
                _, _, wake = await loading
    except Exception as exc:  # noqa: BLE001
        for task in (fetching, speaking, loading, *background):
            if task:
                task.cancel()
        if speaking:
            speaker.interrupt()
        console.print(f"\n[red]Startup failed: {exc}[/]")
        console.print("Run [bold]uv run python -m backend doctor[/] to diagnose.")
        if server:
            await server.stop()
        await llm.aclose()
        store.close()
        return 1

    chime = None
    if s.wakeword.chime:
        chime = functools.partial(earcon.play, device=s.audio.output_device)
    loop = VoiceLoop(
        s,
        conv,
        stt,
        speaker,
        SileroVAD(),
        UtteranceSegmenter(s.vad),
        wake=wake,
        chime=chime,
        bus=bus,
    )
    if speaking:
        loop.brief(speaking)
    if server:
        # After the briefing's fetch: the news are then a cache hit.
        background += start_feeds(s, bus, weather, news)
    mic = Microphone(s.audio.input_device)
    mic.start()

    try:
        async for frame in mic.frames():
            loop.on_frame(frame)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        mic.stop()
        timers.cancel_all()
        if loop.busy:
            speaker.interrupt()
            loop.task.cancel()
        for task in background:
            task.cancel()
        if server:
            await server.stop()
        await llm.aclose()
        store.close()

    addr = f", {s.owner_address}" if s.owner_address else ""
    console.print(f"\n[cyan]{s.assistant_name}:[/] {s.locale.farewell.format(addr=addr)}")
    return 0
