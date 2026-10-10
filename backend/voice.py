"""Voice conversation loop: `uv run python -m backend voice`."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
from rich.console import Console

from backend.audio import earcon
from backend.audio.level import level
from backend.audio.sentences import SentenceSplitter, clean_for_speech
from backend.audio.stt import is_farewell, is_sign_off, is_wake_phrase, only_thanks, thanks
from backend.audio.tts import PiperSpeaker, SaySpeaker, Speaker
from backend.audio.wakeword import WakeWordDetector
from backend.brain.conversation import Conversation
from backend.brain.llm import LLMError, OllamaClient
from backend.config import Settings
from backend.hud import HudBus
from backend.memory.store import MemoryStore
from backend.resources import Resources
from backend.state import State, StateMachine
from backend.tools import Tool, build_registry
from backend.tools.geocode import resolve_location
from backend.tools.timers import TimerManager

console = Console(highlight=False)
log = logging.getLogger(__name__)


def echo(text: str) -> None:
    """Print streamed reply text. soft_wrap: rich would wrap each chunk as if it started a
    line, turning a space into a newline mid-sentence ("São Bernardo do\nCampo")."""
    console.print(text, end="", markup=False, soft_wrap=True)


FOOTPRINT_SETTLE_S = 2  # after an unload or reload, before measuring it for the HUD
ECHO_TAIL_S = 0.4  # ignore the mic briefly after speaking so the room echo dies down


def write_as_spoken(bus: HudBus) -> Callable[[str, float], None]:
    """A speaker's `on_sentence`: the HUD writes each sentence while it is heard, not as
    the model generates it (much faster than speech) (D-42)."""

    def on_sentence(text: str, seconds: float) -> None:
        bus.publish({"type": "reply", "delta": f"{text} ", "duration_s": round(seconds, 2)})

    return on_sentence


def end_reply(bus: HudBus, speaker: Speaker) -> None:
    """Interrupted, the HUD stops writing where the voice stopped; else it writes the rest."""
    bus.publish({"type": "reply_end", "cut": bool(getattr(speaker, "interrupted", False))})


def make_speaker(s: Settings) -> Speaker:
    if s.tts.engine == "say":
        return SaySpeaker(s.tts.say_voice)
    return PiperSpeaker(s.voice_path, s.audio.output_device, s.tts.length_scale)


def speech_sentences(text: str, min_chars: int, lang: str) -> list[str]:
    """The sentences `speak_tokens` makes of `text` arriving at once, to synthesize ahead."""
    splitter = SentenceSplitter(min_chars)
    pieces = [*splitter.feed(text), splitter.flush()]
    return [spoken for p in pieces if p and (spoken := clean_for_speech(p, lang))]


async def _once(text: str) -> AsyncIterator[str]:
    yield text


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
            on_token=echo,
            on_start=on_start,
        )
    finally:
        end_reply(bus, speaker)
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


def autostart_switch(s: Settings, bus: HudBus) -> None:
    """The HUD's "launch at login" switch: show its state and flip it on request."""
    import sys

    from backend import autostart

    if sys.platform != "darwin":
        return
    bus.publish({"type": "autostart", "enabled": autostart.enabled()})
    tasks: set[asyncio.Task] = set()

    async def flip(on: bool) -> None:
        try:
            now = await asyncio.to_thread(autostart.set_enabled, s, on)
            bus.publish({"type": "autostart", "enabled": now})
            console.print(f"\n[dim]Launch at login {'on' if now else 'off'} (from the HUD)[/]")
        except Exception as exc:  # noqa: BLE001
            log.exception("Launch at login not changed")
            bus.publish({"type": "autostart", "enabled": autostart.enabled(), "error": str(exc)})

    def on_request(value) -> None:
        task = asyncio.create_task(flip(bool(value)))
        tasks.add(task)  # a reference, or the task may be collected mid-way
        task.add_done_callback(tasks.discard)

    bus.handle("autostart", on_request)


def export_button(s: Settings, bus: HudBus, store: MemoryStore, conv: Conversation) -> None:
    """The HUD's "export" button: the conversation as Markdown in Downloads, shown in Finder."""
    from backend.memory.export import export_conversation

    tasks: set[asyncio.Task] = set()

    async def export() -> None:
        try:
            path = await asyncio.to_thread(export_conversation, s, store, conv.id)
        except OSError as exc:
            log.exception("Conversation not exported")
            bus.publish({"type": "exported", "error": str(exc)})
            return
        console.print(f"\n[dim]Conversation exported: {path}[/]")
        bus.publish({"type": "exported", "path": str(path), "name": path.name})
        with contextlib.suppress(OSError):
            await asyncio.create_subprocess_exec("open", "-R", str(path))

    def on_request(_) -> None:
        task = asyncio.create_task(export())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    bus.handle("export", on_request)


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

        async def sentences() -> AsyncIterator[str]:
            for text in texts:
                yield text

        try:
            await self.speaker.speak(sentences())
        finally:
            end_reply(self.bus, self.speaker)

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

    async def _say_by_code(self, user_text: str, signing_off: bool) -> None:
        """Fixed lines for a sign-off or a lone thank-you: the model echoed the sign-off
        ("Pode descansar, senhora") and said "Por nada, senhora. Disponha." (D-43)."""
        s = self.s
        addr = f", {s.owner_address}" if s.owner_address else ""
        if not signing_off:
            line = s.locale.welcome
        else:
            line = s.locale.sign_off_thanked if thanks(user_text) else s.locale.sign_off
        text = line.format(addr=addr)
        echo(text)
        self.conv.add_exchange(user_text, text)

        def on_start() -> None:
            if self.state.state is State.THINKING:
                self.state.to(State.SPEAKING)

        try:
            await self.speaker.speak(_once(text), on_start=on_start)
        finally:
            end_reply(self.bus, self.speaker)
        console.print()

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
        signing_off = is_sign_off(text, s.assistant_name)
        if signing_off:
            # Decided here, not by the model: qwen3 often replied "I'll rest now" without
            # calling go_to_sleep. We sleep after the farewell.
            self.request_sleep()
        console.print(f"\n[bold]{s.locale.you_label} ›[/] {text}")
        console.print(f"[bold cyan]{s.assistant_name} ›[/] ", end="")
        self.bus.publish({"type": "user", "text": text})
        if s.locale.native_prompt and (signing_off or only_thanks(text, s.assistant_name)):
            await self._say_by_code(text, signing_off)
            return

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
                on_token=echo,
                on_start=on_start,
            )
        except LLMError as exc:
            console.print(f"\n[red]Model error: {exc}[/]")
            return
        finally:
            end_reply(self.bus, self.speaker)
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


OLLAMA_WAIT_AT_LOGIN_S = 120
WINDOW_QUIT = "hud window quit"  # cancel message: Cmd+Q in the HUD window ends Jarvis
MIC_DENIED = (
    "The microphone delivers only silence: macOS has probably denied access to the app "
    "running Jarvis. Allow it in System Settings → Privacy & Security → Microphone."
)


async def run_voice(s: Settings, at_login: bool = False, **kwargs) -> int:
    """`voice`, one instance at a time. At login, an instance already open is not an error."""
    from backend.instance import InstanceLock

    lock = InstanceLock(s.data_path / "jarvis.lock")
    if not lock.acquire():
        pid = lock.owner()
        running = f" (pid {pid})" if pid else ""
        console.print(f"[yellow]{s.assistant_name} is already running{running}.[/]")
        console.print("[dim]Launched at login? `uv run python -m backend autostart stop`[/]")
        return 0 if at_login else 1
    try:
        return await _run_voice(s, at_login=at_login, **kwargs)
    except asyncio.CancelledError as exc:
        if WINDOW_QUIT not in exc.args:
            raise
        return 0  # quit from the HUD window while loading: exit 0, so launchd lets it be
    finally:
        lock.release()


async def _run_voice(
    s: Settings,
    resume: bool = False,
    wake_word: bool | None = None,
    briefing: bool | None = None,
    started_at: float | None = None,
    hud: bool | None = None,
    at_login: bool = False,
) -> int:
    from backend.audio.capture import Microphone, SilenceWatch
    from backend.audio.stt import WhisperSTT
    from backend.audio.vad import SileroVAD, UtteranceSegmenter
    from backend.audio.wakeword import load_wake_model
    from backend.boot import BootProgress, wait_for_internet
    from backend.brain import briefing as brief
    from backend.hud.feeds import footprint_panel, start_feeds
    from backend.hud.server import HudServer, open_hud
    from backend.tools.news import NewsService
    from backend.tools.weather import WeatherService

    started_at = started_at or time.perf_counter()
    use_wake = s.wakeword.enabled if wake_word is None else wake_word
    use_briefing = s.briefing.enabled if briefing is None else briefing
    use_hud = s.hud.enabled if hud is None else hud

    bus = HudBus()
    bus.publish({"type": "state", "state": "booting"})
    autostart_switch(s, bus)
    server = HudServer(s, bus) if use_hud else None
    if server and not await server.start():
        server = None
    window = None
    if server and s.hud.window:
        from backend.app_bundle import launcher
        from backend.hud.window import HudWindow

        window = HudWindow(server.url, s.assistant_name, *launcher(s))
        # Cmd+Q in the window quits Jarvis: the main loop below ends as with Ctrl+C.
        main_task = asyncio.current_task()
        window.on_quit = lambda: main_task.cancel(WINDOW_QUIT)
    if server and s.hud.open_on_start:
        background = [asyncio.create_task(open_hud(bus, server.url, window))]
    else:
        background = []

    steps = ["ollama", "location", "voice", "whisper", "wake", "model"]
    if use_briefing:
        steps[2:2] = ["internet", "weather", "news"]
        steps += ["briefing", "speech"]
    boot = BootProgress(bus, steps)

    # The profile decides how long Ollama keeps the model (D-37).
    llm = OllamaClient(s.llm.model_copy(update={"keep_alive": s.resources.keep_alive}))
    boot.start("ollama")
    if at_login and not await llm.wait_ready(0):
        console.print("[dim]Waiting for Ollama to start...[/]")
        if not await llm.wait_ready(OLLAMA_WAIT_AT_LOGIN_S):
            console.print(f"[red]Ollama did not answer at {s.llm.host}[/]")
            for task in background:
                task.cancel()
            if window:
                await window.close()
            if server:
                await server.stop()
            await llm.aclose()
            return 1
    boot.done("ollama")
    # Before the model: the system prompt names the place (cached after the first run).
    await boot.track("location", resolve_location(s))
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
    tools = build_registry(s, timers, extra, weather, news, store)
    conv_id = store.last_conversation() if resume else None
    conv = Conversation(s, llm, store, conv_id, tools, warm_after_turn=True)
    conv.on_tool = lambda name: bus.publish({"type": "tool", "name": name})
    export_button(s, bus, store, conv)
    stt = WhisperSTT(s.stt.model, s.stt_language, s.stt_hint, s.resources.mlx_cache_mb)
    speaker = make_speaker(s)
    speaker.on_sentence = write_as_spoken(bus)
    if isinstance(speaker, PiperSpeaker):
        speaker.on_level = lambda v: bus.publish({"type": "level", "out": v})
    aec = None
    if s.audio.barge_in and s.audio.echo_cancellation:
        if isinstance(speaker, PiperSpeaker):
            from backend.audio.aec import load_echo_canceller

            aec = load_echo_canceller()
            if aec:
                speaker.on_audio = aec.play
        else:
            console.print("[yellow]Echo cancellation needs the Piper voice: use headphones[/]")
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

    mode = "half duplex"
    if s.audio.barge_in:
        mode = "barge-in on · echo cancellation" if aec else "barge-in on (headphones)"
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
        result = await boot.track(name, work)
        timings[name] = time.perf_counter() - started_at
        return result

    def load_wake() -> WakeWordDetector | None:
        if not use_wake:
            return None
        return WakeWordDetector(
            load_wake_model(s.wakeword_path, s.wakeword_dir), s.wakeword.threshold
        )

    async def gather_data() -> brief.BriefingData:
        """Weather and news, after waiting for the network (Wi-Fi joins late at login)."""
        boot.start("internet")
        if not await wait_for_internet(s.briefing.network_wait_s):
            for name in ("internet", "weather", "news"):
                boot.set(name, "offline")
            return brief.BriefingData()
        boot.done("internet")
        data = await brief.fetch(conv, weather, news, on_step=boot.set)
        timings["data"] = time.perf_counter() - started_at
        return data

    # Everything loads at once behind the loading screen; the briefing is then written
    # and synthesized in full, so at 100% it is spoken without a pause (D-38).
    fetching = asyncio.create_task(gather_data()) if use_briefing else None
    loading = asyncio.gather(
        ready("voice", speaker.warmup()),
        ready("model", conv.prime()),
        ready("whisper", stt.warmup()),
        ready("wake", asyncio.to_thread(load_wake)),
    )
    speaking: asyncio.Task | None = None
    briefing_text = ""
    try:
        with console.status("Loading language model, Whisper and voice..."):
            _, _, _, wake = await loading
            if fetching:
                data = await fetching
                briefing_text = await ready("briefing", brief.compose(conv, data))
                sentences = speech_sentences(briefing_text, s.tts.sentence_min_chars, s.locale.lang)
                await ready("speech", speaker.prepare(sentences))
    except Exception as exc:  # noqa: BLE001
        for task in (fetching, loading, *background):
            if task:
                task.cancel()
        console.print(f"\n[red]Startup failed: {exc}[/]")
        console.print("Run [bold]uv run python -m backend doctor[/] to diagnose.")
        if window:
            await window.close()
        if server:
            await server.stop()
        await llm.aclose()
        store.close()
        if aec:
            aec.close()
        return 1
    boot.finish()
    if briefing_text:
        speaking = asyncio.create_task(
            speak_briefing(s, speaker, _once(briefing_text), started_at, timings, bus)
        )

    chime = None
    if s.wakeword.chime:
        on_audio = aec.play if aec else None
        chime = functools.partial(earcon.play, device=s.audio.output_device, on_audio=on_audio)
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

    async def show_footprint() -> None:
        await asyncio.sleep(FOOTPRINT_SETTLE_S)
        with contextlib.suppress(Exception):
            bus.publish(await footprint_panel(s))

    def on_resources(_: bool) -> None:
        if server:
            background.append(asyncio.create_task(show_footprint()))

    resources = Resources(s, conv, stt, on_change=on_resources)
    loop.state.add_listener(resources.on_state)
    if loop.state.state is State.SLEEPING:
        resources.on_state(None, State.SLEEPING)
    if speaking:
        loop.brief(speaking)
    if server:
        # After the briefing's fetch: the news are then a cache hit.
        background += start_feeds(s, bus, weather, news)
    mic = Microphone(s.audio.input_device)
    mic.start()
    silence = SilenceWatch()

    try:
        async for frame in mic.frames():
            if silence(frame):
                console.print(f"[red]{MIC_DENIED}[/]")
            for cleaned in aec.process(frame) if aec else (frame,):
                loop.on_frame(cleaned)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        mic.stop()
        timers.cancel_all()
        resources.close()
        if loop.busy:
            speaker.interrupt()
            loop.task.cancel()
        for task in background:
            task.cancel()
        if window:
            await window.close()
        if server:
            await server.stop()
        await llm.aclose()
        store.close()
        if aec:
            aec.close()

    addr = f", {s.owner_address}" if s.owner_address else ""
    console.print(f"\n[cyan]{s.assistant_name}:[/] {s.locale.farewell.format(addr=addr)}")
    return 0
