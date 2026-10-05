"""Terminal text chat: `uv run python -m backend chat`."""

from __future__ import annotations

import asyncio

from rich.console import Console

from backend.brain.conversation import Conversation
from backend.brain.llm import LLMError, OllamaClient
from backend.config import Settings
from backend.memory.store import MemoryStore
from backend.tools import build_registry
from backend.tools.geocode import resolve_location
from backend.tools.timers import Timer, TimerManager

console = Console(highlight=False)


def echo(text: str) -> None:
    """Print streamed reply text. soft_wrap: rich would wrap each chunk as if it started a
    line, turning a space into a newline mid-sentence ("São Bernardo do\nCampo")."""
    console.print(text, end="", markup=False, soft_wrap=True)


HELP = "[dim]/new  new conversation · /help  commands · /quit  exit[/]"
QUIT = {"/quit", "/exit", "/q", "/sair"}
NEW = {"/new", "/novo"}
HELP_CMDS = {"/help", "/ajuda"}


async def _once(text: str):
    yield text


async def run_chat(s: Settings, resume: bool = False, speak: bool = False) -> int:
    await resolve_location(s)
    llm = OllamaClient(s.llm)
    speaker = None
    if speak:
        from backend.voice import make_speaker

        speaker = make_speaker(s)
    store = MemoryStore(s.db_path)
    conv_id = store.last_conversation() if resume else None

    def on_timer(timer: Timer) -> None:
        text = timers.announcement(timer)
        console.print(f"\n[bold cyan]{s.assistant_name} ›[/] {text}")
        conv.add_assistant_note(text)
        if speaker:
            asyncio.create_task(speaker.speak(_once(text)))

    timers = TimerManager(s, on_fire=on_timer)
    conv = Conversation(s, llm, store, conv_id, build_registry(s, timers), warm_after_turn=True)

    try:
        with console.status(f"Loading {s.llm.model}..."):
            await conv.prime()
            if speaker:
                await speaker.warmup()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not reach Ollama: {exc}[/]")
        console.print("Run [bold]uv run python -m backend doctor[/] to diagnose.")
        await llm.aclose()
        store.close()
        return 1

    resumed = f" (resumed, {store.count(conv.id)} messages)" if resume and conv_id else ""
    console.print(f"[bold cyan]{s.assistant_name}[/] online · {s.llm.model}{resumed}")
    console.print(HELP)

    try:
        while True:
            try:
                text = (
                    await asyncio.to_thread(console.input, f"\n[bold]{s.locale.you_label} ›[/] ")
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in QUIT:
                break
            if text in NEW:
                conv.reset()
                console.print("[dim]New conversation started.[/]")
                continue
            if text in HELP_CMDS:
                console.print(HELP)
                continue

            console.print(f"[bold cyan]{s.assistant_name} ›[/] ", end="")
            try:
                if speaker:
                    from backend.voice import reply_aloud

                    await reply_aloud(
                        conv,
                        speaker,
                        text,
                        s.tts.sentence_min_chars,
                        on_token=echo,
                    )
                else:
                    async for token in conv.reply(text):
                        echo(token)
            except LLMError as exc:
                console.print(f"\n[red]Model error: {exc}[/]")
                continue
            except KeyboardInterrupt:
                if speaker:
                    speaker.interrupt()
                console.print("\n[dim](interrupted)[/]")
                continue
            st = conv.last_stats
            details = []
            if st.first_token_s is not None:
                details.append(f"first token {st.first_token_s:.2f}s")
            details.append(f"total {st.total_s:.1f}s")
            if st.tokens_per_second:
                details.append(f"{st.tokens_per_second:.0f} tok/s")
            if st.tools:
                details.append(f"tools {', '.join(st.tools)} {st.tools_s:.2f}s")
            console.print(f"\n[dim]{' · '.join(details)}[/]")
    finally:
        timers.cancel_all()
        await llm.aclose()
        store.close()

    addr = f", {s.owner_address}" if s.owner_address else ""
    console.print(f"\n[cyan]{s.assistant_name}:[/] {s.locale.farewell.format(addr=addr)}")
    return 0


async def run_briefing(s: Settings, speak: bool = False) -> int:
    """`python -m backend briefing`: the voice start-up briefing, timed, for tuning it."""
    import time

    from backend.brain import briefing
    from backend.brain.prompts import build_greeting
    from backend.tools.news import NewsService
    from backend.tools.weather import WeatherService

    start = time.perf_counter()
    await resolve_location(s)
    llm = OllamaClient(s.llm)
    store = MemoryStore(s.db_path)
    weather = WeatherService(s) if "weather" in s.tools.enabled else None
    news = NewsService(s) if "news" in s.tools.enabled else None
    conv = Conversation(s, llm, store, tools=build_registry(s, weather=weather, news=news))
    speaker = None
    try:
        with console.status("Fetching weather and news, loading the model..."):
            fetching = asyncio.create_task(briefing.fetch(conv, weather, news))
            await conv.prime()
            primed = time.perf_counter() - start
            data = await fetching
            fetched = time.perf_counter() - start
        console.print(
            f"[dim]weather {'ok' if data.weather else 'missing'} · "
            f"{len(data.news or [])} headlines · model {primed:.2f}s · ready {fetched:.2f}s[/]"
        )
        console.print(f"[bold cyan]{s.assistant_name} ›[/] ", end="")
        times: list[float] = []  # of each piece: greeting and weather (code), then the model

        def on_token(text: str) -> None:
            times.append(time.perf_counter() - start)
            echo(text)

        tokens = briefing.stream(conv, data)
        if speak:
            from backend.voice import make_speaker, speak_tokens

            speaker = make_speaker(s)
            await speaker.warmup()
            await speak_tokens(
                speaker, tokens, s.tts.sentence_min_chars, s.locale.lang, on_token=on_token
            )
        else:
            async for token in tokens:
                on_token(token)
        total = time.perf_counter() - start
        fixed = bool(build_greeting(s)) + bool(briefing.weather_sentence(conv, data))
        model_first = times[fixed] if len(times) > fixed else None
        details = [f"first text {times[0] if times else 0:.2f}s"]
        if model_first is not None:
            details.append(f"model first word {model_first:.2f}s")
        details.append(f"total {total:.2f}s")
        console.print(f"\n[dim]{' · '.join(details)}[/]")
    except LLMError as exc:
        console.print(f"\n[red]Model error: {exc}[/]")
        return 1
    finally:
        await llm.aclose()
        store.close()
    return 0
