"""`python -m backend hud-demo`: the HUD driven by a scripted conversation.

No microphone, models or voice: for working on the interface and for a quick look.
The panels are real (weather, news and system), or fixed with `--sample` (offline,
and the same every time: screenshots).
"""

from __future__ import annotations

import asyncio
import math
import random
import time

from rich.console import Console

from backend.config import Settings
from backend.hud.bus import HudBus
from backend.hud.feeds import start_feeds
from backend.hud.server import HudServer, open_hud
from backend.tools.geocode import resolve_location
from backend.tools.news import NewsService
from backend.tools.weather import WeatherService

console = Console(highlight=False)

FRAME_S = 0.032  # one microphone frame
SCRIPT = {
    "pt": [
        (
            "Me lembre em 10 minutos de tirar o bolo do forno.",
            "set_timer",
            "Pois não. Daqui a 10 minutos eu aviso para tirar o bolo do forno.",
        ),
        (
            "Como está o tempo agora?",
            "get_weather",
            "Agora faz 14 graus, com algumas nuvens, senhora. Amanhã deve chover, "
            "então convém deixar o guarda-chuva à mão.",
        ),
    ],
    "en": [
        (
            "Remind me in 10 minutes to take the cake out.",
            "set_timer",
            "Certainly. I shall remind you to take the cake out in 10 minutes.",
        ),
        (
            "How's the weather right now?",
            "get_weather",
            "It is 14 degrees and partly cloudy, madam. "
            "Light rain is likely tomorrow, so an umbrella would be wise.",
        ),
    ],
}


SAMPLE_PANELS = [
    {
        "type": "weather",
        "place": "London, England, United Kingdom",
        "now": {
            "conditions": "partly cloudy",
            "temperature": 14,
            "feels_like": 13,
            "humidity_percent": 72,
            "wind_kmh": 11,
        },
        "daytime": True,
        "sunrise": "07:07",
        "sunset": "18:29",
        "days": [
            {
                "day": "today",
                "conditions": "overcast",
                "min": 11,
                "max": 17,
                "rain_chance_percent": 20,
            },
            {
                "day": "tomorrow",
                "conditions": "light rain",
                "min": 10,
                "max": 15,
                "rain_chance_percent": 70,
            },
            {
                "day": "Wednesday",
                "conditions": "sunny",
                "min": 9,
                "max": 18,
                "rain_chance_percent": 5,
            },
        ],
    },
    {
        "type": "news",
        "items": [
            {
                "title": "Open-weight models close the gap on reasoning benchmarks",
                "source": "Sample Tech",
                "hours_ago": 1,
            },
            {
                "title": "On-device speech recognition gets faster on laptop chips",
                "source": "Sample Daily",
                "hours_ago": 3,
            },
            {
                "title": "Researchers propose a new test for AI agents that use tools",
                "source": "Sample Review",
                "hours_ago": 6,
            },
            {
                "title": "Small language models find a home in home assistants",
                "source": "Sample Wire",
                "hours_ago": 9,
            },
        ],
    },
    {
        "type": "system",
        "battery": {"percent": 82, "status": "discharging", "on_power_adapter": False},
        "cpu_percent": 18,
        "memory_percent": 54,
        "memory_total_gb": 16.0,
    },
]


class Demo:
    def __init__(self, s: Settings, bus: HudBus) -> None:
        self.s = s
        self.bus = bus
        self.woken = asyncio.Event()
        bus.on_command = self.command

    def command(self, name: str) -> None:
        if name == "wake":
            self.woken.set()

    def state(self, name: str) -> None:
        self.bus.publish({"type": "state", "state": name})

    async def levels(self, key: str, seconds: float, voiced: bool) -> None:
        """Fake loudness: a syllable-like wobble when voiced, room noise otherwise."""
        start = time.monotonic()
        while (t := time.monotonic() - start) < seconds:
            if voiced:
                v = 0.55 + 0.3 * math.sin(t * 17) * math.sin(t * 3.1) + random.uniform(-0.1, 0.1)
            else:
                v = random.uniform(0.05, 0.15)
            self.bus.publish({"type": "level", key: round(max(0.0, min(1.0, v)), 3)})
            await asyncio.sleep(FRAME_S)

    async def turn(self, question: str, tool: str, answer: str) -> None:
        self.state("listening")
        await self.levels("mic", 0.8, voiced=False)
        await self.levels("mic", 1.6, voiced=True)
        await self.levels("mic", 0.6, voiced=False)
        self.state("thinking")
        self.bus.publish({"type": "user", "text": question})
        await asyncio.sleep(0.6)
        self.bus.publish({"type": "tool", "name": tool})
        if tool == "set_timer":
            ends = round((time.time() + 600) * 1000)
            item = {"id": 1, "label": "bolo" if self.s.locale.lang == "pt" else "cake"}
            self.bus.publish(
                {"type": "timers", "items": [{**item, "seconds": 600, "ends_at_ms": ends}]}
            )
        await asyncio.sleep(0.9)
        self.state("speaking")
        words = answer.split(" ")
        talking = asyncio.create_task(self.levels("out", len(words) * 0.22, voiced=True))
        for word in words:
            self.bus.publish({"type": "reply", "delta": f"{word} "})
            await asyncio.sleep(0.22)
        await talking
        self.bus.publish({"type": "level", "out": 0.0})
        self.bus.publish({"type": "reply_end"})
        self.bus.publish(
            {
                "type": "stats",
                "stt_s": 0.31,
                "first_token_s": 0.42,
                "voice_s": 1.12,
                "tokens_per_second": 24,
                "tools": [tool],
            }
        )

    async def run(self) -> None:
        script = SCRIPT.get(self.s.locale.lang, SCRIPT["en"])
        while True:
            self.state("sleeping")
            self.woken.clear()
            try:
                await asyncio.wait_for(self.woken.wait(), 4)
            except TimeoutError:
                pass
            for question, tool, answer in script:
                await self.turn(question, tool, answer)
            self.state("listening")
            await self.levels("mic", 3, voiced=False)


async def run_demo(s: Settings, sample: bool = False) -> int:
    await resolve_location(s)
    bus = HudBus()
    server = HudServer(s, bus)
    if not await server.start():
        console.print(f"[red]Port {s.server.port} is busy: is Jarvis already running?[/]")
        return 1
    bus.publish(
        {
            "type": "hello",
            "name": s.assistant_name,
            "language": s.language,
            "model": s.llm.model,
            "owner": s.owner_address,
            "place": s.location.display_name,
            "tools": ["get_weather", "get_ai_news", "get_system_status", "set_timer"],
            "wake_word": True,
        }
    )
    console.print(f"[bold cyan]HUD demo[/] · {server.url} · Ctrl+C to quit")
    if sample:
        tasks = []
        for panel in SAMPLE_PANELS:
            bus.publish(panel)
    else:
        news = NewsService(s) if s.news.feeds else None
        tasks = start_feeds(s, bus, WeatherService(s), news)
    if s.hud.open_browser:
        tasks.append(asyncio.create_task(open_hud(bus, server.url)))
    try:
        await Demo(s, bus).run()
    except asyncio.CancelledError:
        pass
    finally:
        for task in tasks:
            task.cancel()
        await server.stop()
    return 0
