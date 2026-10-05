"""Panel data refreshed in the background: weather, AI news and the Mac's vitals."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from backend.config import Settings
from backend.hud.bus import HudBus
from backend.tools.news import NewsService
from backend.tools.registry import ToolError
from backend.tools.system import system_status
from backend.tools.weather import CACHE_S, WeatherService

log = logging.getLogger(__name__)

SYSTEM_EVERY_S = 10
RETRY_AFTER_S = 60  # a source that failed is tried again sooner than its usual period
WEATHER_DAYS = 3
NEWS_ITEMS = 5


def weather_panel(forecast: dict[str, Any]) -> dict[str, Any]:
    now = forecast.get("now") or {}
    keep = ("day", "weekday", "conditions", "min", "max", "rain_chance_percent")
    days = [{k: d.get(k) for k in keep} for d in forecast.get("days") or []]
    today = (forecast.get("days") or [{}])[0]
    return {
        "type": "weather",
        "place": forecast.get("place"),
        "now": {
            k: now.get(k)
            for k in ("conditions", "temperature", "feels_like", "humidity_percent", "wind_kmh")
        },
        "daytime": now.get("daytime"),
        "sunrise": today.get("sunrise"),
        "sunset": today.get("sunset"),
        "days": days,
    }


def news_panel(found: dict[str, Any]) -> dict[str, Any]:
    items = [
        {"title": i["title"], "source": i["source"], "hours_ago": i.get("hours_ago")}
        for i in found.get("items") or []
    ]
    return {"type": "news", "items": items}


def system_panel(status: dict[str, Any]) -> dict[str, Any]:
    battery = status.get("battery")
    memory = status.get("memory") or {}
    free = memory.get("free_percent")
    return {
        "type": "system",
        "battery": battery if isinstance(battery, dict) else None,
        "cpu_percent": status.get("cpu_load_percent"),
        "memory_percent": None if free is None else 100 - free,
        "memory_total_gb": memory.get("total_gb"),
    }


async def _every(
    name: str, period_s: float, fetch: Callable[[], Awaitable[dict]], bus: HudBus
) -> None:
    while True:
        wait = period_s
        try:
            bus.publish(await fetch())
        except (ToolError, OSError) as exc:
            log.debug("HUD %s unavailable: %s", name, exc)
            wait = min(period_s, RETRY_AFTER_S)
        except Exception:  # noqa: BLE001 - a panel must never stop the assistant
            log.exception("HUD %s failed", name)
            wait = min(period_s, RETRY_AFTER_S)
        await asyncio.sleep(wait)


def start_feeds(
    s: Settings,
    bus: HudBus,
    weather: WeatherService | None,
    news: NewsService | None,
) -> list[asyncio.Task]:
    async def get_weather() -> dict:
        return weather_panel(await weather.forecast(days=WEATHER_DAYS))

    async def get_news() -> dict:
        return news_panel(await news.headlines(count=NEWS_ITEMS))

    async def get_system() -> dict:
        topics = await asyncio.gather(*(system_status(t) for t in ("battery", "cpu", "memory")))
        return system_panel({k: v for t in topics for k, v in t.items()})

    jobs = [_every("system", SYSTEM_EVERY_S, get_system, bus)]
    if weather is not None and s.location.resolved:
        jobs.append(_every("weather", CACHE_S, get_weather, bus))
    if news is not None:
        jobs.append(_every("news", s.news.cache_minutes * 60, get_news, bus))
    return [asyncio.create_task(job) for job in jobs]
