"""Tools the assistant can call: weather, AI news, the Mac itself, timers, clock, memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.config import Settings
from backend.tools.registry import Tool, ToolError, ToolRegistry
from backend.tools.timers import TimerManager

if TYPE_CHECKING:
    from backend.memory.store import MemoryStore
    from backend.tools.news import NewsService
    from backend.tools.weather import WeatherService

__all__ = ["Tool", "ToolError", "ToolRegistry", "build_registry"]


def build_registry(
    s: Settings,
    timers: TimerManager | None = None,
    extra: list[Tool] | None = None,
    weather: WeatherService | None = None,
    news: NewsService | None = None,
    store: MemoryStore | None = None,
) -> ToolRegistry:
    """The tools enabled in `tools.enabled`, plus any session-specific `extra` ones.

    Pass the services to share their cache, e.g. with the briefing. Memory needs the
    conversation's `store`; without it, the memory tools are left out.
    """
    from backend.tools.clock import clock_tools
    from backend.tools.memory import memory_tools
    from backend.tools.news import news_tools
    from backend.tools.system import system_tools
    from backend.tools.timers import timer_tools
    from backend.tools.weather import weather_tools

    enabled = set(s.tools.enabled)
    tools: list[Tool] = []
    if "weather" in enabled:
        tools += weather_tools(s, weather)
    if "news" in enabled:
        tools += news_tools(s, news)
    if "system" in enabled:
        tools += system_tools(s)
    if "timers" in enabled:
        tools += timer_tools(timers or TimerManager(s))
    if "clock" in enabled:
        tools += clock_tools(s)
    if "memory" in enabled and store is not None:
        tools += memory_tools(store)
    return ToolRegistry([*tools, *(extra or [])])
