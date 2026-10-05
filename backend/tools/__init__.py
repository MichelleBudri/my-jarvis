"""Tools the assistant can call: weather, AI news, the Mac itself and timers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.config import Settings
from backend.tools.registry import Tool, ToolError, ToolRegistry
from backend.tools.timers import TimerManager

if TYPE_CHECKING:
    from backend.tools.news import NewsService
    from backend.tools.weather import WeatherService

__all__ = ["Tool", "ToolError", "ToolRegistry", "build_registry"]


def build_registry(
    s: Settings,
    timers: TimerManager | None = None,
    extra: list[Tool] | None = None,
    weather: WeatherService | None = None,
    news: NewsService | None = None,
) -> ToolRegistry:
    """The tools enabled in `tools.enabled`, plus any session-specific `extra` ones.

    Pass the services to share their cache, e.g. with the briefing.
    """
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
    return ToolRegistry([*tools, *(extra or [])])
