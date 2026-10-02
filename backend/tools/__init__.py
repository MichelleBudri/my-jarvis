"""Tools the assistant can call: weather, AI news, the Mac itself and timers."""

from __future__ import annotations

from backend.config import Settings
from backend.tools.registry import Tool, ToolError, ToolRegistry
from backend.tools.timers import TimerManager

__all__ = ["Tool", "ToolError", "ToolRegistry", "build_registry"]


def build_registry(
    s: Settings, timers: TimerManager | None = None, extra: list[Tool] | None = None
) -> ToolRegistry:
    """The tools enabled in `tools.enabled`, plus any session-specific `extra` ones."""
    from backend.tools.news import news_tools
    from backend.tools.system import system_tools
    from backend.tools.timers import timer_tools
    from backend.tools.weather import weather_tools

    enabled = set(s.tools.enabled)
    tools: list[Tool] = []
    if "weather" in enabled:
        tools += weather_tools(s)
    if "news" in enabled:
        tools += news_tools(s)
    if "system" in enabled:
        tools += system_tools(s)
    if "timers" in enabled:
        tools += timer_tools(timers or TimerManager(s))
    return ToolRegistry([*tools, *(extra or [])])
