"""Time arithmetic done by code: "how long until 6 pm?", "how many days until Christmas?".

The context note gives the model the time, but qwen3:8b got the subtraction wrong: at
17:41 it said 49 minutes were left until 18:00.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta

from backend.config import Settings
from backend.tools.registry import Tool, ToolError

midnight, almost_midnight = time(0, 0), time(23, 59)
_TIME = re.compile(r"^\s*(\d{1,2})\s*(?:[:h]\s*(\d{2})?)?\s*$", re.IGNORECASE)
_ISO_DATE = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$")
_DAY_MONTH = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})\s*$")


def parse_time(text: str) -> time:
    """ "18:00", "18h", "18h30", "6" → a time of day (24-hour clock)."""
    m = _TIME.match(text or "")
    if not m:
        raise ToolError(f"time must be HH:MM (24-hour clock), got {text!r}")
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    if hour == 24 and minute == 0:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ToolError(f"not a time of day: {text!r}")
    return time(hour, minute)


def parse_date(text: str, today: date) -> date:
    """ "2026-12-25", or "25/12" (the next one from today)."""
    if m := _ISO_DATE.match(text or ""):
        y, mo, d = (int(g) for g in m.groups())
    elif m := _DAY_MONTH.match(text or ""):
        d, mo = (int(g) for g in m.groups())
        y = today.year
        try:
            if date(y, mo, d) < today:
                y += 1
        except ValueError:
            pass  # reported below
    else:
        raise ToolError(f"date must be YYYY-MM-DD, got {text!r}")
    try:
        return date(y, mo, d)
    except ValueError:
        raise ToolError(f"not a date: {text!r}") from None


class Clock:
    def __init__(self, s: Settings, now: Callable[[], datetime] | None = None) -> None:
        self.s = s
        self._now = now or (lambda: datetime.now(s.location.tz))

    def _date_words(self, d: date) -> str:
        loc = self.s.locale
        wd, month = loc.weekdays[d.weekday()], loc.months[d.month - 1]
        if loc.lang == "pt":
            return f"{wd}, {d.day} de {month} de {d.year}"
        return f"{wd}, {d.day} {month} {d.year}"

    async def time_until(self, time: str | None = None, date: str | None = None) -> dict:
        if not time and not date:
            raise ToolError("pass a time (HH:MM), a date (YYYY-MM-DD) or both")
        now = self._now().replace(second=0, microsecond=0)  # "17:41" means 17:41:00 to people
        out: dict = {"now": f"{now:%H:%M}"}
        day = parse_date(date, now.date()) if date else None
        if not time:  # a whole day: count days
            days = (day - now.date()).days
            out["date"] = self._date_words(day)
            out["days_until" if days >= 0 else "days_ago"] = abs(days)
            return out
        at = parse_time(time)
        # "Até a meia-noite": the coming one. qwen3 sends 23:59 for it even when told 00:00.
        if day is None and at in (midnight, almost_midnight):
            at, day = midnight, now.date() + timedelta(days=1)
        target = datetime.combine(day or now.date(), at, now.tzinfo)
        out["target"] = f"{target:%H:%M}"
        if day:
            out["date"] = self._date_words(day)
        fmt = self.s.locale.format_duration
        if target >= now:
            out["remaining"] = fmt((target - now).total_seconds())
        elif day:
            out["already_passed_by"] = fmt((now - target).total_seconds())
        else:  # earlier today: they probably mean tomorrow
            out["already_passed_today_by"] = fmt((now - target).total_seconds())
            out["remaining_until_tomorrow"] = fmt(
                (target + timedelta(days=1) - now).total_seconds()
            )
        return out


def clock_tools(s: Settings, now: Callable[[], datetime] | None = None) -> list[Tool]:
    clock = Clock(s, now)
    return [
        Tool(
            name="time_until",
            description=(
                "How long until a time of day or a date, calculated exactly: 'quanto falta "
                "para as 18h?', 'how many days until Christmas?'. Always use it instead of "
                "subtracting times or dates yourself; read out its `remaining` or days as is."
            ),
            handler=clock.time_until,
            parameters={
                "time": {
                    "type": "string",
                    "description": "HH:MM, 24-hour clock, e.g. 18:00; midnight is 00:00",
                },
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD; leave out for today",
                },
            },
            capability={
                "pt": "calcular quanto tempo falta até um horário ou uma data",
                "en": "work out how long until a time or a date",
            },
        )
    ]
