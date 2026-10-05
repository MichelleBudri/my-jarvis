"""Countdown timers that announce themselves when they finish."""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from backend.config import Settings
from backend.tools.registry import Tool, ToolError

log = logging.getLogger(__name__)

MAX_SECONDS = 24 * 3600


@dataclass
class Timer:
    id: int
    seconds: float
    ends_at: float  # monotonic clock
    label: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)

    def remaining(self, now: float) -> float:
        return max(0.0, self.ends_at - now)


class TimerManager:
    def __init__(
        self,
        s: Settings,
        on_fire: Callable[[Timer], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.s = s
        self.on_fire = on_fire
        self.on_change = on_change  # the HUD redraws its timer panel
        self.clock = clock
        self.timers: dict[int, Timer] = {}
        self._ids = itertools.count(1)

    def announcement(self, timer: Timer) -> str:
        loc = self.s.locale
        addr = f", {self.s.owner_address}" if self.s.owner_address else ""
        duration = loc.format_duration(timer.seconds)
        if timer.label:
            return loc.timer_done_label.format(addr=addr, duration=duration, label=timer.label)
        return loc.timer_done.format(addr=addr, duration=duration)

    def _describe(self, t: Timer) -> dict:
        left = t.remaining(self.clock())
        ends = datetime.now(self.s.location.tz) + timedelta(seconds=left)
        return {
            "id": t.id,
            "label": t.label,
            "duration": self.s.locale.format_duration(t.seconds),
            "remaining": self.s.locale.format_duration(left),
            "ends_at": f"{ends:%H:%M}",
        }

    def snapshot(self) -> list[dict]:
        """Active timers with their end as a Unix time in ms, for a live countdown."""
        now, wall = self.clock(), time.time()
        return [
            {
                "id": t.id,
                "label": t.label,
                "seconds": t.seconds,
                "ends_at_ms": round((wall + t.remaining(now)) * 1000),
            }
            for t in sorted(self.timers.values(), key=lambda t: t.ends_at)
        ]

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()

    async def _run(self, timer: Timer) -> None:
        await asyncio.sleep(timer.seconds)
        self.timers.pop(timer.id, None)
        log.debug("Timer %s finished", timer.id)
        self._changed()
        if self.on_fire:
            self.on_fire(timer)

    async def set_timer(
        self,
        hours: float = 0,
        minutes: float = 0,
        seconds: float = 0,
        label: str | None = None,
    ) -> dict:
        total = float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)
        if total <= 0:
            raise ToolError("the duration must be positive")
        if total > MAX_SECONDS:
            raise ToolError("timers can last at most 24 hours")
        label = (label or "").strip() or None
        # "Make it 10 minutes instead": the model calls set_timer again with the same
        # label rather than cancelling first, which left two timers for one task.
        replaced = [
            t for t in self.timers.values() if label and (t.label or "").lower() == label.lower()
        ]
        for t in replaced:
            self._cancel(t)
        timer = Timer(next(self._ids), total, self.clock() + total, label)
        timer.task = asyncio.create_task(self._run(timer))
        self.timers[timer.id] = timer
        self._changed()
        out = {"ok": True, **self._describe(timer)}
        if replaced:
            out["replaced_previous_timer"] = True
        return out

    async def list_timers(self) -> dict:
        timers = sorted(self.timers.values(), key=lambda t: t.ends_at)
        return {"active": [self._describe(t) for t in timers]}

    def _find(self, id: int | None, label: str | None) -> list[Timer]:
        if not self.timers:
            raise ToolError("there are no active timers")
        if id is not None:
            targets = [t for t in self.timers.values() if t.id == int(id)]
        elif label:
            key = label.strip().lower()
            targets = [t for t in self.timers.values() if t.label and key in t.label.lower()]
        elif len(self.timers) == 1:
            targets = list(self.timers.values())
        else:
            raise ToolError("several timers are active; say which one (id or label)")
        if not targets:
            raise ToolError("no matching timer")
        return targets

    async def change_timer(
        self,
        id: int | None = None,
        label: str | None = None,
        hours: float = 0,
        minutes: float = 0,
        seconds: float = 0,
    ) -> dict:
        """Restart a timer with a new duration, keeping its label."""
        targets = self._find(id, label)
        if len(targets) > 1:
            raise ToolError("several timers match; say which one (id or label)")
        old = targets[0]
        if (old.label or None) is None:
            self._cancel(old)  # set_timer replaces by label only
        out = await self.set_timer(hours, minutes, seconds, old.label)
        out.pop("replaced_previous_timer", None)
        return {**out, "changed_from": self.s.locale.format_duration(old.seconds)}

    async def cancel_timer(self, id: int | None = None, label: str | None = None) -> dict:
        targets = self._find(id, label)
        for t in targets:
            self._cancel(t)
        self._changed()
        return {"ok": True, "cancelled": [{"id": t.id, "label": t.label} for t in targets]}

    def _cancel(self, t: Timer) -> None:
        self.timers.pop(t.id, None)
        if t.task:
            t.task.cancel()

    def cancel_all(self) -> None:
        for t in self.timers.values():
            if t.task:
                t.task.cancel()
        self.timers.clear()
        self._changed()


def timer_tools(manager: TimerManager) -> list[Tool]:
    cap = {
        "pt": "criar, mudar, listar e cancelar timers e lembretes curtos (avisos falados)",
        "en": "set, change, list and cancel timers and short reminders (spoken alerts)",
    }
    return [
        Tool(
            name="set_timer",
            description=(
                "Start a countdown timer; when it ends you will announce it aloud. Also use for "
                "'remind me in N minutes to ...', putting what to remember in the label."
            ),
            handler=manager.set_timer,
            parameters={
                "hours": {"type": "number"},
                "minutes": {"type": "number"},
                "seconds": {"type": "number"},
                "label": {
                    "type": "string",
                    "description": (
                        "Only if the user says what it is for, e.g. 'tirar o bolo'; "
                        "never repeat the duration here"
                    ),
                },
            },
            capability=cap,
        ),
        # With only set/cancel, qwen3 answered "make it two minutes" with cancel_timer and
        # then reported the cancellation: the timer was gone.
        Tool(
            name="change_timer",
            description=(
                "Change an active timer's duration ('make it 10 minutes instead'), keeping its "
                "label; it restarts from now. Pick it by id or label; with neither, the only "
                "active one."
            ),
            handler=manager.change_timer,
            parameters={
                "id": {"type": "integer"},
                "label": {"type": "string"},
                "hours": {"type": "number"},
                "minutes": {"type": "number"},
                "seconds": {"type": "number"},
            },
            capability=cap,
        ),
        Tool(
            name="list_timers",
            description="Active timers with the time remaining on each.",
            handler=manager.list_timers,
            capability=cap,
        ),
        Tool(
            name="cancel_timer",
            description=(
                "Cancel a timer by id or label; with no arguments, the only active one. To "
                "change its duration instead, use change_timer."
            ),
            handler=manager.cancel_timer,
            parameters={
                "id": {"type": "integer"},
                "label": {"type": "string"},
            },
            capability=cap,
        ),
    ]
