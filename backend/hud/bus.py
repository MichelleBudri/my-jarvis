"""In-process event bus between the assistant and the HUD clients.

The assistant publishes plain dicts with a "type"; each connected client has a queue.
The latest panel data and the recent transcript are kept, so a page opened (or
reloaded) mid-conversation shows everything at once instead of waiting for updates.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

Event = dict[str, Any]

# Kept and replayed to new clients; anything else (audio levels, tool calls) is live only.
STICKY = {
    "hello",
    "state",
    "weather",
    "news",
    "system",
    "timers",
    "stats",
    "autostart",
    "footprint",
    "boot",
}
TRANSCRIPT_LINES = 8
QUEUE_SIZE = 512  # ~15 s of audio levels; a stalled client loses the oldest events


class HudBus:
    def __init__(self) -> None:
        self._clients: set[asyncio.Queue[Event]] = set()
        self._sticky: dict[str, Event] = {}
        self._lines: list[dict[str, Any]] = []  # {"role", "text", "open"}
        self.on_command: Callable[[str], None] | None = None
        self._handlers: dict[str, Callable[[Any], None]] = {}

    @property
    def clients(self) -> int:
        return len(self._clients)

    def publish(self, event: Event) -> None:
        self._remember(event)
        for queue in self._clients:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)

    def _remember(self, event: Event) -> None:
        kind = event["type"]
        if kind in STICKY:
            self._sticky[kind] = event
        elif kind == "user":
            self._close_reply()
            self._lines.append({"role": "user", "text": event["text"], "open": False})
        elif kind == "reply":
            last = self._lines[-1] if self._lines else None
            if last and last["role"] == "assistant" and last["open"]:
                last["text"] += event["delta"]
            else:
                self._lines.append({"role": "assistant", "text": event["delta"], "open": True})
        elif kind == "reply_end":
            self._close_reply()
        del self._lines[:-TRANSCRIPT_LINES]

    def _close_reply(self) -> None:
        if self._lines and self._lines[-1]["open"]:
            self._lines[-1]["open"] = False

    def snapshot(self) -> list[Event]:
        """What a new client needs to catch up: panels, state and the recent transcript."""
        lines = [{"role": x["role"], "text": x["text"], "open": x["open"]} for x in self._lines]
        return [*self._sticky.values(), {"type": "transcript", "lines": lines}]

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._clients.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        self._clients.discard(queue)

    def command(self, name: str) -> None:
        """A button pressed on the HUD ("wake", "sleep")."""
        if self.on_command is None:
            return
        try:
            self.on_command(name)
        except Exception:  # noqa: BLE001 - a bad click must not take the assistant down
            log.exception("HUD command %r failed", name)

    def handle(self, name: str, handler: Callable[[Any], None]) -> None:
        """Answer a request from the HUD: a switch ("autostart") or a button ("export")."""
        self._handlers[name] = handler

    def request(self, name: str, value: Any = None) -> None:
        handler = self._handlers.get(name)
        if handler is None:
            return
        try:
            handler(value)
        except Exception:  # noqa: BLE001
            log.exception("HUD request %r failed", name)
