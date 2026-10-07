"""Startup progress for the HUD's loading screen, and waiting for the network at login.

The bar moves only when a step really ends, never on a timer. Each step has a weight
close to its share of a typical start (MacBook Air M4): the model's prompt prefix is
most of it (~15 s), the briefing's news next (D-38).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from backend.hud.bus import HudBus

log = logging.getLogger(__name__)

WEIGHTS = {
    "ollama": 1,
    "internet": 1,
    "location": 1,
    "weather": 1,
    "news": 1,
    "voice": 1,
    "whisper": 2,
    "wake": 1,
    "model": 8,
    "briefing": 4,
    "speech": 2,
}
FINISHED = {"done", "skipped", "offline", "failed"}

PROBE_HOST = "api.open-meteo.com"  # the weather source: reachable means the briefing can be
PROBE_TIMEOUT_S = 2


class BootProgress:
    def __init__(self, bus: HudBus, steps: list[str]) -> None:
        self.bus = bus
        self.steps = {name: "pending" for name in steps}
        self.started = time.perf_counter()
        self.publish()

    @property
    def progress(self) -> int:
        total = sum(WEIGHTS.get(n, 1) for n in self.steps)
        done = sum(WEIGHTS.get(n, 1) for n, st in self.steps.items() if st in FINISHED)
        return round(100 * done / total) if total else 100

    def set(self, name: str, status: str) -> None:
        if name not in self.steps or self.steps[name] in FINISHED:
            return
        self.steps[name] = status
        self.publish()

    def start(self, name: str) -> None:
        self.set(name, "running")

    def done(self, name: str) -> None:
        self.set(name, "done")

    def finish(self) -> None:
        """Whatever has not ended by now was not needed."""
        for name, status in self.steps.items():
            if status not in FINISHED:
                self.steps[name] = "skipped"
        self.publish()

    async def track(self, name: str, work):
        """Await `work` as step `name`: running, then done or failed."""
        self.start(name)
        try:
            result = await work
        except BaseException:
            self.set(name, "failed")
            raise
        self.done(name)
        return result

    def publish(self) -> None:
        self.bus.publish(
            {
                "type": "boot",
                "steps": [{"id": n, "status": st} for n, st in self.steps.items()],
                "progress": self.progress,
            }
        )


async def online(host: str = PROBE_HOST, timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, 443), timeout_s)
    except (OSError, TimeoutError):
        return False
    writer.close()
    return True


async def wait_for_internet(
    timeout_s: float,
    probe: Callable[[], object] = online,
    retry_s: float = 1.0,
) -> bool:
    """At login Wi-Fi may join seconds after Jarvis starts: try until `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while True:
        if await probe():
            return True
        if time.monotonic() + retry_s >= deadline:
            log.warning("No internet after %.0f s: briefing without weather and news", timeout_s)
            return False
        await asyncio.sleep(retry_s)
