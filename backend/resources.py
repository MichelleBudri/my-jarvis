"""Memory held only while it is useful: the model and Whisper leave while Jarvis sleeps.

Asleep, Jarvis needs only the wake word and the VAD (tens of MB). After
`resources.idle_minutes` asleep (at once with the `economy` profile) the language model
is unloaded from Ollama (~5.4 GB) and Whisper from MLX (~0.5 GB). On waking they come
back while the person is still speaking: Whisper in ~0.3 s, the model and its prompt
prefix in ~15 s, most of it re-processing the system prompt and tools (D-37).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from backend.config import Settings
from backend.state import State

log = logging.getLogger(__name__)


class Resources:
    def __init__(
        self,
        s: Settings,
        conv,
        stt,
        on_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.s = s
        self.conv = conv
        self.stt = stt
        self.on_change = on_change  # True = loaded
        self.loaded = True
        self._timer: asyncio.Task | None = None
        self._work: asyncio.Task | None = None
        self._lock = asyncio.Lock()  # an unload and a reload never overlap

    def on_state(self, old: State | None, new: State) -> None:
        """A StateMachine listener."""
        if new is State.SLEEPING:
            self._schedule_unload()
            return
        self._cancel_timer()
        # Woken, or spoken to after a timer went off. Announcing the timer needs neither.
        if new in (State.LISTENING, State.THINKING) and not self.loaded:
            self._work = asyncio.create_task(self.reload())

    def _schedule_unload(self) -> None:
        delay = self.s.resources.unload_after_s
        if delay is None or not self.loaded:
            return
        self._cancel_timer()
        self._timer = asyncio.create_task(self._unload_after(delay))

    def _cancel_timer(self) -> None:
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    async def _unload_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        # Shielded: once started, an unload finishes even if Jarvis wakes meanwhile.
        await asyncio.shield(self.unload())

    async def unload(self) -> None:
        async with self._lock:
            if not self.loaded:
                return
            await self.conv.settle()  # a warm-up landing after the unload would reload it
            results = await asyncio.gather(
                self.conv.llm.unload(), self.stt.unload(), return_exceptions=True
            )
            for part, result in zip(("model", "whisper"), results, strict=True):
                if isinstance(result, Exception):
                    log.warning("Could not unload the %s: %s", part, result)
            self.loaded = False
            log.info("Asleep: model and Whisper unloaded")
        self._changed()

    async def reload(self) -> None:
        """Whisper and the model's prompt prefix, in parallel with the person speaking."""
        async with self._lock:
            if self.loaded:
                return
            self.loaded = True  # set first: a reply needs no further reload
            self._changed()
            results = await asyncio.gather(
                self.stt.warmup(), self.conv.warm(), return_exceptions=True
            )
            for part, result in zip(("whisper", "model"), results, strict=True):
                if isinstance(result, Exception):
                    log.warning("Could not reload the %s: %s", part, result)

    def _changed(self) -> None:
        if self.on_change:
            try:
                self.on_change(self.loaded)
            except Exception:  # noqa: BLE001
                log.exception("Resources listener failed")

    def close(self) -> None:
        self._cancel_timer()
        if self._work and not self._work.done():
            self._work.cancel()
