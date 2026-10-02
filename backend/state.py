"""Assistant state machine: sleeping → listening → thinking → speaking."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum


class State(StrEnum):
    SLEEPING = "sleeping"  # waiting for the wake word
    LISTENING = "listening"  # waiting for, or recording, a request
    THINKING = "thinking"  # transcribing and waiting for the model
    SPEAKING = "speaking"


TRANSITIONS: dict[State, set[State]] = {
    State.SLEEPING: {State.LISTENING},
    State.LISTENING: {State.SLEEPING, State.THINKING},
    # Back to listening when the turn ends, fails or is interrupted.
    State.THINKING: {State.SPEAKING, State.LISTENING},
    State.SPEAKING: {State.LISTENING},
}

Listener = Callable[[State, State], None]


class InvalidTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self, initial: State, clock: Callable[[], float] = time.monotonic) -> None:
        self.state = initial
        self._clock = clock
        self.since = clock()
        self._listeners: list[Listener] = []

    def add_listener(self, listener: Listener) -> None:
        """Called with (old, new) on every change; the HUD will subscribe here."""
        self._listeners.append(listener)

    def to(self, new: State) -> bool:
        """Move to `new`. Returns False if already there."""
        old = self.state
        if new == old:
            return False
        if new not in TRANSITIONS[old]:
            raise InvalidTransition(f"{old} → {new}")
        self.state, self.since = new, self._clock()
        for listener in self._listeners:
            listener(old, new)
        return True

    @property
    def elapsed(self) -> float:
        return self._clock() - self.since
