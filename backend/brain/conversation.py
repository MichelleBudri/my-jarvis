"""A conversation: system prompt + history + LLM."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from backend.brain.llm import OllamaClient
from backend.brain.prompts import build_system_prompt
from backend.config import Settings
from backend.memory.store import MemoryStore


@dataclass
class TurnStats:
    first_token_s: float | None = None
    total_s: float = 0.0
    tokens_per_second: float | None = None


class Conversation:
    def __init__(
        self,
        settings: Settings,
        llm: OllamaClient,
        store: MemoryStore,
        conversation_id: int | None = None,
    ) -> None:
        self.s = settings
        self.llm = llm
        self.store = store
        self.id = conversation_id or store.new_conversation()
        self.last_stats = TurnStats()

    def reset(self) -> None:
        self.id = self.store.new_conversation()

    def _messages(self, user_text: str) -> list[dict[str, str]]:
        history = self.store.recent(self.id, self.s.llm.context_messages)
        return [
            {"role": "system", "content": build_system_prompt(self.s)},
            *history,
            {"role": "user", "content": user_text},
        ]

    async def reply(self, user_text: str) -> AsyncIterator[str]:
        """Stream the reply and persist the turn once it finishes."""
        messages = self._messages(user_text)
        stats = TurnStats()
        start = time.perf_counter()
        parts: list[str] = []
        try:
            async for chunk in self.llm.chat_stream(messages):
                if chunk.content:
                    if stats.first_token_s is None:
                        stats.first_token_s = time.perf_counter() - start
                    parts.append(chunk.content)
                    yield chunk.content
                if chunk.done:
                    stats.tokens_per_second = chunk.tokens_per_second
        finally:
            stats.total_s = time.perf_counter() - start
            self.last_stats = stats
            answer = "".join(parts).strip()
            if answer:
                self.store.add(self.id, "user", user_text)
                self.store.add(self.id, "assistant", answer)
