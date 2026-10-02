"""A conversation: system prompt + history + LLM."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from backend.brain.llm import OllamaClient
from backend.brain.prompts import build_context_note, build_system_prompt
from backend.config import Settings
from backend.memory.store import MemoryStore


@dataclass
class TurnStats:
    first_token_s: float | None = None
    total_s: float = 0.0
    tokens_per_second: float | None = None
    prompt_tokens: int | None = None  # tokens Ollama had to (re)process; low = cache hit
    prompt_eval_s: float | None = None


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
        self.system_prompt = build_system_prompt(settings)

    async def prime(self) -> None:
        await self.llm.prime(self.system_prompt)

    def reset(self) -> None:
        self.id = self.store.new_conversation()

    def _messages(self, user_text: str) -> list[dict[str, str]]:
        history = self.store.recent(self.id, self.s.llm.context_messages)
        note = build_context_note(self.s, first_turn=not history)
        return [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": f"{note}\n{user_text}"},
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
                    stats.prompt_tokens = chunk.stats.get("prompt_eval_count")
                    if ns := chunk.stats.get("prompt_eval_duration"):
                        stats.prompt_eval_s = ns / 1e9
        finally:
            stats.total_s = time.perf_counter() - start
            self.last_stats = stats
            answer = "".join(parts).strip()
            if answer:
                self.store.add(self.id, "user", user_text)
                self.store.add(self.id, "assistant", answer)
