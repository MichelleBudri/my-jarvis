"""Async Ollama client with streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.config import LLMConfig

Message = dict[str, Any]


class LLMError(RuntimeError):
    pass


@dataclass
class ChatChunk:
    content: str = ""
    done: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float | None:
        count, ns = self.stats.get("eval_count"), self.stats.get("eval_duration")
        return count / (ns / 1e9) if count and ns else None


class OllamaClient:
    def __init__(self, cfg: LLMConfig, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self._client = client or httpx.AsyncClient(
            base_url=cfg.host, timeout=httpx.Timeout(300, connect=5)
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, messages: list[Message], stream: bool, tools: list | None) -> dict:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": stream,
            "think": self.cfg.think,
            "keep_alive": self.cfg.keep_alive,
            "options": {"temperature": self.cfg.temperature},
        }
        if tools:
            payload["tools"] = tools
        return payload

    async def warmup(self) -> None:
        """Load the model into memory (an empty message list only loads it)."""
        resp = await self._client.post(
            "/api/chat",
            json={"model": self.cfg.model, "messages": [], "keep_alive": self.cfg.keep_alive},
        )
        self._raise_for(resp)

    async def prime(self, system_prompt: str) -> None:
        """Load the model and pre-process the system prompt so the first reply is fast."""
        payload = self._payload([{"role": "system", "content": system_prompt}], False, None)
        payload["options"]["num_predict"] = 1
        self._raise_for(await self._client.post("/api/chat", json=payload))

    async def chat_stream(
        self, messages: list[Message], tools: list | None = None
    ) -> AsyncIterator[ChatChunk]:
        payload = self._payload(messages, stream=True, tools=tools)
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                self._raise_for(resp)
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise LLMError(data["error"])
                msg = data.get("message") or {}
                chunk = ChatChunk(
                    content=msg.get("content", ""),
                    done=bool(data.get("done")),
                    tool_calls=msg.get("tool_calls") or [],
                )
                if chunk.done:
                    chunk.stats = {
                        k: v for k, v in data.items() if k.endswith(("count", "duration"))
                    }
                yield chunk

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        raise LLMError(f"Ollama {resp.status_code}: {detail}")
