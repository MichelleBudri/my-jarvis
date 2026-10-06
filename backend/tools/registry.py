"""Tools the LLM can call: schema for Ollama, dispatch and error handling."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# What the user said this turn, set by Conversation.reply: a tool can check that the user
# really asked for what the model is doing (see `remember`). None outside a conversation.
user_text: ContextVar[str | None] = ContextVar("user_text", default=None)

Handler = Callable[..., Awaitable[Any]]


class ToolError(RuntimeError):
    """An expected failure the model should hear about (bad argument, service down)."""


@dataclass
class Tool:
    name: str
    description: str
    handler: Handler
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema properties
    required: list[str] = field(default_factory=list)
    # What the system prompt says the assistant can do, per language ("pt", "en").
    capability: dict[str, str] = field(default_factory=dict)
    slow: bool = False  # goes to the network: say "one moment" while it runs

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def is_slow(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.slow)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def capabilities(self, lang: str) -> list[str]:
        out: list[str] = []
        for tool in self._tools.values():
            text = tool.capability.get(lang) or tool.capability.get("en")
            if text and text not in out:
                out.append(text)
        return out

    async def call(self, name: str, arguments: dict[str, Any] | str | None) -> str:
        """Run a tool and return JSON for the model. Never raises: errors become data."""
        tool = self._tools.get(name)
        if tool is None:
            return _dump({"error": f"unknown tool {name!r}"})
        if isinstance(arguments, str):  # some models send the arguments as a JSON string
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return _dump({"error": "arguments are not valid JSON"})
        args = {k: v for k, v in (arguments or {}).items() if k in tool.parameters}
        try:
            result = await tool.handler(**args)
        except ToolError as exc:
            return _dump({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.exception("Tool %s failed", name)
            return _dump({"error": f"{type(exc).__name__}: {exc}"})
        return _dump(result)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
