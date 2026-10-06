"""A conversation: system prompt + history + LLM."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from backend.brain.llm import ChatChunk, OllamaClient
from backend.brain.prompts import build_context_note, build_greeting, build_system_prompt
from backend.config import Settings
from backend.memory.store import MemoryStore
from backend.tools.registry import ToolRegistry
from backend.tools.registry import user_text as user_text_var

log = logging.getLogger(__name__)

_END = object()  # end of the model's stream

# Tool results longer than this are not kept in history: news and weather run to ~1,500
# tokens and go stale anyway. Short ones (a timer's id and label) help with follow-ups.
HISTORY_RESULT_MAX_CHARS = 300
STALE_RESULT = json.dumps(
    {"omitted": "result from an earlier turn; call the tool again for current data"}
)

# A reply that opens with this much of an earlier reply, word for word, is a copy. After
# speech it could not make sense of (a conversation in the background), qwen3 repeated
# the weather report it had given earlier; once in history, it repeated it on every turn.
COPY_MIN_CHARS = 40
_ASKS_REPEAT = re.compile(
    r"\b(?:repet|de novo|outra vez|novamente|repeat|again|say that)", re.IGNORECASE
)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class TurnStats:
    first_token_s: float | None = None
    total_s: float = 0.0
    tokens_per_second: float | None = None
    prompt_tokens: int | None = None  # tokens Ollama had to (re)process; low = cache hit
    prompt_eval_s: float | None = None
    tools: list[str] = field(default_factory=list)  # tools called this turn, in order
    tools_s: float = 0.0  # time spent running them


class Conversation:
    def __init__(
        self,
        settings: Settings,
        llm: OllamaClient,
        store: MemoryStore,
        conversation_id: int | None = None,
        tools: ToolRegistry | None = None,
        warm_after_turn: bool = False,
    ) -> None:
        self.s = settings
        self.warm_after_turn = warm_after_turn
        self._warm_task: asyncio.Task | None = None
        self.llm = llm
        self.store = store
        self.tools = tools or ToolRegistry()
        self.id = conversation_id or store.new_conversation()
        self.last_stats = TurnStats()
        self.on_tool: Callable[[str], None] | None = None  # the HUD shows tools as they run
        self._capabilities = self.tools.capabilities(settings.locale.lang)
        # Same list every turn: Ollama renders tools into the cached prompt prefix.
        self._schemas = self.tools.schemas() or None

    @property
    def system_prompt(self) -> str:
        """Rebuilt each turn, but the same text until a fact is remembered or forgotten,
        so Ollama keeps reusing its cached prefix."""
        facts = self.store.facts() if "remember" in self.tools else None
        return build_system_prompt(self.s, self._capabilities, facts)

    @property
    def tool_schemas(self) -> list[dict[str, Any]] | None:
        return self._schemas

    async def prime(self) -> None:
        await self.llm.prime(self.system_prompt, self._schemas)

    async def warm(self) -> None:
        """Pre-process the next request's prefix: system, history and the context note.

        Ollama keeps one cached prefix, and the answer after a tool call is drafted without
        the history (D-18), so the next question used to re-process all of it (~1-2 s).
        Ending with the note, not the last answer, matters: qwen3 renders the final
        assistant message differently from the same message in history.
        """
        history = self.store.recent(self.id, self.s.llm.context_messages)
        note = build_context_note(self.s, first_turn=not history, tools=bool(self._schemas))
        next_turn = [*history, {"role": "user", "content": note}]
        try:
            await self.llm.prime(self.system_prompt, self._schemas, next_turn)
        except Exception as exc:  # noqa: BLE001 - only an optimisation
            log.debug("Cache warm-up failed: %s", exc)

    def schedule_warm(self) -> None:
        """Warm the cache in the background, while the reply is still being spoken."""
        if self.warm_after_turn:
            self._warm_task = asyncio.create_task(self.warm())

    def add_assistant_note(self, text: str) -> None:
        """Something said unprompted (a timer going off), so follow-ups have context."""
        self.store.add(self.id, "assistant", text)

    @staticmethod
    def _for_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """A tool round as it is kept in history: calls without ids, long results dropped."""
        out = []
        for m in messages:
            if m["role"] == "tool":
                content = m["content"]
                if len(content) > HISTORY_RESULT_MAX_CHARS:
                    content = STALE_RESULT
                out.append({"role": "tool", "tool_name": m["tool_name"], "content": content})
            else:
                calls = [{"function": c.get("function") or {}} for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": m["content"], "tool_calls": calls})
        return out

    def _persist(self, user_text: str, trace: list[dict[str, Any]], answer: str) -> None:
        self.store.add(self.id, "user", user_text)
        for m in self._for_history(trace):
            self.store.add(
                self.id, m["role"], m["content"], m.get("tool_calls"), m.get("tool_name")
            )
        if answer:
            self.store.add(self.id, "assistant", answer)

    def reset(self) -> None:
        self.id = self.store.new_conversation()

    def _messages(self, user_text: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
        note = build_context_note(self.s, first_turn=not history, tools=bool(self._schemas))
        return [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": f"{note}\n{user_text}"},
        ]

    async def _run_tools(self, calls: list[dict[str, Any]], stats: TurnStats) -> list[dict]:
        start = time.perf_counter()
        fns = [c.get("function") or {} for c in calls]
        if self.on_tool:
            for fn in fns:
                self.on_tool(fn.get("name", "?"))
        results = await asyncio.gather(
            *(self.tools.call(fn.get("name", ""), fn.get("arguments")) for fn in fns)
        )
        stats.tools_s += time.perf_counter() - start
        messages = []
        for call, fn, result in zip(calls, fns, results, strict=True):
            log.debug("Tool %s(%s) → %s", fn.get("name"), fn.get("arguments"), result[:300])
            stats.tools.append(fn.get("name", "?"))
            msg = {"role": "tool", "tool_name": fn.get("name", ""), "content": result}
            if call.get("id"):
                msg["tool_call_id"] = call["id"]
            messages.append(msg)
        return messages

    async def _stream(
        self, messages: list[dict[str, Any]], quiet_s: float | None
    ) -> AsyncIterator[ChatChunk | None]:
        """The model's chunks, plus one None if `quiet_s` pass with no text and no call yet.

        Ollama holds a tool call back until it is complete, so a turn that ends up calling
        a tool is silent for seconds; the None lets the caller say "one moment" meanwhile.
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def pump() -> None:
            try:
                async for chunk in self.llm.chat_stream(messages, tools=self._schemas):
                    queue.put_nowait(chunk)
            except Exception as exc:  # noqa: BLE001 - re-raised by the consumer
                queue.put_nowait(exc)
                return
            queue.put_nowait(_END)

        task = asyncio.create_task(pump())
        deadline = None if quiet_s is None else time.perf_counter() + quiet_s
        try:
            while True:
                timeout = None if deadline is None else max(0.0, deadline - time.perf_counter())
                try:
                    item = await asyncio.wait_for(queue.get(), timeout)
                except TimeoutError:
                    deadline = None
                    yield None
                    continue
                if item is _END:
                    return
                if isinstance(item, Exception):
                    raise item
                if item.content or item.tool_calls:
                    deadline = None
                yield item
        finally:
            task.cancel()

    @staticmethod
    def _could_be_copy(text: str, earlier: list[str]) -> bool:
        t = _norm(text)
        return bool(t) and any(reply.startswith(t) for reply in earlier)

    def _could_be_tool_name(self, text: str) -> bool:
        t = text.strip().rstrip(".!").lower()
        return bool(t) and any(name.startswith(t) for name in self.tools.names)

    @staticmethod
    def _earlier_replies(user_text: str, history: list[dict[str, Any]]) -> list[str]:
        """What a reply must not copy: the assistant's earlier answers in history."""
        if _ASKS_REPEAT.search(user_text):
            return []  # "pode repetir?": copying is the right answer
        return [_norm(m["content"]) for m in history if m["role"] == "assistant" and m["content"]]

    def _as_tool_name(self, text: str) -> str | None:
        t = text.strip().rstrip(".!").lower()
        return t if t in self.tools else None

    async def reply(self, user_text: str) -> AsyncIterator[str]:
        """Stream the reply, running any tools the model asks for, and persist the turn."""
        user_text_var.set(user_text)  # this task's context: tools see what was asked
        history = self.store.recent(self.id, self.s.llm.context_messages)
        messages: list[dict[str, Any]] = self._messages(user_text, history)
        earlier = self._earlier_replies(user_text, history)
        stats = TurnStats()
        start = time.perf_counter()
        parts: list[str] = []
        # Tool rounds of this turn, persisted with the answer. With only the final text in
        # history, qwen3 saw "set a timer" → "Done." and answered follow-ups ("cancel it",
        # "mute") by claiming the action without calling the tool.
        trace: list[dict[str, Any]] = []
        said: list[str] = []  # this round's text; after the last call, the answer
        one_moment = f"{self.s.locale.one_moment} "
        waited = False  # "One moment" already said this turn
        try:
            # Not persisted: the model would copy it and greet on every turn.
            if not history and (greeting := build_greeting(self.s)):
                yield f"{greeting} "  # not counted in first_token_s: that measures the model
            for _ in range(self.s.tools.max_rounds + 1):
                calls: list[dict[str, Any]] = []
                said = []
                retried_blank = False
                for _attempt in range(3):
                    generated = 0
                    held = ""  # text that may still turn out to be a tool name or a copy
                    copied = False
                    quiet = None
                    if self._schemas and not waited and not parts:
                        quiet = self.s.tools.one_moment_after_s
                    stream = self._stream(messages, quiet)
                    async with contextlib.aclosing(stream):
                        async for chunk in stream:
                            if chunk is None:
                                waited = True
                                yield one_moment
                                continue
                            if chunk.content:
                                held += chunk.content
                                if self._could_be_tool_name(held):
                                    continue
                                if not said and earlier and self._could_be_copy(held, earlier):
                                    if len(_norm(held)) < COPY_MIN_CHARS:
                                        continue
                                    copied = True
                                    break
                                if stats.first_token_s is None:
                                    stats.first_token_s = time.perf_counter() - start
                                said.append(held)
                                yield held
                                held = ""
                            calls.extend(chunk.tool_calls)
                            if chunk.done:
                                generated = chunk.stats.get("eval_count") or 0
                                stats.tokens_per_second = chunk.tokens_per_second
                                stats.prompt_tokens = chunk.stats.get("prompt_eval_count")
                                if ns := chunk.stats.get("prompt_eval_duration"):
                                    stats.prompt_eval_s = ns / 1e9
                    if copied:
                        # Nothing was said yet. Without the history there is nothing to
                        # copy; the question alone gets "could you repeat that?".
                        log.warning("Model copied an earlier reply; answering without history")
                        del messages[1 : 1 + len(history)]
                        history, earlier, calls = [], [], []
                        continue
                    if name := self._as_tool_name(held):
                        # qwen3 sometimes writes "Go_to_sleep." as text instead of calling it;
                        # spoken aloud, that is the tool name read to the user.
                        log.warning("Model wrote the tool name %r as text; calling it", name)
                        calls.append({"function": {"name": name, "arguments": {}}})
                    elif held:  # a short reply that matched an earlier one: not a copy
                        if stats.first_token_s is None:
                            stats.first_token_s = time.perf_counter() - start
                        said.append(held)
                        yield held
                    # Tokens came out but neither text nor a call: Ollama dropped a tool
                    # call it could not parse. Sampling again usually gets it through.
                    if said or calls or not generated or retried_blank:
                        break
                    retried_blank = True
                    log.warning("Blank reply after %d tokens; retrying once", generated)
                parts.extend(said)
                if not calls:
                    break
                if said:  # e.g. "Let me check." before the call: keep the reply speakable
                    yield " "
                    parts.append(" ")
                elif not waited and any(
                    self.tools.is_slow((c.get("function") or {}).get("name", "")) for c in calls
                ):
                    waited = True
                    yield one_moment  # speaks while the network call runs
                if history:
                    # The call already resolved the question in context ("and tomorrow?"
                    # → days=2). Drafting the answer next to earlier replies that open the
                    # same way ("A senhora terá..."), qwen3 copied one instead of the data.
                    del messages[1 : 1 + len(history)]
                    history, earlier = [], []
                round_ = [
                    {"role": "assistant", "content": "".join(said), "tool_calls": calls},
                    *await self._run_tools(calls, stats),
                ]
                messages.extend(round_)
                trace.extend(round_)  # only once the results are in: no orphan call
                said = []  # kept in the call message
            else:
                log.warning("Model kept calling tools after %d rounds", self.s.tools.max_rounds)
        finally:
            stats.total_s = time.perf_counter() - start
            self.last_stats = stats
            if "".join(parts).strip() or trace:
                self._persist(user_text, trace, "".join(said).strip())
                self.schedule_warm()
