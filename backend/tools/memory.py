"""Long-term memory: facts the user asked Jarvis to keep across conversations."""

from __future__ import annotations

import re

from backend.memory.store import MemoryStore
from backend.tools.registry import Tool, ToolError, user_text

MAX_FACTS = 100  # all of them go into the system prompt
MAX_FACT_CHARS = 300
# Without it, qwen3 confirmed with "A nota foi salva com sucesso. Se precisar de algo mais...".
CONFIRM = "confirm in a few words, e.g. 'Anotado.' / 'Noted.'; do not repeat the fact"


_ASKS = re.compile(
    r"\b(?:lembr|record|anot|guard|grav|memoriz|registr|esque[cç]|corrig|corrij|atualiz"
    r"|remember|note|jot|keep in mind|forget|correct|update|save)",
    re.IGNORECASE,
)
_VOCATIVE = r"^(?:[\w ]{1,20},\s*)?"
_RECALL_QUESTION = re.compile(
    _VOCATIVE + r"(?:(?:você|vc|tu)\s+)?(?:(?:ainda|se)\s+)*(?:lembra|recorda|sabe)\b"
    r"|" + _VOCATIVE + r"o\s+que\s+(?:você\s+)?(?:lembra|sabe|guardou|anotou)\b"
    r"|" + _VOCATIVE + r"(?:do|did)\s+you\s+(?:still\s+)?(?:remember|recall|know)\b"
    r"|" + _VOCATIVE + r"what\s+do\s+you\s+(?:remember|know)\b",
    re.IGNORECASE,
)


def asks_to_remember(text: str) -> bool:
    """True when the user asked to keep or correct something, not just whether Jarvis knows.

    Asked "Você lembra como eu tomo café?" with nothing on coffee in memory, qwen3 called
    remember 3 times in 3 and stored a fact it made up.
    """
    t = " ".join(text.split())
    if t.endswith("?") and _RECALL_QUESTION.search(t):
        return False
    return _ASKS.search(t) is not None


class LongTermMemory:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def remember(self, fact: str, replaces: int | None = None) -> dict:
        fact = " ".join((fact or "").split())
        if not fact:
            raise ToolError("nothing to remember: pass the fact as a short sentence")
        if len(fact) > MAX_FACT_CHARS:
            raise ToolError(f"too long: keep it under {MAX_FACT_CHARS} characters")
        if (said := user_text.get()) is not None and not asks_to_remember(said):
            raise ToolError(
                "not saved: the user did not ask you to remember anything. Answer from the "
                "memory list; if it is not there, say you have nothing noted on it"
            )
        # One call for a correction: asked to "fix it", qwen3 added the new fact and kept
        # the old one, then said the user had two daughters, "Ana and Ana Luísa".
        replaced = None
        if replaces is not None:
            replaces = _as_id(replaces)
            old = dict(self.store.facts()).get(replaces)
            if old is None:
                raise ToolError(f"no memory with id {replaces}")
            self.store.forget_fact(replaces)
            replaced = old
        elif len(self.store.facts()) >= MAX_FACTS:
            raise ToolError("memory is full; ask the user which memory to forget first")
        fact_id, new = self.store.add_fact(fact)
        out: dict = {"ok": True, "id": fact_id, "already_known": not new}
        if replaced:
            out["replaced"] = replaced
        return {**out, "note": CONFIRM}

    async def forget(self, id: int | None = None, text: str | None = None) -> dict:
        facts = self.store.facts()
        if id is not None:
            targets = [f for f in facts if f[0] == _as_id(id)]
        elif text and text.strip():
            needle = text.strip().casefold()
            targets = [f for f in facts if needle in f[1].casefold()]
        else:
            raise ToolError("say which memory to forget (its id or part of its text)")
        if not targets:
            raise ToolError("no memory matches")
        if len(targets) > 1:
            raise ToolError("several memories match; pick one by id")
        self.store.forget_fact(targets[0][0])
        return {"ok": True, "forgotten": targets[0][1], "note": CONFIRM}


def _as_id(value: object) -> int:
    """Models sometimes send ids as strings ("3", "[3]")."""
    try:
        return int(str(value).strip("[] "))
    except ValueError:
        raise ToolError(f"not a memory id: {value!r}") from None


def memory_tools(store: MemoryStore) -> list[Tool]:
    memory = LongTermMemory(store)
    cap = {
        "pt": "guardar e esquecer fatos sobre a pessoa entre conversas (memória de longo prazo)",
        "en": "remember and forget facts about the user across conversations (long-term memory)",
    }
    return [
        Tool(
            name="remember",
            description=(
                "Keep a fact for future conversations. Call only when the user asks you to "
                "remember or note something ('lembre que...', 'guarde isso', 'remember that...'), "
                "not for anything said in passing."
            ),
            handler=memory.remember,
            parameters={
                "fact": {
                    "type": "string",
                    "description": (
                        "One short, self-contained sentence in the conversation's language, "
                        "about the user in the third person, e.g. 'Prefere café sem açúcar.'"
                    ),
                },
                "replaces": {
                    "type": "integer",
                    "description": (
                        "When this corrects or updates a remembered fact, that fact's [id]: "
                        "the old one is deleted"
                    ),
                },
            },
            required=["fact"],
            capability=cap,
        ),
        Tool(
            name="forget",
            description=(
                "Delete a remembered fact when the user asks you to forget it or says it is no "
                "longer true. Pick it by its [id] in the memory list, or by part of its text. "
                "To correct a fact, use remember with `replaces` instead."
            ),
            handler=memory.forget,
            parameters={
                "id": {"type": "integer"},
                "text": {"type": "string"},
            },
            capability=cap,
        ),
    ]
