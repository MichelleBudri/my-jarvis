import asyncio
import json

import pytest

from backend.brain.conversation import Conversation
from backend.brain.llm import OllamaClient
from backend.brain.prompts import build_system_prompt
from backend.config import Settings
from backend.memory.cli import run_memory
from backend.memory.store import MemoryStore
from backend.tools import ToolError, build_registry
from backend.tools.memory import MAX_FACTS, LongTermMemory, asks_to_remember
from backend.tools.registry import user_text


def run(coro):
    return asyncio.run(coro)


def test_store_facts_dedupe_and_forget(tmp_path):
    store = MemoryStore(tmp_path / "t.db")
    assert store.add_fact("Prefere  café sem açúcar.") == (1, True)
    assert store.add_fact("prefere café SEM açúcar.") == (1, False)  # same fact, any case
    store.add_fact("A filha se chama Ana.")
    assert store.facts() == [(1, "Prefere café sem açúcar."), (2, "A filha se chama Ana.")]
    assert store.forget_fact(1) and not store.forget_fact(1)
    assert store.forget_all_facts() == 1 and store.facts() == []


def test_facts_survive_new_conversations(tmp_path):
    path = tmp_path / "t.db"
    store = MemoryStore(path)
    store.new_conversation()
    store.add_fact("Mora em Curitiba.")
    store.close()
    store = MemoryStore(path)
    store.new_conversation()
    assert store.facts() == [(1, "Mora em Curitiba.")]


@pytest.mark.parametrize(
    "text",
    [
        "Jarvis, lembre que eu prefiro café sem açúcar.",
        "Anota aí: a minha filha se chama Ana.",
        "Na verdade minha filha se chama Ana Luísa, corrige isso.",
        "Pode lembrar que eu tenho dentista amanhã?",
        "Não esqueça que eu sou alérgica a camarão.",
        "Remember that I take my tea with milk.",
    ],
)
def test_asks_to_remember(text):
    assert asks_to_remember(text)


@pytest.mark.parametrize(
    "text",
    [
        "Você lembra como eu tomo café?",
        "Jarvis, você se lembra do nome da minha filha?",
        "O que você sabe sobre mim?",
        "Do you remember how I take my tea?",
        "Hoje eu tomei um café na padaria.",
        "Qual é a minha cor favorita?",
    ],
)
def test_questions_and_chat_are_not_requests(text):
    assert not asks_to_remember(text)


def test_remember_needs_a_request_from_the_user(tmp_path):
    memory = LongTermMemory(MemoryStore(tmp_path / "t.db"))

    async def as_turn(said: str, fact: str):
        user_text.set(said)
        return await memory.remember(fact)

    with pytest.raises(ToolError, match="did not ask"):
        run(as_turn("Você lembra como eu tomo café?", "Toma café com leite."))
    assert memory.store.facts() == []
    out = run(as_turn("Lembre que eu tomo café com leite.", "Toma café com leite."))
    assert out["ok"] and out["id"] == 1 and "note" in out


def test_remember_replaces_and_forget(tmp_path):
    memory = LongTermMemory(MemoryStore(tmp_path / "t.db"))
    run(memory.remember("A filha se chama Ana."))
    run(memory.remember("Prefere café sem açúcar."))
    out = run(memory.remember("A filha se chama Ana Luísa.", replaces="[1]"))  # id as text
    assert out["replaced"] == "A filha se chama Ana."
    assert [f for _, f in memory.store.facts()] == [
        "Prefere café sem açúcar.",
        "A filha se chama Ana Luísa.",
    ]
    with pytest.raises(ToolError, match="no memory with id"):
        run(memory.remember("x", replaces=99))
    with pytest.raises(ToolError, match="several"):
        run(memory.forget(text="a"))
    assert run(memory.forget(text="CAFÉ"))["forgotten"] == "Prefere café sem açúcar."
    assert run(memory.forget(id=3))["ok"]
    with pytest.raises(ToolError, match="no memory matches"):
        run(memory.forget(id=3))
    with pytest.raises(ToolError, match="which memory"):
        run(memory.forget())


def test_remember_limits(tmp_path):
    memory = LongTermMemory(MemoryStore(tmp_path / "t.db"))
    with pytest.raises(ToolError, match="nothing"):
        run(memory.remember("  "))
    with pytest.raises(ToolError, match="too long"):
        run(memory.remember("x" * 400))
    for i in range(MAX_FACTS):
        memory.store.add_fact(f"Fato {i}.")
    with pytest.raises(ToolError, match="full"):
        run(memory.remember("Mais um."))
    assert run(memory.remember("Fato 0 corrigido.", replaces=1))["ok"]  # a fix still fits


def test_registry_has_memory_only_with_a_store(tmp_path):
    s = Settings()
    assert "remember" not in build_registry(s)
    reg = build_registry(s, store=MemoryStore(tmp_path / "t.db"))
    assert "remember" in reg and "forget" in reg
    s = Settings(tools={"enabled": ["weather"]})
    assert "remember" not in build_registry(s, store=MemoryStore(tmp_path / "t.db"))


@pytest.mark.parametrize("language", ["pt-BR", "en-GB"])
def test_prompt_lists_facts_with_ids(language):
    s = Settings(language=language)
    base = build_system_prompt(s)
    assert build_system_prompt(s, facts=None) == base  # memory off: prompt unchanged
    empty = build_system_prompt(s, facts=[])
    assert "remember" in empty and ("(vazia)" in empty or "(empty)" in empty)
    full = build_system_prompt(s, facts=[(3, "Prefere chá."), (7, "Mora em Lisboa.")])
    assert "- [3] Prefere chá.\n- [7] Mora em Lisboa." in full


def test_conversation_prompt_follows_memory(tmp_path):
    s = Settings()
    store = MemoryStore(tmp_path / "t.db")
    conv = Conversation(s, OllamaClient(s.llm), store, tools=build_registry(s, store=store))
    before = conv.system_prompt
    assert conv.system_prompt == before  # stable between turns: the cached prefix holds
    store.add_fact("Prefere chá.")
    assert "[1] Prefere chá." in conv.system_prompt
    plain = Conversation(s, OllamaClient(s.llm), store, tools=build_registry(s))
    assert "Prefere chá." not in plain.system_prompt


def test_tool_sees_what_the_user_said(tmp_path):
    """Conversation.reply publishes the user's words to the tools it runs."""
    import httpx

    s = Settings()
    store = MemoryStore(tmp_path / "t.db")
    calls = iter(
        [
            {"tool_calls": [{"function": {"name": "remember", "arguments": {"fact": "X."}}}]},
            {"content": "Anotado."},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        msg = {"role": "assistant", "content": "", **next(calls)}
        done = {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 3}
        body = json.dumps({"message": msg, "done": False}) + "\n" + json.dumps(done) + "\n"
        return httpx.Response(200, text=body)

    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    tools = build_registry(s, store=store)
    conv = Conversation(s, OllamaClient(s.llm, client=client), store, tools=tools)

    async def go():
        return "".join([t async for t in conv.reply("Você lembra do X?")])

    run(go())
    assert store.facts() == []  # a question: the model's remember was refused
    tool_msg = store.recent(conv.id, 10)[2]
    assert "did not ask" in tool_msg["content"]


def test_memory_cli(tmp_path, capsys):
    s = Settings()
    store = MemoryStore(s.db_path)
    store.add_fact("Prefere chá.")
    store.add_fact("Mora em Lisboa.")
    store.close()
    assert run_memory(s) == 0
    assert "Prefere chá." in capsys.readouterr().out
    assert run_memory(s, "forget", 1) == 0
    assert run_memory(s, "forget", 1) == 1
    assert run_memory(s, "forget") == 1
    assert run_memory(s, "clear") == 0
    assert MemoryStore(s.db_path).facts() == []
