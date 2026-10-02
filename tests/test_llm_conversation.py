import asyncio
import json

import httpx

from backend.brain.conversation import Conversation
from backend.brain.llm import ChatChunk, OllamaClient
from backend.config import Settings
from backend.memory.store import MemoryStore


def fake_ollama(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        lines = [
            {"message": {"role": "assistant", "content": "Pois não, "}, "done": False},
            {"message": {"role": "assistant", "content": "senhora."}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "eval_count": 20,
                "prompt_eval_count": 12,
                "prompt_eval_duration": 100_000_000,
                "eval_duration": 500_000_000,
            },
        ]
        body = "\n".join(json.dumps(x) for x in lines) + "\n"
        return httpx.Response(200, text=body)

    return handler


def test_conversation_streams_and_persists(tmp_path):
    captured: dict = {}
    s = Settings()
    client = httpx.AsyncClient(
        base_url="http://ollama", transport=httpx.MockTransport(fake_ollama(captured))
    )
    llm = OllamaClient(s.llm, client=client)
    store = MemoryStore(tmp_path / "t.db")
    conv = Conversation(s, llm, store)

    async def go():
        return [t async for t in conv.reply("Olá, Jarvis")]

    tokens = asyncio.run(go())
    assert "".join(tokens) == "Pois não, senhora."

    payload = captured["payload"]
    assert payload["model"] == s.llm.model
    assert payload["stream"] is True and payload["think"] is False
    assert payload["messages"][0]["role"] == "system"
    last = payload["messages"][-1]
    assert last["role"] == "user" and last["content"].startswith("[Contexto:")
    assert last["content"].endswith("\nOlá, Jarvis")

    stored = store.recent(conv.id, 10)
    assert [m["role"] for m in stored] == ["user", "assistant"]
    assert stored[0]["content"] == "Olá, Jarvis"  # the context note is not persisted
    assert conv.last_stats.first_token_s is not None
    assert round(conv.last_stats.tokens_per_second) == 40
    assert conv.last_stats.prompt_tokens == 12


def test_tokens_per_second_none_without_stats():
    assert ChatChunk().tokens_per_second is None


def test_prime_sends_only_the_system_prompt(tmp_path):
    captured: dict = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "."}, "done": True})

    s = Settings()
    client = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    conv = Conversation(s, OllamaClient(s.llm, client=client), MemoryStore(tmp_path / "t.db"))
    asyncio.run(conv.prime())
    payload = captured["payload"]
    assert payload["messages"] == [{"role": "system", "content": conv.system_prompt}]
    assert payload["options"]["num_predict"] == 1 and payload["stream"] is False
