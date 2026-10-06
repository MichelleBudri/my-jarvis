import asyncio
import json

import httpx

from backend.brain.conversation import Conversation
from backend.brain.llm import ChatChunk, OllamaClient
from backend.brain.prompts import build_greeting
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
    greeting = build_greeting(s)
    assert tokens[0] == f"{greeting} "  # first turn: greeted by code, not by the model
    assert "".join(tokens[1:]) == "Pois não, senhora."

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
    assert stored[1]["content"] == "Pois não, senhora."  # the model would copy a greeting

    # Later turns go straight to the model's answer.
    assert "".join(asyncio.run(go())) == "Pois não, senhora."
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


WEATHER = (
    "A senhora está com garoa fraca, temperatura de 17 graus, umidade de 94 por cento "
    "e vento de 12 quilômetros por hora."
)


def scripted_ollama(replies: list[str], payloads: list[dict]):
    """Answers each request with the next reply, streamed in small chunks."""
    queue = iter(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        text = next(queue)
        chunks = [text[i : i + 8] for i in range(0, len(text), 8)]
        lines = [{"message": {"role": "assistant", "content": c}, "done": False} for c in chunks]
        lines.append({"message": {"role": "assistant", "content": ""}, "done": True})
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines) + "\n")

    return handler


def conversation_with_history(tmp_path, replies, payloads, history):
    s = Settings()
    client = httpx.AsyncClient(
        base_url="http://ollama", transport=httpx.MockTransport(scripted_ollama(replies, payloads))
    )
    store = MemoryStore(tmp_path / "t.db")
    conv = Conversation(s, OllamaClient(s.llm, client=client), store)
    for role, content in history:
        store.add(conv.id, role, content)
    return conv, store


def say(conv, text):
    async def go():
        return "".join([t async for t in conv.reply(text)])

    return asyncio.run(go())


def test_a_copied_reply_is_dropped_and_answered_without_history(tmp_path):
    payloads: list[dict] = []
    history = [("user", "Qual é o clima?"), ("assistant", WEATHER)]
    retry = "Desculpe, senhora, pode repetir?"
    conv, store = conversation_with_history(tmp_path, [WEATHER, retry], payloads, history)
    out = say(conv, "Eu amo a própria Alexa.")
    assert out == retry  # nothing of the copy reached the speaker
    assert len(payloads) == 2
    assert [m["role"] for m in payloads[1]["messages"]] == ["system", "user"]  # no history
    assert store.recent(conv.id, 10)[-1]["content"] == retry


def test_repeating_on_request_and_short_replies_are_not_copies(tmp_path):
    payloads: list[dict] = []
    history = [
        ("user", "Qual é o clima?"),
        ("assistant", WEATHER),
        ("user", "Valeu"),
        ("assistant", "Por nada, senhora."),
    ]
    conv, _ = conversation_with_history(
        tmp_path, [WEATHER, "Por nada, senhora."], payloads, history
    )
    assert say(conv, "Pode repetir, por favor?") == WEATHER
    assert say(conv, "Obrigada") == "Por nada, senhora."  # under COPY_MIN_CHARS
    assert len(payloads) == 2  # no retries
    assert conv.last_stats.first_token_s is not None
