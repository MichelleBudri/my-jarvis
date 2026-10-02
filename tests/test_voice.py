import asyncio
import json

import httpx

from backend.brain.conversation import Conversation
from backend.brain.llm import OllamaClient
from backend.config import Settings
from backend.memory.store import MemoryStore
from backend.voice import reply_aloud


class FakeSpeaker:
    def __init__(self):
        self.spoken = []
        self.speaking = False

    async def warmup(self):
        pass

    async def speak(self, sentences, on_start=None):
        async for sentence in sentences:
            if on_start and not self.spoken:
                on_start()
            self.spoken.append(sentence)
        return True

    def interrupt(self):
        pass


def ollama_streaming(chunks):
    def handler(request):
        lines = [{"message": {"content": c}, "done": False} for c in chunks]
        lines.append({"message": {"content": ""}, "done": True})
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    return handler


def test_reply_aloud_speaks_sentence_by_sentence(tmp_path):
    s = Settings()
    chunks = ["Boa noite, **senhora**. ", "Faz 18 graus ", "lá fora. ", "Algo mais?"]
    client = httpx.AsyncClient(
        base_url="http://ollama", transport=httpx.MockTransport(ollama_streaming(chunks))
    )
    conv = Conversation(s, OllamaClient(s.llm, client=client), MemoryStore(tmp_path / "t.db"))
    speaker, tokens, started = FakeSpeaker(), [], []

    asyncio.run(
        reply_aloud(
            conv, speaker, "Como está o tempo?", 10, tokens.append, lambda: started.append(1)
        )
    )

    assert speaker.spoken == ["Boa noite, senhora.", "Faz 18 graus lá fora.", "Algo mais?"]
    assert "".join(tokens) == "".join(chunks)
    assert started == [1]
    assert conv.store.count(conv.id) == 2
