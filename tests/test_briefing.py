import asyncio
import json
from datetime import datetime

import httpx

from backend.brain import briefing
from backend.brain.briefing import BriefingData, weather_sentence
from backend.brain.conversation import Conversation
from backend.brain.llm import OllamaClient
from backend.config import Settings
from backend.memory.store import MemoryStore
from backend.state import State
from backend.tools import build_registry
from backend.tools.registry import ToolError
from backend.tools.weather import WeatherService

EVENING = datetime(2026, 10, 4, 20, 30)
WEATHER = {
    "now": {"conditions": "nublado", "temperature": 18, "feels_like": 18},
    "today": {"conditions": "trovoada", "min": 17, "max": 26, "rain_chance_percent": 100},
}
NEWS = [{"title": "OpenAI lança modelo", "source": "G1"}, {"title": "Ações sobem", "source": "X"}]


def make_conv(tmp_path, handler=None, payloads=None) -> Conversation:
    def default(request: httpx.Request) -> httpx.Response:
        if payloads is not None:
            payloads.append(json.loads(request.content))
        lines = [
            {"message": {"content": "A OpenAI lançou um modelo."}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    s = Settings()
    transport = httpx.MockTransport(handler or default)
    client = httpx.AsyncClient(base_url="http://o", transport=transport)
    store = MemoryStore(tmp_path / "t.db")
    return Conversation(s, OllamaClient(s.llm, client=client), store, tools=build_registry(s))


async def collect(stream) -> list[str]:
    return [t async for t in stream]


MORNING = datetime(2026, 10, 7, 9, 0)


def test_weather_sentence_sounds_spoken_not_like_a_list(tmp_path):
    conv = make_conv(tmp_path)
    drizzle = {
        "now": {"conditions": "garoa", "temperature": 22, "feels_like": 22},
        "today": {"conditions": "garoa forte", "min": 16, "max": 26, "rain_chance_percent": 100},
    }
    assert weather_sentence(conv, BriefingData(weather=drizzle), MORNING) == (
        "Lá fora estão 22 graus e está garoando. Ao longo do dia, a previsão é de garoa "
        "forte, com a temperatura entre 16 e 26 graus. É bom ter o guarda-chuva à mão."
    )
    dry = {
        "now": {"conditions": "céu limpo", "temperature": 30, "feels_like": 34},
        "today": {"conditions": "céu limpo", "min": 20, "max": 31, "rain_chance_percent": 10},
    }
    assert weather_sentence(conv, BriefingData(weather=dry), MORNING) == (
        "Lá fora estão 30 graus e o céu está limpo, com sensação de 34. Ao longo do dia, a "
        "temperatura fica entre 20 e 31 graus."
    )
    maybe = {
        "now": {"conditions": "parcialmente nublado", "temperature": 1},
        "today": {"min": -2, "max": 8, "rain_chance_percent": 50},
    }
    assert weather_sentence(conv, BriefingData(weather=maybe), MORNING) == (
        "Lá fora está 1 grau com algumas nuvens. Ao longo do dia, a temperatura fica entre "
        "-2 e 8 graus. Há 50 por cento de chance de chuva."
    )
    assert weather_sentence(conv, BriefingData(), MORNING) == ""


def test_weather_sentence_leaves_out_the_days_range_in_the_evening(tmp_path):
    conv = make_conv(tmp_path)
    assert weather_sentence(conv, BriefingData(weather=WEATHER), EVENING) == (
        "Lá fora estão 18 graus e o céu está encoberto. É bom ter o guarda-chuva à mão."
    )


def test_weather_sentence_in_english(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_LANGUAGE", "en-GB")
    conv = make_conv(tmp_path)
    data = BriefingData(
        weather={
            "now": {"temperature": 12, "conditions": "light drizzle"},
            "today": {"min": 8, "max": 14},
        },
    )
    assert weather_sentence(conv, data, MORNING) == (
        "It's 12 degrees out and drizzling. Today it should stay between 8 and 14 degrees."
    )


class FakeWeather:
    def __init__(self, error=None, delay=0.0):
        self.error, self.delay = error, delay

    async def forecast(self, days=2):
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return {"now": WEATHER["now"], "days": [WEATHER["today"]]}


class FakeNews:
    def __init__(self, delay=0.0):
        self.delay = delay

    async def headlines(self, topic=None, count=None):
        await asyncio.sleep(self.delay)
        return {"items": [{**n, "hours_ago": 1, "summary": "..."} for n in NEWS]}


def test_fetch_gets_both_and_trims_them(tmp_path):
    conv = make_conv(tmp_path)
    data = asyncio.run(briefing.fetch(conv, FakeWeather(), FakeNews()))
    assert data.weather == WEATHER
    assert data.news == NEWS  # titles and sources only


def test_fetch_leaves_out_a_failing_or_slow_source(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_BRIEFING__FETCH_TIMEOUT_S", "0.05")
    conv = make_conv(tmp_path)
    data = asyncio.run(briefing.fetch(conv, FakeWeather(error=ToolError("503")), FakeNews(delay=1)))
    assert data.weather is None and data.news is None
    data = asyncio.run(briefing.fetch(conv, None, FakeNews()))
    assert data.weather is None and data.news == NEWS


def test_stream_says_greeting_weather_then_the_model_news(tmp_path, owner_env):
    payloads: list[dict] = []
    conv = make_conv(tmp_path, payloads=payloads)
    data = BriefingData(weather=WEATHER, news=NEWS)
    pieces = asyncio.run(collect(briefing.stream(conv, data, EVENING)))
    assert pieces[0] == "Boa noite, senhora. "
    assert pieces[1].startswith("Lá fora estão 18 graus")
    assert pieces[2:] == ["A OpenAI lançou um modelo."]

    msgs = payloads[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "get_ai_news"
    assert json.loads(msgs[3]["content"])["items"] == NEWS
    assert payloads[0]["tools"] == conv.tool_schemas  # same cached prefix as the chat

    stored = conv.store.recent(conv.id, 10)
    assert stored == [{"role": "assistant", "content": "".join(pieces[1:]).strip()}]


def test_stream_without_news_does_not_call_the_model(tmp_path, owner_env):
    def handler(request):
        raise AssertionError("no model call expected")

    conv = make_conv(tmp_path, handler)
    pieces = asyncio.run(collect(briefing.stream(conv, BriefingData(weather=WEATHER), EVENING)))
    assert len(pieces) == 2
    assert asyncio.run(collect(briefing.stream(conv, BriefingData(), EVENING))) == [
        "Boa noite, senhora. "
    ]


def test_stream_keeps_the_weather_when_the_model_fails(tmp_path, owner_env):
    def handler(request):
        return httpx.Response(500, json={"error": "model not found"})

    conv = make_conv(tmp_path, handler)
    data = BriefingData(weather=WEATHER, news=NEWS)
    pieces = asyncio.run(collect(briefing.stream(conv, data, EVENING)))
    assert len(pieces) == 2 and pieces[1].startswith("Lá fora")
    assert conv.store.recent(conv.id, 10)[0]["content"].startswith("Lá fora")


def test_weather_retries_once_on_a_server_error(owner_env):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"current": {"temperature_2m": 18.4}, "daily": {}})

    s = Settings()
    service = WeatherService(s, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = asyncio.run(service.forecast())
    assert len(calls) == 2 and out["now"]["temperature"] == 18


def test_voice_loop_speaks_the_briefing_then_listens(tmp_path):
    from backend.voice import speak_briefing
    from tests.test_voice_loop import FakeSpeaker, Harness

    async def scenario():
        h = Harness()
        h.loop.speaker = FakeSpeaker()

        async def tokens():
            yield "Boa noite. "
            yield "Agora faz 18 graus, nublado."

        task = asyncio.create_task(
            speak_briefing(h.s, h.loop.speaker, tokens(), 0.0, {"data": 0.5})
        )
        h.loop.brief(task)
        assert h.state is State.SPEAKING
        await h.loop.task
        await asyncio.sleep(0)
        assert h.loop.speaker.said == ["Boa noite.", "Agora faz 18 graus, nublado."]
        assert h.state is State.LISTENING and h.loop.listen_deadline is not None

    asyncio.run(scenario())


def test_fetch_reports_each_source_for_the_loading_screen(tmp_path):
    conv = make_conv(tmp_path)
    steps: list[tuple[str, str]] = []
    weather = FakeWeather(error=ToolError("503"))
    asyncio.run(briefing.fetch(conv, weather, FakeNews(), on_step=lambda *a: steps.append(a)))
    assert ("weather", "running") in steps and ("weather", "failed") in steps
    assert steps[-1] == ("news", "done") or ("news", "done") in steps
    steps.clear()
    asyncio.run(briefing.fetch(conv, None, FakeNews(), on_step=lambda *a: steps.append(a)))
    assert ("weather", "skipped") in steps


def test_compose_writes_the_whole_briefing_before_it_is_spoken(tmp_path):
    conv = make_conv(tmp_path)
    data = BriefingData(weather=WEATHER, news=NEWS)
    text = asyncio.run(briefing.compose(conv, data, EVENING))
    assert text.startswith("Boa noite") and text.endswith("A OpenAI lançou um modelo.")
    assert "18 graus" in text


def test_briefing_sentences_match_what_is_spoken_so_the_prepared_audio_is_used():
    from backend.voice import _once, speak_tokens, speech_sentences

    class Recorder:
        said: list[str] = []

        async def speak(self, sentences, on_start=None):
            self.said = [x async for x in sentences]
            return True

    text = "Boa noite, senhora. Agora faz 18 graus, nublado. A OpenAI lançou um modelo novo."
    rec = Recorder()
    asyncio.run(speak_tokens(rec, _once(text), 20, "pt"))
    assert rec.said == speech_sentences(text, 20, "pt") and len(rec.said) >= 2
