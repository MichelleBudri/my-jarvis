import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from backend.brain.conversation import STALE_RESULT, Conversation
from backend.brain.llm import OllamaClient
from backend.brain.prompts import build_system_prompt
from backend.config import Settings
from backend.i18n import EN, PT_BR
from backend.memory.store import MemoryStore
from backend.tools import Tool, ToolError, ToolRegistry, build_registry
from backend.tools.news import NewsItem, NewsService, merge, parse_feed
from backend.tools.system import parse_battery, parse_volume
from backend.tools.timers import TimerManager
from backend.tools.weather import WeatherService, describe

# Registry


async def _echo(text: str = "") -> dict:
    return {"echo": text}


async def _broken() -> dict:
    raise ToolError("service down")


async def _crash() -> dict:
    raise ValueError("boom")


def make_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "echo",
                "Echo",
                _echo,
                parameters={"text": {"type": "string"}},
                capability={"pt": "ecoar", "en": "echo"},
            ),
            Tool("broken", "Broken", _broken, capability={"en": "echo"}),
            Tool("crash", "Crash", _crash),
        ]
    )


def test_registry_schema_and_capabilities():
    reg = make_registry()
    schema = reg.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["parameters"]["properties"] == {"text": {"type": "string"}}
    assert reg.capabilities("pt") == ["ecoar", "echo"]  # falls back to English, deduplicated
    assert reg.capabilities("en") == ["echo"]


def test_registry_call_never_raises():
    reg = make_registry()

    async def go():
        return [
            await reg.call("echo", {"text": "oi", "junk": 1}),  # unknown args dropped
            await reg.call("echo", '{"text": "json string"}'),
            await reg.call("echo", None),
            await reg.call("broken", {}),
            await reg.call("crash", {}),
            await reg.call("nope", {}),
            await reg.call("echo", "{not json"),
        ]

    out = [json.loads(r) for r in asyncio.run(go())]
    assert out[0] == {"echo": "oi"}
    assert out[1] == {"echo": "json string"}
    assert out[2] == {"echo": ""}
    assert out[3] == {"error": "service down"}
    assert out[4] == {"error": "ValueError: boom"}
    assert "unknown tool" in out[5]["error"]
    assert "not valid JSON" in out[6]["error"]


def test_build_registry_respects_enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_TOOLS__ENABLED", '["weather", "timers"]')
    names = build_registry(Settings()).names
    assert "get_weather" in names and "set_timer" in names
    assert "get_ai_news" not in names and "open_app" not in names


# Conversation with tools


def test_conversation_runs_tools_and_answers(tmp_path):
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            call = {"id": "c1", "function": {"name": "echo", "arguments": {"text": "14"}}}
            lines = [
                {"message": {"content": "", "tool_calls": [call]}, "done": False},
                {"message": {"content": ""}, "done": True},
            ]
        else:
            lines = [
                {"message": {"content": "Fazem 14 graus."}, "done": False},
                {"message": {"content": ""}, "done": True, "eval_count": 5},
            ]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    s = Settings()
    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    store = MemoryStore(tmp_path / "t.db")
    conv = Conversation(s, OllamaClient(s.llm, client=client), store, tools=make_registry())
    conv.add_assistant_note("Boa noite.")  # mid-conversation: no greeting from code

    async def go():
        return "".join([t async for t in conv.reply("Como está o tempo?")])

    assert asyncio.run(go()) == "Fazem 14 graus."
    assert len(payloads) == 2
    assert payloads[0]["tools"] == payloads[1]["tools"]  # stable prefix for the cache
    assert payloads[0]["messages"][1]["content"] == "Boa noite."  # history to pick the call
    msgs = payloads[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]  # not to answer
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["tool_calls"][0]["id"] == "c1"
    assert msgs[-1] == {
        "role": "tool",
        "tool_name": "echo",
        "content": '{"echo": "14"}',
        "tool_call_id": "c1",
    }
    assert conv.last_stats.tools == ["echo"]
    stored = store.recent(conv.id, 10)
    assert stored[1:] == [
        {"role": "user", "content": "Como está o tempo?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "echo", "arguments": {"text": "14"}}}],
        },
        {"role": "tool", "content": '{"echo": "14"}', "tool_name": "echo"},
        {"role": "assistant", "content": "Fazem 14 graus."},
    ]


def test_follow_up_sees_earlier_tool_calls(tmp_path):
    # With only "Timer set." in history, qwen3 answered "cancel it" by claiming it was done.
    payloads: list[dict] = []
    script = [
        {"name": "echo", "arguments": {"text": "timer 1"}},
        "Timer criado.",
        {"name": "echo", "arguments": {"text": "x" * 400}},  # e.g. a news digest
        "Três notícias.",
        "Cancelado.",
    ]

    def handler(request):
        payloads.append(json.loads(request.content))
        step = script[len(payloads) - 1]
        if isinstance(step, dict):
            line = {"message": {"content": "", "tool_calls": [{"function": step}]}, "done": True}
        else:
            line = {"message": {"content": step}, "done": True}
        return httpx.Response(200, text=json.dumps(line))

    s = Settings()
    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    conv = Conversation(
        s, OllamaClient(s.llm, client=client), MemoryStore(tmp_path / "t.db"), tools=make_registry()
    )

    async def go(text):
        return "".join([t async for t in conv.reply(text)])

    assert asyncio.run(go("Timer de 1 minuto")).endswith("Timer criado.")
    assert asyncio.run(go("Notícias?")) == "Três notícias."
    assert asyncio.run(go("Cancele o timer")) == "Cancelado."
    msgs = payloads[-1]["messages"]
    assert [m["role"] for m in msgs] == [
        "system",
        *["user", "assistant", "tool", "assistant"] * 2,
        "user",
    ]
    call = msgs[2]["tool_calls"][0]["function"]
    assert call == {"name": "echo", "arguments": {"text": "timer 1"}}
    assert msgs[3]["content"] == '{"echo": "timer 1"}'  # short: kept for "cancel it"
    assert json.loads(msgs[7]["content"]) == json.loads(STALE_RESULT)  # long and stale


def test_interrupted_answer_after_a_tool_is_kept(tmp_path):
    turn = [0]

    def handler(request):
        turn[0] += 1
        if turn[0] == 1:
            call = {"function": {"name": "echo", "arguments": {}}}
            lines = [{"message": {"content": "", "tool_calls": [call]}, "done": True}]
        else:
            lines = [
                {"message": {"content": "Fazem 14 graus."}, "done": False},
                {"message": {"content": " E venta."}, "done": True},
            ]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    s = Settings()
    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    store = MemoryStore(tmp_path / "t.db")
    conv = Conversation(s, OllamaClient(s.llm, client=client), store, tools=make_registry())
    conv.add_assistant_note("Boa noite.")

    async def go():
        gen = conv.reply("Tempo?")
        async for token in gen:
            if token == "Fazem 14 graus.":
                break  # barge-in: the user spoke over the answer
        await gen.aclose()

    asyncio.run(go())
    assert [m["role"] for m in store.recent(conv.id, 10)][-3:] == ["assistant", "tool", "assistant"]
    assert store.recent(conv.id, 10)[-1]["content"] == "Fazem 14 graus."


def test_conversation_stops_after_max_rounds(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_TOOLS__MAX_ROUNDS", "2")
    calls = [0]

    def handler(request):
        calls[0] += 1
        call = {"function": {"name": "echo", "arguments": {}}}
        lines = [{"message": {"content": "", "tool_calls": [call]}, "done": True}]
        return httpx.Response(200, text=json.dumps(lines[0]))

    s = Settings()
    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    conv = Conversation(
        s, OllamaClient(s.llm, client=client), MemoryStore(tmp_path / "t.db"), tools=make_registry()
    )

    conv.add_assistant_note("Boa noite.")

    async def go():
        return [t async for t in conv.reply("loop")]

    assert asyncio.run(go()) == []
    assert calls[0] == 3  # max_rounds tool rounds + the last chance to answer


def test_prompt_lists_tools_instead_of_not_installed():
    s = Settings()
    without = build_system_prompt(s)
    assert "acesso a clima, notícias" in without
    with_tools = build_system_prompt(s, ["consultar o clima"])
    assert "consultar o clima" in with_tools and "ferramentas" in with_tools
    assert "acesso a clima" not in with_tools
    assert '"14 graus"' in with_tools


# Weather


FORECAST = {
    "current": {
        "time": "2026-10-02T14:45",
        "temperature_2m": 13.7,
        "apparent_temperature": 12.2,
        "relative_humidity_2m": 99,
        "precipitation": 0.0,
        "weather_code": 3,
        "wind_speed_10m": 7.7,
        "is_day": 1,
    },
    "daily": {
        "time": ["2026-10-02", "2026-10-03", "2026-10-04"],
        "weather_code": [51, 0, 95],
        "temperature_2m_max": [14.5, 23.2, 20.0],
        "temperature_2m_min": [10.8, 10.8, 15.1],
        "precipitation_probability_max": [35, 4, 80],
        "precipitation_sum": [0.7, 0.2, 12.0],
        "sunrise": ["2026-10-02T05:55", "2026-10-03T05:54", "2026-10-04T05:53"],
        "sunset": ["2026-10-02T18:30", "2026-10-03T18:31", "2026-10-04T18:31"],
        "uv_index_max": [3.2, 9.1, 5.0],
    },
}


def test_weather_summary(owner_env):
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=FORECAST)

    s = Settings()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    svc = WeatherService(s, client)
    out = asyncio.run(svc.forecast(days=3))
    assert seen["params"]["latitude"] == "-25.43"
    assert seen["params"]["timezone"] == "America/Sao_Paulo"
    assert out["now"]["conditions"] == "nublado"
    assert out["now"]["temperature"] == 14 and out["now"]["feels_like"] == 12
    days = out["days"]
    assert [d["day"] for d in days] == ["hoje", "amanhã", "domingo"]
    assert days[1]["conditions"] == "céu limpo" and days[1]["max"] == 23
    assert days[2]["rain_chance_percent"] == 80 and days[0]["sunset"] == "18:30"

    asyncio.run(svc.forecast(days=3))  # cached: no second request needed
    assert describe(95, "en") == "thunderstorm" and describe(1234, "pt")


def test_weather_without_location_asks_for_city():
    with pytest.raises(ToolError, match="ask which city"):
        asyncio.run(WeatherService(Settings()).forecast())


# News


RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>AI News &amp; More | TechCrunch</title>
<item><title>Robots learn to fold laundry</title>
<pubDate>{recent}</pubDate>
<description><![CDATA[<p>A new <b>model</b> folds shirts.</p>]]></description></item>
<item><title>Old story</title><pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate></item>
</channel></rss>"""

GOOGLE = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Google Notícias</title>
<item><title>IA aprende a dobrar roupas - Folha</title><source url="x">Folha</source>
<pubDate>{recent}</pubDate>
<description>&lt;a href="x"&gt;IA aprende a dobrar roupas&lt;/a&gt; Folha</description></item>
<item><title>Robots learn to fold laundry - Wired</title><source url="y">Wired</source>
<pubDate>{recent}</pubDate></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>AI | The Verge</title>
<entry><title type="html">Newest thing</title>
<link rel="alternate" href="https://v/1"/>
<published>{iso}</published>
<summary type="html">Short &amp; sweet.</summary></entry></feed>"""


def _feeds(now: datetime) -> dict[str, str]:
    recent = (now - timedelta(hours=5)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    iso = (now - timedelta(hours=1)).isoformat()
    return {
        "https://tc/feed": RSS.format(recent=recent),
        "https://gn/feed": GOOGLE.format(recent=recent),
        "https://verge/feed": ATOM.format(iso=iso),
    }


def test_parse_feeds():
    feeds = _feeds(datetime.now(UTC))
    tc = parse_feed(feeds["https://tc/feed"], "https://tc/feed")
    assert tc[0].source == "TechCrunch" and tc[0].summary == "A new model folds shirts."
    gn = parse_feed(feeds["https://gn/feed"])
    assert gn[0].title == "IA aprende a dobrar roupas" and gn[0].source == "Folha"
    assert gn[0].summary == ""  # Google repeats the title; dropped
    verge = parse_feed(feeds["https://verge/feed"])
    assert verge[0].source == "The Verge" and verge[0].link == "https://v/1"
    assert verge[0].summary == "Short & sweet."


def test_merge_sorts_dedupes_and_drops_stale():
    now = datetime.now(UTC)
    feeds = [parse_feed(x) for x in _feeds(now).values()]
    titles = [i.title for i in merge(feeds, now)]
    assert titles[0] == "Newest thing"
    assert titles.count("Robots learn to fold laundry") == 1
    assert "Old story" not in titles


def test_news_service_filters_and_caches(monkeypatch):
    feeds = _feeds(datetime.now(UTC))
    monkeypatch.setenv("JARVIS_NEWS__FEEDS", json.dumps([*feeds, "https://down/feed"]))
    hits = []

    def handler(request):
        url = str(request.url)
        hits.append(url)
        if url in feeds:
            return httpx.Response(200, text=feeds[url])
        return httpx.Response(503)

    svc = NewsService(Settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = asyncio.run(svc.headlines(count=2))
    assert out["found"] == 3 and len(out["items"]) == 2
    assert list(out)[-1] == "how_to_reply"  # last, closest to where the answer starts
    assert out["items"][0] == {
        "title": "Newest thing",
        "source": "The Verge",
        "hours_ago": 1,
        "summary": "Short & sweet.",
    }
    robots = asyncio.run(svc.headlines(topic="robôs laundry"))
    assert robots["found"] == 0
    robots = asyncio.run(svc.headlines(topic="Laundry"))
    assert [i["title"] for i in robots["items"]] == ["Robots learn to fold laundry"]
    assert len(hits) == 4  # one fetch per feed, then served from the cache


def test_news_all_feeds_down(monkeypatch):
    monkeypatch.setenv("JARVIS_NEWS__FEEDS", '["https://a", "https://b"]')
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(ToolError, match="unavailable"):
        asyncio.run(NewsService(Settings(), client).headlines())


# System


def test_parse_battery_and_volume():
    text = (
        "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t85%; discharging; "
        "4:05 remaining present: true"
    )
    assert parse_battery(text) == {
        "percent": 85,
        "status": "discharging",
        "on_power_adapter": False,
        "time_remaining": "4h05",
    }
    charging = "Now drawing from 'AC Power'\n -InternalBattery-0\t100%; charged; 0:00 remaining"
    assert parse_battery(charging) == {
        "percent": 100,
        "status": "charged",
        "on_power_adapter": True,
    }
    assert parse_battery("Now drawing from 'AC Power'") is None
    vol = "output volume:25, input volume:50, alert volume:100, output muted:false"
    assert parse_volume(vol) == {"volume_percent": 25, "muted": False}


# Timers and durations


def test_format_duration():
    assert PT_BR.format_duration(300) == "5 minutos"
    assert PT_BR.format_duration(3725) == "1 hora, 2 minutos e 5 segundos"
    assert EN.format_duration(61) == "1 minute and 1 second"
    assert EN.format_duration(0) == "0 seconds"


def test_timer_fires_and_announces(owner_env):
    async def scenario():
        fired = []
        tm = TimerManager(Settings(), on_fire=fired.append)
        out = await tm.set_timer(seconds=0.01, label="tirar o bolo")
        assert out["ok"] and out["duration"] == "0 segundos"
        assert (await tm.list_timers())["active"][0]["label"] == "tirar o bolo"
        await asyncio.sleep(0.05)
        assert len(fired) == 1 and not tm.timers
        assert tm.announcement(fired[0]) == (
            "Com licença, senhora, o timer de 0 segundos terminou: tirar o bolo."
        )

    asyncio.run(scenario())


def test_timer_validation_and_cancel():
    async def scenario():
        fired = []
        tm = TimerManager(Settings(), on_fire=fired.append)
        with pytest.raises(ToolError):
            await tm.set_timer()
        with pytest.raises(ToolError):
            await tm.set_timer(hours=25)
        with pytest.raises(ToolError, match="no active"):
            await tm.cancel_timer()
        await tm.set_timer(minutes=5, label="chá")
        await tm.set_timer(minutes=10, label="massa")
        with pytest.raises(ToolError, match="several"):
            await tm.cancel_timer()
        out = await tm.cancel_timer(label="Chá")
        assert out["cancelled"] == [{"id": 1, "label": "chá"}]
        assert (await tm.cancel_timer())["cancelled"][0]["label"] == "massa"
        await asyncio.sleep(0)
        assert not tm.timers and not fired
        assert (
            TimerManager(Settings()).announcement(type("T", (), {"seconds": 300, "label": None})())
            == "Com licença, o timer de 5 minutos terminou."
        )

    asyncio.run(scenario())


def test_change_timer_keeps_the_label_and_restarts():
    async def scenario():
        tm = TimerManager(Settings())
        with pytest.raises(ToolError, match="no active"):
            await tm.change_timer(minutes=2)
        await tm.set_timer(minutes=1, label="testar")
        out = await tm.change_timer(minutes=2)  # the only active one
        assert out["label"] == "testar" and out["duration"] == "2 minutos"
        assert out["changed_from"] == "1 minuto"
        assert [t.seconds for t in tm.timers.values()] == [120]
        await tm.set_timer(minutes=5)  # no label
        with pytest.raises(ToolError, match="several"):
            await tm.change_timer(minutes=3)
        await tm.change_timer(id=out["id"] + 1, minutes=3)
        assert sorted(t.seconds for t in tm.timers.values()) == [120, 180]
        tm.cancel_all()

    asyncio.run(scenario())


def test_slow_tool_says_one_moment_without_persisting_it(tmp_path):
    turn = [0]

    def handler(request):
        turn[0] += 1
        if turn[0] == 1:
            call = {"function": {"name": "slow", "arguments": {}}}
            line = {"message": {"content": "", "tool_calls": [call]}, "done": True}
        else:
            line = {"message": {"content": "Pronto."}, "done": True}
        return httpx.Response(200, text=json.dumps(line))

    s = Settings()
    reg = ToolRegistry([Tool("slow", "Slow", _echo, slow=True)])
    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    store = MemoryStore(tmp_path / "t.db")
    conv = Conversation(s, OllamaClient(s.llm, client=client), store, tools=reg)
    conv.add_assistant_note("Boa noite.")

    async def go():
        return [t async for t in conv.reply("Notícias?")]

    assert asyncio.run(go()) == ["Um momento. ", "Pronto."]
    assert store.recent(conv.id, 10)[-1]["content"] == "Pronto."


def test_blank_reply_is_retried_once(tmp_path):
    # Ollama sometimes swallows a malformed tool call: tokens generated, nothing returned.
    turn = [0]

    def handler(request):
        turn[0] += 1
        if turn[0] == 1:
            line = {"message": {"content": ""}, "done": True, "eval_count": 18}
        else:
            line = {"message": {"content": "Oitenta por cento."}, "done": True, "eval_count": 4}
        return httpx.Response(200, text=json.dumps(line))

    s = Settings()
    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    conv = Conversation(
        s, OllamaClient(s.llm, client=client), MemoryStore(tmp_path / "t.db"), tools=make_registry()
    )
    conv.add_assistant_note("Boa noite.")

    async def go():
        return "".join([t async for t in conv.reply("Bateria?")])

    assert asyncio.run(go()) == "Oitenta por cento."
    assert turn[0] == 2


def test_system_status_topic_limits_the_report(monkeypatch):
    from backend.tools import system

    async def fake_run(*cmd):
        return 0, "Now drawing from 'AC Power'\n -InternalBattery-0\t90%; charging; 0:40 remaining"

    monkeypatch.setattr(system, "run", fake_run)
    out = asyncio.run(system.system_status("battery"))
    assert out == {
        "battery": {
            "percent": 90,
            "status": "charging",
            "on_power_adapter": True,
            "time_remaining": "0h40",
        }
    }
    assert "disk" in asyncio.run(system.system_status("bogus"))  # unknown topic → everything


def test_merge_interleaves_feeds():
    now = datetime.now(UTC)

    def item(title, hours):
        return NewsItem(title, "x", now - timedelta(hours=hours))

    busy = [item(f"busy {n}", n) for n in range(1, 6)]
    curated = [item("curated old", 10), item("curated older", 20)]
    titles = [i.title for i in merge([busy, curated], now)]
    assert titles[:4] == ["busy 1", "curated old", "busy 2", "curated older"]
    assert titles[4:] == ["busy 3", "busy 4", "busy 5"]


def test_timer_with_same_label_replaces_the_old_one():
    async def scenario():
        tm = TimerManager(Settings())
        await tm.set_timer(minutes=5, label="ligar para a mãe")
        await tm.set_timer(minutes=3, label="chá")
        out = await tm.set_timer(minutes=10, label="Ligar para a mãe")
        assert out["replaced_previous_timer"] and out["duration"] == "10 minutos"
        assert sorted(t.label for t in tm.timers.values()) == ["Ligar para a mãe", "chá"]
        assert "replaced_previous_timer" not in await tm.set_timer(minutes=1)
        tm.cancel_all()

    asyncio.run(scenario())


def _scripted(rounds, tmp_path, reg=None, delay=0.0):
    """A Conversation whose model streams the given rounds of chunks, one per request."""
    turn = [0]

    async def handler(request):
        lines = rounds[min(turn[0], len(rounds) - 1)]
        turn[0] += 1
        await asyncio.sleep(delay)
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    s = Settings()
    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    conv = Conversation(
        s,
        OllamaClient(s.llm, client=client),
        MemoryStore(tmp_path / "t.db"),
        tools=reg or make_registry(),
    )
    conv.add_assistant_note("Boa noite.")
    return conv, turn


def test_tool_name_written_as_text_is_called_not_spoken(tmp_path):
    slept = []

    async def go_to_sleep():
        slept.append(1)
        return {"ok": True}

    reg = ToolRegistry([Tool("go_to_sleep", "Sleep", go_to_sleep)])
    conv, _ = _scripted(
        [
            [{"message": {"content": "Go"}}, {"message": {"content": "_to_sleep."}, "done": True}],
            [{"message": {"content": "Até logo."}, "done": True}],
        ],
        tmp_path,
        reg,
    )

    async def go():
        return "".join([t async for t in conv.reply("Obrigada.")])

    assert asyncio.run(go()) == "Até logo."
    assert slept == [1]


def test_text_starting_like_a_tool_name_is_spoken(tmp_path):
    conv, _ = _scripted(
        [[{"message": {"content": "Ec"}}, {"message": {"content": "o de quê?"}, "done": True}]],
        tmp_path,
    )

    async def go():
        return "".join([t async for t in conv.reply("Repete?")])

    assert asyncio.run(go()) == "Eco de quê?"


def test_one_moment_when_the_model_is_quiet(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_TOOLS__ONE_MOMENT_AFTER_S", "0.01")
    call = {"function": {"name": "echo", "arguments": {"text": "x"}}}
    conv, _ = _scripted(
        [
            [{"message": {"content": "", "tool_calls": [call]}, "done": True}],
            [{"message": {"content": "Pronto."}, "done": True}],
        ],
        tmp_path,
        delay=0.05,
    )

    async def go():
        return [t async for t in conv.reply("Ecoa x")]

    assert asyncio.run(go()) == ["Um momento. ", "Pronto."]  # said once, not again later
    assert conv.store.recent(conv.id, 10)[-1]["content"] == "Pronto."


def test_warm_primes_history_and_the_next_note(tmp_path):
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "x"}, "done": True})

    s = Settings()
    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    conv = Conversation(
        s,
        OllamaClient(s.llm, client=client),
        MemoryStore(tmp_path / "t.db"),
        tools=make_registry(),
    )
    conv.store.add(conv.id, "user", "Oi?")
    conv.store.add(conv.id, "assistant", "Olá.")
    asyncio.run(conv.warm())
    p = payloads[0]
    assert [m["role"] for m in p["messages"]] == ["system", "user", "assistant", "user"]
    assert p["messages"][-1]["content"].startswith("[Contexto:")
    assert "think" not in p and p["options"]["num_predict"] == 1
    assert p["tools"] == conv._schemas


SANTO_ANDRE = [
    {
        "name": "Santo André",
        "admin1": "Distrito de Setúbal",
        "country": "Portugal",
        "country_code": "PT",
        "latitude": 38.06,
        "longitude": -8.78,
    },
    {
        "name": "Santo André",
        "admin1": "Paraná",
        "country": "Brasil",
        "country_code": "BR",
        "latitude": -25.4,
        "longitude": -49.3,
    },
]


def test_geocode_prefers_the_home_region_on_ties():
    from backend.tools.geocode import pick_result

    assert pick_result(SANTO_ANDRE, [])["country"] == "Portugal"  # Open-Meteo's order
    assert pick_result(SANTO_ANDRE, [], ["PR", "Brasil"])["country"] == "Brasil"
    assert pick_result(SANTO_ANDRE, ["Portugal"], ["PR", "Brasil"])["country"] == "Portugal"


def test_weather_far_from_home_asks_to_name_the_country(owner_env):
    def handler(request):
        if "geocoding" in str(request.url):
            name = request.url.params["name"]
            results = SANTO_ANDRE[:1] if name == "Lisboa" else SANTO_ANDRE
            return httpx.Response(200, json={"results": results})
        return httpx.Response(200, json=FORECAST)

    svc = WeatherService(Settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    near = asyncio.run(svc.forecast(city="Santo André"))
    assert near["place"].endswith("Brasil") and "note" not in near
    far = asyncio.run(svc.forecast(city="Lisboa"))
    assert far["place"].endswith("Portugal") and "country" in far["note"]
    assert list(far)[-1] == "note"
    assert "note" not in asyncio.run(svc.forecast())  # home itself


class FakeMac:
    """Plays osascript's volume commands against an in-memory volume."""

    def __init__(self, level, muted=False):
        self.level, self.muted = level, muted

    async def run(self, *cmd):
        script = " ".join(cmd[2::2])
        if "get volume settings" in script:
            return 0, f"output volume:{self.level}, output muted:{str(self.muted).lower()}"
        for line in cmd[2::2]:
            *_, value = line.split()
            if "output volume" in line:
                self.level = int(value)
            elif "output muted" in line:
                self.muted = value == "true"
        return 0, ""


def test_unmute_restores_the_level_after_muting_with_zero(monkeypatch):
    from backend.tools import system

    mac = FakeMac(20)
    monkeypatch.setattr(system, "run", mac.run)
    monkeypatch.setattr(system, "_restore", {"level": None})
    assert asyncio.run(system.set_volume(level=0))["volume_percent"] == 0
    out = asyncio.run(system.set_volume(mute=False))
    assert out == {"ok": True, "volume_percent": 20, "muted": False}


def test_mute_and_unmute_keep_the_level(monkeypatch):
    from backend.tools import system

    mac = FakeMac(20)
    monkeypatch.setattr(system, "run", mac.run)
    monkeypatch.setattr(system, "_restore", {"level": None})
    assert asyncio.run(system.set_volume(mute=True))["muted"] is True
    out = asyncio.run(system.set_volume(mute=False))
    assert out == {"ok": True, "volume_percent": 20, "muted": False}
