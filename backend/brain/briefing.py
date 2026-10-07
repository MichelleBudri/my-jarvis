"""Activation briefing: weather and AI news, fetched in parallel, told in a few sentences.

The greeting and the weather are written by code and spoken at once; the model only picks
and retells the news, generating while they are spoken (D-24).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from backend.brain.conversation import Conversation
from backend.brain.llm import LLMError
from backend.brain.prompts import build_context_note, build_greeting
from backend.tools.news import NewsService
from backend.tools.registry import ToolError
from backend.tools.weather import WeatherService

log = logging.getLogger(__name__)

RAIN_WORTH_SAYING = 40  # per cent

# The headlines go in as the result of a get_ai_news call made by code: asked to retell
# headlines pasted into the user message, qwen3:8b called get_ai_news 3 times in 3.
REQUEST = {
    "pt": "Briefing de ativação: a saudação e o tempo já foram ditos. Quais são as notícias "
    "mais importantes de inteligência artificial?",
    "en": "Activation briefing: the greeting and the weather have already been said. What "
    "are the most important artificial intelligence news items?",
}
HOW_TO_REPLY = {
    "pt": "Resposta falada, como um mordomo que conta as novidades do dia: no máximo {n} "
    "frases curtas, em português natural do Brasil, com as uma ou duas manchetes mais "
    "importantes sobre inteligência artificial. Comece com uma transição curta, como "
    '"Nas notícias de inteligência artificial," ou "Sobre inteligência artificial,". Cada '
    "manchete vira uma frase própria que diz quem fez o quê, como num telejornal. Traduza "
    "as manchetes em inglês com naturalidade, sem deixar palavras em inglês, e use siglas "
    "em português (FMI, ONU, UE). Escreva frases completas, com os artigos do português "
    'falado ("aprovou uma moratória", "a bolha"), e "inteligência artificial" por extenso, '
    'nunca "IA". '
    "Nunca junte duas manchetes num mesmo fato nem acrescente o que a manchete não diz. "
    "Ignore as que não forem sobre inteligência artificial. Sem cumprimentar, sem falar do "
    'tempo, sem rótulos e sem fórmulas como "a senhora pode acompanhar" ou "vale destacar".',
    "en": "Spoken reply, like a butler telling the day's news: at most {n} short sentences "
    "with the one or two most important artificial intelligence headlines. Open with a "
    'short transition such as "In artificial intelligence news,". Each headline becomes '
    "its own sentence saying who did what, as a newsreader would. Never merge two "
    "headlines into one fact or add anything the headline does not say. Skip any that are "
    "not about artificial intelligence. No greeting, no weather, no labels and no stock "
    'phrases such as "you may wish to follow".',
}


@dataclass
class BriefingData:
    weather: dict[str, Any] | None = None
    news: list[dict[str, Any]] | None = None


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def weather_brief(forecast: dict[str, Any]) -> dict[str, Any]:
    """Just what a sentence about today needs."""
    now = forecast.get("now") or {}
    today = (forecast.get("days") or [{}])[0]
    return {
        "now": _compact(
            {
                "conditions": now.get("conditions"),
                "temperature": now.get("temperature"),
                "feels_like": now.get("feels_like"),
            }
        ),
        "today": _compact(
            {
                "conditions": today.get("conditions"),
                "min": today.get("min"),
                "max": today.get("max"),
                "rain_chance_percent": today.get("rain_chance_percent"),
            }
        ),
    }


async def fetch(
    conv: Conversation,
    weather: WeatherService | None,
    news: NewsService | None,
    on_step: Callable[[str, str], None] | None = None,
) -> BriefingData:
    """Both sources at once; one that fails or is slow is left out, never awaited past it.

    `on_step(source, status)` reports each one for the loading screen: running, then done,
    failed or skipped.
    """
    cfg = conv.s.briefing
    report = on_step or (lambda name, status: None)

    async def get_weather() -> dict[str, Any] | None:
        if weather is None:
            return None
        return weather_brief(await weather.forecast(days=1))

    async def get_news() -> list[dict[str, Any]] | None:
        if news is None:
            return None
        found = await news.headlines(count=cfg.headlines)
        return [{"title": i["title"], "source": i["source"]} for i in found["items"]] or None

    async def guarded(name: str, coro, source) -> Any:
        if source is None:
            coro.close()
            report(name, "skipped")
            return None
        report(name, "running")
        try:
            result = await asyncio.wait_for(coro, cfg.fetch_timeout_s)
            report(name, "done" if result else "failed")
            return result
        except (TimeoutError, ToolError) as exc:
            log.warning("Briefing without %s: %s", name, exc or "timed out")
        except Exception:  # noqa: BLE001 - the briefing never stops the app starting
            log.exception("Briefing without %s", name)
        report(name, "failed")
        return None

    w, n = await asyncio.gather(
        guarded("weather", get_weather(), weather), guarded("news", get_news(), news)
    )
    return BriefingData(weather=w, news=n)


# How the sky is right now, as the end of "It is 22 degrees ...": matched by keyword, so
# "garoa forte" and "garoa" both read "e está garoando". First match wins.
NOW_PHRASES = {
    "pt": [
        ("trovoada", "e há trovoadas"),
        ("granizo", "e está caindo granizo"),
        ("neve", "e está nevando"),
        ("garoa", "e está garoando"),
        ("pancadas", "e há pancadas de chuva"),
        ("chuva", "e está chovendo"),
        ("neblina", "com neblina"),
        ("parcialmente nublado", "com algumas nuvens"),
        ("nublado", "e o céu está encoberto"),
        ("predominantemente limpo", "com poucas nuvens"),
        ("céu limpo", "e o céu está limpo"),
    ],
    "en": [
        ("thunder", "with thunderstorms"),
        ("hail", "with hail"),
        ("snow", "and snowing"),
        ("drizzle", "and drizzling"),
        ("shower", "with showers"),
        ("rain", "and raining"),
        ("fog", "and foggy"),
        ("partly cloudy", "with a few clouds"),
        ("overcast", "under a grey sky"),
        ("mainly clear", "with hardly a cloud"),
        ("clear", "under a clear sky"),
    ],
}
RAIN_LIKELY = 70  # per cent: say to take an umbrella
EVENING_HOUR = 18  # from then on today's low and high are history: left out


def _sky_now(conditions: str | None, lang: str) -> str | None:
    if not conditions:
        return None
    for key, phrase in NOW_PHRASES[lang]:
        if key in conditions.lower():
            return phrase
    return f"com {conditions}" if lang == "pt" else f"with {conditions}"


def _degrees(n: int, lang: str) -> str:
    if lang == "pt":
        return f"{n} grau" if abs(n) == 1 else f"{n} graus"
    return f"{n} degree" if abs(n) == 1 else f"{n} degrees"


def weather_sentence(conv: Conversation, data: BriefingData, now: datetime | None = None) -> str:
    """Today's weather from the numbers: instant, and never made up (D-40).

    Asked to "cover only the news" when the weather had failed, qwen3:8b said "23 graus,
    mínima de 19" anyway; with data, it read "trovoada" as "a senhora está com nublado".
    Written as a person would say it, not as a list: "Lá fora estão 22 graus e está
    garoando", not "Agora faz 22 graus, garoa".
    """
    w = data.weather or {}
    current, today = w.get("now") or {}, w.get("today") or {}
    temp = current.get("temperature")
    if temp is None:
        return ""
    lang = "pt" if conv.s.locale.lang == "pt" else "en"
    pt = lang == "pt"
    now = now or datetime.now(conv.s.location.tz)

    if pt:
        verb = "está" if abs(temp) == 1 else "estão"
        text = f"Lá fora {verb} {_degrees(temp, lang)}"
    else:
        text = f"It's {_degrees(temp, lang)} out"
    if sky := _sky_now(current.get("conditions"), lang):
        text += f" {sky}"
    if (feels := current.get("feels_like")) is not None and abs(feels - temp) >= 3:
        text += f", com sensação de {feels}" if pt else f", though it feels like {feels}"
    sentences = [f"{text}."]

    low, high = today.get("min"), today.get("max")
    later = today.get("conditions")
    changes = later and later != current.get("conditions")
    if now.hour < EVENING_HOUR and low is not None and high is not None:
        if pt:
            span = f"entre {low} e {_degrees(high, lang)}"
            day = (
                f"Ao longo do dia, a previsão é de {later}, com a temperatura {span}."
                if changes
                else f"Ao longo do dia, a temperatura fica {span}."
            )
        else:
            span = f"between {low} and {_degrees(high, lang)}"
            day = (
                f"Later on, expect {later}, {span}." if changes else f"Today it should stay {span}."
            )
        sentences.append(day)

    rain = today.get("rain_chance_percent")
    if rain is not None and rain >= RAIN_LIKELY:
        sentences.append("É bom ter o guarda-chuva à mão." if pt else "An umbrella would be wise.")
    elif rain is not None and rain >= RAIN_WORTH_SAYING:
        sentences.append(
            f"Há {rain} por cento de chance de chuva."
            if pt
            else f"There is a {rain} per cent chance of rain."
        )
    return " ".join(sentences)


def build_messages(
    conv: Conversation, data: BriefingData, now: datetime | None = None
) -> list[dict[str, Any]]:
    s = conv.s
    lang = "pt" if s.locale.lang == "pt" else "en"
    call = {"function": {"name": "get_ai_news", "arguments": {}}}
    result = {
        "items": data.news,
        "how_to_reply": HOW_TO_REPLY[lang].format(n=s.persona.max_sentences),
    }
    return [
        {"role": "system", "content": conv.system_prompt},
        {"role": "user", "content": f"{build_context_note(s, now)}\n{REQUEST[lang]}"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_name": "get_ai_news",
            "content": json.dumps(result, ensure_ascii=False),
        },
    ]


_END = object()


async def _news(conv: Conversation, data: BriefingData, now: datetime | None, out: asyncio.Queue):
    messages = build_messages(conv, data, now)
    try:
        # The tools are sent but never run: they are part of the prompt prefix Ollama has
        # cached, and leaving them out made it re-process the system prompt (~3 s).
        async for chunk in conv.llm.chat_stream(messages, tools=conv.tool_schemas):
            if chunk.content:
                out.put_nowait(chunk.content)
            if chunk.tool_calls:
                log.warning("Briefing model called %s; ignored", chunk.tool_calls)
    except (LLMError, httpx.HTTPError) as exc:
        log.warning("Briefing without news, model failed: %s", exc)
    finally:
        out.put_nowait(_END)


async def stream(
    conv: Conversation, data: BriefingData, now: datetime | None = None
) -> AsyncIterator[str]:
    """The greeting and the weather, then the model's news as it is generated.

    The model starts at once, so its prompt is processed while the rest is spoken. Only
    weather and news are kept in history, as something said unprompted: follow-ups ("tell
    me more about the second one") need them, and a stored greeting would be copied.
    """
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(_news(conv, data, now, queue)) if data.news else None
    said: list[str] = []
    try:
        if greeting := build_greeting(conv.s, now):
            yield f"{greeting} "
        if weather := weather_sentence(conv, data, now):
            said.append(f"{weather} ")
            yield f"{weather} "
        if task:
            while (item := await queue.get()) is not _END:
                said.append(item)
                yield item
    finally:
        if task:
            task.cancel()
        if text := "".join(said).strip():
            conv.add_assistant_note(text)
            conv.schedule_warm()  # the first question then finds the briefing cached


async def compose(conv: Conversation, data: BriefingData, now: datetime | None = None) -> str:
    """The whole briefing as one text, so it can be spoken without pauses (D-38)."""
    return "".join([piece async for piece in stream(conv, data, now)]).strip()
