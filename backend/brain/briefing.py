"""Activation briefing: weather and AI news, fetched in parallel, told in a few sentences.

The greeting and the weather are written by code and spoken at once; the model only picks
and retells the news, generating while they are spoken (D-24).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
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
    "pt": "Resposta falada: no máximo {n} frases curtas de texto corrido com as uma ou duas "
    "manchetes mais importantes sobre inteligência artificial, ignorando as que não forem "
    "sobre inteligência artificial. Sem cumprimentar, sem falar do tempo e sem rótulos "
    'como "Notícias:".',
    "en": "Spoken reply: at most {n} short sentences of flowing text with the one or two "
    "most important artificial intelligence headlines, skipping any that are not about "
    'artificial intelligence. No greeting, no weather and no labels such as "News:".',
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
) -> BriefingData:
    """Both sources at once; one that fails or is slow is left out, never awaited past it."""
    cfg = conv.s.briefing

    async def get_weather() -> dict[str, Any] | None:
        if weather is None:
            return None
        return weather_brief(await weather.forecast(days=1))

    async def get_news() -> list[dict[str, Any]] | None:
        if news is None:
            return None
        found = await news.headlines(count=cfg.headlines)
        return [{"title": i["title"], "source": i["source"]} for i in found["items"]] or None

    async def guarded(name: str, coro) -> Any:
        try:
            return await asyncio.wait_for(coro, cfg.fetch_timeout_s)
        except (TimeoutError, ToolError) as exc:
            log.warning("Briefing without %s: %s", name, exc or "timed out")
        except Exception:  # noqa: BLE001 - the briefing never stops the app starting
            log.exception("Briefing without %s", name)
        return None

    w, n = await asyncio.gather(guarded("weather", get_weather()), guarded("news", get_news()))
    return BriefingData(weather=w, news=n)


def weather_sentence(conv: Conversation, data: BriefingData) -> str:
    """Today's weather from the numbers: instant, and never made up.

    Asked to "cover only the news" when the weather had failed, qwen3:8b said "23 graus,
    mínima de 19" anyway; with data, it read "trovoada" as "a senhora está com nublado".
    """
    w = data.weather or {}
    now, today = w.get("now") or {}, w.get("today") or {}
    temp = now.get("temperature")
    if temp is None:
        return ""
    pt = conv.s.locale.lang == "pt"
    text = f"Agora faz {temp} graus" if pt else f"It is {temp} degrees"
    if (feels := now.get("feels_like")) is not None and abs(feels - temp) >= 3:
        text += f", com sensação de {feels}" if pt else f", feeling like {feels}"
    if now.get("conditions"):
        text += f", {now['conditions']}"
    text += "."
    low, high = today.get("min"), today.get("max")
    if low is None or high is None:
        return text
    later = []
    if (cond := today.get("conditions")) and cond != now.get("conditions"):
        later.append(f"previsão de {cond}" if pt else f"{cond} expected")
    later += (
        [f"mínima de {low}", f"máxima de {high}"]
        if pt
        else [f"a low of {low}", f"a high of {high}"]
    )
    if (rain := today.get("rain_chance_percent")) is not None and rain >= RAIN_WORTH_SAYING:
        later.append(
            f"{rain} por cento de chance de chuva" if pt else f"a {rain} per cent chance of rain"
        )
    joined = f"{', '.join(later[:-1])} {conv.s.locale.and_word} {later[-1]}"
    return f"{text} {'Hoje' if pt else 'Today'}, {joined}."


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
        if weather := weather_sentence(conv, data):
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
