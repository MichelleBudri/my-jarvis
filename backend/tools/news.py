"""AI news from RSS and Atom feeds (no API key), merged, deduplicated and cached."""

from __future__ import annotations

import asyncio
import html
import itertools
import logging
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx

from backend.config import Settings
from backend.tools.registry import Tool, ToolError

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) my-jarvis"
MAX_AGE_H = 72  # search feeds (Google News) also return old articles
SUMMARY_CHARS = 220
REPLY_HINT = (
    "Spoken reply: one or two flowing sentences naming the two or three most relevant "
    "headlines, in the reply language. No list, no numbering, no line breaks."
)

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


@dataclass
class NewsItem:
    title: str
    source: str
    published: datetime | None = None
    summary: str = ""
    link: str = ""

    def hours_ago(self, now: datetime) -> int | None:
        if self.published is None:
            return None
        return max(0, int((now - self.published).total_seconds() // 3600))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, *names: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) in names:
            return c
    return None


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return _SPACE.sub(" ", html.unescape(_TAG.sub(" ", "".join(el.itertext())))).strip()


def _date(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)  # RSS: RFC 2822
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))  # Atom: ISO 8601
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _site_name(channel_title: str, url: str) -> str:
    # "AI News & Artificial Intelligence | TechCrunch" → "TechCrunch"
    name = re.split(r"\s+[|–-]\s+", channel_title)[-1].strip() if channel_title else ""
    return name or urlparse(url).netloc.removeprefix("www.")


def parse_feed(xml_text: str, url: str = "") -> list[NewsItem]:
    root = ET.fromstring(xml_text)
    channel = _child(root, "channel") if _local(root.tag) == "rss" else root
    site = _site_name(_text(_child(channel, "title")) if channel is not None else "", url)
    items: list[NewsItem] = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        title = _text(_child(el, "title"))
        if not title:
            continue
        source = _text(_child(el, "source"))  # Google News names the original outlet
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].rstrip()
        summary = _text(_child(el, "description", "summary"))
        if _norm(summary).startswith(_norm(title)[:40]):
            summary = ""  # Google News repeats the title as an HTML link
        if len(summary) > SUMMARY_CHARS:
            summary = summary[:SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"
        link_el = _child(el, "link")
        link = ""
        if link_el is not None:
            link = link_el.get("href") or (link_el.text or "").strip()
        published = _child(el, "pubDate", "published", "updated", "date")
        items.append(
            NewsItem(
                title=title,
                source=source or site,
                published=_date(published.text or "") if published is not None else None,
                summary=summary,
                link=link,
            )
        )
    return items


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if c.isalnum() or c == " ").strip()


def merge(feeds: list[list[NewsItem]], now: datetime, max_age_h: int = MAX_AGE_H) -> list[NewsItem]:
    """Feeds interleaved, newest first, without duplicates or stale items.

    Round by round, each feed gives its next newest item. Sorting everything by date
    alone let the busy Google News searches (~90% of items, many from small outlets)
    push the curated feeds out of the top five.
    """
    oldest = datetime.min.replace(tzinfo=UTC)

    def newest_first(items: list[NewsItem]) -> list[NewsItem]:
        return sorted(items, key=lambda i: i.published or oldest, reverse=True)

    seen: set[str] = set()
    fresh: list[list[NewsItem]] = []
    for feed in feeds:
        kept = []
        for item in feed:
            key = _norm(item.title)[:60]
            hours = item.hours_ago(now)
            if key in seen or (hours is not None and hours > max_age_h):
                continue
            seen.add(key)
            kept.append(item)
        fresh.append(newest_first(kept))
    out: list[NewsItem] = []
    for round_ in itertools.zip_longest(*fresh):
        out += newest_first([i for i in round_ if i is not None])
    return out


class NewsService:
    def __init__(self, s: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.s = s
        self._client = client
        self._cache: tuple[float, list[NewsItem]] | None = None

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> list[NewsItem]:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return parse_feed(resp.text, url)
        except (httpx.HTTPError, ET.ParseError) as exc:
            log.warning("News feed failed %s: %s", url, exc)
            return []

    async def items(self) -> list[NewsItem]:
        ttl = self.s.news.cache_minutes * 60
        if self._cache and time.monotonic() - self._cache[0] < ttl:
            return self._cache[1]
        feeds = self.s.news.feeds
        if not feeds:
            raise ToolError("no news feeds configured")
        if self._client is not None:
            results = await asyncio.gather(*(self._fetch(self._client, u) for u in feeds))
        else:
            async with httpx.AsyncClient(
                timeout=10, follow_redirects=True, headers={"User-Agent": USER_AGENT}
            ) as client:
                results = await asyncio.gather(*(self._fetch(client, u) for u in feeds))
        if not any(results):
            raise ToolError("news feeds unavailable right now")
        merged = merge(list(results), datetime.now(UTC))
        self._cache = (time.monotonic(), merged)
        return merged

    async def headlines(self, topic: str | None = None, count: int | None = None) -> dict:
        count = max(1, min(10, int(count or self.s.news.max_items)))
        items = await self.items()
        if topic:
            words = _norm(topic).split()
            items = [i for i in items if all(w in _norm(f"{i.title} {i.summary}") for w in words)]
        now = datetime.now(UTC)
        return {
            "topic": topic or "artificial intelligence",
            "found": len(items),
            "items": [
                {
                    "title": i.title,
                    "source": i.source,
                    "hours_ago": i.hours_ago(now),
                    **({"summary": i.summary} if i.summary else {}),
                }
                for i in items[:count]
            ],
            # Last, next to where the answer starts: without it the reply came back as a
            # numbered list 6 times out of 6 once the history was left out (D-18).
            "how_to_reply": REPLY_HINT,
        }


def news_tools(s: Settings, service: NewsService | None = None) -> list[Tool]:
    service = service or NewsService(s)
    return [
        Tool(
            name="get_ai_news",
            description=(
                "Latest artificial intelligence news headlines from several sources, newest "
                "first. Headlines may be in English or Portuguese: retell them in the reply "
                "language. Pick the two or three most relevant and summarise briefly."
            ),
            handler=service.headlines,
            parameters={
                "topic": {
                    "type": "string",
                    "description": "Optional keyword filter, e.g. 'OpenAI' or 'robots'",
                },
                "count": {"type": "integer", "description": "How many headlines (1-10)"},
            },
            slow=True,
            capability={
                "pt": "trazer as últimas notícias de inteligência artificial",
                "en": "fetch the latest artificial intelligence news",
            },
        )
    ]
