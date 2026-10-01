"""Resolve "City, State, Country" to coordinates and timezone via Open-Meteo."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import httpx

from backend.config import LocationConfig, Settings

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

BR_STATES = {
    "ac": "acre",
    "al": "alagoas",
    "ap": "amapa",
    "am": "amazonas",
    "ba": "bahia",
    "ce": "ceara",
    "df": "distrito federal",
    "es": "espirito santo",
    "go": "goias",
    "ma": "maranhao",
    "mt": "mato grosso",
    "ms": "mato grosso do sul",
    "mg": "minas gerais",
    "pa": "para",
    "pb": "paraiba",
    "pr": "parana",
    "pe": "pernambuco",
    "pi": "piaui",
    "rj": "rio de janeiro",
    "rn": "rio grande do norte",
    "rs": "rio grande do sul",
    "ro": "rondonia",
    "rr": "roraima",
    "sc": "santa catarina",
    "sp": "sao paulo",
    "se": "sergipe",
    "to": "tocantins",
}


class GeocodeError(RuntimeError):
    pass


def _norm(text: str | None) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).strip().lower()


def _score(result: dict, hints: list[str]) -> int:
    fields = {
        _norm(result.get(k)) for k in ("country", "country_code", "admin1", "admin2", "admin3")
    }
    score = 0
    for hint in hints:
        h = _norm(hint)
        if h in fields or BR_STATES.get(h) in fields:
            score += 1
    return score


def pick_result(results: list[dict], hints: list[str]) -> dict:
    # max() is stable: ties keep Open-Meteo's order (most populous first)
    return max(results, key=lambda r: _score(r, hints))


def display_name(result: dict) -> str:
    parts = [result.get("name"), result.get("admin1"), result.get("country")]
    seen: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)
    return ", ".join(seen)


async def geocode(
    query: str, client: httpx.AsyncClient | None = None, language: str = "pt"
) -> dict:
    city, *hints = [p.strip() for p in query.split(",") if p.strip()]
    own = client is None
    client = client or httpx.AsyncClient(timeout=10)
    try:
        resp = await client.get(
            GEOCODE_URL, params={"name": city, "count": 10, "language": language, "format": "json"}
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    finally:
        if own:
            await client.aclose()
    if not results:
        raise GeocodeError(f'City not found: "{query}"')
    best = pick_result(results, hints)
    return {
        "name": display_name(best),
        "latitude": best["latitude"],
        "longitude": best["longitude"],
        "timezone": best.get("timezone"),
    }


def _cache_path(s: Settings) -> Path:
    return s.data_path / "location.json"


async def resolve_location(s: Settings, client: httpx.AsyncClient | None = None) -> LocationConfig:
    """Fill in coordinates/timezone from JARVIS_CITY, cached in data/location.json."""
    loc = s.location
    if loc.resolved or not loc.query:
        return loc

    cache = _cache_path(s)
    data: dict | None = None
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("query") == loc.query:
            data = cached["result"]

    if data is None:
        try:
            data = await geocode(loc.query, client, s.locale.lang)
        except (httpx.HTTPError, GeocodeError) as exc:
            log.warning("Could not geocode %r: %s", loc.query, exc)
            return loc
        cache.write_text(json.dumps({"query": loc.query, "result": data}, ensure_ascii=False))

    resolved = loc.model_copy(
        update={
            "name": loc.name or data["name"],
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "timezone": loc.timezone or data.get("timezone"),
        }
    )
    s.location = resolved
    return resolved
