"""Current weather and forecast from Open-Meteo (free, no API key)."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import date

import httpx

from backend.config import Settings
from backend.tools.geocode import GeocodeError, geocode
from backend.tools.registry import Tool, ToolError

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_S = 600
RETRY_AFTER_S = 0.5
MAX_DAYS = 7
FAR_KM = 500
FAR_NOTE = (
    "This place is far from the user's home and speech recognition may have misheard the "
    "city: name the place with its country in the reply."
)

# qwen3:8b read the result as something the user had: "A senhora está com garoa fraca,
# temperatura de 24 graus, umidade de 70 por cento..." when asked about Santo André (D-41).
HOW_TO_REPLY = {
    "pt": (
        "Fale do tempo como um fato sobre o lugar, dizendo o nome dele primeiro, por exemplo "
        '"Em {place}, está garoando e faz 24 graus." Nunca como algo que a pessoa tem '
        '("a senhora está com garoa"). Responda o que foi perguntado: condição e '
        "temperatura, chuva só se for provável ou se perguntarem, umidade e vento só se "
        "perguntarem."
    ),
    "en": (
        'Describe the weather as a fact about the place, naming it first, e.g. "In {place}, '
        "it's drizzling and 24 degrees.\" Never as something the user has. Answer what was "
        "asked: conditions and temperature, rain only if likely or asked, humidity and wind "
        "only if asked."
    ),
}

CURRENT = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
    "weather_code,wind_speed_10m,is_day"
)
DAILY = (
    "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
    "precipitation_sum,sunrise,sunset,uv_index_max"
)

# WMO weather interpretation codes, as used by Open-Meteo.
WMO: dict[int, tuple[str, str]] = {
    0: ("céu limpo", "clear sky"),
    1: ("predominantemente limpo", "mainly clear"),
    2: ("parcialmente nublado", "partly cloudy"),
    3: ("nublado", "overcast"),
    45: ("neblina", "fog"),
    48: ("neblina com geada", "freezing fog"),
    51: ("garoa fraca", "light drizzle"),
    53: ("garoa", "drizzle"),
    55: ("garoa forte", "heavy drizzle"),
    56: ("garoa congelante fraca", "light freezing drizzle"),
    57: ("garoa congelante", "freezing drizzle"),
    61: ("chuva fraca", "light rain"),
    63: ("chuva", "rain"),
    65: ("chuva forte", "heavy rain"),
    66: ("chuva congelante fraca", "light freezing rain"),
    67: ("chuva congelante", "freezing rain"),
    71: ("neve fraca", "light snow"),
    73: ("neve", "snow"),
    75: ("neve forte", "heavy snow"),
    77: ("grãos de neve", "snow grains"),
    80: ("pancadas de chuva fracas", "light showers"),
    81: ("pancadas de chuva", "showers"),
    82: ("pancadas de chuva fortes", "violent showers"),
    85: ("pancadas de neve", "snow showers"),
    86: ("pancadas de neve fortes", "heavy snow showers"),
    95: ("trovoada", "thunderstorm"),
    96: ("trovoada com granizo", "thunderstorm with hail"),
    99: ("trovoada com granizo forte", "thunderstorm with heavy hail"),
}

DAY_LABELS = {"pt": ("hoje", "amanhã"), "en": ("today", "tomorrow")}


def describe(code: int | None, lang: str) -> str | None:
    if code is None:
        return None
    pt, en = WMO.get(code, ("condição desconhecida", "unknown conditions"))
    return pt if lang == "pt" else en


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def _r(value: float | None) -> int | None:
    return None if value is None else round(value)


def summarize(data: dict, s: Settings, place: str | None) -> dict:
    """Trim Open-Meteo's response to what a spoken answer needs."""
    lang = s.locale.lang
    labels = DAY_LABELS.get(lang, DAY_LABELS["en"])
    cur = data.get("current") or {}
    out: dict = {
        "place": place,
        "units": "temperature in degrees Celsius, wind in km/h, precipitation in mm",
        "now": {
            "conditions": describe(cur.get("weather_code"), lang),
            "temperature": _r(cur.get("temperature_2m")),
            "feels_like": _r(cur.get("apparent_temperature")),
            "humidity_percent": cur.get("relative_humidity_2m"),
            "wind_kmh": _r(cur.get("wind_speed_10m")),
            "precipitation_mm": cur.get("precipitation"),
            "daytime": bool(cur.get("is_day")) if "is_day" in cur else None,
        },
        "days": [],
    }
    daily = data.get("daily") or {}
    for i, day in enumerate(daily.get("time") or []):
        d = date.fromisoformat(day)
        name = labels[i] if i < len(labels) else s.locale.weekdays[d.weekday()]

        def col(key: str, i: int = i):
            values = daily.get(key) or []
            return values[i] if i < len(values) else None

        out["days"].append(
            {
                "day": name,
                "date": day,
                "weekday": s.locale.weekdays[d.weekday()],
                "conditions": describe(col("weather_code"), lang),
                "min": _r(col("temperature_2m_min")),
                "max": _r(col("temperature_2m_max")),
                "rain_chance_percent": col("precipitation_probability_max"),
                "rain_mm": col("precipitation_sum"),
                "uv_index": _r(col("uv_index_max")),
                "sunrise": (col("sunrise") or "")[11:16] or None,
                "sunset": (col("sunset") or "")[11:16] or None,
            }
        )
    return out


class WeatherService:
    def __init__(self, s: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.s = s
        self._client = client
        self._cache: dict[tuple, tuple[float, dict]] = {}

    async def _get(self, url: str, params: dict) -> dict:
        # Open-Meteo answers 503 now and then (1 in 6 starts); the next try usually works.
        for attempt in range(2):
            if self._client is not None:
                resp = await self._client.get(url, params=params)
            else:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(url, params=params)
            if resp.status_code < 500 or attempt:
                break
            await asyncio.sleep(RETRY_AFTER_S)
        resp.raise_for_status()
        return resp.json()

    async def _where(self, city: str | None) -> tuple[float, float, str | None, str | None]:
        loc = self.s.location
        if not city:
            if not loc.resolved:
                raise ToolError("no home location configured; ask which city")
            return loc.latitude, loc.longitude, loc.timezone, loc.display_name
        # "Santo André" alone should be the one next door, not one in Portugal.
        home = [p.strip() for p in (loc.display_name or "").split(",")[1:]]
        try:
            found = await geocode(city, self._client, self.s.locale.lang, home)
        except GeocodeError as exc:
            raise ToolError(str(exc)) from exc
        return found["latitude"], found["longitude"], found.get("timezone"), found["name"]

    def _far_from_home(self, lat: float, lon: float) -> bool:
        loc = self.s.location
        if loc.latitude is None or loc.longitude is None:
            return False
        return distance_km(loc.latitude, loc.longitude, lat, lon) > FAR_KM

    async def forecast(self, city: str | None = None, days: int = 2) -> dict:
        days = max(1, min(MAX_DAYS, int(days or 2)))
        try:
            lat, lon, tz, place = await self._where(city)
            key = (round(lat, 2), round(lon, 2), days)
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < CACHE_S:
                data = hit[1]
            else:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "timezone": tz or "auto",
                    "forecast_days": days,
                    "current": CURRENT,
                    "daily": DAILY,
                }
                data = await self._get(FORECAST_URL, params)
                self._cache[key] = (time.monotonic(), data)
        except httpx.HTTPError as exc:
            raise ToolError(f"weather service unavailable: {exc}") from exc
        out = summarize(data, self.s, place)
        lang = "pt" if self.s.locale.lang == "pt" else "en"
        short = (place or "").split(",")[0].strip() or ("aqui" if lang == "pt" else "here")
        out["how_to_reply"] = HOW_TO_REPLY[lang].format(place=short)
        if city and self._far_from_home(lat, lon):
            # "Santandre" (Santo André misheard) matched only a Romanian village: saying
            # the country lets the user catch it.
            out["note"] = FAR_NOTE
        return out


def weather_tools(s: Settings, service: WeatherService | None = None) -> list[Tool]:
    service = service or WeatherService(s)
    return [
        Tool(
            name="get_weather",
            description=(
                "Current weather and daily forecast. Use for any question about weather, "
                "temperature, rain, wind, sunrise or sunset. Without a city, uses the user's "
                "home location."
            ),
            handler=service.forecast,
            parameters={
                "city": {
                    "type": "string",
                    "description": "Only if the user names another place, e.g. 'Lisbon, Portugal'",
                },
                "days": {
                    "type": "integer",
                    "description": "Days of forecast starting today (1-7); 2 covers tomorrow",
                },
            },
            slow=True,
            capability={
                "pt": "consultar o clima e a previsão do tempo",
                "en": "check the weather and forecast",
            },
        )
    ]
