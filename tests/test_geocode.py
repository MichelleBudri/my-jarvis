import asyncio
import json

import httpx

from backend.config import Settings
from backend.tools.geocode import pick_result, resolve_location

RESULTS = [
    {
        "name": "Santa Maria",
        "latitude": -15.98,
        "longitude": -48.01,
        "country": "Brasil",
        "country_code": "BR",
        "admin1": "Distrito Federal",
        "timezone": "America/Sao_Paulo",
    },
    {
        "name": "Santa Maria",
        "latitude": -29.68,
        "longitude": -53.81,
        "country": "Brasil",
        "country_code": "BR",
        "admin1": "Rio Grande do Sul",
        "timezone": "America/Sao_Paulo",
    },
]


def test_pick_result_uses_state_abbreviation():
    assert pick_result(RESULTS, ["RS", "Brasil"])["admin1"] == "Rio Grande do Sul"
    assert pick_result(RESULTS, ["DF"])["admin1"] == "Distrito Federal"
    assert pick_result(RESULTS, [])["admin1"] == "Distrito Federal"  # tie: first result wins


def test_resolve_location_geocodes_and_caches(monkeypatch):
    monkeypatch.setenv("JARVIS_CITY", "Santa Maria, RS, Brasil")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["name"])
        return httpx.Response(200, json={"results": RESULTS})

    def client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    s = Settings()
    loc = asyncio.run(resolve_location(s, client()))
    assert calls == ["Santa Maria"]
    assert loc.latitude == -29.68 and loc.timezone == "America/Sao_Paulo"
    assert loc.name == "Santa Maria, Rio Grande do Sul, Brasil"
    assert s.location == loc

    cached = json.loads((s.data_path / "location.json").read_text())
    assert cached["query"] == "Santa Maria, RS, Brasil"

    asyncio.run(resolve_location(Settings(), client()))
    assert len(calls) == 1  # served from cache


def test_resolve_location_offline_keeps_going(monkeypatch):
    monkeypatch.setenv("JARVIS_CITY", "Nowhere")

    def handler(request):
        raise httpx.ConnectError("offline")

    s = Settings()
    loc = asyncio.run(
        resolve_location(s, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    )
    assert not loc.resolved and loc.query == "Nowhere"
