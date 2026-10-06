import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from backend.config import Settings
from backend.tools import ToolError, build_registry
from backend.tools.clock import Clock, parse_time

TZ = ZoneInfo("America/Sao_Paulo")


def clock_at(hour, minute, second=0, day=5, month=10):
    s = Settings()  # pt-BR
    return Clock(s, now=lambda: datetime(2026, month, day, hour, minute, second, tzinfo=TZ))


def until(clock, **kwargs):
    return asyncio.run(clock.time_until(**kwargs))


def test_minutes_until_a_time_today():
    # The bug: at 17:41 qwen3 said 49 minutes were left until 18:00.
    out = until(clock_at(17, 41, 13), time="18:00")
    assert out == {"now": "17:41", "target": "18:00", "remaining": "19 minutos"}
    assert until(clock_at(9, 0), time="18h30")["remaining"] == "9 horas e 30 minutos"


def test_a_time_already_passed_today_points_to_tomorrow():
    out = until(clock_at(17, 45), time="17:00")
    assert out["already_passed_today_by"] == "45 minutos"
    assert out["remaining_until_tomorrow"] == "23 horas e 15 minutos"


@pytest.mark.parametrize("midnight", ["00:00", "24:00", "23:59"])
def test_midnight_is_the_coming_one(midnight):
    assert until(clock_at(17, 45), time=midnight)["remaining"] == "6 horas e 15 minutos"


def test_days_until_a_date():
    out = until(clock_at(17, 45), date="2026-12-25")
    assert out["days_until"] == 81 and out["date"] == "sexta-feira, 25 de dezembro de 2026"
    assert until(clock_at(17, 45), date="25/12")["days_until"] == 81
    assert until(clock_at(17, 45, day=26, month=12), date="25/12")["days_until"] == 364
    assert until(clock_at(17, 45), date="2026-10-01")["days_ago"] == 4


def test_time_on_a_date():
    out = until(clock_at(17, 45), time="09:00", date="2026-10-06")
    assert out["remaining"] == "15 horas e 15 minutos" and "terça-feira" in out["date"]


@pytest.mark.parametrize("bad", ["25:00", "18:75", "dezoito", ""])
def test_bad_times(bad):
    with pytest.raises(ToolError):
        parse_time(bad)


def test_needs_a_time_or_a_date_and_valid_dates():
    with pytest.raises(ToolError, match="pass a time"):
        until(clock_at(17, 45))
    with pytest.raises(ToolError, match="not a date"):
        until(clock_at(17, 45), date="2026-02-30")
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        until(clock_at(17, 45), date="Natal")


def test_registered_by_default():
    reg = build_registry(Settings())
    assert "time_until" in reg
    assert "time_until" not in build_registry(Settings(tools={"enabled": ["weather"]}))
