from datetime import datetime
from zoneinfo import ZoneInfo

from backend.brain.prompts import build_system_prompt, period_key, season_key
from backend.config import Settings
from backend.i18n import EN, PT_BR, get_locale

TZ = ZoneInfo("America/Sao_Paulo")


def test_format_datetime():
    dt = datetime(2026, 10, 1, 9, 5, tzinfo=TZ)
    assert PT_BR.format_datetime(dt) == "quinta-feira, 1 de outubro de 2026, 09:05"
    assert EN.format_datetime(dt) == "Thursday, 1 October 2026, 09:05"


def test_season_respects_hemisphere():
    assert season_key(datetime(2026, 10, 1, tzinfo=TZ), -23.7) == "spring"
    assert season_key(datetime(2026, 1, 15, tzinfo=TZ), -23.7) == "summer"
    assert season_key(datetime(2026, 7, 10, tzinfo=TZ), -23.7) == "winter"
    assert season_key(datetime(2026, 4, 10, tzinfo=TZ), -23.7) == "autumn"
    assert season_key(datetime(2026, 10, 1, tzinfo=TZ), 40.0) == "autumn"


def test_period_key():
    assert period_key(datetime(2026, 10, 1, 0, 48)) == "dawn"
    assert period_key(datetime(2026, 10, 1, 9)) == "morning"
    assert period_key(datetime(2026, 10, 1, 15)) == "afternoon"
    assert period_key(datetime(2026, 10, 1, 20)) == "night"


def test_get_locale_normalizes():
    assert get_locale("pt-br").code == "pt-BR"
    assert get_locale("pt_PT").lang == "pt"
    assert get_locale("en-US").language_name == "English"
    assert get_locale("fr").native_prompt is False


def test_prompt_for_owner(owner_env):
    prompt = build_system_prompt(Settings(), now=datetime(2026, 10, 1, 0, 48, tzinfo=TZ))
    assert "computador de Alice" in prompt
    assert '"senhora" ou "senhora Alice"' in prompt
    assert "feminino" in prompt
    assert "primavera" in prompt
    assert "madrugada" in prompt and '"boa noite"' in prompt
    assert "Curitiba" in prompt
    assert "NÃO tem acesso" in prompt


def test_prompt_without_owner_is_neutral():
    prompt = build_system_prompt(Settings(), now=datetime(2026, 10, 1, 9, tzinfo=TZ))
    assert "quem o utiliza" in prompt
    assert "não marquem gênero" in prompt
    assert "Estação" not in prompt  # no location, no season


def test_prompt_in_english(owner_env, monkeypatch):
    monkeypatch.setenv("JARVIS_LANGUAGE", "en-GB")
    prompt = build_system_prompt(Settings(), now=datetime(2026, 10, 1, 9, tzinfo=TZ))
    assert "You are Jarvis" in prompt
    assert 'address her as "madam"' in prompt
    assert "spring (southern hemisphere)" in prompt
    assert '"good morning"' in prompt
    assert "British butler" in prompt


def test_prompt_other_language_asks_reply_in_it(monkeypatch):
    monkeypatch.setenv("JARVIS_LANGUAGE", "es")
    prompt = build_system_prompt(Settings(), now=datetime(2026, 10, 1, 9, tzinfo=TZ))
    assert "Always reply in español" in prompt
