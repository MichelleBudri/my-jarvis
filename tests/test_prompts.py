import re
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.brain.prompts import build_context_note, build_system_prompt, period_key, season_key
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
    s = Settings()
    prompt = build_system_prompt(s)
    assert "computador de Alice" in prompt
    assert '"senhora" ou "senhora Alice"' in prompt
    assert "feminino" in prompt
    assert "Curitiba" in prompt and "Hemisfério sul" in prompt
    assert "NÃO tem acesso" in prompt
    note = build_context_note(s, datetime(2026, 10, 1, 0, 48, tzinfo=TZ), first_turn=True)
    assert note == (
        "[Contexto: quinta-feira, 1 de outubro de 2026, 00:48, madrugada, primavera. "
        'Início da conversa: cumprimente com "boa noite".]'
    )


def test_system_prompt_is_stable_over_time(owner_env):
    # Any change here invalidates Ollama's prompt cache and adds seconds of latency.
    assert build_system_prompt(Settings()) == build_system_prompt(Settings())
    assert not re.search(r"\d{2}:\d{2}", build_system_prompt(Settings()))


def test_prompt_without_owner_is_neutral():
    s = Settings()
    assert "quem o utiliza" in build_system_prompt(s)
    assert "não marquem gênero" in build_system_prompt(s)
    note = build_context_note(s, now=datetime(2026, 10, 1, 9, tzinfo=TZ))
    assert "primavera" not in note  # no location, no season


def test_prompt_in_english(owner_env, monkeypatch):
    monkeypatch.setenv("JARVIS_LANGUAGE", "en-GB")
    s = Settings()
    prompt = build_system_prompt(s)
    assert "You are Jarvis" in prompt
    assert 'address her as "madam"' in prompt
    assert "Southern hemisphere" in prompt and "British butler" in prompt
    note = build_context_note(s, datetime(2026, 10, 1, 9, tzinfo=TZ), first_turn=True)
    assert "spring" in note and '"good morning"' in note


def test_prompt_other_language_asks_reply_in_it(monkeypatch):
    monkeypatch.setenv("JARVIS_LANGUAGE", "es")
    assert "Always reply in español" in build_system_prompt(Settings())


def test_greeting_hint_only_on_first_turn(owner_env):
    s, now = Settings(), datetime(2026, 10, 1, 20, tzinfo=TZ)
    assert '"boa noite"' in build_context_note(s, now, first_turn=True)
    assert "cumprimente" not in build_context_note(s, now, first_turn=False)
