from backend.config import Settings


def test_yaml_defaults_are_generic():
    s = Settings()
    assert s.assistant_name == "Jarvis"
    assert s.user.name is None and s.user.kind == "n"
    assert s.location.query is None and not s.location.resolved
    assert s.llm.model
    assert s.news.feeds


def test_simple_env_vars(owner_env):
    s = Settings()
    assert s.user.name == "Alice"
    assert s.owner_address == "senhora"
    assert s.owner_full_address == "senhora Alice"
    assert s.location.query.startswith("Curitiba")
    assert s.location.latitude == -25.43
    assert s.location.tz_name == "America/Sao_Paulo"


def test_masculine_and_custom_title(monkeypatch):
    monkeypatch.setenv("JARVIS_OWNER_NAME", "Carlos")
    monkeypatch.setenv("JARVIS_OWNER_GENDER", "masculino")
    assert Settings().owner_full_address == "senhor Carlos"
    monkeypatch.setenv("JARVIS_OWNER_TITLE", "doutor")
    assert Settings().owner_full_address == "doutor Carlos"


def test_neutral_uses_name(monkeypatch):
    monkeypatch.setenv("JARVIS_OWNER_NAME", "Alex")
    s = Settings()
    assert s.user.kind == "n" and s.owner_address == "Alex" and s.owner_full_address == "Alex"


def test_nested_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("JARVIS_LLM__MODEL", "modelo-teste")
    monkeypatch.setenv("JARVIS_LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.llm.model == "modelo-teste"
    assert s.log_level == "DEBUG"


def test_dotenv_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("JARVIS_OWNER_NAME=Bia\nJARVIS_CITY=Recife, PE, Brasil\n")
    s = Settings(_env_file=env)
    assert s.user.name == "Bia"
    assert s.location.query == "Recife, PE, Brasil"


def test_voice_follows_language(monkeypatch):
    s = Settings()
    assert s.voice_path.is_absolute()
    assert s.voice_name == "pt_BR-faber-medium" and s.stt_language == "pt"
    monkeypatch.setenv("JARVIS_LANGUAGE", "en-gb")
    s = Settings()
    assert s.language == "en-GB"
    assert s.voice_name == "en_GB-alan-medium" and s.stt_language == "en"
    monkeypatch.setenv("JARVIS_TTS__VOICE", "minha-voz")
    assert Settings().voice_name == "minha-voz"


def test_title_depends_on_language(monkeypatch):
    monkeypatch.setenv("JARVIS_OWNER_NAME", "Alice")
    monkeypatch.setenv("JARVIS_OWNER_GENDER", "feminino")
    monkeypatch.setenv("JARVIS_LANGUAGE", "en-GB")
    s = Settings()
    assert s.owner_address == "madam" and s.owner_full_address == "madam"
    monkeypatch.setenv("JARVIS_OWNER_TITLE", "Miss Alice")
    assert Settings().owner_address == "Miss Alice"


def test_language_in_dotenv(tmp_path):
    env = tmp_path / ".env"
    env.write_text("JARVIS_LANGUAGE=es\n")
    s = Settings(_env_file=env)
    assert s.language == "es" and s.locale.language_name == "español"
    assert not s.locale.native_prompt
