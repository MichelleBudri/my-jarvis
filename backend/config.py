"""Settings, in increasing priority: config.yaml < .env < environment variables."""

from __future__ import annotations

import os
from datetime import tzinfo
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from backend.i18n import Locale, get_locale, normalize_code

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT_DIR / "config.yaml"
ENV_FILE = ROOT_DIR / ".env"
WAKEWORD_VERSION = "v0.1"  # openWakeWord pre-trained model version

# Flat variables for personal data, mapped onto the nested settings.
SIMPLE_ENV: dict[str, tuple[str, str]] = {
    "JARVIS_OWNER_NAME": ("user", "name"),
    "JARVIS_OWNER_GENDER": ("user", "gender"),
    "JARVIS_OWNER_TITLE": ("user", "form_of_address"),
    "JARVIS_LANGUAGE": ("", "language"),
    "JARVIS_CITY": ("location", "query"),
    "JARVIS_LATITUDE": ("location", "latitude"),
    "JARVIS_LONGITUDE": ("location", "longitude"),
    "JARVIS_TIMEZONE": ("location", "timezone"),
}


class LocationConfig(BaseModel):
    query: str | None = None  # e.g. "London, England, United Kingdom"
    name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None  # None = system timezone

    @property
    def resolved(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def display_name(self) -> str | None:
        return self.name or self.query

    @property
    def tz(self) -> tzinfo:
        if self.timezone:
            return ZoneInfo(self.timezone)
        from datetime import datetime

        return datetime.now().astimezone().tzinfo  # type: ignore[return-value]

    @property
    def tz_name(self) -> str:
        return self.timezone or str(self.tz)


class UserConfig(BaseModel):
    name: str | None = None
    gender: str = "neutral"  # female | male | neutral (Portuguese values also accepted)
    form_of_address: str | None = None  # None = language default (madam/sir, senhora/senhor)

    @property
    def kind(self) -> str:
        g = self.gender.strip().lower()
        if g.startswith(("f", "w")):
            return "f"
        if g.startswith("m"):
            return "m"
        return "n"


class PersonaConfig(BaseModel):
    style: str | None = None  # None = British butler, in the chosen language
    max_sentences: int = 2


class LLMConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    keep_alive: str = "30m"
    temperature: float = 0.6
    think: bool = False
    context_messages: int = 30  # a turn with a tool call takes 4: question, call, result, answer


class STTConfig(BaseModel):
    model: str = "mlx-community/whisper-small-mlx"
    language: str | None = None  # None = assistant language


class TTSConfig(BaseModel):
    engine: str = "piper"
    voice: str | None = None  # None = language default
    voices_dir: Path = Path("models/voices")
    say_voice: str | None = None
    length_scale: float | None = None  # speaking speed: < 1 faster, > 1 slower
    sentence_min_chars: int = 20


class AudioConfig(BaseModel):
    input_device: str | int | None = None  # None = system default
    output_device: str | int | None = None
    # Interrupt Jarvis by speaking. Needs headphones, or it will hear itself.
    barge_in: bool = False


class VADConfig(BaseModel):
    threshold: float = 0.5
    start_ms: int = 96
    min_speech_ms: int = 300
    silence_ms: int = 700  # pause that ends an utterance
    preroll_ms: int = 300
    max_utterance_s: float = 30


class WakeWordConfig(BaseModel):
    enabled: bool = True
    model: str = "hey_jarvis"  # openWakeWord pre-trained name, or a path to a custom .onnx
    models_dir: Path = Path("models/wakeword")
    threshold: float = 0.5
    listen_timeout_s: float = 6  # after the wake word, how long to wait for a request
    follow_up_s: float = 8  # after a reply, keep listening without the wake word
    chime: bool = True


class ToolsConfig(BaseModel):
    # weather | news | system | timers. Remove one to hide it from the model.
    enabled: list[str] = Field(default_factory=lambda: ["weather", "news", "system", "timers"])
    max_rounds: int = 3  # tool calls the model may chain before it must answer
    # Say "one moment" when the model is silent this long (it is probably calling a tool).
    one_moment_after_s: float = 1.5


class NewsConfig(BaseModel):
    cache_minutes: int = 30
    max_items: int = 5
    feeds: list[str] = Field(default_factory=list)


class BriefingConfig(BaseModel):
    enabled: bool = True  # weather and AI news on start (`voice --no-briefing` skips it)
    headlines: int = 6  # candidates the model picks the one or two most relevant from
    fetch_timeout_s: float = 5  # a source slower than this is left out


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_nested_delimiter="__",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        yaml_file=CONFIG_FILE,
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    assistant_name: str = "Jarvis"
    language: str = "pt-BR"
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    user: UserConfig = Field(default_factory=UserConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    location: LocationConfig = Field(default_factory=LocationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @property
    def data_path(self) -> Path:
        path = self.data_dir if self.data_dir.is_absolute() else ROOT_DIR / self.data_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, v: str) -> str:
        return normalize_code(v)

    # Language-dependent values
    @property
    def locale(self) -> Locale:
        return get_locale(self.language)

    @property
    def persona_style(self) -> str:
        return self.persona.style or self.locale.persona

    @property
    def owner_address(self) -> str | None:
        """How the assistant addresses the owner: a title (madam, senhora...) or their name."""
        u = self.user
        return u.form_of_address or self.locale.titles.get(u.kind) or u.name

    @property
    def owner_full_address(self) -> str | None:
        return self.locale.full_address(self.owner_address, self.user.name)

    @property
    def voice_name(self) -> str:
        return self.tts.voice or self.locale.voice

    @property
    def voice_path(self) -> Path:
        d = self.tts.voices_dir
        base = d if d.is_absolute() else ROOT_DIR / d
        return base / f"{self.voice_name}.onnx"

    @property
    def stt_language(self) -> str:
        return self.stt.language or self.locale.lang

    @property
    def stt_hint(self) -> str:
        return self.locale.stt_hint.format(name=self.assistant_name)

    @property
    def wakeword_dir(self) -> Path:
        d = self.wakeword.models_dir
        return d if d.is_absolute() else ROOT_DIR / d

    @property
    def wakeword_path(self) -> Path:
        model = self.wakeword.model
        if model.endswith(".onnx"):
            path = Path(model)
            return path if path.is_absolute() else ROOT_DIR / path
        return self.wakeword_dir / f"{model}_{WAKEWORD_VERSION}.onnx"

    @property
    def db_path(self) -> Path:
        return self.data_path / "jarvis.db"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: init args > flat vars > env > .env > config.yaml
        env_file = getattr(dotenv_settings, "env_file", ENV_FILE)
        return (
            init_settings,
            SimpleEnvSource(settings_cls, env_file),
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


class SimpleEnvSource(PydanticBaseSettingsSource):
    """Reads JARVIS_OWNER_NAME, JARVIS_CITY, etc. from the environment and .env."""

    def __init__(self, settings_cls: type[BaseSettings], env_file: Any) -> None:
        super().__init__(settings_cls)
        self.env_file = Path(env_file) if env_file else None

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if self.env_file and self.env_file.exists():
            values.update(dotenv_values(self.env_file))
        values.update(os.environ)
        out: dict[str, dict[str, Any]] = {}
        for var, (section, key) in SIMPLE_ENV.items():
            value = values.get(var)
            if value in (None, ""):
                continue
            if section:
                out.setdefault(section, {})[key] = value.strip()
            else:
                out[key] = value.strip()
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()
