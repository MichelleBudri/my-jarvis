"""Per-language strings, date formatting, default titles and voices."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime


@dataclass(frozen=True)
class Locale:
    code: str  # e.g. pt-BR
    lang: str  # e.g. pt
    language_name: str
    weekdays: tuple[str, ...]
    months: tuple[str, ...]
    seasons: dict[str, str]
    periods: dict[str, str]
    greetings: dict[str, str]  # keyed by period
    titles: dict[str, str]  # keyed by gender kind: f | m
    persona: str
    voice: str  # default Piper voice
    you_label: str
    farewell: str
    stt_hint: str  # primes Whisper with the assistant's name and conversational style
    native_prompt: bool = True  # False: English prompt + reply-in-language rule
    # Spoken durations and timer announcements
    units: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "h": ("hour", "hours"),
            "m": ("minute", "minutes"),
            "s": ("second", "seconds"),
        }
    )
    and_word: str = "and"
    timer_done: str = "Excuse me{addr}, your {duration} timer is up."
    timer_done_label: str = "Excuse me{addr}, your {duration} timer is up: {label}."
    one_moment: str = "One moment."  # while a slow tool runs

    def format_datetime(self, now: datetime) -> str:
        wd, month = self.weekdays[now.weekday()], self.months[now.month - 1]
        if self.lang == "pt":
            return f"{wd}, {now.day} de {month} de {now.year}, {now:%H:%M}"
        return f"{wd}, {now.day} {month} {now.year}, {now:%H:%M}"

    def full_address(self, title: str | None, name: str | None) -> str | None:
        if not title:
            return name
        if not name or title == name:
            return title
        return f"{title} {name}" if self.lang == "pt" else title

    def format_duration(self, seconds: float) -> str:
        """3725 → "1 hour, 2 minutes and 5 seconds", in words a voice can read."""
        total = max(0, round(seconds))
        h, rest = divmod(total, 3600)
        m, s = divmod(rest, 60)
        parts = [
            f"{n} {self.units[k][0] if n == 1 else self.units[k][1]}"
            for n, k in ((h, "h"), (m, "m"), (s, "s"))
            if n
        ] or [f"0 {self.units['s'][1]}"]
        if len(parts) == 1:
            return parts[0]
        return f"{', '.join(parts[:-1])} {self.and_word} {parts[-1]}"


PT_BR = Locale(
    code="pt-BR",
    lang="pt",
    language_name="português do Brasil",
    weekdays=(
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo",
    ),
    months=(
        "janeiro",
        "fevereiro",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    ),
    seasons={"spring": "primavera", "summer": "verão", "autumn": "outono", "winter": "inverno"},
    periods={"dawn": "madrugada", "morning": "manhã", "afternoon": "tarde", "night": "noite"},
    greetings={
        "dawn": "boa noite",
        "morning": "bom dia",
        "afternoon": "boa tarde",
        "night": "boa noite",
    },
    titles={"f": "senhora", "m": "senhor"},
    persona=(
        "mordomo britânico: formal, educado e impecável, com humor seco e sutil, "
        "usado com moderação; nunca bajulador"
    ),
    voice="pt_BR-faber-medium",
    you_label="Você",
    farewell="Às suas ordens{addr}. Até breve.",
    stt_hint="Olá, {name}. Como está o tempo hoje? Quais são as notícias?",
    units={"h": ("hora", "horas"), "m": ("minuto", "minutos"), "s": ("segundo", "segundos")},
    and_word="e",
    timer_done="Com licença{addr}, o timer de {duration} terminou.",
    timer_done_label="Com licença{addr}, o timer de {duration} terminou: {label}.",
    one_moment="Um momento.",
)

EN = Locale(
    code="en-GB",
    lang="en",
    language_name="British English",
    weekdays=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    months=(
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ),
    seasons={"spring": "spring", "summer": "summer", "autumn": "autumn", "winter": "winter"},
    periods={
        "dawn": "small hours",
        "morning": "morning",
        "afternoon": "afternoon",
        "night": "evening",
    },
    greetings={
        "dawn": "good evening",
        "morning": "good morning",
        "afternoon": "good afternoon",
        "night": "good evening",
    },
    titles={"f": "madam", "m": "sir"},
    persona=(
        "a British butler: formal, polite and impeccable, with a dry, subtle wit "
        "used sparingly; never sycophantic"
    ),
    voice="en_GB-alan-medium",
    you_label="You",
    farewell="At your service{addr}. Until next time.",
    stt_hint="Hello, {name}. What's the weather like today? Any news?",
)

OTHER_NAMES = {
    "es": "español",
    "fr": "français",
    "de": "Deutsch",
    "it": "italiano",
    "nl": "Nederlands",
    "ja": "日本語",
    "zh": "中文",
}


def normalize_code(code: str) -> str:
    parts = code.replace("_", "-").strip().split("-")
    lang = parts[0].lower()
    return f"{lang}-{parts[1].upper()}" if len(parts) > 1 and parts[1] else lang


def get_locale(code: str | None) -> Locale:
    """pt-* and en-* are fully supported; other languages get the English prompt."""
    code = normalize_code(code or "pt-BR")
    lang = code.split("-")[0]
    if lang == "pt":
        return replace(PT_BR, code=code)
    if lang == "en":
        return replace(
            EN, code=code, language_name="English" if code != "en-GB" else EN.language_name
        )
    return replace(
        EN,
        code=code,
        lang=lang,
        language_name=OTHER_NAMES.get(lang, code),
        native_prompt=False,
    )
