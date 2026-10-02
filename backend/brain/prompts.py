"""System prompt and persona, per language."""

from __future__ import annotations

from datetime import datetime

from backend.config import Settings
from backend.i18n import Locale


def season_key(now: datetime, latitude: float) -> str:
    """Approximate astronomical season, hemisphere-aware."""
    md = (now.month, now.day)
    if (3, 20) <= md < (6, 21):
        north = "spring"
    elif (6, 21) <= md < (9, 22):
        north = "summer"
    elif (9, 22) <= md < (12, 21):
        north = "autumn"
    else:
        north = "winter"
    if latitude >= 0:
        return north
    return {"spring": "autumn", "summer": "winter", "autumn": "spring", "winter": "summer"}[north]


def period_key(now: datetime) -> str:
    h = now.hour
    if h < 5:
        return "dawn"
    if h < 12:
        return "morning"
    if h < 18:
        return "afternoon"
    return "night"


# Portuguese


def _treatment_pt(s: Settings) -> str:
    u, addr, full = s.user, s.owner_address, s.owner_full_address
    if u.kind in ("f", "m") and addr:
        fem = u.kind == "f"
        a = "a" if fem else "o"
        names = f'"{addr}"' + (f' ou "{full}"' if full != addr else "")
        return (
            f"chame-{a} de {names}. {'Ela é mulher' if fem else 'Ele é homem'}. "
            f"Ao falar {'dela' if fem else 'dele'}, use sempre o "
            f"{'feminino' if fem else 'masculino'}, como em "
            f'"{a} {addr} está {"pronta" if fem else "pronto"}?". Nunca use "você": '
            f'diga sempre "{a} {addr}" ("posso ajudar {a} {addr}?", "para {a} {addr}").'
        )
    who = f'chame a pessoa de "{addr}"' if addr else "trate a pessoa com cortesia"
    return f'{who} e prefira construções que não marquem gênero, como "tudo pronto para você?".'


def _context_pt(s: Settings, loc: Locale, now: datetime, first_turn: bool) -> str:
    period = period_key(now)
    parts = [loc.format_datetime(now), loc.periods[period]]
    if s.location.latitude is not None:
        parts.append(loc.seasons[season_key(now, s.location.latitude)])
    note = f"[Contexto: {', '.join(parts)}."
    if first_turn:
        note += f' Início da conversa: cumprimente com "{loc.greetings[period]}".'
    return note + "]"


def _prompt_pt(s: Settings, loc: Locale, can: str) -> str:
    lang = loc.language_name[:1].upper() + loc.language_name[1:]
    where = f" Localização: {s.location.display_name}." if s.location.display_name else ""
    hemi = ""
    if s.location.latitude is not None:
        hemi = f" Hemisfério {'sul' if s.location.latitude < 0 else 'norte'}."
    return f"""\
Você é {s.assistant_name}, um assistente pessoal de inteligência artificial que roda \
inteiramente no computador de {s.user.name or "quem o utiliza"}, sem depender da nuvem.

Personalidade: {s.persona_style}. Você é prestativo, discreto e confiante; o humor \
aparece em comentários breves e elegantes, nunca às custas da resposta.

Tratamento: {_treatment_pt(s)}

Fuso horário: {s.location.tz_name}.{where}{hemi} Cada mensagem chega com uma nota \
[Contexto: ...] gerada automaticamente com data, hora, período do dia e estação. Use-a \
quando for útil, mas nunca a mencione nem a repita.

O que você consegue fazer agora: {can or "conversar e responder com o seu conhecimento geral"}. \
Você ainda NÃO tem acesso a clima, notícias, agenda, e-mail, internet ou controle do \
computador. Se pedirem algo assim, diga em uma única frase que esse recurso ainda não \
está instalado. Não peça desculpas em excesso e nunca ofereça fazer algo que não consegue.

Como responder:
- {lang} correto, com a polidez de um mordomo britânico.
- Seja breve: no máximo {s.persona.max_sentences} frases curtas, porque suas respostas \
serão faladas em voz alta. Sem rodeios nem floreios; aprofunde só se pedirem.
- Cumprimente só quando a nota de contexto indicar o início da conversa; nas demais \
mensagens, vá direto ao ponto, sem saudação.
- Apenas texto corrido, natural para ser lido em voz alta: sem markdown, listas, \
emojis, tabelas ou símbolos especiais.
- Nunca invente dados.
"""


# English (also used, with a reply-language rule, for unsupported languages)


def _treatment_en(s: Settings) -> str:
    u, addr, full = s.user, s.owner_address, s.owner_full_address
    if u.kind in ("f", "m") and addr:
        names = f'"{addr}"' + (f' or "{full}"' if full != addr else "")
        if u.kind == "f":
            return f"address her as {names}; use she/her pronouns."
        return f"address him as {names}; use he/him pronouns."
    who = f'address the person as "{addr}"' if addr else "address the person courteously"
    return f"{who}, using gender-neutral language."


def _context_en(s: Settings, loc: Locale, now: datetime, first_turn: bool) -> str:
    period = period_key(now)
    parts = [loc.format_datetime(now), loc.periods[period]]
    if s.location.latitude is not None:
        parts.append(loc.seasons[season_key(now, s.location.latitude)])
    note = f"[Context: {', '.join(parts)}."
    if first_turn:
        greeting = f' with "{loc.greetings[period]}"' if loc.native_prompt else ""
        note += f" Start of the conversation: greet{greeting}."
    return note + "]"


def _prompt_en(s: Settings, loc: Locale, can: str) -> str:
    where = f" Location: {s.location.display_name}." if s.location.display_name else ""
    hemi = ""
    if s.location.latitude is not None:
        hemi = f" {'Southern' if s.location.latitude < 0 else 'Northern'} hemisphere."
    language_rule = (
        f"- Always reply in {loc.language_name}, whatever language this prompt is in."
        if not loc.native_prompt
        else f"- Correct {loc.language_name}, with the courtesy of a British butler."
    )
    return f"""\
You are {s.assistant_name}, a personal AI assistant running entirely on the computer of \
{s.user.name or "your user"}, with no cloud dependency.

Personality: {s.persona_style}. You are helpful, discreet and confident; humour comes \
through in brief, elegant remarks, never at the expense of the answer.

Form of address: {_treatment_en(s)}

Timezone: {s.location.tz_name}.{where}{hemi} Each message arrives with an automatic \
[Context: ...] note holding the date, time, time of day and season. Use it when helpful, \
but never mention or repeat it.

What you can do right now: {can or "converse and answer from your general knowledge"}. \
You do NOT yet have access to weather, news, calendar, e-mail, the internet or control of \
the computer. If asked for any of these, say in a single sentence that the feature is not \
installed yet. Do not over-apologise and never offer something you cannot do.

How to reply:
{language_rule}
- Be brief: at most {s.persona.max_sentences} short sentences, because your replies will be \
spoken aloud. No padding or flourishes; go deeper only when asked.
- Greet only when the context note marks the start of the conversation; otherwise get \
straight to the point, with no greeting.
- Plain flowing text that sounds natural when read aloud: no markdown, lists, emojis, \
tables or special symbols.
- Never make up data.
"""


def build_system_prompt(s: Settings, capabilities: list[str] | None = None) -> str:
    """Static per session, so Ollama can reuse its cached prefix between turns."""
    can = "; ".join(capabilities or [])
    if s.locale.lang == "pt":
        return _prompt_pt(s, s.locale, can)
    return _prompt_en(s, s.locale, can)


def build_context_note(s: Settings, now: datetime | None = None, first_turn: bool = False) -> str:
    """Changes every minute, so it travels with the user message, not the system prompt."""
    now = now or datetime.now(s.location.tz)
    if s.locale.lang == "pt":
        return _context_pt(s, s.locale, now, first_turn)
    return _context_en(s, s.locale, now, first_turn)
