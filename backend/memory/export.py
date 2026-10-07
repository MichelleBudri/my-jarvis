"""The conversation as a Markdown file, from the HUD's "export" button."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from backend.config import Settings
from backend.memory.store import MemoryStore

DOWNLOADS = Path.home() / "Downloads"


def to_markdown(s: Settings, lines: list[dict[str, str]], now: datetime | None = None) -> str:
    tz = s.location.tz
    now = now or datetime.now(tz)
    pt = s.locale.lang == "pt"
    title = f"Conversa com {s.assistant_name}" if pt else f"Conversation with {s.assistant_name}"
    stamp = now.strftime("%d/%m/%Y %H:%M" if pt else "%Y-%m-%d %H:%M")
    exported = "Exportada em" if pt else "Exported on"
    out = [f"# {title}", "", f"_{exported} {stamp}_", ""]
    day = None
    for line in lines:
        at = datetime.fromisoformat(line["created_at"]).astimezone(tz)
        if at.date() != day:
            day = at.date()
            out += [f"## {at.strftime('%d/%m/%Y' if pt else '%Y-%m-%d')}", ""]
        who = s.locale.you_label if line["role"] == "user" else s.assistant_name
        text = line["content"].strip().replace("\n", "  \n")
        out += [f"**{who}** · {at:%H:%M}  ", text, ""]
    if not lines:
        out.append("_Nada foi dito ainda._" if pt else "_Nothing has been said yet._")
    return "\n".join(out).rstrip() + "\n"


def export_conversation(
    s: Settings, store: MemoryStore, conversation_id: int, folder: Path = DOWNLOADS
) -> Path:
    """Write the conversation to Downloads (or data/exports without one); returns the file."""
    now = datetime.now(s.location.tz)
    if not folder.is_dir():
        folder = s.data_path / "exports"
        folder.mkdir(parents=True, exist_ok=True)
    name = "conversa" if s.locale.lang == "pt" else "conversation"
    path = folder / f"{s.assistant_name} - {name} {now:%Y-%m-%d %H-%M-%S}.md"
    path.write_text(to_markdown(s, store.transcript(conversation_id), now), encoding="utf-8")
    return path
