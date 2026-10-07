from datetime import datetime
from zoneinfo import ZoneInfo

from backend.config import Settings
from backend.memory.export import export_conversation, to_markdown
from backend.memory.store import MemoryStore


def make(tmp_path):
    s = Settings(data_dir=str(tmp_path / "data"))
    s.location = s.location.model_copy(update={"timezone": "America/Sao_Paulo"})
    store = MemoryStore(tmp_path / "t.db")
    conv = store.new_conversation()
    store.add(conv, "assistant", "Bom dia, senhora. Agora faz 24 graus.")
    store.add(conv, "user", "Que horas são?")
    store.add(conv, "assistant", "", tool_calls=[{"function": {"name": "time_until"}}])
    store.add(conv, "tool", '{"time": "11:34"}', tool_name="time_until")
    store.add(conv, "assistant", "São 11h34, senhora.")
    return s, store, conv


def test_transcript_keeps_only_what_was_said(tmp_path):
    _, store, conv = make(tmp_path)
    said = [(x["role"], x["content"]) for x in store.transcript(conv)]
    assert said == [
        ("assistant", "Bom dia, senhora. Agora faz 24 graus."),
        ("user", "Que horas são?"),
        ("assistant", "São 11h34, senhora."),
    ]


def test_markdown_has_who_said_what_and_when_in_local_time(tmp_path):
    s, _, _ = make(tmp_path)
    lines = [
        {"role": "user", "content": "Oi", "created_at": "2026-10-07T14:34:00+00:00"},
        {"role": "assistant", "content": "Olá.", "created_at": "2026-10-07T14:34:05+00:00"},
    ]
    now = datetime(2026, 10, 7, 11, 40, tzinfo=ZoneInfo("America/Sao_Paulo"))
    md = to_markdown(s, lines, now)
    assert md.startswith("# Conversa com Jarvis\n")
    assert "## 07/10/2026" in md
    assert "**Você** · 11:34  \nOi" in md and "**Jarvis** · 11:34  \nOlá." in md
    assert "Nada foi dito" in to_markdown(s, [], now)


def test_export_writes_to_the_folder_or_falls_back_to_data(tmp_path):
    s, store, conv = make(tmp_path)
    folder = tmp_path / "Downloads"
    folder.mkdir()
    path = export_conversation(s, store, conv, folder)
    assert path.parent == folder and path.suffix == ".md"
    assert "São 11h34, senhora." in path.read_text(encoding="utf-8")
    path = export_conversation(s, store, conv, tmp_path / "missing")
    assert path.parent == s.data_path / "exports"
