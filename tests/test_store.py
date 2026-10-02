import sqlite3

from backend.memory.store import MemoryStore


def test_history_order_and_limit(tmp_path):
    store = MemoryStore(tmp_path / "t.db")
    conv = store.new_conversation()
    for i in range(5):
        store.add(conv, "user", f"pergunta {i}")
        store.add(conv, "assistant", f"resposta {i}")
    recent = store.recent(conv, 4)
    assert [m["content"] for m in recent] == [
        "pergunta 3",
        "resposta 3",
        "pergunta 4",
        "resposta 4",
    ]
    assert store.count(conv) == 10
    assert store.last_conversation() == conv


def test_conversations_are_isolated(tmp_path):
    store = MemoryStore(tmp_path / "t.db")
    a, b = store.new_conversation(), store.new_conversation()
    store.add(a, "user", "só na A")
    assert store.recent(b, 10) == []


def test_tool_calls_round_trip_and_cut_turns_are_trimmed(tmp_path):
    store = MemoryStore(tmp_path / "t.db")
    conv = store.new_conversation()
    calls = [{"function": {"name": "set_timer", "arguments": {"minutes": 1}}}]
    store.add(conv, "user", "Timer de 1 minuto")
    store.add(conv, "assistant", "", tool_calls=calls)
    store.add(conv, "tool", '{"ok": true}', tool_name="set_timer")
    store.add(conv, "assistant", "Pronto.")
    assert store.recent(conv, 4)[1] == {"role": "assistant", "content": "", "tool_calls": calls}
    assert store.recent(conv, 4)[2]["tool_name"] == "set_timer"
    # A limit that starts mid-turn never hands the model a call or result without its question.
    assert store.recent(conv, 3) == [{"role": "assistant", "content": "Pronto."}]
    assert store.recent(conv, 2) == [{"role": "assistant", "content": "Pronto."}]


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER,"
        " role TEXT, content TEXT, created_at TEXT);"
        "INSERT INTO conversations (started_at) VALUES ('2026-10-01');"
        "INSERT INTO messages (conversation_id, role, content, created_at)"
        " VALUES (1, 'user', 'antiga', '2026-10-01');"
    )
    db.commit()
    db.close()
    store = MemoryStore(path)
    store.add(1, "tool", "{}", tool_name="echo")
    assert store.recent(1, 10) == [
        {"role": "user", "content": "antiga"},
        {"role": "tool", "content": "{}", "tool_name": "echo"},
    ]
