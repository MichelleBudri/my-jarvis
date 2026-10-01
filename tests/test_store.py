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
