"""Conversation history and long-term facts, stored in SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    tool_calls      TEXT,  -- JSON, on an assistant message that called tools
    tool_name       TEXT   -- on a tool message
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(messages)")}
        with self.db:
            for col in ("tool_calls", "tool_name"):
                if col not in cols:  # databases created before tool calls were persisted
                    self.db.execute(f"ALTER TABLE messages ADD COLUMN {col} TEXT")

    def close(self) -> None:
        self.db.close()

    def new_conversation(self) -> int:
        with self.db:
            cur = self.db.execute("INSERT INTO conversations (started_at) VALUES (?)", (_now(),))
        return int(cur.lastrowid)

    def last_conversation(self) -> int | None:
        row = self.db.execute("SELECT MAX(id) AS id FROM conversations").fetchone()
        return row["id"]

    def add(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_name: str | None = None,
    ) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO messages "
                "(conversation_id, role, content, created_at, tool_calls, tool_name) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    role,
                    content,
                    _now(),
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    tool_name,
                ),
            )

    def recent(self, conversation_id: int, limit: int) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT role, content, tool_calls, tool_name FROM messages "
            "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        messages: list[dict[str, Any]] = []
        for r in reversed(rows):
            msg: dict[str, Any] = {"role": r["role"], "content": r["content"]}
            if r["tool_calls"]:
                msg["tool_calls"] = json.loads(r["tool_calls"])
            if r["tool_name"]:
                msg["tool_name"] = r["tool_name"]
            messages.append(msg)
        # The limit may cut a turn after its question: a call or result without it.
        while messages and (messages[0]["role"] == "tool" or "tool_calls" in messages[0]):
            messages.pop(0)
        return messages

    def transcript(self, conversation_id: int) -> list[dict[str, str]]:
        """What was said, in order: questions and spoken answers, without tool traffic."""
        rows = self.db.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? "
            "AND role IN ('user', 'assistant') AND tool_calls IS NULL AND content != '' "
            "ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # Long-term memory: facts kept across conversations ("remember that...")

    def add_fact(self, content: str) -> tuple[int, bool]:
        """Store a fact; returns its id and whether it is new (False: already known)."""
        content = " ".join(content.split())
        for fact_id, known in self.facts():
            if known.casefold() == content.casefold():
                return fact_id, False
        with self.db:
            cur = self.db.execute(
                "INSERT INTO facts (content, created_at) VALUES (?, ?)", (content, _now())
            )
        return int(cur.lastrowid), True

    def facts(self) -> list[tuple[int, str]]:
        rows = self.db.execute("SELECT id, content FROM facts ORDER BY id").fetchall()
        return [(r["id"], r["content"]) for r in rows]

    def forget_fact(self, fact_id: int) -> bool:
        with self.db:
            cur = self.db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0

    def forget_all_facts(self) -> int:
        with self.db:
            cur = self.db.execute("DELETE FROM facts")
        return cur.rowcount

    def count(self, conversation_id: int) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return int(row["n"])
