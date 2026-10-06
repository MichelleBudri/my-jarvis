"""`python -m backend memory`: see and delete what Jarvis remembers, without asking it."""

from __future__ import annotations

from rich.console import Console

from backend.config import Settings
from backend.memory.store import MemoryStore

console = Console(highlight=False)


def run_memory(s: Settings, action: str = "list", fact_id: int | None = None) -> int:
    store = MemoryStore(s.db_path)
    try:
        if action == "forget":
            if fact_id is None:
                console.print("[red]Which one? `memory forget <id>` (ids from `memory`)[/]")
                return 1
            if not store.forget_fact(fact_id):
                console.print(f"[red]No memory with id {fact_id}[/]")
                return 1
            console.print(f"[green]Forgot memory {fact_id}[/]")
            return 0
        if action == "clear":
            n = store.forget_all_facts()
            console.print(f"[green]Forgot {n} memories[/]")
            return 0
        facts = store.facts()
        if not facts:
            console.print(
                f'[dim]Nothing remembered yet. Ask: "{s.assistant_name}, remember that..."[/]'
            )
        for i, text in facts:
            console.print(f"[cyan]{i:>4}[/]  {text}")
        return 0
    finally:
        store.close()
