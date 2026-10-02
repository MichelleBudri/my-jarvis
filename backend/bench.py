"""Compare LLM latency across models: `uv run python -m backend bench qwen3:8b qwen3.5:4b`."""

from __future__ import annotations

import time

import httpx
from rich.console import Console
from rich.table import Table

from backend.brain.conversation import Conversation
from backend.brain.llm import OllamaClient
from backend.config import Settings
from backend.memory.store import MemoryStore

console = Console(highlight=False)

SCRIPT = {
    "pt": [
        "Olá, tudo bem?",
        "Me conte uma curiosidade sobre o espaço.",
        "E quanto tempo a luz do Sol leva para chegar até nós?",
        "Obrigada, era só isso.",
    ],
    "en": [
        "Hello, how are you?",
        "Tell me a fun fact about space.",
        "And how long does sunlight take to reach us?",
        "Thanks, that's all.",
    ],
}


async def _unload(llm: OllamaClient) -> None:
    # Free the RAM before loading the next model (16 GB fills up quickly).
    await llm._client.post("/api/generate", json={"model": llm.cfg.model, "keep_alive": 0})


async def run_bench(s: Settings, models: list[str]) -> int:
    models = models or [s.llm.model]
    installed = {m["name"] for m in httpx.get(f"{s.llm.host}/api/tags", timeout=5).json()["models"]}
    script = SCRIPT.get(s.locale.lang, SCRIPT["en"])

    table = Table(title="LLM latency per turn")
    for col in ("Model", "Turn", "First word", "tok/s", "Reply"):
        table.add_column(col, overflow="fold")

    for model in models:
        if model not in installed and f"{model}:latest" not in installed:
            console.print(f"[yellow]Skipping {model}: run `ollama pull {model}`[/]")
            continue
        llm = OllamaClient(s.llm.model_copy(update={"model": model}))
        conv = Conversation(s, llm, MemoryStore(":memory:"))
        with console.status(f"Loading {model}..."):
            t0 = time.perf_counter()
            await conv.prime()
            load_s = time.perf_counter() - t0
        for i, prompt in enumerate(script, 1):
            with console.status(f"{model} · turn {i}/{len(script)}"):
                reply = "".join([t async for t in conv.reply(prompt)]).strip()
            st = conv.last_stats
            table.add_row(
                f"{model}\n[dim]load + prime {load_s:.1f}s[/]" if i == 1 else "",
                str(i),
                f"{st.first_token_s:.2f}s" if st.first_token_s else "-",
                f"{st.tokens_per_second:.0f}" if st.tokens_per_second else "-",
                reply[:90] + ("…" if len(reply) > 90 else ""),
            )
        table.add_section()
        await _unload(llm)
        await llm.aclose()

    console.print(table)
    console.print(
        "[dim]First word is what you wait for before Jarvis starts talking. Models whose "
        "prompt cache works stay well under 1 s on every turn.[/]"
    )
    return 0
