"""How much memory Jarvis holds: its own processes and the model loaded in Ollama.

`python -m backend footprint` measures a running Jarvis from outside; the HUD's system
panel shows the same numbers from inside.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import httpx

from backend.config import Settings

# `top` reports the physical footprint, which counts Metal buffers (Whisper's weights on
# the GPU) that RSS leaves out: "1234M", "56K", "1.2G", sometimes with a "+" or "-".
_TOP_MEM = re.compile(r"^(\d+)\s+([\d.]+)([BKMG])", re.MULTILINE)
_UNIT = {"B": 1 / 1024**2, "K": 1 / 1024, "M": 1.0, "G": 1024.0}


@dataclass
class Footprint:
    jarvis_mb: float | None  # this process and its children (the HUD window)
    model_mb: float | None  # 0 when Ollama has the model unloaded; None when unreachable
    model: str | None

    def as_event(self) -> dict:
        return {
            "type": "footprint",
            "jarvis_mb": self.jarvis_mb,
            "model_mb": self.model_mb,
            "model": self.model,
        }


def parse_top(text: str) -> dict[int, float]:
    """`top -l 1 -stats pid,mem` → {pid: MB}."""
    return {int(pid): float(n) * _UNIT[unit] for pid, n, unit in _TOP_MEM.findall(text)}


async def _out(*cmd: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    out, _ = await asyncio.wait_for(proc.communicate(), 10)
    return out.decode(errors="replace")


async def processes_mb(pid: int) -> float | None:
    """A process and its direct children (the HUD window), as Activity Monitor counts them."""
    try:
        children = [int(p) for p in (await _out("pgrep", "-P", str(pid))).split()]
        pids = [pid, *children]
        args = [a for p in pids for a in ("-pid", str(p))]
        found = parse_top(await _out("top", "-l", "1", "-stats", "pid,mem", *args))
    except (OSError, TimeoutError, ValueError):
        return None
    total = sum(found.get(p, 0.0) for p in pids)
    return round(total, 1) if pid in found else None


def model_mb(ps: dict, model: str) -> float:
    """Ollama's /api/ps → MB held by `model` (0 when not loaded)."""
    for m in ps.get("models") or []:
        if m.get("name") == model or m.get("model") == model:
            return round(m.get("size", 0) / 1024**2, 1)
    return 0.0


async def measure(s: Settings, pid: int | None) -> Footprint:
    async def ollama() -> float | None:
        try:
            async with httpx.AsyncClient(base_url=s.llm.host, timeout=3) as client:
                resp = await client.get("/api/ps")
                resp.raise_for_status()
                return model_mb(resp.json(), s.llm.model)
        except (httpx.HTTPError, ValueError):
            return None

    async def nothing() -> None:
        return None

    own = processes_mb(pid) if pid else nothing()
    jarvis, model = await asyncio.gather(own, ollama())
    return Footprint(jarvis, model, s.llm.model)


def run_footprint(s: Settings) -> int:
    """Measure the running Jarvis (the pid in its lock file) and the model."""
    from rich.console import Console

    from backend.instance import InstanceLock

    console = Console(highlight=False)
    lock = InstanceLock(s.data_path / "jarvis.lock")
    pid = None
    if lock.acquire():  # free: the pid written in it is from a run that has ended
        lock.release()
        console.print("[yellow]Jarvis is not running: only the model is measured[/]")
    else:
        pid = lock.owner()
    fp = asyncio.run(measure(s, pid))
    jarvis = "-" if pid is None or fp.jarvis_mb is None else f"{fp.jarvis_mb:,.0f} MB"
    if fp.model_mb is None:
        model = "Ollama unreachable"
    elif fp.model_mb == 0:
        model = "not loaded"
    else:
        model = f"{fp.model_mb:,.0f} MB"
    console.print(f"Jarvis (with the HUD window): {jarvis}")
    console.print(f"Model {fp.model}: {model}")
    return 0
