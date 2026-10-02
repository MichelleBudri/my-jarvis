"""The Mac itself: battery, memory, disk, uptime, volume and opening apps."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time

from backend.config import Settings
from backend.tools.registry import Tool, ToolError

TIMEOUT_S = 5

_BATTERY = re.compile(r"(\d+)%;\s*([^;]+);\s*(?:(\d+:\d+) remaining)?")
_MEM_FREE = re.compile(r"free percentage:\s*(\d+)%")
_BOOT = re.compile(r"sec = (\d+)")
_VOLUME = re.compile(r"output volume:(\d+|missing value).*output muted:(true|false)")


async def run(*cmd: str) -> tuple[int, str]:
    """Run a command without a shell; returns (exit code, stdout + stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except FileNotFoundError as exc:
        raise ToolError(f"{cmd[0]} not available") from exc
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), TIMEOUT_S)
    except TimeoutError as exc:
        proc.kill()
        raise ToolError(f"{cmd[0]} timed out") from exc
    return proc.returncode or 0, out.decode(errors="replace").strip()


def parse_battery(text: str) -> dict | None:
    m = _BATTERY.search(text)
    if not m:
        return None  # desktop Mac
    percent, status, remaining = m.groups()
    out: dict = {
        "percent": int(percent),
        "status": status.strip(),  # charging | discharging | charged | finishing charge
        "on_power_adapter": "AC Power" in text,
    }
    if remaining and remaining != "0:00":
        h, mins = remaining.split(":")
        out["time_remaining"] = f"{int(h)}h{mins}"
    return out


def parse_volume(text: str) -> dict | None:
    m = _VOLUME.search(text)
    if not m:
        return None
    level, muted = m.groups()
    return {
        "volume_percent": None if level == "missing value" else int(level),
        "muted": muted == "true",
    }


def _gb(n: int) -> float:
    return round(n / 1024**3, 1)


TOPICS = ("battery", "cpu", "memory", "disk", "uptime", "volume", "all")


async def system_status(topic: str = "all") -> dict:
    topic = topic if topic in TOPICS else "all"

    def wants(name: str) -> bool:
        return topic in (name, "all")

    out: dict = {}
    if wants("battery"):
        _, batt = await run("pmset", "-g", "batt")
        out["battery"] = parse_battery(batt) or "no battery (desktop Mac)"

    if wants("cpu"):
        load1, _, _ = os.getloadavg()
        cores = os.cpu_count() or 1
        out["cpu_load_percent"] = min(100, round(100 * load1 / cores))

    if wants("memory"):
        _, mem = await run("memory_pressure", "-Q")
        _, total = await run("sysctl", "-n", "hw.memsize")
        if (m := _MEM_FREE.search(mem)) and total.isdigit():
            out["memory"] = {"total_gb": _gb(int(total)), "free_percent": int(m.group(1))}

    if wants("disk"):
        disk = shutil.disk_usage("/")
        out["disk"] = {"free_gb": _gb(disk.free), "total_gb": _gb(disk.total)}

    if wants("uptime"):
        _, boot = await run("sysctl", "-n", "kern.boottime")
        if m := _BOOT.search(boot):
            mins = int((time.time() - int(m.group(1))) // 60)
            out["uptime"] = f"{mins // 1440}d {mins // 60 % 24}h {mins % 60}min"

    if wants("volume"):
        _, vol = await run("osascript", "-e", "get volume settings")
        if volume := parse_volume(vol):
            out["sound"] = volume
    return out


# Level to bring back on unmute when the sound was "muted" by setting the volume to 0.
_restore = {"level": None}


async def _volume_settings() -> dict:
    _, vol = await run("osascript", "-e", "get volume settings")
    return parse_volume(vol) or {}


async def set_volume(level: int | None = None, mute: bool | None = None) -> dict:
    if level is None and mute is None:
        raise ToolError("give a level (0-100) or mute true/false")
    before = await _volume_settings()
    current = before.get("volume_percent")
    if level is not None:
        level = max(0, min(100, int(level)))
    if (mute or level == 0) and current and not before.get("muted"):
        _restore["level"] = current
    if mute is False and level is None and not current and _restore["level"]:
        level = _restore["level"]  # unmuting from 0 would stay silent: bring the old level
    script: list[str] = []
    if level is not None:
        script.append(f"set volume output volume {level}")
    if mute is not None:
        script.append(f"set volume output muted {'true' if mute else 'false'}")
    elif level:
        script.append("set volume output muted false")  # raising the volume implies unmute
    code, out = await run("osascript", *(arg for line in script for arg in ("-e", line)))
    if code:
        raise ToolError(out or "could not change the volume")
    return {"ok": True, **await _volume_settings()}


async def open_app(name: str) -> dict:
    name = (name or "").strip()
    if not name or name.startswith("-"):
        raise ToolError("give an application name")
    code, out = await run("open", "-a", name)
    if code:
        raise ToolError(f"application not found: {name}")
    return {"ok": True, "opened": name}


def system_tools(s: Settings) -> list[Tool]:
    cap = {
        "pt": "ver bateria, memória, disco e volume do computador, ajustar o volume e abrir apps",
        "en": "check the computer's battery, memory, disk and volume, set the volume and open apps",
    }
    return [
        Tool(
            name="get_system_status",
            description=(
                "Status of this Mac: battery, CPU load, memory, free disk space, uptime and "
                "sound volume. Use for questions about the computer itself."
            ),
            handler=system_status,
            # Without a slot for "battery", qwen3 often wrote a call that Ollama 0.35
            # dropped (blank reply, no tool call); with it, 10 of 10 calls came through.
            parameters={
                "topic": {
                    "type": "string",
                    "enum": list(TOPICS),
                    "description": "What the user asked about; 'all' for a general check",
                }
            },
            required=["topic"],
            capability=cap,
        ),
        Tool(
            name="set_volume",
            # qwen3 unmuted with level=50 ("volte o som" after "tire o som" at 20): say
            # explicitly that unmuting alone restores the previous level.
            description=(
                "Set the Mac's output volume and/or mute or unmute it. To silence the sound, "
                "pass mute=true (not level 0). To bring the sound back, pass only "
                "mute=false, with no level: the previous volume comes back by itself. Pass a "
                "level only when the user says a number or asks for louder/quieter."
            ),
            handler=set_volume,
            parameters={
                "level": {"type": "integer", "description": "Volume from 0 to 100"},
                "mute": {"type": "boolean", "description": "true to mute, false to unmute"},
            },
            capability=cap,
        ),
        Tool(
            name="open_app",
            description="Open (or bring to front) a Mac application by name, e.g. 'Safari'.",
            handler=open_app,
            parameters={"name": {"type": "string", "description": "Application name"}},
            required=["name"],
            capability=cap,
        ),
    ]
