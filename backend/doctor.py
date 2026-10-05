"""Environment checks: `uv run python -m backend doctor`."""

from __future__ import annotations

import asyncio
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from backend.config import Settings, get_settings
from backend.tools.geocode import resolve_location

console = Console()


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def check_platform() -> Check:
    is_mac_arm = sys.platform == "darwin" and platform.machine() == "arm64"
    detail = f"{platform.system()} {platform.release()} ({platform.machine()})"
    return Check("macOS Apple Silicon", is_mac_arm, detail, required=False)


def check_python() -> Check:
    ok = (3, 12) <= sys.version_info[:2] < (3, 14)
    return Check("Python 3.12–3.13", ok, platform.python_version())


def check_audio(s: Settings) -> list[Check]:
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        return [Check("Audio devices", False, f"sounddevice unavailable: {exc}", required=False)]
    checks = []
    for kind, label, device in (
        ("input", "Microphone", s.audio.input_device),
        ("output", "Speakers", s.audio.output_device),
    ):
        try:
            info = sd.query_devices(device, kind=kind)
            checks.append(Check(label, True, info["name"], required=False))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(label, False, str(exc), required=False))
    return checks


def check_speech_models(s: Settings) -> list[Check]:
    checks = []
    try:
        import pysilero_vad  # noqa: F401

        checks.append(Check("Voice activity detection", True, "Silero VAD", required=False))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Voice activity detection", False, str(exc), required=False))
    try:
        import mlx_whisper  # noqa: F401
        from huggingface_hub import try_to_load_from_cache

        cached = isinstance(try_to_load_from_cache(s.stt.model, "config.json"), str)
        detail = s.stt.model if cached else "not downloaded: run ./scripts/setup.sh"
        checks.append(Check("Speech recognition", cached, detail, required=False))
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check("Speech recognition", False, f"mlx-whisper unavailable: {exc}", required=False)
        )
    checks.append(check_wakeword(s))
    return checks


def check_wakeword(s: Settings) -> Check:
    if not s.wakeword.enabled:
        return Check("Wake word", True, "disabled: always listening", required=False)
    files = [s.wakeword_path, s.wakeword_dir / "melspectrogram.onnx"]
    files.append(s.wakeword_dir / "embedding_model.onnx")
    missing = [f.name for f in files if not f.exists()]
    if missing:
        return Check(
            "Wake word",
            False,
            f"missing {', '.join(missing)}: run ./scripts/setup.sh",
            required=False,
        )
    try:
        import openwakeword  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return Check("Wake word", False, f"openwakeword unavailable: {exc}", required=False)
    detail = f"{s.wakeword_path.name} · threshold {s.wakeword.threshold}"
    return Check("Wake word", True, detail, required=False)


def check_ollama(s: Settings) -> list[Check]:
    checks: list[Check] = []
    try:
        version = httpx.get(f"{s.llm.host}/api/version", timeout=3).json()["version"]
        checks.append(Check("Ollama server", True, f"v{version} at {s.llm.host}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Ollama server", False, f"unreachable: {exc}"))
        return checks

    tags = httpx.get(f"{s.llm.host}/api/tags", timeout=5).json().get("models", [])
    names = {m["name"] for m in tags} | {m.get("model", "") for m in tags}
    has_model = s.llm.model in names or f"{s.llm.model}:latest" in names
    checks.append(
        Check(
            f"Model {s.llm.model}",
            has_model,
            "installed" if has_model else f"run: ollama pull {s.llm.model}",
        )
    )
    if not has_model:
        return checks

    payload = {
        "model": s.llm.model,
        "messages": [
            {"role": "system", "content": f"You are {s.assistant_name}."},
            {"role": "user", "content": "Reply with exactly: systems online."},
        ],
        "stream": False,
        "think": s.llm.think,
        "keep_alive": s.llm.keep_alive,
        "options": {"temperature": 0},
    }
    start = time.perf_counter()
    try:
        resp = httpx.post(f"{s.llm.host}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
        elapsed = time.perf_counter() - start
        checks.append(Check("LLM response", bool(text), f'"{text[:60]}" ({elapsed:.1f}s)'))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("LLM response", False, str(exc)))
    return checks


def check_language(s: Settings) -> Check:
    loc = s.locale
    detail = f"{loc.code} · {loc.language_name}"
    if not loc.native_prompt:
        detail += " (English prompt, replies in this language)"
    return Check("Language", True, detail, required=False)


def check_voice(s: Settings) -> Check:
    path = s.voice_path
    ok = path.exists() and Path(f"{path}.json").exists()
    detail = str(path.relative_to(path.parents[2])) if ok else "run: ./scripts/setup.sh"
    return Check(f"Piper voice {s.voice_name}", ok, detail, required=False)


def check_owner(s: Settings) -> Check:
    u = s.user
    if not u.name and u.kind == "n":
        return Check("Owner", False, "set JARVIS_OWNER_NAME in .env", required=False)
    detail = f"{u.name or '-'} · addressed as: {s.owner_full_address or 'neutral'}"
    return Check("Owner", True, detail, required=False)


def check_location(s: Settings) -> Check:
    loc = asyncio.run(resolve_location(s))
    if not loc.query and not loc.resolved:
        return Check("Location", False, "set JARVIS_CITY in .env", required=False)
    if not loc.resolved:
        return Check("Location", False, f'could not find "{loc.query}"', required=False)
    detail = f"{loc.display_name} ({loc.latitude:.4f}, {loc.longitude:.4f}) · {loc.tz_name}"
    return Check("Location", True, detail, required=False)


def check_hud(s: Settings) -> Check:
    from backend.hud.server import DIST_DIR

    if not s.hud.enabled:
        return Check("HUD", True, "disabled", required=False)
    if not (DIST_DIR / "index.html").exists():
        return Check("HUD", False, "not built: run ./scripts/setup.sh", required=False)
    return Check("HUD", True, f"http://{s.server.host}:{s.server.port}", required=False)


def run() -> int:
    s = get_settings()
    checks = [
        check_platform(),
        check_python(),
        check_language(s),
        check_owner(s),
        check_location(s),
        *check_ollama(s),
        check_voice(s),
        *check_speech_models(s),
        *check_audio(s),
        check_hud(s),
    ]

    table = Table(title=f"{s.assistant_name} · doctor", show_lines=False)
    table.add_column("", width=2)
    table.add_column("Item")
    table.add_column("Detail", overflow="fold")
    for c in checks:
        icon = "[green]✔[/]" if c.ok else ("[red]✘[/]" if c.required else "[yellow]![/]")
        table.add_row(icon, c.name, c.detail)
    console.print(table)

    failed = [c for c in checks if c.required and not c.ok]
    if failed:
        console.print(f"[red]{len(failed)} required check(s) failed.[/]")
        return 1
    console.print("[green]All set.[/]")
    return 0
