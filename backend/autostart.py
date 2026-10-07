"""Launch at login with a macOS LaunchAgent: `python -m backend autostart install`."""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from backend import app_bundle
from backend.config import ROOT_DIR, Settings

console = Console(highlight=False)

LABEL = "local.my-jarvis.voice"
AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LOG_MAX_BYTES = 5_000_000  # rotated to jarvis.log.1 on start
# launchd starts agents with a bare PATH; the tools need `say`, `osascript`, `open`...
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def plist_path(agents_dir: Path = AGENTS_DIR) -> Path:
    return agents_dir / f"{LABEL}.plist"


def log_path(s: Settings) -> Path:
    path = s.data_path / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / "jarvis.log"


def rotate_log(path: Path, max_bytes: int = LOG_MAX_BYTES) -> None:
    if path.exists() and path.stat().st_size > max_bytes:
        path.replace(path.with_name(path.name + ".1"))


def build_plist(
    s: Settings, voice_args: list[str], program: str, python_env: dict[str, str]
) -> dict:
    log = str(log_path(s))
    return {
        "Label": LABEL,
        "ProgramArguments": [program, "-m", "backend", "voice", "--at-login", *voice_args],
        # Login Items shows the app's name and icon instead of the executable's.
        "AssociatedBundleIdentifiers": [app_bundle.BUNDLE_ID],
        "WorkingDirectory": str(ROOT_DIR),
        "RunAtLoad": True,
        # Restart after a crash, not after a normal exit (Ctrl+C, `autostart stop`).
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        # Audio work: no background throttling (App Nap) of timers and I/O.
        "ProcessType": "Interactive",
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": {"PATH": PATH, "PYTHONUNBUFFERED": "1", **python_env},
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _domain() -> str:
    return f"gui/{os.getuid()}"


@dataclass
class Status:
    installed: bool
    loaded: bool
    pid: int | None
    last_exit: str | None


def status(agents_dir: Path = AGENTS_DIR) -> Status:
    installed = plist_path(agents_dir).exists()
    out = _launchctl("print", f"{_domain()}/{LABEL}")
    if out.returncode != 0:
        return Status(installed, False, None, None)
    pid = re.search(r"^\s*pid = (\d+)", out.stdout, re.MULTILINE)
    last = re.search(r"^\s*last exit code = (.+)$", out.stdout, re.MULTILINE)
    return Status(installed, True, int(pid.group(1)) if pid else None, last and last.group(1))


def start(agents_dir: Path = AGENTS_DIR) -> bool:
    path = plist_path(agents_dir)
    _launchctl("bootout", f"{_domain()}/{LABEL}")  # reload if already loaded
    out = _launchctl("bootstrap", _domain(), str(path))
    if out.returncode != 0:
        console.print(f"[red]launchctl bootstrap failed: {out.stderr.strip()}[/]")
    return out.returncode == 0


def stop() -> None:
    """Unload until the next login: launchd sends SIGTERM and does not restart it."""
    _launchctl("bootout", f"{_domain()}/{LABEL}")


def install(s: Settings, voice_args: list[str], agents_dir: Path = AGENTS_DIR) -> int:
    agents_dir.mkdir(parents=True, exist_ok=True)
    app = app_bundle.build(s)  # shown as "Jarvis", not "python"
    console.print(f"[green]✔[/] App: [dim]{app}[/]")
    program = str(app_bundle.executable(s))
    path = plist_path(agents_dir)
    with path.open("wb") as f:
        plistlib.dump(build_plist(s, voice_args, program, app_bundle.python_env()), f)
    console.print(f"[green]✔[/] Launch at login installed: [dim]{path}[/]")
    if not start(agents_dir):
        return 1
    console.print(
        f"{s.assistant_name} is starting in the background now, and will on every login.\n"
        f"[dim]Log: {log_path(s)} · stop: `autostart stop` · remove: `autostart uninstall`[/]"
    )
    console.print(
        f"[dim]The first time, macOS asks for microphone access for {s.assistant_name}. If it "
        "does not hear you, allow it in System Settings → Privacy & Security → Microphone. "
        "Run `autostart install` again after updating Python (`uv python upgrade`).[/]"
    )
    return 0


def uninstall(s: Settings, agents_dir: Path = AGENTS_DIR) -> int:
    stop()
    app_bundle.remove(s)
    path = plist_path(agents_dir)
    if path.exists():
        path.unlink()
        console.print("[green]✔[/] Launch at login removed")
    else:
        console.print("[dim]Launch at login was not installed[/]")
    return 0


def enabled(agents_dir: Path = AGENTS_DIR) -> bool:
    return plist_path(agents_dir).exists()


def _parked(s: Settings) -> Path:
    """Where a switched-off agent waits, so switching it on keeps its `voice` flags."""
    return s.data_path / f"{LABEL}.plist"


def set_enabled(s: Settings, on: bool, agents_dir: Path = AGENTS_DIR) -> bool:
    """The HUD switch: takes effect at the next login and leaves this run alone.

    No `bootout` or `bootstrap`: a Jarvis started by launchd would be ended by the first,
    and a second one started (and refused by the lock) by the second. Returns the new state.
    """
    path, parked = plist_path(agents_dir), _parked(s)
    if on and not path.exists():
        agents_dir.mkdir(parents=True, exist_ok=True)
        if parked.exists():
            parked.replace(path)
        else:
            if not app_bundle.executable(s).exists():
                app_bundle.build(s)  # never rebuilt here: this process may be running it
            plist = build_plist(s, [], str(app_bundle.executable(s)), app_bundle.python_env())
            with path.open("wb") as f:
                plistlib.dump(plist, f)
    elif not on and path.exists():
        parked.parent.mkdir(parents=True, exist_ok=True)
        path.replace(parked)
    return path.exists()


def run_autostart(s: Settings, action: str, voice_args: list[str]) -> int:
    if sys.platform != "darwin":
        console.print("[red]Launch at login uses launchd and only works on macOS[/]")
        return 1
    if action == "install":
        return install(s, voice_args)
    if action == "uninstall":
        return uninstall(s)
    if action == "start":
        if not plist_path().exists():
            console.print("[red]Not installed: run `autostart install` first[/]")
            return 1
        return 0 if start() else 1
    if action == "stop":
        stop()
        console.print("[dim]Stopped until the next login (or `autostart start`)[/]")
        return 0
    st = status()
    if not st.installed:
        console.print("Launch at login: [yellow]off[/] · turn on with `autostart install`")
        return 0
    state = f"[green]running[/] (pid {st.pid})" if st.pid else "[yellow]not running[/]"
    if st.pid is None and st.last_exit:
        state += f" · last exit: {st.last_exit}"
    console.print(f"Launch at login: [green]on[/] · {state}")
    console.print(f"[dim]Log: {log_path(s)}[/]")
    return 0
