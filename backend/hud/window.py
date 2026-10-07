"""The HUD in a native macOS window (WebKit through pywebview), without browser chrome.

The window runs in a child process: Cocoa needs the main thread, which in `voice` belongs to
the asyncio loop. The page is the same one the browser gets, from the same server.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading
import time

log = logging.getLogger(__name__)

BACKGROUND = "#02050a"  # the HUD's, so the window never flashes white while loading


class _Api:
    """Exposed to the page as `window.pywebview.api`: WebKit's Fullscreen API is off in
    pywebview windows, so the HUD's F key asks the window instead, and copying and opening
    links too."""

    def __init__(self) -> None:
        self.window = None

    def toggle_fullscreen(self) -> None:
        if self.window is not None:
            self.window.toggle_fullscreen()

    def open_url(self, url: str) -> bool:
        """A news item clicked in the HUD: read it in the default browser, not in here."""
        from urllib.parse import urlparse

        if urlparse(url).scheme not in ("http", "https"):
            return False
        import webbrowser

        return webbrowser.open(url)

    def copy(self, text: str) -> bool:
        """The HUD's copy buttons: WebKit in a pywebview window may refuse the clipboard."""
        return copy_to_clipboard(text)


def copy_to_clipboard(text: str) -> bool:
    try:
        from AppKit import NSPasteboard, NSPasteboardTypeString

        board = NSPasteboard.generalPasteboard()
        board.clearContents()
        return bool(board.setString_forType_(text, NSPasteboardTypeString))
    except Exception:  # noqa: BLE001 - no PyObjC: pbcopy, told the text is UTF-8
        import subprocess

        env = {**os.environ, "LANG": "en_US.UTF-8"}
        done = subprocess.run(["pbcopy"], input=text.encode(), env=env, check=False)
        return done.returncode == 0


def _name_app(name: str) -> None:
    """Show the assistant's name in the menu bar instead of "Python"."""
    with contextlib.suppress(Exception):
        from Foundation import NSBundle

        info = NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = name


def _watch_parent(parent_pid: int, window) -> None:
    """Close the window when Jarvis exits, even if it was killed without cleaning up."""
    while os.getppid() == parent_pid:
        time.sleep(1)
    window.destroy()


def run_window(url: str, title: str, parent_pid: int | None = None) -> int:
    """Blocks until the window is closed (child side)."""
    import webview

    _name_app(title)
    api = _Api()
    window = webview.create_window(
        title,
        url,
        js_api=api,
        width=1440,
        height=900,
        min_size=(720, 540),
        background_color=BACKGROUND,
        text_select=False,
    )
    api.window = window
    if parent_pid:
        threading.Thread(target=_watch_parent, args=(parent_pid, window), daemon=True).start()
    webview.start()
    return 0


class HudWindow:
    """The window process, seen from `voice` (parent side)."""

    def __init__(
        self, url: str, title: str, program: str | None = None, env: dict | None = None
    ) -> None:
        """`program`: the Python to run it with; the app bundle's shows "Jarvis" in the
        Dock and menu bar instead of "python" (it then needs `env`)."""
        self.url = url
        self.title = title
        self.program = program or sys.executable
        self.env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._drain: asyncio.Task | None = None

    async def open(self) -> bool:
        if self._proc is not None and self._proc.returncode is None:
            return True  # already open
        args = ["-m", "backend", "hud-window", "--url", self.url, "--parent", str(os.getpid())]
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.program,
                *args,
                env=self.env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            log.warning("HUD window failed to start: %s", exc)
            return False
        # A window that cannot open (no GUI session, pywebview missing) exits at once.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._proc.wait(), 1.5)
            err = (await self._proc.stderr.read()).decode(errors="replace").strip()
            log.warning("HUD window closed on start: %s", err.splitlines()[-1] if err else "")
            return False
        self._drain = asyncio.create_task(self._log_stderr())  # a full pipe would block it
        return True

    async def _log_stderr(self) -> None:
        while line := await self._proc.stderr.readline():
            log.debug("HUD window: %s", line.decode(errors="replace").rstrip())

    async def close(self) -> None:
        if self._drain:
            self._drain.cancel()
        if self._proc is None or self._proc.returncode is not None:
            return
        self._proc.terminate()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._proc.wait(), 2)
            return
        self._proc.kill()
