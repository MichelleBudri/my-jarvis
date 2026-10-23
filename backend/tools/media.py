"""Music on Spotify: play what was asked for, pause, skip, volume, what is playing.

Playback goes through Spotify's AppleScript dictionary, which has no search. To play
"Stray Kids", Jarvis opens the app's own search page and clicks the round green play button
of the top result, found on a screenshot of the window (D-45). The page itself is out of
reach otherwise: Spotify shows no web content to Accessibility, and the Tab order changes
with the state of its panels.

Needs, for the app running Jarvis: Automation (control Spotify and System Events),
Accessibility (the click) and Screen Recording (the screenshot).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

from backend.config import Settings
from backend.state import State
from backend.tools.registry import Tool, ToolError
from backend.tools.system import run

log = logging.getLogger(__name__)

APP = "Spotify"
LAUNCH_WAIT_S = 10
POLL_S = 0.3
# Spotify's green (#1ED760) and the size of the top result's play button, in points (48).
BUTTON_PT = (30, 72)

NOT_ALLOWED = (
    "macOS did not let Jarvis control {what}: allow it in System Settings → Privacy & "
    "Security → {where}"
)


async def osa(*lines: str) -> str:
    """Run AppleScript lines; the error the model hears names the permission to grant."""
    code, out = await run("osascript", *(arg for line in lines for arg in ("-e", line)))
    if code:
        if "-1743" in out:
            raise ToolError(NOT_ALLOWED.format(what=APP, where="Automation"))
        if "-25211" in out or "-1719" in out:
            raise ToolError(NOT_ALLOWED.format(what="the screen", where="Accessibility"))
        raise ToolError(out or "AppleScript failed")
    return out


def tell(*lines: str) -> list[str]:
    return [f'tell application "{APP}"', *lines, "end tell"]


# --- Finding the play button on a screenshot


def read_bmp(data: bytes) -> np.ndarray:
    """An uncompressed BMP (what `screencapture -t bmp` writes) as an RGB array."""
    offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    channels = struct.unpack_from("<H", data, 28)[0] // 8
    stride = (width * channels + 3) & ~3
    rows = abs(height)
    pixels = np.frombuffer(data, np.uint8, count=stride * rows, offset=offset)
    pixels = pixels.reshape(rows, stride)[:, : width * channels].reshape(rows, width, channels)
    if height > 0:  # stored bottom-up
        pixels = pixels[::-1]
    return pixels[..., 2::-1]  # BGR(A) → RGB


def find_play_button(rgb: np.ndarray, scale: float) -> tuple[float, float] | None:
    """Centre, in points from the image's top left, of the topmost big green round button.

    On the search page that is the top result's. The small green ticks ("saved") and the
    category tiles are left out by size and shape.
    """
    from scipy import ndimage

    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    green = (g > 170) & (r < 110) & (b < 150) & (g - r > 90)
    labels, _ = ndimage.label(green)
    best: tuple[float, float, float] | None = None
    lo, hi = BUTTON_PT
    for i, box in enumerate(ndimage.find_objects(labels), 1):
        h = (box[0].stop - box[0].start) / scale
        w = (box[1].stop - box[1].start) / scale
        if not (lo <= w <= hi and 0.8 <= w / h <= 1.25):
            continue
        if (labels[box] == i).mean() < 0.5:  # a disc with a dark triangle, not a ring
            continue
        top = box[0].start / scale
        x = (box[1].start + box[1].stop) / 2 / scale
        y = (box[0].start + box[0].stop) / 2 / scale
        if best is None or top < best[0]:
            best = (top, x, y)
    return None if best is None else (best[1], best[2])


# Clicks through CoreGraphics (JXA), then puts the pointer back where the person left it.
CLICK_JS = """ObjC.import('CoreGraphics');
var back = $.CGEventGetLocation($.CGEventCreate(null));
function post(type, p) {
  $.CGEventPost($.kCGHIDEventTap, $.CGEventCreateMouseEvent(null, type, p, $.kCGMouseButtonLeft));
  delay(0.05);
}
var p = {x: %f, y: %f};
post($.kCGEventMouseMoved, p);
post($.kCGEventLeftMouseDown, p);
post($.kCGEventLeftMouseUp, p);
post($.kCGEventMouseMoved, back);"""


# --- Spotify


@dataclass
class Track:
    state: str  # playing | paused | stopped
    name: str = ""
    artist: str = ""
    album: str = ""
    url: str = ""
    volume: int = 0
    shuffling: bool = False
    repeating: bool = False

    def describe(self) -> dict:
        if self.state == "stopped" or not self.name:
            return {"state": self.state}
        # No position or duration: qwen3 read "0:01 of 3:05" aloud after every command.
        return {"state": self.state, "track": self.name, "artist": self.artist, "album": self.album}


NOW_PLAYING = tell(
    "set s to player state as text",
    'if s is "stopped" then return s',
    "set t to current track",
    "return s & linefeed & (name of t) & linefeed & (artist of t) & linefeed & "
    "(album of t) & linefeed & (spotify url of t) & linefeed & sound volume & linefeed & "
    "shuffling & linefeed & repeating",
)


def parse_track(text: str) -> Track:
    lines = text.split("\n")
    if len(lines) < 8:
        return Track(state=lines[0].strip() or "stopped")
    state, name, artist, album, url, vol, shuffling, repeating = lines[:8]
    volume = int(vol) if vol.strip().isdigit() else 0
    return Track(state, name, artist, album, url, volume, shuffling == "true", repeating == "true")


class Spotify:
    async def running(self) -> bool:
        # `is running` asks without launching it, unlike a `tell`.
        return await osa(f'application "{APP}" is running') == "true"

    async def now_playing(self) -> Track:
        if not await self.running():
            return Track(state="closed")
        return parse_track(await osa(*NOW_PLAYING))

    async def volume(self) -> int:
        return int(await osa(*tell("sound volume")))

    async def set_volume(self, level: int) -> None:
        level = max(0, min(100, int(level)))
        await osa(*tell(f"set sound volume to {level}"))
        # Spotify often keeps one less than asked (50 → 49): ask once more, one higher.
        if 0 < level < 100 and await self.volume() < level:
            await osa(*tell(f"set sound volume to {level + 1}"))

    async def command(self, *lines: str) -> None:
        await osa(*tell(*lines))

    async def launch(self) -> None:
        """Open Spotify if needed and wait until it answers AppleScript."""
        if await self.running():
            return
        code, out = await run("open", "-a", APP)
        if code:
            raise ToolError(f"{APP} is not installed")
        deadline = time.monotonic() + LAUNCH_WAIT_S
        while time.monotonic() < deadline:
            with contextlib.suppress(ToolError):
                await osa(*tell("player state"))
                return
            await asyncio.sleep(0.5)
        raise ToolError(f"{APP} did not open in {LAUNCH_WAIT_S} s")

    async def window(self) -> tuple[int, int, int, int]:
        """Bring the window to the front; its (x, y, width, height) in points."""
        await osa(*tell("reopen", "activate"))  # reopen: a window even if it was closed
        out = await osa(
            f'tell application "System Events" to tell process "{APP}" to '
            "get {position, size} of front window"
        )
        x, y, w, h = (int(float(v)) for v in out.split(","))
        return x, y, w, h

    async def screenshot(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.bmp"
            code, out = await run(
                "screencapture", "-x", "-o", "-t", "bmp", "-R", f"{x},{y},{w},{h}", str(path)
            )
            if code or not path.exists():
                raise ToolError(out or "could not take a screenshot of Spotify")
            return read_bmp(path.read_bytes())

    async def search_and_play(self, query: str, wait_s: float) -> Track:
        """Open the search page for `query` and click the top result's play button."""
        await self.launch()
        before = await self.now_playing()
        if before.state == "playing":
            # If the top result is what is playing, its button is a pause button.
            await self.command("pause")
            before.state = "paused"
        # The empty search page has no green buttons: the old results never get clicked.
        await osa(*tell("activate"), 'open location "spotify:search:"')
        await asyncio.sleep(POLL_S)
        await osa(f'open location "spotify:search:{quote(query)}"')
        x, y, w, h = await self.window()

        # The same spot on two screenshots in a row: the page has stopped moving.
        deadline, last = time.monotonic() + wait_s, None
        while time.monotonic() < deadline:
            await asyncio.sleep(POLL_S)
            rgb = await self.screenshot(x, y, w, h)
            spot = await asyncio.to_thread(find_play_button, rgb, rgb.shape[1] / w)
            if spot and last and abs(spot[0] - last[0]) < 2 and abs(spot[1] - last[1]) < 2:
                break
            last = spot
        else:
            raise ToolError(
                f"found nothing to play for {query!r} on {APP}'s search page. If the window "
                "shows results, macOS may be blocking the screenshot: allow Jarvis in System "
                "Settings → Privacy & Security → Screen Recording"
            )
        click = CLICK_JS % (x + spot[0], y + spot[1])
        code, _ = await run("osascript", "-l", "JavaScript", "-e", click)
        if code:
            raise ToolError(NOT_ALLOWED.format(what="the mouse", where="Accessibility"))

        # Confirm by AppleScript that something else is playing now.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            await asyncio.sleep(POLL_S)
            now = await self.now_playing()
            if now.state == "playing":
                return now
        raise ToolError(
            f"clicked play on {APP}, but nothing started playing. macOS drops the click "
            "unless Jarvis is allowed in System Settings → Privacy & Security → Accessibility"
        )


# --- Lower the music while Jarvis listens and speaks


class Ducker:
    """Spotify at a fraction of its volume while Jarvis is awake (D-46).

    Otherwise Whisper transcribes the song along with the request, and the reply is lost in
    it. Echo cancellation removes only Jarvis's own voice, the one signal it knows.
    """

    def __init__(self, spotify: Spotify, ratio: float, wake: bool = True) -> None:
        self.spotify = spotify
        self.ratio = ratio
        self.wake = wake
        self.awake = False
        self.saved: int | None = None  # the volume to bring back
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    def on_state(self, _: State | None, new: State) -> None:
        # With the wake word: from "Hey Jarvis" until it sleeps again. Always listening:
        # only while it thinks and speaks, or the music would never come back up.
        awake = new is not State.SLEEPING if self.wake else new in (State.THINKING, State.SPEAKING)
        if awake == self.awake:
            return
        self.awake = awake
        task = asyncio.get_running_loop().create_task(self.duck() if awake else self.restore())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def duck(self, even_paused: bool = False) -> None:
        async with self._lock:
            if not self.awake or self.saved is not None or not self.ratio:
                return
            try:
                if not await self.spotify.running():
                    return
                if not even_paused and (await self.spotify.now_playing()).state != "playing":
                    return
                level = await self.spotify.volume()
                await self.spotify.set_volume(round(level * self.ratio))
                self.saved = level
            except ToolError as exc:
                log.info("Could not lower Spotify: %s", exc)

    async def restore(self) -> None:
        async with self._lock:
            if self.saved is None:
                return
            level, self.saved = self.saved, None
            try:
                if await self.spotify.running():
                    await self.spotify.set_volume(level)
            except ToolError as exc:
                log.info("Could not restore Spotify's volume: %s", exc)

    async def set_level(self, level: int) -> int:
        """Change the music's volume; while ducked, the level it comes back to."""
        level = max(0, min(100, int(level)))
        async with self._lock:
            if self.saved is not None:
                self.saved = level
                await self.spotify.set_volume(round(level * self.ratio))
            else:
                await self.spotify.set_volume(level)
        return level

    async def level(self) -> int:
        return self.saved if self.saved is not None else await self.spotify.volume()


# --- Tools


ACTIONS = {
    "pause": ["pause"],
    "resume": ["play"],
    "next": ["next track"],
    "previous": ["previous track"],
    "shuffle_on": ["set shuffling to true"],
    "shuffle_off": ["set shuffling to false"],
    "repeat_on": ["set repeating to true"],
    "repeat_off": ["set repeating to false"],
}


class MediaService:
    def __init__(self, s: Settings, spotify: Spotify | None = None, wake: bool = True) -> None:
        self.s = s
        self.spotify = spotify or Spotify()
        self.ducker = Ducker(self.spotify, s.media.duck_to, wake)

    def on_state(self, old: State | None, new: State) -> None:
        self.ducker.on_state(old, new)

    async def play_music(self, query: str = "") -> dict:
        query = " ".join((query or "").split())
        if not query:
            return await self.media_control("resume")
        await self.spotify.launch()
        await self.ducker.duck(even_paused=True)  # so it never starts over Jarvis's reply
        track = await self.spotify.search_and_play(query, self.s.media.search_wait_s)
        return {"ok": True, **track.describe()}

    async def media_control(self, action: str) -> dict:
        lines = ACTIONS.get(action)
        if lines is None:
            raise ToolError(f"action must be one of {', '.join(ACTIONS)}")
        if not await self.spotify.running():
            raise ToolError(f"{APP} is closed: nothing is playing")
        await self.spotify.command(*lines)
        if action not in ("next", "previous"):
            return {"ok": True, "action": action}
        await asyncio.sleep(0.4)  # the new track's name arrives a moment later
        return {"ok": True, "action": action, **(await self.spotify.now_playing()).describe()}

    async def now_playing(self) -> dict:
        track = await self.spotify.now_playing()
        if track.state == "closed":
            return {"state": f"{APP} is closed"}
        out = track.describe()
        if "track" in out:
            out |= {
                "spotify_volume": await self.ducker.level(),  # not the lowered one
                "shuffle": track.shuffling,
                "repeat": track.repeating,
            }
        return out

    async def media_volume(self, level: int | None = None, change: int | None = None) -> dict:
        if level is None and change is None:
            raise ToolError("give a level (0-100) or a change such as -20 or +20")
        if not await self.spotify.running():
            raise ToolError(f"{APP} is closed")
        if level is None:
            level = await self.ducker.level() + int(change or 0)
        return {"ok": True, "spotify_volume": await self.ducker.set_level(level)}


def media_tools(s: Settings, service: MediaService | None = None) -> list[Tool]:
    media = service or MediaService(s)
    cap = {
        "pt": "tocar músicas, artistas, álbuns e playlists no Spotify e controlar a reprodução",
        "en": "play songs, artists, albums and playlists on Spotify and control playback",
    }
    return [
        Tool(
            name="play_music",
            description=(
                "Open Spotify and play what the user asks for: an artist, song, album or "
                "playlist ('toque Stray Kids', 'coloca Maniac', 'play some lo-fi'). Plays "
                "Spotify's top search result, so put the artist in the query when the user "
                "names one ('bohemian rhapsody queen'). With no query, resumes playback."
            ),
            handler=media.play_music,
            parameters={
                "query": {
                    "type": "string",
                    "description": "What to search for, in the user's words, without 'play'",
                }
            },
            required=["query"],
            capability=cap,
            slow=True,
        ),
        Tool(
            name="media_control",
            description=(
                "Control Spotify's playback: pause, resume, next or previous track, shuffle "
                "and repeat on/off ('pausa a música', 'próxima', 'volta uma')."
            ),
            handler=media.media_control,
            parameters={
                "action": {"type": "string", "enum": list(ACTIONS)},
            },
            required=["action"],
            capability=cap,
        ),
        Tool(
            name="now_playing",
            description="What Spotify is playing now: song, artist, album and position.",
            handler=media.now_playing,
            capability=cap,
        ),
        # qwen3 mixes up the Mac's volume and the music's: each description names the other.
        Tool(
            name="media_volume",
            description=(
                "Spotify's own volume, for 'abaixa a música' or 'aumenta a música'. Not the "
                "Mac's volume: for that, or for 'abaixa o volume' with no music mentioned, use "
                "set_volume. Give a level from 0 to 100, or a change such as -20 or +20."
            ),
            handler=media.media_volume,
            parameters={
                "level": {"type": "integer", "description": "Volume from 0 to 100"},
                "change": {"type": "integer", "description": "Relative change, e.g. -20"},
            },
            capability=cap,
        ),
    ]
