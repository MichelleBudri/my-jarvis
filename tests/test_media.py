import asyncio
import json
import struct

import numpy as np
import pytest

from backend.config import Settings
from backend.state import State
from backend.tools import ToolError, build_registry
from backend.tools import media as media_mod
from backend.tools.media import (
    Ducker,
    MediaService,
    Spotify,
    Track,
    find_play_button,
    parse_track,
    read_bmp,
)

GREEN = (30, 215, 96)


def disc(img: np.ndarray, cx: int, cy: int, radius: int, color=GREEN) -> None:
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = color


def search_page(scale: int = 2) -> np.ndarray:
    """A dark window with the top result's play button, saved ticks and a green tile."""
    img = np.full((400 * scale, 600 * scale, 3), 18, np.uint8)
    disc(img, 500 * scale, 100 * scale, 24 * scale)  # top result, 48 pt
    img[95 * scale : 105 * scale, 498 * scale : 504 * scale] = 0  # the play triangle
    disc(img, 500 * scale, 60 * scale, 8 * scale)  # a "saved" tick, above it but small
    img[200 * scale : 330 * scale, 40 * scale : 240 * scale] = GREEN  # a category tile
    disc(img, 500 * scale, 300 * scale, 24 * scale)  # a lower one (a hovered card)
    return img


def to_bmp(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    stride = (w * 3 + 3) & ~3
    rows = b"".join(
        rgb[y, :, ::-1].tobytes() + b"\0" * (stride - w * 3) for y in range(h - 1, -1, -1)
    )
    header = struct.pack("<2sIHHI", b"BM", 54 + len(rows), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(rows), 0, 0, 0, 0)
    return header + info + rows


def test_finds_the_top_result_play_button():
    for scale in (1, 2):
        assert find_play_button(search_page(scale), scale) == pytest.approx((500, 100), abs=1)


def test_no_button_on_a_page_without_one():
    img = np.full((400, 600, 3), 18, np.uint8)
    img[50:300, 40:400] = GREEN  # a big tile is not a button
    disc(img, 300, 350, 8)  # neither is a tick
    assert find_play_button(img, 1) is None


def test_read_bmp_round_trip():
    img = search_page(1)
    assert np.array_equal(read_bmp(to_bmp(img)), img)


def test_parse_track():
    text = "playing\nMANIAC\nStray Kids\nODDINARY\nspotify:track:x\n97\ntrue\nfalse"
    t = parse_track(text)
    assert (t.state, t.name, t.artist, t.volume, t.shuffling, t.repeating) == (
        "playing", "MANIAC", "Stray Kids", 97, True, False,
    )  # fmt: skip
    assert t.describe() == {
        "state": "playing", "track": "MANIAC", "artist": "Stray Kids", "album": "ODDINARY",
    }  # fmt: skip
    assert parse_track("stopped").describe() == {"state": "stopped"}


def test_applescript_errors_name_the_permission(monkeypatch):
    async def denied(*cmd):
        return 1, "execution error: Not authorized to send Apple events to Spotify. (-1743)"

    monkeypatch.setattr(media_mod, "run", denied)
    with pytest.raises(ToolError, match="Automation"):
        asyncio.run(media_mod.osa('tell application "Spotify" to pause'))


class FakeSpotify(Spotify):
    def __init__(self, running=True, state="playing", volume=80):
        self.is_running, self.state, self.vol = running, state, volume
        self.commands: list[tuple[str, ...]] = []
        self.searched: list[str] = []

    async def running(self):
        return self.is_running

    async def now_playing(self):
        if not self.is_running:
            return Track(state="closed")
        return Track(self.state, "MANIAC", "Stray Kids", "ODDINARY", "spotify:track:x")

    async def volume(self):
        return self.vol

    async def set_volume(self, level):
        self.vol = level

    async def command(self, *lines):
        self.commands.append(lines)

    async def launch(self):
        self.is_running = True

    async def search_and_play(self, query, wait_s):
        self.searched.append(query)
        self.state = "playing"
        return await self.now_playing()


def test_ducks_while_awake_and_restores_on_sleep():
    async def scenario():
        spotify = FakeSpotify(volume=80)
        ducker = Ducker(spotify, 0.2)
        ducker.on_state(State.SLEEPING, State.LISTENING)
        await asyncio.sleep(0)
        await asyncio.gather(*ducker._tasks)
        assert spotify.vol == 16
        ducker.on_state(State.LISTENING, State.THINKING)  # still awake: nothing changes
        assert all(t.done() for t in ducker._tasks)  # no new task
        assert await ducker.set_level(50) == 50  # "abaixa a música" while ducked
        assert spotify.vol == 10 and await ducker.level() == 50
        ducker.on_state(State.SPEAKING, State.SLEEPING)
        await asyncio.gather(*ducker._tasks)
        assert spotify.vol == 50

    asyncio.run(scenario())


def test_does_not_duck_paused_or_closed_spotify():
    async def scenario():
        for spotify in (FakeSpotify(state="paused"), FakeSpotify(running=False)):
            ducker = Ducker(spotify, 0.2)
            ducker.awake = True
            await ducker.duck()
            await ducker.restore()
            assert spotify.vol == 80 and ducker.saved is None

    asyncio.run(scenario())


def test_always_listening_ducks_only_while_thinking_and_speaking():
    ducker = Ducker(FakeSpotify(), 0.2, wake=False)

    async def scenario():
        ducker.on_state(State.LISTENING, State.LISTENING)
        assert not ducker.awake
        ducker.on_state(State.LISTENING, State.THINKING)
        assert ducker.awake
        ducker.on_state(State.SPEAKING, State.LISTENING)
        assert not ducker.awake
        await asyncio.gather(*ducker._tasks)

    asyncio.run(scenario())


def test_play_music_lowers_the_music_before_it_starts():
    async def scenario():
        spotify = FakeSpotify(running=False, state="paused", volume=90)
        media = MediaService(Settings(), spotify)
        media.ducker.awake = True  # asked by voice: Jarvis is about to reply
        out = await media.play_music("  stray   kids ")
        assert spotify.searched == ["stray kids"]
        assert out["track"] == "MANIAC" and spotify.vol == 18
        assert await media.play_music("") == {"ok": True, "action": "resume"}
        assert spotify.commands[-1] == ("play",)

    asyncio.run(scenario())


def test_media_control_and_volume():
    async def scenario():
        spotify = FakeSpotify(volume=60)
        media = MediaService(Settings(), spotify)
        assert (await media.media_control("next"))["track"] == "MANIAC"
        assert await media.media_control("pause") == {"ok": True, "action": "pause"}
        with pytest.raises(ToolError):
            await media.media_control("louder")
        assert (await media.media_volume(change=-20))["spotify_volume"] == 40
        assert (await media.media_volume(level=150))["spotify_volume"] == 100
        assert (await media.now_playing())["spotify_volume"] == 100
        spotify.is_running = False
        assert await media.now_playing() == {"state": "Spotify is closed"}
        with pytest.raises(ToolError, match="closed"):
            await media.media_control("pause")

    asyncio.run(scenario())


def test_registered_and_told_apart_from_the_mac_volume():
    reg = build_registry(Settings(), media=MediaService(Settings(), FakeSpotify()))
    assert {"play_music", "media_control", "now_playing", "media_volume"} <= set(reg.names)
    assert reg.is_slow("play_music") and not reg.is_slow("media_control")
    schemas = {t["function"]["name"]: t["function"] for t in reg.schemas()}
    assert "set_volume" in schemas["media_volume"]["description"]
    assert "media_volume" in schemas["set_volume"]["description"]
    out = json.loads(asyncio.run(reg.call("media_control", {"action": "previous"})))
    assert out["action"] == "previous"
