import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.audio.level import level
from backend.config import Settings
from backend.hud.bus import QUEUE_SIZE, HudBus
from backend.hud.feeds import news_panel, system_panel, weather_panel
from backend.hud.server import allowed_origins, create_app
from backend.state import State
from backend.tools.timers import TimerManager
from tests.test_voice_loop import FakeConversation, Harness

ORIGIN = "http://127.0.0.1:8765"


def kinds(events):
    return [e["type"] for e in events]


def test_level_maps_silence_to_zero_and_loud_to_one():
    assert level(np.zeros(512, dtype=np.int16)) == 0.0
    assert level(np.array([], dtype=np.int16)) == 0.0
    loud = (np.sin(np.linspace(0, 60, 512)) * 30000).astype(np.int16)
    assert level(loud) == 1.0
    quiet = (np.sin(np.linspace(0, 60, 512)) * 300).astype(np.int16)
    assert 0.0 < level(quiet) < 0.5


def test_snapshot_keeps_latest_panels_and_the_transcript():
    bus = HudBus()
    bus.publish({"type": "state", "state": "sleeping"})
    bus.publish({"type": "state", "state": "listening"})
    bus.publish({"type": "level", "mic": 0.4})  # live only
    bus.publish({"type": "user", "text": "What time is it?"})
    bus.publish({"type": "reply", "delta": "It is "})
    bus.publish({"type": "reply", "delta": "ten o'clock."})
    snap = bus.snapshot()
    assert kinds(snap) == ["state", "transcript"]
    assert snap[0]["state"] == "listening"
    assert snap[1]["lines"] == [
        {"role": "user", "text": "What time is it?", "open": False},
        {"role": "assistant", "text": "It is ten o'clock.", "open": True},
    ]
    bus.publish({"type": "reply_end"})
    bus.publish({"type": "reply", "delta": "The timer is done."})  # an announcement
    lines = bus.snapshot()[-1]["lines"]
    assert [x["open"] for x in lines] == [False, False, True]
    for i in range(20):
        bus.publish({"type": "user", "text": str(i)})
    assert len(bus.snapshot()[-1]["lines"]) == 8


def test_a_stalled_client_loses_the_oldest_events():
    bus = HudBus()
    queue = bus.subscribe()
    for i in range(QUEUE_SIZE + 10):
        bus.publish({"type": "level", "mic": i})
    assert queue.qsize() == QUEUE_SIZE
    assert queue.get_nowait()["mic"] == 10
    bus.unsubscribe(queue)
    assert bus.clients == 0


def test_bad_command_handler_does_not_raise():
    bus = HudBus()
    bus.command("wake")  # no handler yet
    bus.on_command = lambda name: 1 / 0
    bus.command("wake")


def test_websocket_sends_snapshot_then_live_events_and_takes_commands(tmp_path):
    bus = HudBus()
    got: list[str] = []
    bus.on_command = got.append
    bus.publish({"type": "state", "state": "sleeping"})
    client = TestClient(create_app(bus, allowed_origins("127.0.0.1", 8765), tmp_path))
    with client.websocket_connect("/ws", headers={"origin": ORIGIN}) as ws:
        assert ws.receive_json() == {"type": "state", "state": "sleeping"}
        assert ws.receive_json()["type"] == "transcript"
        bus.publish({"type": "tool", "name": "get_weather"})
        assert ws.receive_json() == {"type": "tool", "name": "get_weather"}
        ws.send_text("not json")
        ws.send_json({"type": "shutdown"})  # not a HUD command
        ws.send_json({"type": "wake"})
        bus.publish({"type": "reply_end"})
        ws.receive_json()  # the server has read the messages sent before it
    assert got == ["wake"]
    assert bus.clients == 0


def test_websocket_refuses_other_sites(tmp_path):
    bus = HudBus()
    client = TestClient(create_app(bus, allowed_origins("127.0.0.1", 8765), tmp_path))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "https://example.com"}):
            pass
    assert bus.clients == 0
    assert "http://localhost:5173" in allowed_origins("127.0.0.1", 8765)


def test_page_explains_how_to_build_when_missing(tmp_path):
    client = TestClient(create_app(HudBus(), set(), tmp_path))
    assert "npm" in client.get("/").text
    (tmp_path / "index.html").write_text("<title>built</title>")
    client = TestClient(create_app(HudBus(), set(), tmp_path))
    assert "built" in client.get("/").text


def test_panels_from_tool_results():
    forecast = {
        "place": "London, England, United Kingdom",
        "now": {"conditions": "overcast", "temperature": 17, "feels_like": 16, "daytime": True},
        "days": [
            {"day": "today", "conditions": "rain", "min": 12, "max": 20, "sunrise": "06:01"},
            {"day": "tomorrow", "conditions": "sunny", "min": 11, "max": 22},
        ],
    }
    w = weather_panel(forecast)
    assert w["now"]["temperature"] == 17 and w["sunrise"] == "06:01" and len(w["days"]) == 2
    n = news_panel({"items": [{"title": "T", "source": "S", "hours_ago": 2, "summary": "x"}]})
    assert n["items"] == [{"title": "T", "source": "S", "hours_ago": 2}]
    s = system_panel(
        {
            "battery": "no battery (desktop Mac)",
            "cpu_load_percent": 12,
            "memory": {"total_gb": 16.0, "free_percent": 70},
        }
    )
    assert s["battery"] is None and s["memory_percent"] == 30 and s["cpu_percent"] == 12


def test_timer_changes_are_reported(owner_env):
    async def scenario():
        changes = []
        timers = TimerManager(Settings(), on_change=lambda: changes.append(timers.snapshot()))
        await timers.set_timer(minutes=5, label="cake")
        assert changes[-1][0]["label"] == "cake" and changes[-1][0]["seconds"] == 300
        await timers.cancel_timer(label="cake")
        assert changes[-1] == []
        await timers.set_timer(seconds=0.01)
        await asyncio.sleep(0.05)
        assert changes[-1] == [] and len(changes) == 4

    asyncio.run(scenario())


def test_hud_commands_wake_and_sleep():
    async def scenario():
        h = Harness()
        events = h.loop.bus.subscribe()
        h.loop.bus.command("wake")
        assert h.state is State.LISTENING
        h.loop.bus.command("wake")  # already awake: nothing happens
        h.loop.bus.command("sleep")
        assert h.state is State.SLEEPING
        states = []
        while not events.empty():
            e = events.get_nowait()
            if e["type"] == "state":
                states.append(e["state"])
        assert states == ["listening", "sleeping"]

    asyncio.run(scenario())


class SlowSpeaker:
    """Speaks until interrupted."""

    def __init__(self):
        self.stop = asyncio.Event()

    async def speak(self, sentences, on_start=None):
        async for _ in sentences:
            pass
        await self.stop.wait()
        return False

    def interrupt(self):
        self.stop.set()


def test_sleep_command_cuts_a_reply_short():
    async def scenario():
        h = Harness()
        h.loop.speaker = SlowSpeaker()
        h.loop.conv = FakeConversation()
        h.loop.announce("The timer is done.")
        assert h.state is State.SPEAKING
        h.loop.bus.command("sleep")
        await asyncio.sleep(0.01)
        assert h.state is State.SLEEPING

    asyncio.run(scenario())


def test_hud_commands_ignored_without_wake_word():
    h = Harness(wake=False)
    h.loop.bus.command("sleep")
    assert h.state is State.LISTENING


class FakeWindow:
    def __init__(self, opens: bool) -> None:
        self.opens = opens
        self.opened = 0

    async def open(self) -> bool:
        self.opened += 1
        return self.opens


def test_hud_opens_in_the_window_and_falls_back_to_the_browser(monkeypatch):
    from backend.hud import server

    browser = []
    monkeypatch.setattr(server, "RECONNECT_S", 0)
    monkeypatch.setattr(server.webbrowser, "open", browser.append)
    window = FakeWindow(opens=True)
    asyncio.run(server.open_hud(HudBus(), "http://hud", window))
    assert window.opened == 1 and browser == []
    asyncio.run(server.open_hud(HudBus(), "http://hud", FakeWindow(opens=False)))
    assert browser == ["http://hud"]
    bus = HudBus()
    bus.subscribe()  # a page left open reconnected: open nothing
    window = FakeWindow(opens=True)
    asyncio.run(server.open_hud(bus, "http://hud", window))
    assert window.opened == 0 and browser == ["http://hud"]


def test_hud_window_reports_a_window_that_cannot_open(monkeypatch):
    import sys

    from backend.hud import window

    monkeypatch.setattr(sys, "executable", "/usr/bin/false")  # exits at once
    win = window.HudWindow("http://hud", "Jarvis")
    assert asyncio.run(win.open()) is False


def test_old_open_browser_setting_still_works(monkeypatch):
    monkeypatch.setenv("JARVIS_HUD__OPEN_BROWSER", "false")
    assert Settings().hud.open_on_start is False
    assert Settings().hud.window is True
