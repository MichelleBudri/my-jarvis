import asyncio
import os
import plistlib

import httpx
import numpy as np

from backend import app_bundle, autostart
from backend.audio.capture import SilenceWatch
from backend.brain.llm import OllamaClient
from backend.config import ROOT_DIR, Settings
from backend.instance import InstanceLock


def test_plist_runs_voice_at_login_and_restarts_only_after_a_crash(tmp_path):
    s = Settings()
    env = {"PYTHONHOME": "/py", "PYTHONPATH": "/venv/site-packages"}
    plist = autostart.build_plist(
        s, ["--no-briefing"], "/data/Jarvis.app/Contents/MacOS/Jarvis", env
    )
    assert plist["ProgramArguments"] == [
        "/data/Jarvis.app/Contents/MacOS/Jarvis",
        "-m",
        "backend",
        "voice",
        "--at-login",
        "--no-briefing",
    ]
    assert plist["AssociatedBundleIdentifiers"] == [app_bundle.BUNDLE_ID]
    assert plist["WorkingDirectory"] == str(ROOT_DIR)
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["StandardOutPath"] == str(s.data_path / "logs" / "jarvis.log")
    assert "/usr/bin" in plist["EnvironmentVariables"]["PATH"]
    assert plist["EnvironmentVariables"]["PYTHONHOME"] == "/py"
    path = tmp_path / "a.plist"
    path.write_bytes(plistlib.dumps(plist))  # serialisable as launchd expects
    assert plistlib.loads(path.read_bytes())["Label"] == autostart.LABEL


def test_install_builds_the_app_and_writes_the_agent(tmp_path, monkeypatch):
    s = Settings()
    started, built, removed = [], [], []
    monkeypatch.setattr(autostart, "start", lambda d: started.append(d) or True)
    monkeypatch.setattr(autostart, "stop", lambda: None)
    monkeypatch.setattr(app_bundle, "build", lambda s: built.append(s) or tmp_path / "J.app")
    monkeypatch.setattr(app_bundle, "remove", lambda s: removed.append(s))
    assert autostart.install(s, [], agents_dir=tmp_path) == 0
    assert autostart.plist_path(tmp_path).exists() and started == [tmp_path] and built
    program = plistlib.loads(autostart.plist_path(tmp_path).read_bytes())["ProgramArguments"][0]
    assert program == str(app_bundle.executable(s))
    assert autostart.uninstall(s, agents_dir=tmp_path) == 0
    assert not autostart.plist_path(tmp_path).exists() and removed


def test_switch_writes_and_parks_the_agent_without_touching_launchd(tmp_path, monkeypatch):
    s = Settings(data_dir=str(tmp_path / "data"))
    agents = tmp_path / "agents"
    built = []
    monkeypatch.setattr(autostart, "_launchctl", lambda *a: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(app_bundle, "build", lambda s: built.append(s))
    assert autostart.set_enabled(s, True, agents) is True
    assert autostart.enabled(agents) and built  # the app was missing, so it was built
    plist = plistlib.loads(autostart.plist_path(agents).read_bytes())
    assert plist["ProgramArguments"][0] == str(app_bundle.executable(s))

    # Flags given to `autostart install` survive switching off and on again.
    plist["ProgramArguments"].append("--no-briefing")
    autostart.plist_path(agents).write_bytes(plistlib.dumps(plist))
    assert autostart.set_enabled(s, False, agents) is False
    assert not autostart.enabled(agents)
    assert autostart.set_enabled(s, False, agents) is False  # already off
    assert autostart.set_enabled(s, True, agents) is True
    args = plistlib.loads(autostart.plist_path(agents).read_bytes())["ProgramArguments"]
    assert args[-1] == "--no-briefing"


def test_app_bundle_is_named_after_the_assistant():
    s = Settings(assistant_name="Friday")
    info = app_bundle.info_plist(s)
    assert info["CFBundleName"] == "Friday" and info["CFBundleExecutable"] == "Friday"
    assert "NSMicrophoneUsageDescription" in info  # macOS ends apps that lack it
    assert app_bundle.executable(s).parts[-4:] == ("Friday.app", "Contents", "MacOS", "Friday")


def test_python_env_points_at_this_environment(monkeypatch):
    monkeypatch.delenv("PYTHONHOME", raising=False)
    env = app_bundle.python_env()
    assert os.path.isdir(os.path.join(env["PYTHONHOME"], "lib"))
    assert any(p.endswith("site-packages") for p in env["PYTHONPATH"].split(os.pathsep))
    monkeypatch.setenv("PYTHONHOME", "/inside/the/app")  # already the app: passed on
    assert app_bundle.python_env()["PYTHONHOME"] == "/inside/the/app"


def test_icon_is_a_valid_png():
    pixels = app_bundle.icon_pixels(64)
    assert pixels.shape == (64, 64, 4)
    assert pixels[0, 0, 3] == 0 and pixels[32, 32, 3] == 255  # transparent corner, opaque centre
    png = app_bundle.png_bytes(pixels)
    assert png.startswith(b"\x89PNG") and b"IEND" in png


def test_log_rotation(tmp_path):
    log = tmp_path / "jarvis.log"
    log.write_text("x" * 100)
    autostart.rotate_log(log, max_bytes=200)
    assert log.exists()
    autostart.rotate_log(log, max_bytes=50)
    assert not log.exists() and (tmp_path / "jarvis.log.1").read_text() == "x" * 100


def test_instance_lock(tmp_path):
    path = tmp_path / "jarvis.lock"
    first, second = InstanceLock(path), InstanceLock(path)
    assert first.acquire()
    assert not second.acquire()
    assert second.owner() == os.getpid()
    first.release()
    assert second.acquire()
    second.release()


def test_silence_watch():
    zeros, sound = np.zeros(512, np.int16), np.ones(512, np.int16)
    watch = SilenceWatch(frames=3)
    assert [watch(zeros) for _ in range(5)] == [False, False, True, False, False]  # once
    watch = SilenceWatch(frames=3)
    assert not any(watch(f) for f in (zeros, sound, zeros, zeros, zeros))  # a live mic


def test_wait_ready():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"version": "0.1"})

    client = httpx.AsyncClient(base_url="http://o", transport=httpx.MockTransport(handler))
    llm = OllamaClient(Settings().llm, client=client)
    assert asyncio.run(llm.wait_ready(5, interval_s=0.01))
    assert len(attempts) == 3
    down = OllamaClient(
        Settings().llm,
        client=httpx.AsyncClient(
            base_url="http://o",
            transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))),
        ),
    )
    assert not asyncio.run(down.wait_ready(0.03, interval_s=0.01))
