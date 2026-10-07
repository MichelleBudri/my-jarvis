import asyncio

from backend.config import ResourcesConfig, Settings
from backend.footprint import model_mb, parse_top
from backend.resources import Resources
from backend.state import State


class FakeLLM:
    def __init__(self, log):
        self.log = log

    async def unload(self):
        self.log.append("llm unload")


class FakeConv:
    def __init__(self, log):
        self.log = log
        self.llm = FakeLLM(log)

    async def settle(self):
        self.log.append("settle")

    async def warm(self):
        self.log.append("llm warm")


class FakeSTT:
    def __init__(self, log):
        self.log = log

    async def unload(self):
        self.log.append("stt unload")

    async def warmup(self):
        self.log.append("stt warm")


def make(profile="balanced", idle_minutes=10.0):
    s = Settings()
    s.resources = ResourcesConfig(profile=profile, idle_minutes=idle_minutes)
    log: list[str] = []
    changes: list[bool] = []
    res = Resources(s, FakeConv(log), FakeSTT(log), on_change=changes.append)
    return res, log, changes


def test_profiles_set_the_delay_and_ollamas_safety_net():
    assert ResourcesConfig(profile="performance").unload_after_s is None
    assert ResourcesConfig(profile="performance").keep_alive == "-1"
    assert ResourcesConfig(profile="balanced", idle_minutes=10).unload_after_s == 600
    assert ResourcesConfig(profile="balanced", idle_minutes=10).keep_alive == "15m"
    assert ResourcesConfig(profile="economy").unload_after_s == 0
    assert ResourcesConfig(profile="economy").keep_alive == "5m"


def test_unloads_after_sleeping_long_enough_and_reloads_on_waking():
    async def go():
        res, log, changes = make(idle_minutes=0.001)  # 60 ms
        res.on_state(State.LISTENING, State.SLEEPING)
        await asyncio.sleep(0.15)
        assert not res.loaded
        assert log[0] == "settle" and set(log[1:]) == {"llm unload", "stt unload"}
        log.clear()
        res.on_state(State.SLEEPING, State.LISTENING)
        await res._work
        assert res.loaded and set(log) == {"stt warm", "llm warm"}
        assert changes == [False, True]

    asyncio.run(go())


def test_waking_before_the_delay_keeps_everything_loaded():
    async def go():
        res, log, _ = make(idle_minutes=0.002)
        res.on_state(State.LISTENING, State.SLEEPING)
        await asyncio.sleep(0.02)
        res.on_state(State.SLEEPING, State.LISTENING)
        await asyncio.sleep(0.2)
        assert res.loaded and log == []

    asyncio.run(go())


def test_performance_never_unloads_and_a_timer_announcement_does_not_reload():
    async def go():
        res, log, _ = make(profile="performance")
        res.on_state(State.LISTENING, State.SLEEPING)
        assert res._timer is None

        res, log, _ = make(profile="economy")
        res.on_state(State.LISTENING, State.SLEEPING)
        await asyncio.sleep(0.05)
        assert not res.loaded
        res.on_state(State.SLEEPING, State.SPEAKING)  # a timer going off
        assert res._work is None
        res.on_state(State.SPEAKING, State.LISTENING)  # then the person answers
        await res._work
        assert res.loaded

    asyncio.run(go())


def test_a_failed_unload_does_not_stop_jarvis():
    async def go():
        res, log, _ = make(profile="economy")

        async def broken():
            raise OSError("Ollama gone")

        res.conv.llm.unload = broken
        await res.unload()
        assert not res.loaded and "stt unload" in log

    asyncio.run(go())


def test_parses_top_memory_and_ollama_ps():
    out = "Processes: 2 total\n\nPID   MEM\n5338  2368K\n5394  1765M+\n12    1.2G-\n"
    found = parse_top(out)
    assert round(found[5338], 2) == 2.31 and found[5394] == 1765 and found[12] == 1228.8
    ps = {"models": [{"name": "qwen3:8b", "size": 5645229096}]}
    assert model_mb(ps, "qwen3:8b") == 5383.7
    assert model_mb({"models": []}, "qwen3:8b") == 0
