import asyncio

from backend.boot import BootProgress, wait_for_internet
from backend.hud.bus import HudBus


def last_boot(bus: HudBus) -> dict:
    return next(e for e in bus.snapshot() if e["type"] == "boot")


def test_progress_moves_only_when_steps_end_weighted_by_their_cost():
    bus = HudBus()
    boot = BootProgress(bus, ["ollama", "model", "speech"])  # weights 1 + 8 + 2
    assert last_boot(bus)["progress"] == 0
    boot.start("model")
    assert boot.progress == 0
    boot.done("ollama")
    assert boot.progress == round(100 / 11)
    boot.done("model")
    assert boot.progress == round(900 / 11)
    boot.set("model", "failed")  # an ended step stays ended
    assert boot.steps["model"] == "done"
    boot.finish()
    event = last_boot(bus)
    assert event["progress"] == 100
    assert {"id": "speech", "status": "skipped"} in event["steps"]


def test_track_marks_a_failed_step():
    async def go():
        boot = BootProgress(HudBus(), ["model"])

        async def broken():
            raise RuntimeError("Ollama 500")

        try:
            await boot.track("model", broken())
        except RuntimeError:
            pass
        assert boot.steps["model"] == "failed" and boot.progress == 100

    asyncio.run(go())


def test_waits_for_the_network_then_gives_up():
    async def go():
        answers = iter([False, False, True])

        async def probe():
            return next(answers)

        assert await wait_for_internet(1, probe, retry_s=0.01)

        async def never():
            return False

        assert not await wait_for_internet(0.05, never, retry_s=0.01)

    asyncio.run(go())
