"""Live wake word scores, for tuning the threshold: `uv run python -m backend wake-test`."""

from __future__ import annotations

import asyncio
import time

from rich.console import Console

from backend.audio.wakeword import WakeWordDetector, load_wake_model
from backend.config import Settings

console = Console(highlight=False)
BAR_WIDTH = 40


async def run_wake_test(s: Settings) -> int:
    from backend.audio.capture import Microphone

    try:
        model = load_wake_model(s.wakeword_path, s.wakeword_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    detector = WakeWordDetector(model, s.wakeword.threshold)
    mic = Microphone(s.audio.input_device)
    mic.start()
    threshold = s.wakeword.threshold
    mark = round(threshold * BAR_WIDTH)
    console.print(f'Say "Hey Jarvis" · model {s.wakeword_path.name} · threshold {threshold}')
    console.print("[dim]Only scores above 0.05 are shown. Ctrl+C to quit.[/]")

    peak, hits, last_hit = 0.0, 0, 0.0
    try:
        async for frame in mic.frames():
            for score in detector.scores(frame):
                now = time.monotonic()
                if score >= 0.05:
                    filled = round(score * BAR_WIDTH)
                    bar = "".join(
                        "█" if i < filled else ("|" if i == mark else "·") for i in range(BAR_WIDTH)
                    )
                    color = "green" if score >= threshold else "yellow"
                    console.print(f"[{color}]{bar}[/] {score:.3f}")
                peak = max(peak, score)
                if score >= threshold and now - last_hit > 1.5:  # one count per phrase
                    hits, last_hit = hits + 1, now
                    console.print(f"[bold green]✔ detected ({hits})[/]")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        mic.stop()
    console.print(f"\nPeak score {peak:.3f} · detections {hits}")
    return 0
