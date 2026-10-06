"""Entry point: `uv run python -m backend <command>`."""

from __future__ import annotations

import argparse
import os
import sys
import time

from backend.config import get_settings
from backend.logging_setup import setup_logging

STARTED_AT = time.perf_counter()  # the briefing reports its latency from here

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _run(coro) -> int:
    import asyncio

    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check Ollama, model, voice and location")
    sub.add_parser("config", help="print the effective configuration")
    chat = sub.add_parser("chat", help="text chat in the terminal")
    chat.add_argument("-r", "--resume", action="store_true", help="resume the last conversation")
    chat.add_argument("-s", "--speak", action="store_true", help="read replies aloud")
    voice = sub.add_parser("voice", help="talk to Jarvis with your voice")
    voice.add_argument("-r", "--resume", action="store_true", help="resume the last conversation")
    voice.add_argument(
        "--no-wake", action="store_true", help="always listen, without the wake word"
    )
    voice.add_argument("--no-briefing", action="store_true", help="skip the weather and news")
    voice.add_argument("--no-hud", action="store_true", help="no holographic interface")
    voice.add_argument("--at-login", action="store_true", help=argparse.SUPPRESS)
    autostart = sub.add_parser("autostart", help="launch Jarvis at login (macOS LaunchAgent)")
    autostart.add_argument(
        "action",
        nargs="?",
        choices=["status", "install", "uninstall", "start", "stop"],
        default="status",
    )
    autostart.add_argument("--no-briefing", action="store_true", help="no briefing at login")
    autostart.add_argument("--no-hud", action="store_true", help="no HUD at login")
    briefing = sub.add_parser("briefing", help="print the activation briefing, with timings")
    briefing.add_argument("-s", "--speak", action="store_true", help="read it aloud too")
    demo = sub.add_parser("hud-demo", help="show the HUD with a scripted conversation (no mic)")
    demo.add_argument("--sample", action="store_true", help="fixed panels, no network")
    window = sub.add_parser("hud-window", help="open the HUD in a native window")
    window.add_argument("--url", help="default: the configured server address")
    window.add_argument("--parent", type=int, help=argparse.SUPPRESS)  # close when it exits
    sub.add_parser("wake-test", help="show live wake word scores to tune the threshold")
    sub.add_parser("echo-test", help="measure how well echo cancellation works in this room")
    memory = sub.add_parser("memory", help="list or forget what Jarvis remembers about you")
    memory.add_argument("action", nargs="?", choices=["list", "forget", "clear"], default="list")
    memory.add_argument("id", nargs="?", type=int, help="the memory to forget (see `memory`)")
    sub.add_parser("prompt", help="print the current system prompt")
    bench = sub.add_parser("bench", help="compare LLM latency across models")
    bench.add_argument("models", nargs="*", help="Ollama models (default: the configured one)")
    bench_stt = sub.add_parser("bench-stt", help="compare Whisper models on one recorded phrase")
    bench_stt.add_argument("models", nargs="*", help="Hugging Face repos (default: 3 sizes)")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)

    if args.command == "doctor":
        from backend import doctor

        return doctor.run()
    if args.command == "chat":
        from backend.chat import run_chat

        return _run(run_chat(settings, resume=args.resume, speak=args.speak))
    if args.command == "voice":
        from backend.voice import run_voice

        if args.at_login:
            from backend.autostart import log_path, rotate_log

            rotate_log(log_path(settings))
        wake_word = False if args.no_wake else None
        briefing = False if args.no_briefing else None
        return _run(
            run_voice(
                settings,
                resume=args.resume,
                wake_word=wake_word,
                briefing=briefing,
                started_at=STARTED_AT,
                hud=False if args.no_hud else None,
                at_login=args.at_login,
            )
        )
    if args.command == "autostart":
        from backend.autostart import run_autostart

        voice_args = [
            flag
            for flag, on in (("--no-briefing", args.no_briefing), ("--no-hud", args.no_hud))
            if on
        ]
        return run_autostart(settings, args.action, voice_args)
    if args.command == "briefing":
        from backend.chat import run_briefing

        return _run(run_briefing(settings, speak=args.speak))
    if args.command == "hud-demo":
        from backend.hud.demo import run_demo

        return _run(run_demo(settings, sample=args.sample))
    if args.command == "hud-window":
        from backend.hud.window import run_window

        url = args.url or f"http://{settings.server.host}:{settings.server.port}"
        return run_window(url, settings.assistant_name, args.parent)
    if args.command == "wake-test":
        from backend.wake_test import run_wake_test

        return _run(run_wake_test(settings))
    if args.command == "echo-test":
        from backend.echo_test import run_echo_test

        return _run(run_echo_test(settings))
    if args.command == "bench":
        from backend.bench import run_bench

        return _run(run_bench(settings, args.models))
    if args.command == "bench-stt":
        from backend.bench_stt import run_bench_stt

        return _run(run_bench_stt(settings, args.models))
    if args.command == "memory":
        from backend.memory.cli import run_memory

        return run_memory(settings, args.action, args.id)
    if args.command in {"prompt", "config"}:
        import asyncio

        from backend.tools.geocode import resolve_location

        asyncio.run(resolve_location(settings))
    if args.command == "prompt":
        from backend.brain.prompts import build_context_note, build_system_prompt
        from backend.memory.store import MemoryStore

        facts = None
        if "memory" in settings.tools.enabled:
            store = MemoryStore(settings.db_path)
            facts = store.facts()
            store.close()
        print(build_system_prompt(settings, facts=facts))
        print(build_context_note(settings, first_turn=True))
        return 0
    if args.command == "config":
        print(settings.model_dump_json(indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
