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
    briefing = sub.add_parser("briefing", help="print the activation briefing, with timings")
    briefing.add_argument("-s", "--speak", action="store_true", help="read it aloud too")
    sub.add_parser("wake-test", help="show live wake word scores to tune the threshold")
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

        wake_word = False if args.no_wake else None
        briefing = False if args.no_briefing else None
        return _run(
            run_voice(
                settings,
                resume=args.resume,
                wake_word=wake_word,
                briefing=briefing,
                started_at=STARTED_AT,
            )
        )
    if args.command == "briefing":
        from backend.chat import run_briefing

        return _run(run_briefing(settings, speak=args.speak))
    if args.command == "wake-test":
        from backend.wake_test import run_wake_test

        return _run(run_wake_test(settings))
    if args.command == "bench":
        from backend.bench import run_bench

        return _run(run_bench(settings, args.models))
    if args.command == "bench-stt":
        from backend.bench_stt import run_bench_stt

        return _run(run_bench_stt(settings, args.models))
    if args.command in {"prompt", "config"}:
        import asyncio

        from backend.tools.geocode import resolve_location

        asyncio.run(resolve_location(settings))
    if args.command == "prompt":
        from backend.brain.prompts import build_context_note, build_system_prompt

        print(build_system_prompt(settings))
        print(build_context_note(settings, first_turn=True))
        return 0
    if args.command == "config":
        print(settings.model_dump_json(indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
