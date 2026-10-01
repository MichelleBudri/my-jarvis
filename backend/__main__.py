"""Entry point: `uv run python -m backend <command>`."""

from __future__ import annotations

import argparse
import sys

from backend.config import get_settings
from backend.logging_setup import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check Ollama, model, voice and location")
    sub.add_parser("config", help="print the effective configuration")
    chat = sub.add_parser("chat", help="text chat in the terminal")
    chat.add_argument("-r", "--resume", action="store_true", help="resume the last conversation")
    sub.add_parser("prompt", help="print the current system prompt")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)

    if args.command == "doctor":
        from backend import doctor

        return doctor.run()
    if args.command == "chat":
        import asyncio

        from backend.chat import run_chat

        return asyncio.run(run_chat(settings, resume=args.resume))
    if args.command in {"prompt", "config"}:
        import asyncio

        from backend.tools.geocode import resolve_location

        asyncio.run(resolve_location(settings))
    if args.command == "prompt":
        from backend.brain.prompts import build_system_prompt

        print(build_system_prompt(settings))
        return 0
    if args.command == "config":
        print(settings.model_dump_json(indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
