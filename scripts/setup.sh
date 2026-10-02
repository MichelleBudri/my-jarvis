#!/usr/bin/env bash
# Installs Jarvis dependencies on macOS (Apple Silicon). Usage: ./scripts/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

info()  { printf '\033[36m▸ %s\033[0m\n' "$*"; }
ok()    { printf '\033[32m✔ %s\033[0m\n' "$*"; }
fail()  { printf '\033[31m✘ %s\033[0m\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] \
  || fail "This setup targets macOS on Apple Silicon."

command -v brew >/dev/null || fail "Homebrew not found. Install it from https://brew.sh"

if [[ ! -f .env ]]; then
  cp .env.example .env
  info "Created .env from .env.example. Edit it with your name, city and language."
fi

if ! command -v uv >/dev/null; then
  info "Installing uv..."
  brew install uv
fi
ok "uv $(uv --version | awk '{print $2}')"

if ! command -v ollama >/dev/null; then
  info "Installing Ollama..."
  brew install ollama
fi
ok "ollama installed"

if ! curl -sf http://localhost:11434/api/version >/dev/null; then
  info "Starting the Ollama server..."
  brew services start ollama >/dev/null 2>&1 || (nohup ollama serve >/tmp/ollama.log 2>&1 &)
  for _ in {1..30}; do
    curl -sf http://localhost:11434/api/version >/dev/null && break
    sleep 1
  done
  curl -sf http://localhost:11434/api/version >/dev/null || fail "Ollama did not respond on :11434"
fi
ok "Ollama server running"

info "Syncing the Python environment..."
uv sync
ok "Python dependencies"

read_cfg() { uv run --quiet python -c "from backend.config import get_settings as g; s=g(); print($1)"; }

MODEL="$(read_cfg 's.llm.model')"
info "Pulling model $MODEL (the first download takes a while)..."
ollama pull "$MODEL"
ok "model $MODEL"

VOICE="$(read_cfg 's.voice_name')"
VOICE_PATH="$(read_cfg 's.voice_path')"
if [[ ! -f "$VOICE_PATH" ]]; then
  # pt_BR-faber-medium -> pt/pt_BR/faber/medium on Hugging Face
  IFS='-' read -r LOCALE SPEAKER QUALITY <<<"$VOICE"
  BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LOCALE%%_*}/${LOCALE}/${SPEAKER}/${QUALITY}/${VOICE}"
  info "Downloading voice $VOICE..."
  mkdir -p "$(dirname "$VOICE_PATH")"
  curl -fL --progress-bar -o "$VOICE_PATH" "$BASE.onnx"
  curl -fsL -o "$VOICE_PATH.json" "$BASE.onnx.json"
fi
ok "voice $VOICE"

STT_MODEL="$(read_cfg 's.stt.model')"
info "Downloading speech recognition model $STT_MODEL..."
uv run --quiet python -c "from huggingface_hub import snapshot_download as d; d('$STT_MODEL')"
ok "speech model $STT_MODEL"

echo
uv run python -m backend doctor
