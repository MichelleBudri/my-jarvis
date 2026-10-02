<div align="center">

<img src="docs/assets/banner.svg" width="100%" alt="JARVIS: a fully local voice assistant">

<br>

![macOS](https://img.shields.io/badge/macOS-Apple_Silicon-00e5ff?style=for-the-badge&logo=apple&logoColor=00e5ff&labelColor=0a1420)
![Python](https://img.shields.io/badge/Python-3.12+-00e5ff?style=for-the-badge&logo=python&logoColor=00e5ff&labelColor=0a1420)
![Ollama](https://img.shields.io/badge/LLM-Ollama-00e5ff?style=for-the-badge&logo=ollama&logoColor=00e5ff&labelColor=0a1420)
![Status](https://img.shields.io/badge/Phase-3_of_7-ffb340?style=for-the-badge&labelColor=0a1420)
![License](https://img.shields.io/badge/License-MIT-3dffb0?style=for-the-badge&labelColor=0a1420)

**A voice assistant that lives on your Mac, not in someone else's data center.**<br>
The language model, speech recognition and voice all run locally: no API keys, no tokens, no cloud.

[Features](#-features) · [Architecture](#-architecture) · [Quick start](#-quick-start) · [Personalization](#-personalization) · [Roadmap](#-roadmap)

</div>

<br>

## ◈ Features

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>🎙️ Voice-first</h3>
      Wake word, speech-to-text and a natural voice, all on-device. Replies stream sentence by sentence, so Jarvis starts talking before it finishes thinking.
    </td>
    <td width="33%" valign="top">
      <h3>🔒 100% local</h3>
      Runs on Apple Silicon through Ollama and MLX. Your conversations never leave your machine. Only weather and news touch the internet, through free, keyless APIs.
    </td>
    <td width="33%" valign="top">
      <h3>🎩 A butler's manners</h3>
      Formal, polite and dry-witted. Addresses you by the title and name you choose, with correct grammatical gender.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>🌦️ Morning briefing</h3>
      On activation, Jarvis reports the local weather and the latest AI headlines in a few spoken sentences.
    </td>
    <td valign="top">
      <h3>🌍 Multilingual</h3>
      Full support for English and Portuguese, including dates, seasons by hemisphere and voices. Other languages work too.
    </td>
    <td valign="top">
      <h3>🛰️ Futuristic HUD</h3>
      An original holographic interface: spinning rings that react to the voice, live panels and per-state animations.
    </td>
  </tr>
</table>

## ◈ Architecture

```mermaid
flowchart LR
    MIC(["🎙️ Microphone"]) --> WAKE["Wake word<br/><sub>openWakeWord</sub>"]
    WAKE --> VAD["Speech detection<br/><sub>Silero VAD</sub>"]
    VAD --> STT["Transcription<br/><sub>mlx-whisper</sub>"]
    STT --> LLM{{"🧠 Brain<br/><sub>Ollama · qwen3</sub>"}}
    LLM <--> TOOLS["Tools<br/><sub>weather · news · system</sub>"]
    LLM --> TTS["Voice<br/><sub>Piper</sub>"]
    TTS --> SPK(["🔊 Speaker"])
    LLM -. WebSocket .-> HUD[["🛰️ HUD"]]

    classDef core fill:#0a1420,stroke:#00e5ff,color:#eafcff,stroke-width:2px
    classDef io fill:#02050a,stroke:#ffb340,color:#ffd59a
    class WAKE,VAD,STT,TOOLS,TTS,HUD core
    class LLM core
    class MIC,SPK io
```

## ◈ Quick start

> [!NOTE]
> Requires macOS on Apple Silicon and [Homebrew](https://brew.sh). Python 3.12 is installed automatically by uv. Tested on an M4 with 16 GB running macOS 26.

```bash
git clone https://github.com/MichelleBudri/my-jarvis.git && cd my-jarvis
cp .env.example .env        # your name, city and language
./scripts/setup.sh          # installs uv + Ollama, pulls the model and voice, runs a health check
uv run python -m backend voice  # say "Hey Jarvis", then ask
```

The setup downloads the language model (`qwen3:8b`, about 5.2 GB), the speech recognition model (`whisper-small`, about 0.5 GB), the Piper voice for your language and the "Hey Jarvis" wake word model (about 4 MB).

### Commands

| Command | What it does |
|---|---|
| `uv run python -m backend voice` | Voice conversation: say "Hey Jarvis", ask, and Jarvis answers out loud (`--no-wake` listens all the time) |
| `uv run python -m backend chat` | Text chat with streaming replies (`-s` reads them aloud, `-r` resumes the last conversation) |
| `uv run python -m backend doctor` | Health check: Ollama, models, voice, wake word, microphone, owner, location |
| `uv run python -m backend wake-test` | Live wake word scores, to tune the detection threshold |
| `uv run python -m backend bench [models...]` | Compare LLM latency across Ollama models over a short scripted conversation |
| `uv run python -m backend bench-stt [models...]` | Record one phrase and compare Whisper models on it |
| `uv run python -m backend prompt` | Print the current system prompt |
| `uv run python -m backend config` | Print the effective configuration |
| `uv run pytest` · `uv run ruff check .` | Tests and lint |

## ◈ Personalization

Personal settings live in `.env`, which is never committed. See [`.env.example`](.env.example).

| Variable | Example | Effect |
|---|---|---|
| `JARVIS_LANGUAGE` | `en-GB` · `en-US` · `pt-BR` · `es` | Persona, dates, titles, voice and transcription |
| `JARVIS_OWNER_NAME` | `Michelle` | Your name |
| `JARVIS_OWNER_GENDER` | `female` · `male` · `neutral` | Default title (madam / sir) and grammatical agreement |
| `JARVIS_OWNER_TITLE` | `madam` · `Dr. Smith` | How Jarvis addresses you; empty = default for language and gender |
| `JARVIS_CITY` | `London, England, United Kingdom` | Weather, timezone and season |
| `JARVIS_LATITUDE` · `_LONGITUDE` · `_TIMEZONE` | `51.51` · `-0.13` · `Europe/London` | Optional: skips the city lookup |

<details>
<summary><b>Languages</b></summary>
<br>

English and Portuguese are fully supported. In English, Jarvis uses the British voice `en_GB-alan-medium`. Other languages use the English prompt plus an instruction to reply in the chosen language; set a compatible [Piper voice](https://huggingface.co/rhasspy/piper-voices) in `JARVIS_TTS__VOICE`.

</details>

<details>
<summary><b>Location</b></summary>
<br>

The city is resolved to coordinates and a timezone with the [Open-Meteo geocoding API](https://open-meteo.com/en/docs/geocoding-api) (free, no key) and cached in `data/location.json`. Adding the state or country helps pick the right city when names repeat.

</details>

<details>
<summary><b>Project defaults</b></summary>
<br>

The model, voice and news feeds are set in [`config.yaml`](config.yaml). Any key can be overridden with an environment variable, using `__` between sections:

```bash
JARVIS_LLM__MODEL=qwen3.5:4b uv run python -m backend voice
```

</details>

<details>
<summary><b>Voice and microphone</b></summary>
<br>

The first time Jarvis captures audio, macOS asks for microphone access for the app running it (Terminal, iTerm, VS Code...). If you deny it by mistake, enable it in **System Settings → Privacy & Security → Microphone**.

Jarvis sleeps until it hears **"Hey Jarvis"** (a rising chime confirms it). You can ask in the same breath ("Hey Jarvis, what time is it?") or wait for the chime. After each reply it keeps listening for 8 seconds, so follow-up questions need no wake word; then a falling chime means it went back to sleep. "Ei Jarvis" works too, so the wake word is fine in Portuguese.

```
sleeping ──"Hey Jarvis"──▶ listening ──speech──▶ thinking ──▶ speaking
    ▲                       │     ▲                              │
    └── silence (6 s) ──────┘     └──── reply done (8 s window) ─┘
```

If it misses you or wakes up by itself, run `uv run python -m backend wake-test` and adjust `JARVIS_WAKEWORD__THRESHOLD` (default 0.5; lower is more sensitive). The timings are `JARVIS_WAKEWORD__LISTEN_TIMEOUT_S` and `_FOLLOW_UP_S`, and `JARVIS_WAKEWORD__CHIME=false` silences the chimes.

By default Jarvis is half duplex: it stops listening while it thinks and speaks, so it never hears itself through the speakers. With headphones you can turn on barge-in and interrupt it mid-sentence:

```bash
JARVIS_AUDIO__BARGE_IN=true uv run python -m backend voice
```

Other knobs: `JARVIS_VAD__SILENCE_MS` (how long a pause ends your turn, default 700), `JARVIS_TTS__LENGTH_SCALE` (speaking speed, below 1 is faster), and `JARVIS_AUDIO__INPUT_DEVICE` / `_OUTPUT_DEVICE` (list devices with `uv run python -m sounddevice`).

</details>

## ◈ Roadmap

| | Phase | Status |
|:-:|---|---|
| 0 | Foundation: settings, setup script, health check | ✅ Done |
| 1 | Text chat: streaming LLM, persona, history, personalization | ✅ Done |
| 2 | Voice in and out: VAD, Whisper, streaming Piper, barge-in | ✅ Done |
| 3 | Wake word: "Hey Jarvis", state machine | ✅ Done |
| 4 | Tools: weather, AI news, system, timers | 🔜 Next |
| 5 | Activation briefing | ⏳ Planned |
| 6 | Holographic HUD | ⏳ Planned |
| 7 | Polish: long-term memory, launch at login, echo cancellation | ⏳ Planned |


## ◈ Stack

<div align="center">

![Ollama](https://img.shields.io/badge/Ollama-0a1420?style=flat-square&logo=ollama&logoColor=00e5ff)
![MLX Whisper](https://img.shields.io/badge/MLX_Whisper-0a1420?style=flat-square&logo=apple&logoColor=00e5ff)
![openWakeWord](https://img.shields.io/badge/openWakeWord-0a1420?style=flat-square)
![Silero VAD](https://img.shields.io/badge/Silero_VAD-0a1420?style=flat-square)
![Piper](https://img.shields.io/badge/Piper_TTS-0a1420?style=flat-square)
![Open-Meteo](https://img.shields.io/badge/Open--Meteo-0a1420?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0a1420?style=flat-square&logo=fastapi&logoColor=00e5ff)
![TypeScript](https://img.shields.io/badge/TypeScript-0a1420?style=flat-square&logo=typescript&logoColor=00e5ff)
![SQLite](https://img.shields.io/badge/SQLite-0a1420?style=flat-square&logo=sqlite&logoColor=00e5ff)
![uv](https://img.shields.io/badge/uv-0a1420?style=flat-square&logo=uv&logoColor=00e5ff)

</div>

## ◈ License

[MIT](LICENSE). Jarvis is an independent hobby project and is not affiliated with Marvel or Disney.

<div align="center">
<br>
<sub><code>// SYSTEMS ONLINE · RUNNING ON YOUR MACHINE</code></sub>
</div>
