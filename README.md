<div align="center">

<img src="docs/assets/banner.svg" width="100%" alt="JARVIS: a fully local voice assistant">

<br>

![macOS](https://img.shields.io/badge/macOS-Apple_Silicon-00e5ff?style=for-the-badge&logo=apple&logoColor=00e5ff&labelColor=0a1420)
![Python](https://img.shields.io/badge/Python-3.12+-00e5ff?style=for-the-badge&logo=python&logoColor=00e5ff&labelColor=0a1420)
![Ollama](https://img.shields.io/badge/LLM-Ollama-00e5ff?style=for-the-badge&logo=ollama&logoColor=00e5ff&labelColor=0a1420)
![Status](https://img.shields.io/badge/Phase-7_of_7-3dffb0?style=for-the-badge&labelColor=0a1420)
![License](https://img.shields.io/badge/License-MIT-3dffb0?style=for-the-badge&labelColor=0a1420)

**A voice assistant that lives on your Mac, not in someone else's data center.**<br>
The language model, speech recognition and voice all run locally: no API keys, no tokens, no cloud.

[Features](#-features) · [HUD](#-hud) · [Architecture](#-architecture) · [Quick start](#-quick-start) · [Personalization](#-personalization) · [Roadmap](#-roadmap)

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
      <h3>🌦️ Activation briefing</h3>
      On start, Jarvis greets you with the local weather and the one or two AI headlines that matter, fetched while the models load.
    </td>
    <td valign="top">
      <h3>🌍 Multilingual</h3>
      Full support for English and Portuguese, including dates, seasons by hemisphere and voices. Other languages work too.
    </td>
    <td valign="top">
      <h3>🛰️ Futuristic HUD</h3>
      An original holographic interface in its own window: rings that react to your voice and Jarvis's, a live transcript and panels for weather, news, system and timers.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>🧠 Remembers you</h3>
      Ask it to remember something and it will know it in every conversation after that. You can list and delete what it keeps at any time.
    </td>
    <td valign="top">
      <h3>🗣️ Interrupt it</h3>
      Echo cancellation removes Jarvis's own voice from the microphone, so you can cut in mid-sentence without headphones.
    </td>
    <td valign="top">
      <h3>🔁 Always there</h3>
      Starts with your Mac and waits quietly for "Hey Jarvis". Only one copy runs at a time.
    </td>
  </tr>
</table>

## ◈ HUD

<div align="center">
<img src="docs/assets/hud.png" width="100%" alt="The Jarvis HUD: a glowing core of rings in the centre, weather and AI news on the left, system, timers and latency on the right, the conversation below">
</div>

<br>

When `voice` starts, the HUD opens in its own window. It is served by Jarvis itself at <http://127.0.0.1:8765>, so it also works in any browser tab, on another monitor or in full screen.

- **The core** changes with the state: dim and slow while it sleeps, cyan while it listens, amber and spinning fast while it thinks, bright while it speaks. The ring of bars follows the loudness of your voice, then of Jarvis's.
- **The transcript** shows what you said and the reply as it is generated, with the tool being used ("checking the weather").
- **The panels** show the weather, the latest AI headlines, battery, CPU and memory, timers counting down, and how long each reply took (speech recognition, first word from the model, first sound).
- **Tap the core** (or press <kbd>Space</kbd>) to wake Jarvis without the wake word, or to send it back to sleep mid-sentence. <kbd>F</kbd> toggles full screen.
- **Closed the window?** Jarvis keeps running; `uv run python -m backend hud-window` opens it again.

To see it without the microphone or the models, run `uv run python -m backend hud-demo`: a scripted conversation drives the interface (`--sample` uses fixed panel data and works offline).

## ◈ Architecture

```mermaid
flowchart LR
    MIC(["🎙️ Microphone"]) --> WAKE["Wake word<br/><sub>openWakeWord</sub>"]
    WAKE --> VAD["Speech detection<br/><sub>Silero VAD</sub>"]
    VAD --> STT["Transcription<br/><sub>mlx-whisper</sub>"]
    STT --> LLM{{"🧠 Brain<br/><sub>Ollama · qwen3</sub>"}}
    LLM <--> TOOLS["Tools<br/><sub>weather · news · system · timers</sub>"]
    LLM <--> MEM[("Memory<br/><sub>SQLite</sub>")]
    LLM --> TTS["Voice<br/><sub>Piper</sub>"]
    TTS --> SPK(["🔊 Speaker"])
    TTS -. reference .-> AEC["Echo cancellation<br/><sub>WebRTC AEC3</sub>"]
    MIC --> AEC
    AEC -. barge-in .-> VAD
    LLM -. WebSocket .-> HUD[["🛰️ HUD<br/><sub>window · browser</sub>"]]

    classDef core fill:#0a1420,stroke:#00e5ff,color:#eafcff,stroke-width:2px
    classDef io fill:#02050a,stroke:#ffb340,color:#ffd59a
    class WAKE,VAD,STT,TOOLS,MEM,TTS,AEC,HUD core
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

The setup downloads the language model (`qwen3:8b`, about 5.2 GB), the speech recognition model (`whisper-small`, about 0.5 GB), the Piper voice for your language and the "Hey Jarvis" wake word model (about 4 MB). It also installs Node.js if needed and builds the HUD.

### Commands

| Command | What it does |
|---|---|
| `uv run python -m backend voice` | Voice conversation: say "Hey Jarvis", ask, and Jarvis answers out loud. Starts with the weather and AI news and opens the HUD (`--no-briefing` and `--no-hud` skip them, `--no-wake` listens all the time, `-r` resumes the last conversation) |
| `uv run python -m backend chat` | Text chat with streaming replies (`-s` reads them aloud, `-r` resumes the last conversation) |
| `uv run python -m backend autostart install` | Launch Jarvis at login, in the background (`status`, `stop`, `start`, `uninstall`; `--no-briefing` to start silently) |
| `uv run python -m backend memory` | List what Jarvis remembers about you (`memory forget 3`, `memory clear`) |
| `uv run python -m backend hud-window` | Open the HUD window again while Jarvis runs |
| `uv run python -m backend hud-demo` | The HUD with a scripted conversation: no microphone or models needed (`--sample` for fixed, offline panel data) |
| `uv run python -m backend doctor` | Health check: Ollama, models, voice, wake word, microphone, owner, location |
| `uv run python -m backend briefing` | Print the activation briefing with its timings (`-s` reads it aloud) |
| `uv run python -m backend wake-test` | Live wake word scores, to tune the detection threshold |
| `uv run python -m backend echo-test` | Jarvis speaks for 10 s and measures how much of its own voice the microphone still hears |
| `uv run python -m backend bench [models...]` | Compare LLM latency across Ollama models over a short scripted conversation |
| `uv run python -m backend bench-stt [models...]` | Record one phrase and compare Whisper models on it |
| `uv run python -m backend prompt` | Print the current system prompt |
| `uv run python -m backend config` | Print the effective configuration |
| `uv run pytest` · `uv run ruff check .` | Tests and lint |
| `npm --prefix frontend run dev` | Work on the HUD with hot reload at <http://localhost:5173> (with `voice` or `hud-demo` running) |

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
<summary><b>Tools</b></summary>
<br>

Jarvis calls tools on its own when a question needs live data or an action:

| Tool | Ask things like | Source |
|---|---|---|
| Weather | "How's the weather?" · "Will it rain tomorrow?" · "And in Lisbon?" | [Open-Meteo](https://open-meteo.com) (free, no key) |
| AI news | "Any AI news?" · "Anything about robots?" | RSS feeds in `config.yaml` (`news.feeds`) |
| System | "How much battery is left?" · "Set the volume to 30" · "Open Safari" | macOS (`pmset`, `osascript`, `open`) |
| Timers | "Remind me in 10 minutes to take the cake out" · "Make it 15" · "Cancel the timer" | Local; announced out loud when they end |
| Clock | "How long until 6 pm?" · "How many days until Christmas?" | Local; the code does the arithmetic, not the model |
| Memory | "Remember that I take my coffee black" · "Actually, make it with milk" · "Forget that" | Local (`data/jarvis.db`) |

When a tool takes a moment (a slow news feed, for example), Jarvis says "One moment" so you are not left in silence. In voice mode, saying goodbye, "that's all" or "you can rest" sends Jarvis back to sleep. Unmuting brings back the volume you had before. Turn tools off with `tools.enabled` in `config.yaml` (for example `JARVIS_TOOLS__ENABLED='["weather", "timers"]'`). Timers live in memory and are cleared when Jarvis quits.

</details>

<details>
<summary><b>Memory</b></summary>
<br>

Jarvis keeps a fact only when you ask it to ("remember that...", "note that...", "don't forget..."). Questions such as "do you remember how I take my coffee?" never store anything: that is checked in code, not left to the model, because the model sometimes guessed an answer and saved the guess. Corrections replace the old fact.

Everything it keeps goes into every conversation, up to 100 short facts, stored locally in `data/jarvis.db`. `uv run python -m backend memory` lists them; `memory forget <id>` and `memory clear` delete them without asking Jarvis. Turn it off by removing `memory` from `tools.enabled`.

</details>

<details>
<summary><b>Launch at login</b></summary>
<br>

```bash
uv run python -m backend autostart install   # starts now and on every login
uv run python -m backend autostart status    # running? where is the log?
uv run python -m backend autostart stop      # until the next login
uv run python -m backend autostart uninstall
```

It builds a small `Jarvis.app` in `data/` and installs a LaunchAgent (`~/Library/LaunchAgents/local.my-jarvis.voice.plist`) that runs `voice` through it in the background, waits up to 2 minutes for Ollama to start and restarts Jarvis if it crashes. The app is what makes macOS show "Jarvis", with its own icon, in Activity Monitor, Login Items, the Dock (for the HUD window) and the microphone permission, instead of "python". Output goes to `data/logs/jarvis.log`. Add `--no-briefing` to `install` if you prefer it to start silently. After updating Python (`uv python upgrade`), run `autostart install` again.

The first time, macOS asks for microphone access for Jarvis. If Jarvis does not hear you, allow it in **System Settings → Privacy & Security → Microphone**; the log says so when the microphone delivers only silence. Only one Jarvis runs at a time, so a `voice` started in the terminal while it runs in the background tells you so and exits.

</details>

<details>
<summary><b>HUD</b></summary>
<br>

The HUD settings live in `config.yaml`: `hud.enabled`, `hud.window` (a native window, or `false` for a browser tab), `hud.open_on_start` and the address in `server.host` / `server.port` (default `127.0.0.1:8765`). If a HUD page is already open, Jarvis reconnects it instead of opening another. If the window cannot open, the HUD opens in the browser. If the port is busy, Jarvis runs without the HUD and says so.

Only pages served from this address (or the Vite dev server) may connect: the WebSocket checks the page's origin, so other websites open in your browser cannot read the conversation. To open the HUD from another device on your network, set `JARVIS_SERVER__HOST` to this Mac's address, and keep in mind that anyone on that network can then open it too.

After changing the frontend, rebuild with `npm --prefix frontend run build`.

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

By default Jarvis is half duplex: it stops listening while it thinks and speaks, so it never hears itself through the speakers. Turn on barge-in to interrupt it mid-sentence:

```bash
JARVIS_AUDIO__BARGE_IN=true uv run python -m backend voice
```

No headphones needed: echo cancellation (WebRTC's AEC3, running locally) receives everything Jarvis plays and removes it from the microphone. Run `uv run python -m backend echo-test` to check your room: on a MacBook Air at normal volume, the speech detector went from hearing Jarvis in 88% of the frames to 0%. At high volume you may need to speak up, or keep talking for a second or two, before Jarvis notices you; with the `say` voice there is no echo cancellation, so use headphones.

On start, Jarvis says the briefing: a greeting and today's weather, written from the forecast numbers, then the most relevant AI headlines picked and retold by the model, which writes them while the first part is spoken. It starts about 2 seconds after launch, before speech recognition has finished loading, and ends with a line showing when each part was ready. A source that fails or takes over `JARVIS_BRIEFING__FETCH_TIMEOUT_S` (default 5) is left out. Turn it off with `JARVIS_BRIEFING__ENABLED=false` or `voice --no-briefing`.

Other knobs: `JARVIS_VAD__SILENCE_MS` (how long a pause ends your turn, default 700), `JARVIS_TTS__LENGTH_SCALE` (speaking speed, below 1 is faster), and `JARVIS_AUDIO__INPUT_DEVICE` / `_OUTPUT_DEVICE` (list devices with `uv run python -m sounddevice`).

</details>

## ◈ Roadmap

| | Phase | Status |
|:-:|---|---|
| 0 | Foundation: settings, setup script, health check | ✅ Done |
| 1 | Text chat: streaming LLM, persona, history, personalization | ✅ Done |
| 2 | Voice in and out: VAD, Whisper, streaming Piper, barge-in | ✅ Done |
| 3 | Wake word: "Hey Jarvis", state machine | ✅ Done |
| 4 | Tools: weather, AI news, system, timers | ✅ Done |
| 5 | Activation briefing | ✅ Done |
| 6 | Holographic HUD | ✅ Done |
| 7 | Polish: long-term memory, launch at login, echo cancellation, native HUD window | ✅ Done |


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
![Vite](https://img.shields.io/badge/Vite-0a1420?style=flat-square&logo=vite&logoColor=00e5ff)
![pywebview](https://img.shields.io/badge/pywebview-0a1420?style=flat-square)
![WebRTC AEC3](https://img.shields.io/badge/WebRTC_AEC3-0a1420?style=flat-square&logo=webrtc&logoColor=00e5ff)
![SQLite](https://img.shields.io/badge/SQLite-0a1420?style=flat-square&logo=sqlite&logoColor=00e5ff)
![uv](https://img.shields.io/badge/uv-0a1420?style=flat-square&logo=uv&logoColor=00e5ff)

</div>

## ◈ License

[MIT](LICENSE). Jarvis is an independent hobby project and is not affiliated with Marvel or Disney.

<div align="center">
<br>
<sub><code>// SYSTEMS ONLINE · RUNNING ON YOUR MACHINE</code></sub>
</div>
