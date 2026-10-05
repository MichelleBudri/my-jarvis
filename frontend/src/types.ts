// Events sent by backend/hud (see backend/hud/bus.py and backend/voice.py).

export type AssistantState = "booting" | "sleeping" | "listening" | "thinking" | "speaking";

export interface Hello {
  type: "hello";
  name: string;
  language: string;
  model: string;
  owner: string | null;
  place: string | null;
  tools: string[];
  wake_word: boolean;
}

export interface StateEvent {
  type: "state";
  state: AssistantState;
}

export interface LevelEvent {
  type: "level";
  mic?: number;
  out?: number;
}

export interface TranscriptLine {
  role: "user" | "assistant";
  text: string;
  open: boolean;
}

export interface WeatherDay {
  day: string;
  weekday: string;
  conditions: string | null;
  min: number | null;
  max: number | null;
  rain_chance_percent: number | null;
}

export interface WeatherEvent {
  type: "weather";
  place: string | null;
  now: {
    conditions: string | null;
    temperature: number | null;
    feels_like: number | null;
    humidity_percent: number | null;
    wind_kmh: number | null;
  };
  daytime: boolean | null;
  sunrise: string | null;
  sunset: string | null;
  days: WeatherDay[];
}

export interface NewsEvent {
  type: "news";
  items: { title: string; source: string; hours_ago: number | null }[];
}

export interface SystemEvent {
  type: "system";
  battery: { percent: number; status: string; on_power_adapter: boolean } | null;
  cpu_percent: number | null;
  memory_percent: number | null;
  memory_total_gb: number | null;
}

export interface Timer {
  id: number;
  label: string | null;
  seconds: number;
  ends_at_ms: number;
}

export interface StatsEvent {
  type: "stats";
  stt_s: number | null;
  first_token_s: number | null;
  voice_s: number | null;
  tokens_per_second: number | null;
  tools: string[];
}

export type HudEvent =
  | Hello
  | StateEvent
  | LevelEvent
  | WeatherEvent
  | NewsEvent
  | SystemEvent
  | StatsEvent
  | { type: "timers"; items: Timer[] }
  | { type: "transcript"; lines: TranscriptLine[] }
  | { type: "user"; text: string }
  | { type: "reply"; delta: string }
  | { type: "reply_end" }
  | { type: "tool"; name: string };
