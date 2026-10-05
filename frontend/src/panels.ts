// Side panels and the transcript. Everything from the backend goes in as text, never
// as HTML: news titles come from the internet.

import { fill, type Strings } from "./i18n";
import type {
  NewsEvent,
  StatsEvent,
  SystemEvent,
  Timer,
  TranscriptLine,
  WeatherEvent,
} from "./types";

const SPARK_POINTS = 16;
const SHOWN_LINES = 4;

function $(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} missing`);
  return node;
}

function el(tag: string, className = "", text = ""): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

const dash = (v: number | null | undefined, unit = "") => (v == null ? "--" : `${v}${unit}`);

function meter(label: string, percent: number | null, text: string, warn = false): HTMLElement {
  const row = el("div", "meter");
  const bar = el("div", "bar");
  const fillBar = el("i");
  fillBar.style.width = `${Math.max(0, Math.min(100, percent ?? 0))}%`;
  if (warn) row.classList.add("warn");
  bar.append(fillBar);
  row.append(el("span", "label", label), bar, el("span", "value", text));
  return row;
}

function clock(ms: number): string {
  const s = Math.max(0, Math.ceil(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

export class Panels {
  private t: Strings;
  private timers: Timer[] = [];
  private voiceTimes: number[] = [];
  private lines: TranscriptLine[] = [];
  private lastStats: StatsEvent | null = null;

  constructor(t: Strings) {
    this.t = t;
    setInterval(() => this.tickTimers(), 250);
    this.renderTimers();
    this.renderStats();
  }

  setStrings(t: Strings): void {
    this.t = t;
    document.querySelectorAll<HTMLElement>("[data-label]").forEach((node) => {
      const key = node.dataset.label as keyof Strings["panels"];
      node.textContent = t.panels[key] ?? "";
    });
    $("footer").textContent = `// ${t.footer}`;
    this.renderTimers();
    this.renderStats();
    this.renderTranscript();
  }

  weather(w: WeatherEvent): void {
    const t = this.t;
    $("w-place").textContent = (w.place ?? "").split(",")[0];
    $("w-temp").textContent = dash(w.now.temperature);
    $("w-cond").textContent = w.now.conditions ?? "";
    const feels = w.now.feels_like;
    $("w-feels").textContent = feels == null ? "" : `${t.feels} ${feels}°`;

    const details = $("w-details");
    details.replaceChildren();
    const pairs: [string, string][] = [
      [t.humidity, dash(w.now.humidity_percent, "%")],
      [t.wind, dash(w.now.wind_kmh, " km/h")],
    ];
    if (w.sunrise && w.sunset) pairs.push([t.sun, `${w.sunrise} – ${w.sunset}`]);
    for (const [k, v] of pairs) details.append(el("dt", "", k), el("dd", "", v));

    const days = $("w-days");
    days.replaceChildren(
      ...w.days.map((d) => {
        const row = el("li");
        const rain = d.rain_chance_percent;
        row.append(
          el("span", "day", d.day.replace(/-feira$/i, "")),
          el("span", "cond dim", d.conditions ?? ""),
          el("span", "rain", rain != null && rain >= 30 ? `${rain}%` : ""),
          el("span", "range", `${dash(d.min)}° / ${dash(d.max)}°`),
        );
        return row;
      }),
    );
  }

  news(n: NewsEvent): void {
    const list = $("news-list");
    if (!n.items.length) {
      list.replaceChildren(el("li", "empty", this.t.noNews));
      return;
    }
    list.replaceChildren(
      ...n.items.map((item) => {
        const row = el("li");
        const when =
          item.hours_ago == null
            ? ""
            : item.hours_ago < 1
              ? this.t.justNow
              : fill(this.t.hoursAgo, { n: item.hours_ago });
        const meta = el("div", "meta");
        meta.append(el("span", "", item.source), el("span", "", when));
        row.append(el("div", "title", item.title), meta);
        return row;
      }),
    );
  }

  system(s: SystemEvent): void {
    const t = this.t;
    const rows: HTMLElement[] = [];
    if (s.battery) {
      const b = s.battery;
      const charging = b.status.startsWith("charg") && b.status !== "charged";
      const text = charging ? `${b.percent}% ⚡` : `${b.percent}%`;
      rows.push(meter(t.battery, b.percent, text, b.percent <= 15 && !b.on_power_adapter));
    }
    rows.push(meter(t.cpu, s.cpu_percent, dash(s.cpu_percent, "%"), (s.cpu_percent ?? 0) > 85));
    const memText = s.memory_total_gb
      ? `${dash(s.memory_percent, "%")} · ${s.memory_total_gb} GB`
      : dash(s.memory_percent, "%");
    rows.push(meter(t.memory, s.memory_percent, memText, (s.memory_percent ?? 0) > 85));
    $("meters").replaceChildren(...rows);
  }

  setTimers(items: Timer[]): void {
    this.timers = items;
    this.renderTimers();
  }

  private renderTimers(): void {
    const list = $("timer-list");
    if (!this.timers.length) {
      list.replaceChildren(el("li", "empty", this.t.noTimers));
      return;
    }
    list.replaceChildren(
      ...this.timers.map((timer) => {
        const row = el("li");
        row.dataset.id = String(timer.id);
        const head = el("div", "head");
        head.append(el("span", "label", timer.label ?? this.t.timer), el("span", "left"));
        const bar = el("div", "bar");
        bar.append(el("i"));
        row.append(head, bar);
        return row;
      }),
    );
    this.tickTimers();
  }

  private tickTimers(): void {
    const now = Date.now();
    for (const timer of this.timers) {
      const row = document.querySelector<HTMLElement>(`#timer-list li[data-id="${timer.id}"]`);
      if (!row) continue;
      const left = timer.ends_at_ms - now;
      row.querySelector(".left")!.textContent = clock(left);
      const done = 1 - left / (timer.seconds * 1000);
      row.querySelector<HTMLElement>(".bar i")!.style.width = `${Math.min(100, done * 100)}%`;
      row.classList.toggle("ending", left < 10_000);
    }
  }

  stats(s: StatsEvent): void {
    this.lastStats = s;
    if (s.voice_s != null) {
      this.voiceTimes = [...this.voiceTimes, s.voice_s].slice(-SPARK_POINTS);
    }
    this.renderStats();
  }

  private renderStats(): void {
    const s = this.lastStats;
    const readouts: [string, number | null][] = [
      ["stt", s?.stt_s ?? null],
      ["llm", s?.first_token_s ?? null],
      [this.t.voice, s?.voice_s ?? null],
    ];
    $("readouts").replaceChildren(
      ...readouts.map(([label, value]) => {
        const box = el("div", "readout");
        box.append(el("span", "value", value == null ? "--" : value.toFixed(2)), el("span", "label", label));
        return box;
      }),
    );
    $("tps").textContent = s?.tokens_per_second ? `${s.tokens_per_second} tok/s` : "";
    const max = Math.max(2, ...this.voiceTimes);
    $("spark").replaceChildren(
      ...this.voiceTimes.map((v) => {
        const bar = el("i");
        bar.style.height = `${Math.max(6, (v / max) * 100)}%`;
        if (v > 2) bar.classList.add("slow"); // the target: voice in under 2 s
        return bar;
      }),
    );
  }

  setTranscript(lines: TranscriptLine[]): void {
    this.lines = lines.map((l) => ({ ...l }));
    this.renderTranscript();
  }

  user(text: string): void {
    this.closeReply();
    this.lines.push({ role: "user", text, open: false });
    this.renderTranscript();
  }

  reply(delta: string): void {
    const last = this.lines.at(-1);
    if (last?.role === "assistant" && last.open) last.text += delta;
    else this.lines.push({ role: "assistant", text: delta, open: true });
    this.renderTranscript();
  }

  replyEnd(): void {
    this.closeReply();
    this.renderTranscript();
  }

  private closeReply(): void {
    const last = this.lines.at(-1);
    if (last?.open) last.open = false;
  }

  private renderTranscript(): void {
    this.lines = this.lines.slice(-SHOWN_LINES);
    const box = $("transcript");
    box.replaceChildren(
      ...this.lines.map((line, i) => {
        const row = el("p", `line ${line.role}`);
        row.style.opacity = String(0.35 + (0.65 * (i + 1)) / this.lines.length);
        if (line.role === "user") row.append(el("span", "who", `${this.t.you} ›`));
        row.append(el("span", "text", line.text.trim()));
        if (line.open) row.append(el("span", "cursor"));
        return row;
      }),
    );
  }
}
