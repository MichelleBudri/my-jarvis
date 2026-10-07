// Side panels and the transcript. Everything from the backend goes in as text, never
// as HTML: news titles come from the internet.

import { copyText } from "./clipboard";
import { fill, type Strings } from "./i18n";
import type {
  FootprintEvent,
  NewsEvent,
  StatsEvent,
  SystemEvent,
  Timer,
  TranscriptLine,
  WeatherEvent,
} from "./types";

const SPARK_POINTS = 16;
const SHOWN_LINES = 4;
// A sentence is typed over this share of its spoken length, so the text never trails
// the voice (Piper's audio ends with a short silence).
const WRITE_SHARE = 0.9;
const TYPE_TICK_MS = 30; // how often typed characters are added
// Typing time of a character, in letters: the voice pauses after commas and full stops,
// and so does the typing.
const PAUSE_AFTER: Record<string, number> = { ",": 4, ";": 4, ":": 4, ".": 7, "!": 7, "?": 7 };

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

const gb = (mb: number) => (mb < 1024 ? `${Math.round(mb)} MB` : `${(mb / 1024).toFixed(1)} GB`);

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

type OpenUrl = { api?: { open_url?: (url: string) => Promise<boolean> } };

/** A headline that opens its article: in the default browser from the native window
 * (it would otherwise load inside the HUD), in a new tab from a browser. */
function newsLink(title: string, url: string, hint: string): HTMLElement {
  const safe = /^https?:\/\//i.test(url) ? url : null;
  if (!safe) return el("div", "title", title);
  const a = el("a", "title link", title) as HTMLAnchorElement;
  a.href = safe;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.title = hint;
  a.addEventListener("click", (e) => {
    const native = (window as unknown as { pywebview?: OpenUrl }).pywebview?.api;
    if (native?.open_url) {
      e.preventDefault();
      void native.open_url(safe);
    }
  });
  return a;
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
  private lastSystem: SystemEvent | null = null;
  private lastFootprint: FootprintEvent | null = null;
  private voiceTimes: number[] = [];
  private lines: TranscriptLine[] = [];
  // Spoken sentences waiting to be typed, and the one being typed.
  private queue: { text: string; ms: number }[] = [];
  private typing: { text: string; ms: number; at: number[]; shown: number; start: number } | null =
    null;
  private writeTimer = 0;
  private lastStats: StatsEvent | null = null;

  constructor(t: Strings) {
    this.t = t;
    setInterval(() => this.tickTimers(), 250);
    this.renderTimers();
    this.renderStats();
    this.renderSystem();
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
        const title = item.link
          ? newsLink(item.title, item.link, this.t.openNews)
          : el("div", "title", item.title);
        row.append(title, meta);
        return row;
      }),
    );
  }

  system(s: SystemEvent): void {
    this.lastSystem = s;
    this.renderSystem();
  }

  footprint(f: FootprintEvent): void {
    this.lastFootprint = f;
    this.renderSystem();
  }

  private renderSystem(): void {
    const s = this.lastSystem;
    if (!s) return;
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
    const f = this.lastFootprint;
    const totalMb = (s.memory_total_gb ?? 0) * 1024;
    const share = (mb: number | null) => (mb && totalMb ? (mb / totalMb) * 100 : null);
    if (f?.jarvis_mb != null) {
      rows.push(meter(t.jarvisMemory, share(f.jarvis_mb), gb(f.jarvis_mb)));
    }
    if (f?.model_mb != null) {
      const text = f.model_mb === 0 ? t.modelResting : gb(f.model_mb);
      rows.push(meter(t.modelMemory, share(f.model_mb), text));
    }
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
    this.stopWriting();
    this.lines = lines.map((l) => ({ ...l }));
    this.renderTranscript();
  }

  user(text: string): void {
    this.flushWords();
    this.closeReply();
    this.lines.push({ role: "user", text, open: false });
    this.renderTranscript();
  }

  /** Reply text. With `durationS` (a sentence as the voice starts it), it is typed
   * letter by letter over the time it takes to say; without it, at once. */
  reply(delta: string, durationS?: number): void {
    if (!durationS) {
      this.append(delta);
      return;
    }
    this.queue.push({ text: delta, ms: durationS * 1000 * WRITE_SHARE });
    if (!this.writeTimer) this.typeNext();
  }

  /** `cut`: the voice was interrupted, so what was not yet heard is dropped. */
  replyEnd(cut = false): void {
    if (cut) this.stopWriting();
    else this.flushWords();
    this.closeReply();
    this.renderTranscript();
  }

  private append(delta: string): void {
    const last = this.lines.at(-1);
    if (last?.role === "assistant" && last.open) last.text += delta;
    else this.lines.push({ role: "assistant", text: delta, open: true });
    this.renderTranscript();
  }

  private typeNext(): void {
    const next = this.queue.shift();
    if (!next) {
      this.typing = null;
      this.writeTimer = 0;
      this.renderTranscript(); // the caret blinks again until the next sentence
      return;
    }
    // When each character is due, as a share of the sentence: pauses after punctuation.
    const chars = [...next.text];
    const weights = chars.map((_, i) => 1 + (PAUSE_AFTER[chars[i - 1]] ?? 0));
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    let sum = 0;
    const at = weights.map((w) => ((sum += w) / total) * next.ms);
    this.typing = { ...next, at, shown: 0, start: performance.now() };
    this.typeTick();
  }

  private typeTick(): void {
    const job = this.typing;
    if (!job) return;
    const elapsed = performance.now() - job.start;
    let n = job.shown;
    while (n < job.at.length && job.at[n] <= elapsed) n++;
    if (n > job.shown) {
      this.append([...job.text].slice(job.shown, n).join(""));
      job.shown = n;
    }
    if (job.shown >= job.at.length) {
      this.typeNext();
      return;
    }
    this.writeTimer = window.setTimeout(() => this.typeTick(), TYPE_TICK_MS);
  }

  private flushWords(): void {
    const job = this.typing;
    const rest =
      (job ? [...job.text].slice(job.shown).join("") : "") +
      this.queue.map((q) => q.text).join("");
    this.stopWriting();
    if (rest) this.append(rest);
  }

  private stopWriting(): void {
    clearTimeout(this.writeTimer);
    this.writeTimer = 0;
    this.queue = [];
    this.typing = null;
  }

  private closeReply(): void {
    const last = this.lines.at(-1);
    if (last?.open) last.open = false;
  }

  private copyButton(line: TranscriptLine): HTMLElement {
    const button = el("button", "copy", `⧉ ${this.t.copy}`) as HTMLButtonElement;
    button.type = "button";
    button.title = this.t.copy;
    button.addEventListener("click", async () => {
      const ok = await copyText(line.text.trim());
      button.textContent = ok ? `✓ ${this.t.copied}` : `✖ ${this.t.copyFailed}`;
      button.classList.toggle("done", ok);
      window.setTimeout(() => {
        button.textContent = `⧉ ${this.t.copy}`;
        button.classList.remove("done");
      }, 1500);
    });
    return button;
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
        if (line.open) row.append(el("span", this.typing ? "cursor typing" : "cursor"));
        else if (line.role === "assistant" && line.text.trim()) row.append(this.copyButton(line));
        return row;
      }),
    );
  }
}
