// The holographic core: concentric rings drawn on a canvas, reacting to the
// assistant's state and to the loudness of the microphone or the voice.

import type { AssistantState } from "./types";

export type Mode = AssistantState | "offline";

type RGB = [number, number, number];

interface Look {
  color: RGB; // main rings
  accent: RGB; // the fast arc
  glow: number;
  speed: number;
  alpha: number;
}

const CYAN: RGB = [0, 229, 255];
const ICE: RGB = [122, 244, 255];
const AMBER: RGB = [255, 179, 64];
const GREEN: RGB = [61, 255, 176];
const GREY: RGB = [84, 104, 120];

const LOOKS: Record<Mode, Look> = {
  booting: { color: CYAN, accent: AMBER, glow: 0.5, speed: 1.8, alpha: 0.75 },
  sleeping: { color: CYAN, accent: AMBER, glow: 0.25, speed: 0.22, alpha: 0.5 },
  listening: { color: CYAN, accent: GREEN, glow: 0.8, speed: 0.8, alpha: 1 },
  thinking: { color: AMBER, accent: CYAN, glow: 0.75, speed: 3.4, alpha: 1 },
  speaking: { color: ICE, accent: AMBER, glow: 1, speed: 1.1, alpha: 1 },
  offline: { color: GREY, accent: GREY, glow: 0.08, speed: 0.06, alpha: 0.4 },
};

const BARS = 120; // audio bars around the ring (mirrored: 60 distinct samples)
const IDLE_FRAME_MS = 1000 / 15; // frame interval while sleeping or offline
const SAMPLE_MS = 33; // how often the level history advances
const SILENT_VOICE_MS = 300; // speaking with no levels (macOS `say`): animate anyway
const TAU = Math.PI * 2;
const CAPTION_PX = 76; // room under the core for the state caption and hint

const rgba = ([r, g, b]: RGB, a: number) => `rgba(${r | 0},${g | 0},${b | 0},${a})`;
const mix = (a: number, b: number, k: number) => a + (b - a) * k;
const mixRGB = (a: RGB, b: RGB, k: number): RGB => [
  mix(a[0], b[0], k),
  mix(a[1], b[1], k),
  mix(a[2], b[2], k),
];

export class Core {
  private readonly ctx: CanvasRenderingContext2D;
  private mode: Mode = "offline";
  private look: Look = { ...LOOKS.offline, color: [...GREY], accent: [...GREY] };
  private phase = 0;
  private raw = 0;
  private level = 0;
  private lastFeed = -Infinity;
  private readonly history = new Float32Array(BARS / 2);
  private head = 0;
  private lastSample = 0;
  private born = performance.now();
  private last = performance.now();
  private size = { w: 0, h: 0, r: 0, cy: 0 };
  private readonly calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    onTap: () => void,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2D canvas");
    this.ctx = ctx;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    canvas.addEventListener("click", (e) => {
      if (this.onCore(e)) onTap();
    });
    canvas.addEventListener("mousemove", (e) => {
      canvas.style.cursor = this.onCore(e) ? "pointer" : "default";
    });
  }

  setMode(mode: Mode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    if (mode !== "speaking" && mode !== "listening") this.raw = 0;
  }

  /** Loudness 0..1 of whatever the assistant is hearing or saying. */
  feed(value: number): void {
    this.raw = value;
    this.lastFeed = performance.now();
  }

  start(): void {
    let lastDraw = 0;
    let running = false;
    const frame = (now: number) => {
      if (document.hidden) {
        running = false; // resumed by visibilitychange: nothing drawn while unseen
        return;
      }
      // Asleep the core barely moves: a quarter of the frames is enough (D-37).
      const idle = this.mode === "sleeping" || this.mode === "offline";
      if (!idle || now - lastDraw >= IDLE_FRAME_MS) {
        this.draw(now);
        lastDraw = now;
      }
      requestAnimationFrame(frame);
    };
    const resume = () => {
      if (running || document.hidden) return;
      running = true;
      this.last = performance.now(); // no jump after a long pause
      requestAnimationFrame(frame);
    };
    document.addEventListener("visibilitychange", resume);
    resume();
  }

  private onCore(e: MouseEvent): boolean {
    const rect = this.canvas.getBoundingClientRect();
    const dx = e.clientX - rect.left - this.size.w / 2;
    const dy = e.clientY - rect.top - this.size.cy;
    return Math.hypot(dx, dy) < this.size.r * 0.62;
  }

  private resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const free = Math.max(0, height - CAPTION_PX);
    this.size = { w: width, h: height, r: Math.min(width, free) * 0.4, cy: free / 2 };
  }

  private input(now: number): number {
    if (this.mode === "speaking" && now - this.lastFeed > SILENT_VOICE_MS) {
      const t = now / 1000; // no real levels: a speech-like wobble
      return 0.45 + 0.3 * Math.sin(t * 15) * Math.sin(t * 2.7) + 0.1 * Math.sin(t * 41);
    }
    if (this.mode === "listening" || this.mode === "speaking") return this.raw;
    if (this.mode === "thinking") return 0.12 + 0.08 * Math.sin(now / 90);
    return 0;
  }

  private update(now: number, dt: number): void {
    const target = LOOKS[this.mode];
    const k = 1 - Math.exp(-dt * 5);
    const L = this.look;
    L.color = mixRGB(L.color, target.color, k);
    L.accent = mixRGB(L.accent, target.accent, k);
    L.glow = mix(L.glow, target.glow, k);
    L.speed = mix(L.speed, target.speed, k);
    L.alpha = mix(L.alpha, target.alpha, k);
    this.phase += dt * L.speed * (this.calm ? 0.25 : 1);

    const input = Math.max(0, Math.min(1, this.input(now)));
    // Fast attack, slower release: syllables pop, silence settles.
    const rate = input > this.level ? 18 : 5;
    this.level = mix(this.level, input, 1 - Math.exp(-dt * rate));
    if (now - this.lastSample >= SAMPLE_MS) {
      this.lastSample = now;
      this.head = (this.head + 1) % this.history.length;
      this.history[this.head] = this.level;
    }
  }

  private draw(now: number): void {
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    this.update(now, dt);

    const { ctx } = this;
    const { w, h, cy } = this.size;
    const intro = Math.min(1, (now - this.born) / 1400);
    const ease = 1 - (1 - intro) ** 3;
    const R = this.size.r * (0.86 + 0.14 * ease);
    const L = this.look;
    const p = this.phase;
    const breath = 0.5 + 0.5 * Math.sin(now / 1600);

    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(w / 2, cy);
    ctx.globalAlpha = L.alpha * ease;

    const halo = ctx.createRadialGradient(0, 0, R * 0.2, 0, 0, R * 1.3);
    halo.addColorStop(0, rgba(L.color, 0.16 * L.glow + 0.08 * this.level));
    halo.addColorStop(1, rgba(L.color, 0));
    ctx.fillStyle = halo;
    ctx.fillRect(-R * 1.4, -R * 1.4, R * 2.8, R * 2.8);

    if (this.mode === "thinking" && "createConicGradient" in ctx) {
      const sweep = ctx.createConicGradient(p * 1.2, 0, 0);
      sweep.addColorStop(0, rgba(L.color, 0.22));
      sweep.addColorStop(0.12, rgba(L.color, 0));
      sweep.addColorStop(1, rgba(L.color, 0));
      ctx.fillStyle = sweep;
      ctx.beginPath();
      ctx.arc(0, 0, R * 0.98, 0, TAU);
      ctx.arc(0, 0, R * 0.52, 0, TAU, true);
      ctx.fill();
    }

    ctx.lineCap = "butt";
    ctx.shadowColor = rgba(L.color, 0.9);

    this.ring(-p * 0.12, () => {
      for (let i = 0; i < 120; i++) {
        const a = (i / 120) * TAU;
        const long = i % 10 === 0;
        const r0 = R * (long ? 0.955 : 0.975);
        ctx.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
        ctx.lineTo(Math.cos(a) * R, Math.sin(a) * R);
      }
      ctx.strokeStyle = rgba(L.color, 0.45);
      ctx.lineWidth = 1.2;
    });

    ctx.shadowBlur = 12 * L.glow;
    this.arcs(R * 0.9, p * 0.3, [0.42, 0.08, 0.25, 0.08, 0.6, 0.12, 0.45, 0.1], rgba(L.color, 0.9), 3);

    const accents = this.mode === "thinking" ? 2 : 1;
    for (let i = 0; i < accents; i++) {
      const start = p * 1.3 + (i * TAU) / accents;
      ctx.beginPath();
      ctx.arc(0, 0, R * 0.83, start, start + 0.38);
      ctx.strokeStyle = rgba(L.accent, 0.95);
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }
    ctx.shadowBlur = 0;

    this.circle(R * 0.77, rgba(L.color, 0.35), 1);
    ctx.save();
    ctx.rotate(-p * 0.2);
    ctx.setLineDash([2, 7]);
    this.circle(R * 0.71, rgba(L.color, 0.7), 1.5);
    ctx.restore();

    // Audio bars, mirrored left/right, newest at the top
    const n = this.history.length;
    const base = R * 0.53;
    const reach = R * 0.16;
    ctx.beginPath();
    for (let i = 0; i < BARS; i++) {
      const j = i < n ? i : BARS - 1 - i;
      const v = this.history[(this.head - j + n) % n];
      const len = 1.5 + reach * v;
      const a = -Math.PI / 2 + (i / BARS) * TAU;
      ctx.moveTo(Math.cos(a) * base, Math.sin(a) * base);
      ctx.lineTo(Math.cos(a) * (base + len), Math.sin(a) * (base + len));
    }
    ctx.strokeStyle = rgba(L.color, 0.85);
    ctx.lineWidth = Math.max(1.5, (TAU * base) / BARS - 2.5);
    ctx.shadowBlur = 8 * L.glow;
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.save();
    ctx.rotate(p * 0.18);
    ctx.setLineDash([R * 0.09, R * 0.035]);
    this.circle(R * 0.47, rgba(L.color, 0.28), R * 0.05);
    ctx.restore();

    this.circle(R * 0.36, rgba(ICE, 0.85 * L.alpha), 2);
    const pulse = 1 + this.level * 0.3 + (this.mode === "sleeping" ? breath * 0.06 : 0);
    const core = ctx.createRadialGradient(0, 0, 0, 0, 0, R * 0.31 * pulse);
    core.addColorStop(0, rgba([242, 254, 255], 0.95));
    core.addColorStop(0.35, rgba(mixRGB(ICE, L.color, 0.5), 0.75));
    core.addColorStop(1, rgba(L.color, 0));
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(0, 0, R * 0.31 * pulse, 0, TAU);
    ctx.fill();

    ctx.beginPath();
    for (const [x, y] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.moveTo(x * R * 1.06, y * R * 1.06);
      ctx.lineTo(x * R * 1.16, y * R * 1.16);
    }
    ctx.strokeStyle = rgba(L.color, 0.5);
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.restore();
  }

  private ring(rotation: number, path: () => void): void {
    const { ctx } = this;
    ctx.save();
    ctx.rotate(rotation);
    ctx.beginPath();
    path();
    ctx.stroke();
    ctx.restore();
  }

  private circle(r: number, color: string, width: number): void {
    const { ctx } = this;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, TAU);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  /** Arcs with gaps: `pattern` alternates arc and gap lengths, in radians. */
  private arcs(r: number, rotation: number, pattern: number[], color: string, width: number) {
    const { ctx } = this;
    const total = pattern.reduce((a, b) => a + b, 0);
    const scale = TAU / total;
    let a = rotation;
    ctx.beginPath();
    pattern.forEach((len, i) => {
      if (i % 2 === 0) {
        ctx.moveTo(Math.cos(a) * r, Math.sin(a) * r);
        ctx.arc(0, 0, r, a, a + len * scale);
      }
      a += len * scale;
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }
}
