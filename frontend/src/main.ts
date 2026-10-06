import "./style.css";

import { Core, type Mode } from "./core";
import { fill, stringsFor } from "./i18n";
import { Panels } from "./panels";
import { Link } from "./socket";
import type { Hello, HudEvent } from "./types";

const hud = document.querySelector<HTMLElement>(".hud")!;
const byId = (id: string) => document.getElementById(id)!;

let t = stringsFor(navigator.language);
let hello: Hello | null = null;
let mode: Mode = "offline";
let toolTimer = 0;

const panels = new Panels(t);
panels.setStrings(t);

function tap(): void {
  if (mode === "sleeping") link.send("wake");
  else if (mode === "listening" || mode === "thinking" || mode === "speaking") link.send("sleep");
}

const core = new Core(byId("core") as HTMLCanvasElement, tap);

function setMode(next: Mode): void {
  mode = next;
  hud.dataset.state = next;
  core.setMode(next);
  const label = t.states[next];
  byId("state-label").textContent = label;
  byId("caption-state").textContent = label;
  const name = hello?.name ?? "Jarvis";
  let hint = "";
  if (next === "sleeping") hint = hello?.wake_word === false ? "" : fill(t.hints.sleeping, { name });
  else if (next === "listening") hint = t.hints.listening;
  else if (next === "offline") hint = t.hints.offline;
  byId("caption-hint").textContent = hint;
  if (next !== "thinking") showTool(null);
}

function showTool(name: string | null): void {
  const box = byId("tool");
  clearTimeout(toolTimer);
  if (!name) {
    box.classList.remove("on");
    return;
  }
  box.textContent = `⟳ ${t.tools[name] ?? name}`;
  box.classList.add("on");
  toolTimer = window.setTimeout(() => box.classList.remove("on"), 6000);
}

function greet(h: Hello): void {
  hello = h;
  t = stringsFor(h.language);
  document.documentElement.lang = h.language;
  panels.setStrings(t);
  byId("name").textContent = h.name.toUpperCase();
  byId("model").textContent = h.model;
  document.title = `${h.name} HUD`;
  byId("owner").textContent = [h.owner, h.place].filter(Boolean).join(" · ");
  setMode(mode);
  tickClock();
}

function onEvent(e: HudEvent): void {
  switch (e.type) {
    case "hello":
      return greet(e);
    case "state":
      return setMode(e.state);
    case "level":
      if (e.mic !== undefined && (mode === "listening" || mode === "sleeping")) core.feed(e.mic);
      if (e.out !== undefined && mode === "speaking") core.feed(e.out);
      return;
    case "weather":
      return panels.weather(e);
    case "news":
      return panels.news(e);
    case "system":
      return panels.system(e);
    case "timers":
      return panels.setTimers(e.items);
    case "stats":
      return panels.stats(e);
    case "transcript":
      return panels.setTranscript(e.lines);
    case "user":
      return panels.user(e.text);
    case "reply":
      return panels.reply(e.delta);
    case "reply_end":
      return panels.replyEnd();
    case "tool":
      return showTool(e.name);
  }
}

const link = new Link(onEvent, (online) => {
  if (!online) setMode("offline");
});

function tickClock(): void {
  const now = new Date();
  const lang = hello?.language ?? navigator.language;
  byId("time").textContent = now.toLocaleTimeString(lang, { hour: "2-digit", minute: "2-digit" });
  byId("date").textContent = now.toLocaleDateString(lang, {
    weekday: "long",
    day: "numeric",
    month: "long",
  });
}

// In the native window (pywebview) WebKit's Fullscreen API is off: the window does it.
type PyWebView = { api?: { toggle_fullscreen?: () => Promise<void> } };

function toggleFullscreen(): void {
  const native = (window as unknown as { pywebview?: PyWebView }).pywebview?.api;
  if (native?.toggle_fullscreen) void native.toggle_fullscreen();
  else if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen();
}

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.code === "Space") {
    e.preventDefault();
    tap();
  } else if (e.key === "f" || e.key === "F") {
    toggleFullscreen();
  }
});

setMode("offline");
tickClock();
setInterval(tickClock, 1000);
core.start();
link.connect();
