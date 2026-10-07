import type { HudEvent } from "./types";

const RETRY_MS = 1000; // the backend opens no new tab if a page reconnects within 1.5 s

/** A WebSocket to the assistant that reconnects forever. */
export class Link {
  private ws: WebSocket | null = null;

  constructor(
    private readonly onEvent: (event: HudEvent) => void,
    private readonly onStatus: (online: boolean) => void,
  ) {}

  connect(): void {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${location.host}/ws`);
    this.ws = ws;
    ws.onopen = () => this.onStatus(true);
    ws.onmessage = (msg) => {
      try {
        this.onEvent(JSON.parse(msg.data) as HudEvent);
      } catch (err) {
        console.warn("Bad HUD event", err);
      }
    };
    ws.onclose = () => {
      this.ws = null;
      this.onStatus(false);
      setTimeout(() => this.connect(), RETRY_MS);
    };
  }

  send(type: "wake" | "sleep" | "export"): void {
    this.post({ type });
  }

  /** Flip a switch; the backend answers with the new state. */
  set(type: "autostart", value: boolean): void {
    this.post({ type, value });
  }

  private post(msg: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }
}
