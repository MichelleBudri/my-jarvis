import { defineConfig } from "vite";

// `npm run dev` serves the HUD with hot reload on :5173 and forwards the events from a
// running `python -m backend voice` (or `hud-demo`) on :8765.
export default defineConfig({
  server: {
    proxy: { "/ws": { target: "ws://127.0.0.1:8765", ws: true } },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
