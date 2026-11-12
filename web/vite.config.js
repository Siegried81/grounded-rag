import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api is proxied to the FastAPI server so the browser sees one origin.
// Fixed ports so it runs beside dreamjob (5173/8000) and route-or-roam (5181/8001);
// strictPort fails loudly instead of silently moving to a port another app owns.
//
// When the API is not listening yet, Vite 6 answers 500, which the UI cannot
// tell from a real server error. The handler below answers 502 Bad Gateway
// instead (Vite registers its own 500 handler after `configure` and skips it
// once headers are sent), so the UI's startup retry sees the same status as
// behind Render's edge.
const api = {
  target: "http://127.0.0.1:8002",
  changeOrigin: true,
  configure: (proxy) => {
    proxy.on("error", (_err, _req, res) => {
      if (res && "writeHead" in res && !res.headersSent && !res.writableEnded) {
        res.writeHead(502, { "Content-Type": "text/plain" }).end();
      }
    });
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    proxy: { "/api": api },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
