import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api is proxied to the FastAPI server so the browser sees one origin.
// Fixed ports so it runs beside dreamjob (5173/8000) and route-or-roam (5181/8001);
// strictPort fails loudly instead of silently moving to a port another app owns.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8002" },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
