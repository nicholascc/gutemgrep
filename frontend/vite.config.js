import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "/static/",
  server: {
    proxy: {
      "/query": "http://127.0.0.1:5000",
      "/api": "http://127.0.0.1:5000",
      "/book": "http://127.0.0.1:5000",
      "/health": "http://127.0.0.1:5000"
    }
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
    assetsDir: "assets",
    rollupOptions: {
      output: {
        entryFileNames: "assets/app.js",
        chunkFileNames: "assets/chunk-[name].js",
        assetFileNames: (assetInfo) => {
          if (assetInfo?.name?.endsWith(".css")) return "assets/app.css";
          return "assets/[name][extname]";
        }
      }
    }
  }
});
