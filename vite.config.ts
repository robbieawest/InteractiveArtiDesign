import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import vue from "@vitejs/plugin-vue";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

/** Starts the surfacing job server (surfacing-server/) alongside the dev
 *  server and stops it with it, so `npm run dev` is the only command needed.
 *  Skipped (with a hint) until its venv exists; if the port is already
 *  taken — e.g. the server runs in your own terminal — uvicorn just exits
 *  and the proxy keeps talking to the existing instance. */
function surfacingServer(): Plugin {
  let child: ChildProcess | undefined;
  return {
    name: "surfacing-server",
    apply: "serve",
    configureServer(server) {
      // vitest boots a vite server with this config too; no sidecar there
      if (process.env.VITEST) return;
      const dir = path.join(rootDir, "surfacing-server");
      const uvicorn = path.join(dir, ".venv", "bin", "uvicorn");
      if (!existsSync(uvicorn)) {
        server.config.logger.warn(
          "[surfacing] surfacing-server/.venv not found — the Surfacer " +
            "panel will report the server offline (setup: see " +
            "surfacing-server/README.md)",
        );
        return;
      }
      child = spawn(uvicorn, ["server:app", "--port", "8801"], {
        cwd: dir,
        stdio: "inherit",
      });
      // don't let the child's process handle keep vite alive on shutdown
      child.unref();
      const stop = () => {
        child?.kill();
        child = undefined;
      };
      server.httpServer?.on("close", stop);
      process.on("exit", stop);
    },
  };
}

export default defineConfig({
  plugins: [vue(), surfacingServer()],
  // relative asset paths so the build works both locally and when served
  // from a subpath like https://<user>.github.io/InteractiveArtiDesign/
  base: "./",
  server: {
    // the surfacing job server (surfacing-server/) runs alongside vite;
    // proxying keeps the browser same-origin so there is no CORS setup
    proxy: {
      "/api": "http://127.0.0.1:8801",
    },
    watch: {
      // never watch the python sidecar: its venvs hold tens of thousands of
      // files, its method submodules another few hundred thousand (the
      // TRELLIS-AMD one alone carries a 16GB env), and jobs/ writes
      // checkpoints throughout an optimization — enough to exhaust inotify
      // watches and crash vite (ENOSPC). None of it is read by the browser
      // build.
      ignored: ["**/surfacing-server/**"],
    },
  },
  test: {
    // the app's own tests only. Method venvs live inside surfacing-server/
    // and their site-packages contain third-party .test.ts files (gradio's
    // frontend suite, vendored under TRELLIS-AMD), which vitest would
    // otherwise collect and fail on. Matched by venv rather than by method
    // name so the next method to arrive is covered without an edit.
    exclude: ["**/node_modules/**", "**/dist/**", "**/.venv*/**"],
  },
});
