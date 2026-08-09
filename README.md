# InteractiveArtiDesign

3D sketching on canvas surfaces, evolving toward rapid design of articulated
objects: draw strokes in space, segment them into parts, rig those parts with
joints, pose and explode the result, and surface the sketch into a mesh.

A live build of the editor is at
<https://robbieawest.github.io/InteractiveArtiDesign/>. See `ARCHITECTURE.md`
for the module layout and what is implemented.

## Requirements

- **Node 22+** (CI builds on 22). If you use nvm, `nvm use 22`.
- A browser with **WebGL2** — any current Chrome, Firefox or Safari.
- **Python 3.10+**, only if you want the surfacing sidecar (see below). The
  editor runs fine without it.

## Quick start

```bash
git clone https://github.com/robbieawest/InteractiveArtiDesign.git
cd InteractiveArtiDesign
npm install
npm run dev
```

Then open the URL Vite prints (`http://localhost:5173`). That is the whole
setup for the editor — no submodules, no Python, no configuration.

### Scripts

| command | what it does |
| --- | --- |
| `npm run dev` | Vite dev server, plus the surfacing sidecar if it is installed |
| `npm test` | vitest, run mode |
| `npm run typecheck` | `vue-tsc --noEmit` |
| `npm run build` | typecheck + production build into `dist/` |
| `npm run preview` | serve the built `dist/` |

If your shell does not source nvm automatically (this is the usual case for
non-interactive shells), prefix commands with the path to your node:

```bash
export PATH="$HOME/.nvm/versions/node/v22.23.1/bin:$PATH"
```

### Running the dev server on a remote machine

Vite binds to `127.0.0.1:5173`, so the simplest route is an SSH tunnel — no
config change, and the browser still sees a `localhost` origin (which keeps
secure-context APIs and the `/api` proxy working):

```bash
# on the remote
npm run dev

# on your machine
ssh -N -L 5173:localhost:5173 user@<remote-ip>
```

Then browse to `http://localhost:5173`. The alternative, `npm run dev -- --host`
plus an open firewall port, exposes the server on the network and is not a
secure context; prefer the tunnel.

## Surfacing sidecar (optional)

`surfacing-server/` is a local FastAPI process that turns a sketch into a mesh
using one of several surfacing methods. `npm run dev` starts and stops it
alongside Vite once its venv exists; until then the Surfacer panel simply
reports the server as offline and everything else works normally.

```bash
cd surfacing-server
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

That gets you the server and its built-in `bbox` adapter, a stand-in that boxes
each part so the whole pipeline can be exercised end to end.

### Real surfacing methods

Each real method lives in its own repository (a submodule under
`surfacing-server/methods/`) with its own Python environment, so that the
server's env stays torch-free and method dependencies never conflict. They are
**opt-in one at a time** — initialize only the submodules you intend to run:

```bash
git submodule update --init surfacing-server/methods/NeuralSketch2Surf
```

Per-method setup — venv, torch build, checkpoints to download, and the handful
of methods that need something compiled — is documented method by method in
[`surfacing-server/README.md`](surfacing-server/README.md), which is also where
the job protocol and the benchmark harness are described.

GPU-vendor-dependent settings (which torch wheel index to install from, which
defines each method subprocess needs) all live in
`surfacing-server/backends.json`, keyed by `SURFACING_GPU_BACKEND`. It defaults
to `rocm`; on an NVIDIA machine export `SURFACING_GPU_BACKEND=cuda` **before**
installing the method venvs, so the install and the runtime agree.

`cluster/` holds scripts for running the same methods as batch sweeps on a
Slurm cluster; see `cluster/README.md`.

## Deployment

Pushes to `main` are built and published to GitHub Pages by
`.github/workflows/deploy.yml`. `vite.config.ts` sets `base: "./"` so the built
assets resolve under the repository subpath — keep it.
