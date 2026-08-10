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

### Set the GPU backend first

Everything vendor-dependent — which torch wheel index to install from, which
defines a method subprocess needs, and for TRELLIS which *fork* to run — lives
in `surfacing-server/backends.json`, keyed by `SURFACING_GPU_BACKEND`.

**It defaults to `rocm`.** On an NVIDIA machine:

```bash
export SURFACING_GPU_BACKEND=cuda
```

Set it in two places or you will get confusing failures:

- **before installing** any method venv, so the CUDA wheels are the ones that
  get fetched. Setting it afterwards does not re-install anything, and nothing
  complains until the first job runs.
- **in the environment that starts the server** — the shell you run `npm run
  dev` from (the sidecar inherits it), or the one you run uvicorn in. Miss this
  and the adapters resolve the ROCm row: TRELLIS in particular will report that
  *TRELLIS-AMD* is missing on a machine that has upstream TRELLIS checked out.

`cluster/env.sh` already exports `cuda`, so a shell that sourced it is fine.
An unknown value is an error in every consumer rather than a silent fallback.

### Real surfacing methods

Each real method lives in its own repository (a submodule under
`surfacing-server/methods/`) with its own Python environment, so the server's
env stays torch-free and method dependencies never conflict. They are **opt-in
one at a time** — set up only the ones you intend to run.

| adapter | submodule | env | needs beyond `pip install` |
| --- | --- | --- | --- |
| `bbox` | — | server env | nothing; test stand-in, always available |
| `ns2s` | `NeuralSketch2Surf` | `.venv-ns2s` | S2V-Net checkpoint |
| `vns` | `vns` | `.venv-vns` | system SuiteSparse for scikit-sparse |
| `neuvas` | `NeuVAS` | `.venv-neuvas` | nothing — optimizes per sketch |
| `vrs2s` | `VRSketch2Shape` | `.venv-vrs2s` | sketch2model checkpoint |
| `sf3d` | `surface-fitting-3d-sketches` | `.venv-sf3d` | build `external/pygco`; CPU only |
| `trellis` | `TRELLIS` / `TRELLIS-AMD` | *inside the checkout* | upstream's own setup — see below |

The shape of a method install is the same for all but TRELLIS: init the
submodule, make a venv beside the server, install torch **first** from the
index this backend calls for, then the method's requirements file.

```bash
git submodule update --init surfacing-server/methods/NeuralSketch2Surf
cd surfacing-server
python3 -m venv .venv-ns2s
.venv-ns2s/bin/pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128     # rocm6.4 on AMD
.venv-ns2s/bin/pip install -r requirements-ns2s.txt
```

torch is deliberately absent from every requirements file — the index above is
the only thing deciding whether a venv ends up CUDA or ROCm, which is why the
backend has to be settled first. The per-backend indices are in
`backends.json` (`cu128`/`rocm6.4`, and `cu121`/`rocm6.2` for `vns`, which pins
an older torch).

`cluster/setup_venvs.sh` does all of this for every method in one pass, reading
the indices from `backends.json` — worth using even off-cluster. Run
`cluster/setup_native.sh` first: the `vns` venv compiles scikit-sparse against
system SuiteSparse and fails without it.

Checkpoints are not in the repos and are not downloaded automatically:

| method | file | from | into |
| --- | --- | --- | --- |
| `ns2s` | `best_model_jit.pt` | [HongshengY/S2V_Net](https://huggingface.co/HongshengY/S2V_Net) | `methods/NeuralSketch2Surf/checkpoints/` |
| `vrs2s` | `df_epoch_best_multicls.pth` | [YiziChen/sketch2model](https://huggingface.co/YiziChen/sketch2model) | `methods/VRSketch2Shape/weights/all_class/` |

`sf3d` needs one compiled dependency (`external/pygco`, an alpha-expansion
graph cut whose sources are downloaded by its own makefile) and `neuvas` needs
nothing at all. Both are written out step by step, along with the parameters,
quirks and runtimes of every method, in
[`surfacing-server/README.md`](surfacing-server/README.md) — that file is the
reference; this is the map.

### TRELLIS

TRELLIS is the exception to all of the above, in three ways:

- **Two checkouts, one adapter.** Upstream is CUDA-only (custom kernels,
  xformers, flash-attn), so AMD runs the TRELLIS-AMD fork instead. Which one
  you get is decided by `SURFACING_GPU_BACKEND`, so initialize the matching
  submodule and only that one:
  ```bash
  git submodule update --init --recursive surfacing-server/methods/TRELLIS  # cuda
  git submodule update --init surfacing-server/methods/TRELLIS-AMD          # rocm
  ```
  `--recursive` is required for upstream, which carries FlexiCubes as a nested
  submodule; the AMD fork vendors it. Without it the mesh decoder fails at
  import with a missing `trellis.representations.mesh.flexicubes` module.
- **The venv lives inside the checkout** (`methods/TRELLIS/.venv`), not beside
  the server, because the two forks need incompatible torch builds. There is no
  `requirements-trellis.txt` — follow each repo's own setup instructions.
  Override the pair with `TRELLIS_REPO` / `TRELLIS_PYTHON`.
- **Weights download on first use** — ~2.9 GB into `~/.cache/huggingface` plus
  ~1.2 GB of DINOv2 into `~/.cache/torch`. On a machine with a small home
  quota, point `HF_HOME` and `TORCH_HOME` at scratch before the first job.

Comparing results across the two backends? Set the `fill_holes` parameter to
`off` explicitly. It defaults to `auto`, which follows `backends.json` and is
on for CUDA and off for ROCm, so the meshes are not cleaned alike otherwise.

`cluster/` holds scripts for running the same methods as batch sweeps on a
Slurm cluster; see `cluster/README.md`. Note that those scripts name the
server's own environment `.venv-server`, while `vite.config.ts` looks for a
bare `.venv` — on a machine that has both roles, symlink one to the other.

## Deployment

Pushes to `main` are built and published to GitHub Pages by
`.github/workflows/deploy.yml`. `vite.config.ts` sets `base: "./"` so the built
assets resolve under the repository subpath — keep it.
