# Surfacing job server

A local FastAPI sidecar that runs surfacing methods on the sketch. The Vite
app reaches it through the `/api` proxy in `vite.config.ts`.

One-time setup:

```bash
cd surfacing-server
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

After that, `npm run dev` starts and stops it together with Vite (the
`surfacingServer` plugin in `vite.config.ts`). To run it by hand instead —
e.g. to watch its logs separately or restart it after editing an adapter:

```bash
.venv/bin/uvicorn server:app --port 8801
```

(If it's already running when Vite starts, the spawned copy exits on the
taken port and the proxy talks to yours.)

The app works fine without it — the Surfacer panel just reports the server
as offline.

## Protocol

- `GET  /api/health` → `{ status, methods: [{ name, params }] }` — `params`
  are the adapter's user-editable parameter declarations (see
  `adapters/base.py`); the Surfacer panel renders them generically and sends
  the chosen values back as the job's `options`
- `POST /api/jobs` with `{ method, sketch, options }` → `{ jobId }`
- `GET  /api/jobs/{id}` → `{ status: pending|running|done|error, progress, message, error }`
- `GET  /api/jobs/{id}/log?after=N` → `{ lines, next }` — free-form adapter
  log lines from index N on (shown in the Surfacer panel's log window)
- `GET  /api/jobs/{id}/partials?after=N` → `{ names, next }` and
  `GET /api/jobs/{id}/partials/{index}` → `.glb` — pieces of the result (one
  per part) published while the job is still running, so the benchmark grid
  can show geometry as it lands. Names first, bytes on request, so a client
  that ignores partials never pays for them.
- `GET  /api/jobs/{id}/result` → binary glTF (`.glb`)

Jobs live in memory only, and hold no geometry longer than they must: fetching
`/result` frees that job's result *and* its partials on the spot, which is safe
because the protocol is one-shot in both directions (partials are pulled before
the result, the result is fetched once — a second fetch gets a 410). Only jobs
nobody collected — a failure, or a run the client abandoned — still hold their
meshes, and all but the newest few of those are forgotten as each new job
completes (`MAX_FINISHED_JOBS`). `POST /api/jobs/{id}/cancel` kills a job: its method
processes (adapters spawn through `common.spawn`, which attaches each process
to the job on whose thread it started) and *every* resident worker, since a
job's GPU work is not always its own — sf3d's proxy step runs another
adapter's worker. The client sends it automatically when its abort signal
fires, so the benchmark window's Stop frees the card immediately; the job
lands in `error` with "cancelled". None of the methods poll for cancellation,
so this is a kill, not a request — which is why Pause, which waits for the
current surface, is still the gentler option.

`POST /api/jobs` also takes an optional `save: { benchmarkId, adapter, run,
sketch }`; when present the server writes the finished result to
`benchmarks/<benchmarkId>/<adapter>/<run>/<sketch>.glb` itself, so benchmark
output survives the browser tab.

### Benchmarks

`benchmarks.py` backs the app's benchmark window — standardized surfacing
comparison across adapters and parameter permutations. The split of labour is
forced by what each side can do: only the server can read a folder of sketches
or write next to the repo, and only the client can run the glTF importer (it
is three.js). So the server scans and serves raw bytes, the client converts,
and the converted documents come back to be stored.

- `GET  /api/benchmark/scan?dir=` → the surfaceable inputs in a folder: loose
  `.json` sketch documents, and subfolders containing a `.gltf` (the
  `SampleModels/` layout)
- `GET  /api/benchmark/file?path=` → raw bytes of one source file (a `.gltf`
  and its sidecar `.bin` are pulled through this into the importer)
- `POST/GET /api/benchmark/{id}/sketches` → the preprocessed sketch documents
- `POST /api/benchmark/{id}/copy` `{ target }` → a clean copy: the sketches
  alone, into a new folder that must not already exist. The client writes the
  new `progress.json` (same runs, empty status), so a copy is a fresh sweep
  over the same inputs with the source folder's surfaces left intact
- `PUT/GET /api/benchmark/{id}/progress` → bench state, opaque to the server
- `GET  /api/benchmark` → the reopenable folders: id, sketch count, finished
  surface count, and whether a `progress.json` is there to restore
- `GET  /api/benchmark/{id}/results/{adapter}/{run}/{sketch}` → a stored `.glb`

One benchmark is one timestamped folder under `<repo>/benchmarks` (gitignored):

```
benchmarks/2026-07-28T15-04-22/
  sketches/          preprocessed inputs — sketch .json with articulation,
                     in the pose they are to be surfaced in. Point the source
                     picker here to rerun the same set with no preprocessing.
  progress.json
  ns2s/run-1/excavator.glb
  vns/run-1/excavator.glb
```

Those three things are the whole record, which is what makes a benchmark
reopenable from a cold server: `sketches/` holds the inputs, `progress.json`
the run configuration and per-cell status (written after every finished
surface, not at the end), and the `.glb` tree the results. Nothing is held
only in the tab, and no shutdown hook is needed — a benchmark paused or
interrupted at any point reopens from the Load list and resumes at the first
cell that is not already done. A cell that was in flight when the server died
was never recorded, so it simply reruns. Rerunning deliberately — the whole
bench or one run — just clears the affected cells' status; their `.glb` files
are overwritten when the cells are redone, and nothing reads a surface whose
cell is not `done`.

Only finished surfaces are written here. Partials — the per-part and
converging-snapshot geometry adapters publish mid-run — stay in the job's
memory and are never saved.

Method scratch is separate and disposable: each job gets `jobs/<method>-<id>/`
for input `.obj` files, per-iteration mesh snapshots, optimizer checkpoints,
tensorboard events and the probability volumes ns2s writes. A part-based ns2s
run leaves ~160MB of it, so the newest few directories are kept and the rest
pruned as each job starts (`SURFACING_KEEP_JOBS`, default 3). A failed run's
scratch therefore survives long enough to inspect, and a directory whose tree
has been written to in the last 15 minutes is never pruned — that grace is
what stops a long NeuVAS run, which writes deep inside its tree, from being
deleted underneath itself.

Not to be confused with `bench_vns/`, which is the VNS *performance* harness
(iteration-speed measurements on one sketch) and unrelated.

`sketch` is built by `src/surfacing/client.ts`: world-space stroke
centerlines with part ids, plus the part and joint (screw) tables, so
articulation-aware methods get the full picture and baselines ignore what
they don't need.

## Adapters

One surfacing method = one adapter in `adapters/` (register it in
`adapters/__init__.py`). Adapters run in a background thread per job; slow
methods should call `report(progress, message)` as they go. Helpers shared
between the subprocess-backed adapters — the sketch → curve-network `.obj`
writer, part grouping, the per-part mesh merge, the ROCm defines — live in
`adapters/common.py`.

Real methods live in their own repos (`methods/`, as submodules) and their own
python environments, and are invoked by their adapter as a subprocess — this
server's env stays torch-free so method dependencies never conflict. The
included `bbox` adapter is a stand-in that boxes each part, to exercise the
whole pipeline end to end.

| adapter | method | env | notes |
| --- | --- | --- | --- |
| `bbox` | — | server env | test stand-in, one padded box per part |
| `ns2s` | NeuralSketch2Surf | `.venv-ns2s` | seconds per sketch, closed surfaces |
| `vns` | Variational Neural Surfacing | `.venv-vns` | minutes per sketch, handles open sheets |
| `neuvas` | NeuVAS | `.venv-neuvas` | ~45 min per sketch, no weights needed |
| `vrs2s` | VRSketch2Shape | `.venv-vrs2s` | *generates* a shape rather than fitting one |
| `sf3d` | Piecewise-Smooth Surface Fitting | `.venv-sf3d` | CPU only; *refines* another adapter's output |

Every real adapter takes a **part-based** toggle: surface each part separately
and merge the results (boolean union when every part came out watertight,
plain concatenation otherwise) instead of fitting one surface to the whole
sketch. Unassigned strokes are ignored in that mode.

### NeuralSketch2Surf (`ns2s`)

Two things beyond `pip install`, both one-time:

```bash
python3 -m venv .venv-ns2s
.venv-ns2s/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
.venv-ns2s/bin/pip install -r requirements-ns2s.txt
```

and the S2V-Net checkpoint, which is not in the repo — download
`best_model_jit.pt` from <https://huggingface.co/HongshengY/S2V_Net> into
`methods/NeuralSketch2Surf/checkpoints/`. Override either location with
`NS2S_PYTHON` / `NS2S_CHECKPOINT` if they live elsewhere.

The adapter shells out to the submodule's `inference.py` unmodified. Because
that script takes an *input directory*, part-based mode is still a single
subprocess and a single model load however many parts there are — and each
part gets its own full 112³ grid, so small parts come out sharper than they do
in the whole-object pass.

Inference runs in a **resident worker** (`adapters/ns2s_worker.py`), spawned on
first use and reused for every later job. A fresh interpreter costs ~15s before
it can do ~1s of work — 12s of that is the first `.cuda()` building the ROCm
context — so a benchmark over N sketches went from `N × 16s` to `15s + N × 1s`
(measured: 14.6s for the first job, 1.0s for the second). The worker speaks
line-delimited JSON over stdin/stdout, emitting one event per finished input so
per-part geometry still streams out. `NS2S_WORKER=0` falls back to one
interpreter per job, which is the easier thing to debug when the method itself
misbehaves. VNS needs none of this: it spends minutes per sketch, so its
startup is noise.

A resident worker holds its whole model in VRAM for its lifetime (the ns2s one
sits on ~13GB), so on a single card it starves every other method — VNS died in
`loss.backward()` with a HIP OOM once its training schedule's demand peaked,
with the idle ns2s worker owning the card. So **the running method owns the
GPU**: `server._run_job` calls `common.release_other_workers` before any
adapter with `uses_gpu` starts, terminating every *other* method's resident
worker and waiting for it to actually exit (VRAM comes back on reap, not on
SIGTERM). Successive runs of one method — the benchmark case the residency
exists for — never evict anything; a mixed sweep pays one reload per method
switch. `sf3d` is `uses_gpu = False` because its own fitting is pure CPU; the
GPU half of an sf3d job is the proxy method it invokes directly, so
`_build_proxy` makes that eviction call on the proxy's behalf instead. Claiming
the card for sf3d itself would evict the very worker the proxy then reloads. A new resident worker registers itself with
`register_resident_worker(METHOD_NAME, WORKER)` at module scope and implements
`stop() -> bool`; a CPU-only adapter sets `uses_gpu = False` (only `bbox`) so
running it never costs a warm worker.

`img_size` (112) and `feature_size` (24) are fixed by the trained checkpoint
and are not exposed. The user-facing knobs are the marching-cubes
`threshold`, the voxel-grid `margin`, and an optional `smooth` pass — the
paper's fidelity-vs-smoothness post-process, which the adapter runs through a
`--headless` flag added to `smooth.py` in our fork (upstream it is a Polyscope
GUI only). Note that `inference.py` reports a failed reconstruction on stdout
and still exits 0, so the adapter treats a missing `_recon.obj` as the failure
signal.

### NeuVAS (`neuvas`)

```bash
python3 -m venv .venv-neuvas
.venv-neuvas/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
.venv-neuvas/bin/pip install -r requirements-neuvas.txt
```

No checkpoint: NeuVAS optimizes a neural SDF per sketch, so there is nothing to
download and nothing to go stale. The CUDA extension vendored under
`code/reconstruction/frnn` is **not** built — it is only imported by
`utils/point_processing.py`, which the reconstruction path never touches.

Closest in spirit to VNS — a per-shape neural SDF — but with a different
smoothness term: a thin-plate energy (k1² + k2², from the Hessian of the field)
minimized over the zero level set, reweighted by distance to the *feature*
curves so sharp G0 creases survive along the strokes. The paper's own example
passes the input cloud as its own feature set, which is what the
`sharp_features` toggle does.

**It is slow**: ~0.27 s/epoch measured on the sample data, so the paper's 10000
epochs is roughly 45 minutes for one sketch, plus a marching-cubes snapshot
every 1000 epochs (~35 s each at 512³, ~8 s at 256³). A few thousand epochs
already gives a recognisable surface. Snapshots are published as they land, so
a run can be watched converging in the benchmark grid.

Effective parameters are `epochs`, `smoothness`, `sharp_features`, `fidelity`,
`eikonal`, `samples`, `resolution` and `snapshot_every`. Worth knowing that
upstream's `decay_params` is a 5-tuple of which **only the first three are
read** — the trainer never touches `decay_params[3]`, `decay_params[4]` or
`--use_decay_devlop_lambda`, despite the driver script passing all of them. The
thin-plate weight is also modulated by `cos(2π · epoch/1000)`, so stopping
mid-period lands on a different effective smoothness than stopping on a
multiple of 1000.

Unlike every other adapter, NeuVAS does **not** normalize its input: the
network's `scale`, the sampler's `global_sigma` and the fixed [-1.2, 1.2]
marching-cubes grid all assume a shape about the size of its sample data. The
adapter scales the sketch to a half-extent of 0.5 to match, and maps the mesh
back to world coordinates itself. It also takes a point cloud rather than a
curve network, so the adapter resamples the strokes at even arc length —
otherwise densely-sampled stretches of a stroke would dominate the fit.

Fork changes: `--exps_dir` (write output to a job directory instead of
`<repo>/exps`, keeping the checkout clean), `--plot_frequency` and
`--plot_resolution` (snapshot cadence and resolution, previously hard-coded at
1000 epochs and forced to 512³ on epoch 10000). The upstream entry point
`run_abc_recon.py` is not used — it hard-codes a single ABC filename that isn't
in the repo, so it doesn't run on the shipped sample as-is; the adapter drives
`k1_k2_square_sum.py` directly.

### VRSketch2Shape (`vrs2s`)

```bash
python3 -m venv .venv-vrs2s
.venv-vrs2s/bin/pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/rocm6.4
.venv-vrs2s/bin/pip install -r requirements-vrs2s.txt
```

and one checkpoint, `df_epoch_best_multicls.pth` from
<https://huggingface.co/YiziChen/sketch2model>, into
`methods/VRSketch2Shape/weights/all_class/`. Override with `VRS2S_PYTHON` /
`VRS2S_CHECKPOINT`. The upstream README also points at
`saved_ckpt/vqvae-snet-all.pth`; that one is **not** needed — the full
checkpoint carries its own VQVAE weights, and our fork builds the VQVAE
unloaded when no standalone file is given.

**This method does something different from the other two.** An order-aware
BERT encoder embeds the stroke sequence, a latent diffusion model samples a
shape conditioned on it, and a VQVAE decodes a 64³ truncated SDF that marching
cubes turns into a mesh. Nothing constrains the output to pass through the
strokes: it generates a plausible complete object from a prior over the four
ShapeNet categories it was trained on (airplane, cabinet, chair, table). On a
sketch outside those it returns the nearest thing it knows. It is also
stochastic, so `seed` is a real parameter — a run is one sample, not the
answer. Read its benchmark numbers as "what does a generative prior do here",
not as a surfacing baseline.

The knobs are `ddim_steps` (runtime is linear in it), `seed`, `guidance`,
`simplify`, `iso`, `up_axis` and `fit_to_sketch`; the encoder shape (1200
tokens, 63-wide NeRF encoding at nerf_L 10, 256 hidden, 6 layers) is fixed by
the checkpoint. Guidance defaults to 1.0 (off) because the model was trained
without condition dropout — its unconditional branch is an all-zero sketch it
never saw, so classifier-free guidance is off-distribution here and the paper's
own evaluation does not use it.

Fork changes, all additive: `headless.py` is a new single-sketch entry point
(the released `infer.py` is a dataset evaluation script that needs ground-truth
SDFs and renders through pytorch3d); `models/sketch2shape_model.py` takes
`is_train` / `headless` / `vq_ckpt` / `device` and resolves its configs
relative to the repo instead of to `scripts/`; `utils/util_3d.py` makes the
pytorch3d and open3d imports optional; and the DDIM sampler no longer forces
its buffers onto cuda. Inference runs in a resident worker
(`adapters/vrs2s_worker.py`) for the same reason ns2s does — `VRS2S_WORKER=0`
falls back to one interpreter per generation.

One trap worth knowing if you touch the preprocessing: `simplifyline` 0.0.7's
`MatrixDouble` reads uninitialized memory over the first few doubles when it is
constructed from a python list instead of a numpy array, which silently
corrupts the start of every stroke. `headless.py` always hands it contiguous
float64 arrays. Its `simplify_line_3d` also decimates far harder than its
tolerance suggests (a 60-point half circle comes back as 3 points at any
tolerance) — that is what the model was trained on, so we reproduce it rather
than fix it.

### Piecewise-Smooth Surface Fitting (`sf3d`)

[Yu et al., SIGGRAPH 2022](https://em-yu.github.io/research/surfacing_3d_sketches/),
MIT-licensed, originally at <https://gitlab.inria.fr/D3/surface-fitting-3d-sketches>.

```bash
python3 -m venv .venv-sf3d
.venv-sf3d/bin/pip install -r requirements-sf3d.txt

cd methods/surface-fitting-3d-sketches
# the fork's .gitmodules uses an SSH URL for pygco; this avoids needing a key
git config submodule.external/pygco.url https://github.com/em-yu/pygco.git
git submodule update --init external/pygco
cd external/pygco && make download && make all
```

CPU only — no torch, no checkpoint. The `pygco` build is the one step pip
cannot do: the segmentation is an alpha-expansion graph cut over the gco-v3.0
library, whose sources the authors' fork downloads from UWaterloo rather than
vendoring. `make` warns loudly about `register` storage specifiers — that is
2010 C++ under C++17, not a problem. Override the checkout and env with
`SF3D_REPO` / `SF3D_PYTHON`. Only `external/pygco` is needed; `VIPSS` and
`instant-meshes` are for preprocessing this adapter replaces, and the
`3d-sketches-curated-dataset` submodule (35 MB, gitlab) is only worth
initializing to test against the paper's own inputs.

The method is 2021 code and three things have moved under it since. All three
are handled in `adapters/sf3d_worker.py` rather than by patching the submodule
or pinning the env backwards, so they survive a re-clone:

* **numpy 2 removed `np.float` / `np.int`**, which the method and pygco use in
  12 places. `numpy_compat()` restores them before the method imports; they
  were aliases for the builtins, so nothing changes but the lookup.
* **scipy rejects a 2-D `x0`** since 1.11. The mesh optimization hands L-BFGS
  its `(N, 3)` vertex array and relied on scipy ravelling it, while its own
  energy and gradient already take the flat vector — so the worker's
  `minimize` wrapper flattens it, which is what the code means.
* **pymeshlab renamed `AbsoluteValue` to `PureValue`** in 2023.12; the worker
  resolves whichever exists.

**This method does not surface a sketch from nothing.** It starts from a
*proxy* — a manifold mesh of about the right shape and the exact right
topology — and reshapes it: segment the proxy into regions by graph cut, fit a
low-degree implicit polynomial to the strokes over each region, repeat until
the labelling settles, then pull the proxy onto the fitted patches with L-BFGS
while snapping patch boundaries onto strokes. That last part is the point:
creases come out sharp where a stroke says there is one, instead of rounded
off like every neural SDF here does.

The paper builds proxies with VIPSS or by hand in Blender over a few minutes.
Neither works unattended, so the adapter runs **another adapter first** and
remeshes its output — `proxy_method` picks which, default `ns2s` (seconds).
So `sf3d` is best read as a sharpening pass over another method rather than a
competitor to it, and a benchmark row for it is really a row for the pair.

Runtime, per the paper's Table 1 over 17 sketches: 9–146 s to associate
strokes with the proxy, 0.8–79 s for the segmentation (under 10 s on all but
two), 15–68 s for the mesh optimization. Call it 1–4 minutes on top of the
proxy method. Nothing is GPU-accelerated, so the ROCm defines in
`adapters/common.py` are irrelevant here.

Measured on this box, `onshape_bishop` from the authors' dataset (7632 stroke
points) with an `ns2s` proxy at `proxy_resolution` 0.012: 59 s all in, of which
36 s was the projection, converging in 6 iterations onto 9 patches. The paper's
own row for that sketch is 9.2 s / 1.2 s / 14.8 s over 4 iterations — same
ballpark, and the gap is mostly proxy resolution, which drives everything
downstream. At the default 0.007 the proxy runs to tens of thousands of
vertices and the projection to minutes, so it is the first knob to turn when a
run feels slow.

Three limits are structural, not tuning problems:

* **The proxy fixes the topology.** This method moves a surface; it never
  changes its genus, closes a gap or splits a component. A proxy with the
  wrong number of holes gives a result with the wrong number of holes.
* **Non-manifold output is impossible** — the whole pipeline is a PyGEL
  halfedge mesh. The adapter checks manifoldness after remeshing and fails
  with that explanation rather than deep inside the method.
* **Surfaces always come out closed.** The paper's `open_boundary` trimming
  needs per-point border markings hand-picked in Blender, which has no
  sensible UI here, so the adapter leaves it off.

`part_based` gives each part its own proxy and its own segmentation, which
also keeps one part's strokes from pulling patches across a joint — at the
cost of a full proxy run per part.

The proxy method's own parameters are copied wholesale into this adapter's
list with a `<method>_` prefix (`ns2s_threshold`, `vns_grid_res`, …), each
gated on `proxy_method` so only the selected method's rows are editable. The
copy is generated by `proxy_params()`, so adding a knob to `ns2s` gives `sf3d`
that knob for free. It drops `part_based` and anything gated on it being true:
the proxy adapter is handed one unit's strokes and must surface them whole,
since this adapter has already done any splitting. Doing this the dumb
mechanical way beats curating a shortlist, because `ns2s_threshold` — the
marching-cubes isovalue — sets the proxy's *topology*, and topology is the one
thing this method cannot repair later. A result that looks like a failed fit
is often a proxy with the wrong genus.

Two of the paper's parameters are stated for sketches normalized to a unit
bounding-box diagonal (`proxy_resolution` 0.007, `sketch_error_dist` 0.01).
The adapter exposes them as fractions and multiplies by the real diagonal of
whatever it is given, so they mean the same thing at any scale.

No fork changes are needed — but the method's own drivers are unusable as
they stand: `run_segment_and_fit.py` and `run_projection.py` walk a
`metadata.csv` of dataset sketch names, resolve every path relative to the
repo root, and call `polyscope.init()` unconditionally. So
`adapters/sf3d_worker.py` skips them and calls the library underneath
(`main.build_proxy.build`, `main.segment_and_fit.SegmentFit`,
`run.run_projection.run_projection`), all of which take arrays and objects.
The worker gets partial output by wrapping two symbols for the duration of the
projection: `run_projection.optimize_mesh`, to capture the post-insertion face
list, and `mesh_optimization.minimize`, to hook L-BFGS's per-iteration
callback. The proxy is published as soon as it is remeshed, then a snapshot
every `snapshot_every` steps, so the surface is watchable pulling onto the
strokes from the first second rather than after the last.

### TRELLIS (`trellis`)

[Xiang et al., 2024](https://trellis3d.github.io/), MIT-licensed, at
<https://github.com/microsoft/TRELLIS>. The only *generative* method here: it
does not fit the strokes, it samples an object conditioned on images of them.

**Two checkouts, one adapter.** Upstream TRELLIS is CUDA-only (custom kernels,
xformers, flash-attn), so an AMD machine runs the
[TRELLIS-AMD](https://github.com/CalebisGross/TRELLIS-AMD) fork instead — a
different repo, a different venv, HIP defines. Which one this machine uses
comes from `backends.json` under the active `SURFACING_GPU_BACKEND`, next to
every other vendor-dependent answer:

| backend | submodule | notes |
| --- | --- | --- |
| `cuda` | `methods/TRELLIS` | upstream; spconv + flash-attn |
| `rocm` | `methods/TRELLIS-AMD` | fork; torchsparse + sdpa |

Both are submodules like every other method, so a machine only initializes the
one its backend needs:

```bash
git submodule update --init surfacing-server/methods/TRELLIS-AMD   # or .../TRELLIS
```

Override either with `TRELLIS_REPO` / `TRELLIS_PYTHON`. The venv lives *inside*
the checkout (`<repo>/.venv`), not beside the server's, because the two forks
need incompatible torch builds — and because that keeps a 16 GB environment
inside the submodule that owns it. The parameters, the worker protocol and the
output are identical across the two, so results are comparable.

There is nothing to `pip install -r` here — follow each repo's own setup. On
ROCm only `torchsparse` has to build from source, and it needs an explicit
prefix because the installer hardcodes one that may not exist:

```bash
# derive the prefix rather than hardcoding it: ROCm installs to a VERSIONED
# /opt/rocm-X.Y.Z with no unversioned symlink, so anything that assumes
# /opt/rocm fails deep in a build as "hipcc: not found", and a hardcoded
# version silently rots the next time ROCm updates
ROCM=$(dirname "$(dirname "$(readlink -f "$(which hipcc)")")")
export ROCM_HOME=$ROCM ROCM_PATH=$ROCM CUDA_HOME=$ROCM
export PYTORCH_ROCM_ARCH="gfx1100;gfx1101" FORCE_CUDA=1

.venv/bin/python -m pip install --no-build-isolation \
  git+https://github.com/mit-han-lab/torchsparse.git
```

Use `python -m pip`, not `.venv/bin/pip`: console scripts carry an absolute
shebang, so they break if the submodule is ever moved while `bin/python`
keeps working.

Weights are pulled on first use — ~2.9 GB into `~/.cache/huggingface`, plus
~1.2 GB of DINOv2 into `~/.cache/torch`. Geometry only (`formats=['mesh']`),
which is also what keeps AMD viable: neither nvdiffrast nor
diff-gaussian-rasterization is imported on that path.

**The sketch is not the input.** Every other adapter consumes strokes as
geometry; TRELLIS consumes images, and DINOv2 patch tokens are the only channel
through which anything about the sketch reaches the model. So a *conditioner*
turns strokes into images first, and which one to use is the open experimental
question rather than an implementation detail — hence the `CONDITIONERS`
registry in `adapters/trellis.py`. One exists so far:

* **`views`** — multi-view renders of the strokes, rasterized by the editor
  (`src/engine/strokeViews.ts`) and sent with the job as PNG data URLs in
  `options.views`, either a flat list for the whole sketch or `{part: [...]}`
  for a part-based run. Client-side because the app already owns a three.js
  view of the document; a second renderer in python would be a second thing to
  keep honest about stroke width, framing and pose.

Those renders are not free-form. `preprocess_image` premultiplies alpha onto
black (so strokes must be **light** — the benchmark thumbnails' `0x333333`
disappears), crops to an `alpha > 0.8 * 255` bbox (so they must be **opaque**),
and resamples to 518 px against a 37×37 DINOv2 patch grid (so hairlines are
close to invisible — strokes are drawn as **tubes**, not lines).

Those numbers are facts about TRELLIS, not about drawing, so they live in the
conditioner's `view_spec` and travel to the client with the method
declaration (`SurfacingAdapter.view_spec`, served as `viewSpec` from
`/api/health`):

```python
view_spec = {"size": 518, "count": 4, "pitch": 0.35,
             "strokeColor": "#dcdcdc", "strokeThickness": 0.012,
             "margin": 1.15}
```

The client renders whatever the selected strategy asks for and never knows
which method it is serving — so a new conditioner changes the renders by
changing one dict, and a method that wants no images declares nothing and gets
none. `strokeThickness` is a fraction of the sketch's bounding radius rather
than pixels, so line weight is independent of the units a sketch was drawn in.

### Mesh cleanup, and why it differs by backend

Raw FlexiCubes output is not watertight and arrives in many connected
components — measured on one run, 265 fragments holding 3.7% of the area, all
within 2.5% of the body, which read as shimmer along the silhouette.
`postprocess_mesh` has two stages and only one is portable:

| stage | what it does | AMD |
| --- | --- | --- |
| `simplify` | quadric edge collapse (pyvista, CPU) | works |
| `fill_holes` | rasterize 100 views, drop rarely-visible components, mincut interior shells | **broken** |

`fill_holes` is the stage that clears the fragments, and on AMD the fork's
simplified coarse rasterizer returns empty visibility, so it marks every face
invisible and deletes the mesh (`AMD_GPU_GUIDE.md` §3.3, "Disable fill_holes
(Critical!)"). That is not a configuration problem — nvdiffrast-hip builds and
`backend='gl'` is already applied in the checkout; the stage itself is wrong
there. So the `fill_holes` param defaults to `auto`, following
`backends.json`, and the `min_component` param clears fragments by area
instead where it cannot run. Area is a cruder rule than visibility: it cannot
tell a small part from a stray shell, only small from large.

Set `fill_holes` to `off` explicitly when comparing the two backends, or they
are not cleaned alike and the meshes are not comparable.

One more caveat before reading a bad result as a bad method:
`run_multi_image` takes **no camera poses** — it
reconciles views purely from their tokens — so more views is not monotonically
better, and three or four well-spread ones beat a dozen.

Measured on this box (RX 7800 XT, 8 + 8 steps, 3 views): ~35 s cold, of which
almost all is the pipeline load; both flow stages together run in about 8 s and
decode ~124 k vertices. The result lands in a normalized cube, so `fit_to_sketch`
scales it uniformly into the strokes' bounding box — uniformly, because fitting
each axis independently would shear the shape the model inferred.
