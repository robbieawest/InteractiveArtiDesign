# Running surfacing benchmarks on the ICF cluster

The browser drives a benchmark one cell at a time against the local
`surfacing-server`. This runs the same cells as independent Slurm array tasks,
with no server and no browser involved.

Nothing here is specific to a particular benchmark. What runs comes entirely
from `benchmarks/<id>/progress.json` and `benchmarks/<id>/sketches/`:

* adapters are `progress.json`'s own keys,
* runs and their options are its run lists,
* whether a run splits into per-part tasks is that **run's** `part_based`
  flag — flip NeuVAS to part-based in the browser and it splits, with no
  change here,
* parts are whatever carries strokes in each sketch (a sketch can declare 19
  parts and populate 14; the empty ones are not submitted).

`profiles.json` contributes resource requests only — time, `--gres`, throttle,
and a preference for splitting — and has a `default` entry, so an adapter it
has never heard of still schedules.

## What a cell is

One Slurm task, one surface, one file:

```
benchmarks/<id>/<adapter>/<run>/<sketch>.glb          whole-sketch cell
benchmarks/<id>/<adapter>/<run>/<sketch>/part_07.glb  one part of a split cell
```

Split cells are merged into the same `<sketch>.glb` the client already reads,
so nothing on the browser side changes. The per-part files are kept.

## First time

```bash
# locally
cluster/pack.sh <benchmark-id>
scp -o ProxyJump=$UUN@student.ssh.inf.ed.ac.uk icf-bundle.tar.gz $UUN@icf:~/

# on the HEAD node — setup needs outbound network (pip, micromamba, the gco
# sources, the checkpoint) and compute nodes are commonly airgapped. This is
# not the sort of work the head node's banner is warning about: it is network
# and disk, plus a few seconds of compiling one Cython extension.
mkdir -p ~/InteractiveArtiDesign && tar xzf ~/icf-bundle.tar.gz -C ~/InteractiveArtiDesign
cd ~/InteractiveArtiDesign
$EDITOR cluster/env.sh                       # UUN, REPO, SCRATCH_ROOT
cluster/setup_native.sh                      # SuiteSparse, pygco, checkpoints
TORCH_INDEX=cu128 TORCH_INDEX_VNS=cu121 cluster/setup_venvs.sh
```

Then verify on a compute node, the only place with a GPU:

```bash
srun -p Teaching --gres=gpu:h200_1g.18gb:1 --pty \
  surfacing-server/.venv-ns2s/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`setup_venvs.sh` ends with a preflight that proves the mesh-boolean backend
works. Do not skip it: `combine_meshes` catches a failed union, logs, and
concatenates instead — a valid `.glb` and a zero exit code — so a missing
`manifold3d` degrades every part-based result silently.

## Each sweep

```bash
cd ~/InteractiveArtiDesign
surfacing-server/.venv-server/bin/python cluster/plan_sweep.py <benchmark-id>
```

It writes one manifest per adapter and prints the `sbatch` lines. Run them,
note each job id, then run the `finalize.sh` line it prints with those ids
substituted. `finalize.sh` merges per-part surfaces and rebuilds
`progress.json` so the benchmark reopens correctly in the browser.

## Resuming

`run_cell.py` skips any cell whose `.glb` already exists, so **resubmitting the
same arrays picks up exactly what is missing**. A preempted, timed-out or
OOM-killed task costs one part, not a sketch. Re-running `plan_sweep.py` is
safe and regenerates the manifests whole.

## Where things live

| | |
|---|---|
| `/home/<UUN>` (NFS) | repo, venvs, SuiteSparse prefix, results |
| `/disk/scratch/<UUN>/<job>_<task>` | method scratch, per task, deleted on success |
| `/afs/...` | archive, copied by hand from the head node |

Scratch is per task deliberately: two array tasks do land on one node, and a
shared root would have them pruning each other's working files
(`SURFACING_KEEP_JOBS=-1` disables that pruning entirely). A failed task keeps
its scratch for inspection and says where.

AFS is written from the head node, never from a job — a batch job's Kerberos
token may not reach it, and finding out costs you the run. `archive.sh` makes
the tarball; you copy it.

## GPU choice

The Teaching partition is mostly 2080 Ti (**11 GB**), which is under both the
ns2s resident worker (~13 GB) and VNS's measured 12.4 GB. So the profiles ask
for the H200 MIG slices (`h200_1g.18gb`, 18 GB, 35 of them) and the A6000s
(48 GB, 8 of them) instead. The slices are 1/7 of the H200's compute, so they
suit short or IO-bound work; the A6000s are full cards and better for long
compute-bound loops like VNS and NeuVAS. One field per adapter in
`profiles.json`.

## Files

| | |
|---|---|
| `env.sh` | paths and interpreters, sourced by every job |
| `profiles.json` | per-adapter scheduling policy, with a `default` |
| `cells.py` | what a cell is; shared by the planner and the runner |
| `plan_sweep.py` | benchmark → manifests + the `sbatch` lines. Submits nothing |
| `job.sh` | array task body: pick the row, make scratch, run the cell |
| `run_cell.py` | one cell: read sketch → `adapter.run` → save. Skips if done |
| `finalize.py` / `.sh` | merge per-part surfaces, rebuild `progress.json` |
| `setup_native.sh` | SuiteSparse, pygco, checkpoints |
| `setup_venvs.sh` | the five venvs + the boolean preflight |
| `pack.sh` / `archive.sh` | bundle up, bundle back |
