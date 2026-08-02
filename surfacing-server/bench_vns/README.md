# VNS performance benchmark

Measures whether the two rounds of performance work on the VNS fork actually
speed up iteration, on the excavator sketch from `SampleModels/`.

```bash
cd surfacing-server/bench_vns
python3 bench_vns.py setup          # build the three bench forks (once)
python3 bench_vns.py run --repeats 2
```

`setup` and `run` use the system python; the trainer itself always runs in
`../.venv-vns` (override with `--python`). Results land in
`results/<timestamp>/` — `results.json` (every timing block), `report.md`, the
raw trainer logs and the generated `sketch.obj`. Both `forks/` and `results/`
are gitignored.

## Hardware and the ROCm defines

The machine is an **AMD Radeon RX 7800 XT (gfx1101) on ROCm** — there is no
CUDA. `torch==2.5.1+rocm6.2` comes from the ROCm wheel index (see
`../requirements-vns.txt`); HIP then masquerades as `torch.cuda`, so the VNS
code needs no changes, but two defines must be in the environment **before any
torch code runs**:

```
HSA_OVERRIDE_GFX_VERSION=11.0.0
HIP_VISIBLE_DEVICES=0
```

They live in `ROCM_ENV` at the top of `bench_vns.py` and are injected into
every trainer subprocess (the same pair `adapters/vns.py` sets for real jobs),
so there is nothing to export by hand; anything already exported in the calling
shell wins. They are also recorded into `results.json` with each run. The
`hipBLASLt ... unsupported architecture` warning at startup is benign.

## What each step is

Each step is its own **bench fork**: a git worktree of `../methods/vns` on a
`bench/*` branch, built by `setup` and left on disk for inspection.

| fork | base | what it adds |
|---|---|---|
| `forks/step1-upstream` | `3e9cd19` | the paper's code as published |
| `forks/step2-fork-main` | `ec4f30c` | DPSR module built once instead of per iteration, `ras_p` kept on the GPU, `--slim_output` |
| `forks/step3-pytorch` | `ec4f30c` + `losses_torch` working tree | torch marching cubes + surface sampler (`utils/mc_torch.py`), so the grid never leaves the device; per-iteration open3d distance mask and cKDTree k-NN dropped; loss schedule parameterised |

The thing under test: the stock trainer copies the whole `res^3` Poisson grid
to the CPU **every iteration** so numpy/skimage/open3d can extract and sample
the zero-isosurface. At `grid_res 256` that is a ~64 MB synchronous
device→host transfer per iteration — it not only costs bus time, it drains the
GPU pipeline, since the whole graph must finish before the copy returns. Step 3
removes it by doing marching cubes and the area-weighted sampling in torch;
only the 15 000 sampled points come back. Step 2 attacks the smaller siblings
of the same problem (a spectral filter rebuilt on the CPU and re-uploaded every
step, and `ras_p` bounced to numpy mid-solve).

Step 3 is a *snapshot* of the uncommitted `losses_torch` working tree taken
when `setup` ran. Re-run `python3 bench_vns.py setup --force` after changing
that tree, or the benchmark keeps measuring the old code.

## Protocol

100 iterations: **50 of initialization** (no isosurface extracted, `L_smooth`
off) then **50 with `L_smooth` on** — `--iters 100 --init-iters 50`, and
`--grid-res` defaults to the adapter's 256.

Steps 1 and 2 hardcode those stage boundaries (200 in `models/losses.py`, 100
in `recon_dataset.py`), so `setup` backports exactly two things into them, and
nothing else — see `backport.py`, where every replacement is asserted to match
once:

- `--iso_after` / `--smooth_after`, from the `losses_torch` branch, so the
  stages can be moved;
- `--slim_output`, from the fork's *Slim output* commit, so no fork writes
  checkpoints, `grid_values`, `ras_p` or `samplings` during a timed run
  (step 1 only — steps 2 and 3 already have it).

Everything else in steps 1 and 2 is deliberately left slow: they keep the k-NN
search, the open3d distance mask, the numpy grid round-trip and the
skimage/open3d isosurface sampler, because removing those *is* step 3's
contribution. Two step-3-only flags are simply not passed to the older forks,
which is behaviour-neutral here: `smooth_refine_after` (600) is past the end of
a 100-iteration run either way, and `cc_weight` only *disables* the
curve-consistency work in step 3, which steps 1 and 2 always do.

Everything else comes from `adapters/vns.py`'s `DEFAULTS` — the hyperparameters
the app really surfaces with — so the benchmark measures the configuration that
matters, and the command is built the way the adapter builds it.

## How it is timed

No instrumentation is added to the forks; all three run exactly the code they
ship. The driver timestamps the trainer's own progress lines (printed every 10
iterations) as they arrive, with the child launched under `python -u` so the
pipe is not block-buffered. Each of those lines calls `.item()` on the loss, so
the GPU is synchronised at every mark and the deltas are real elapsed work.

Reported per step (median over repeats):

- **total** — subprocess wall time;
- **startup** — launch through the first completed iteration (imports, curve
  network load, HIP init);
- **init ms/iter** and **L_smooth ms/iter** — mean over the 10-iteration blocks
  falling wholly inside each stage. The final block is excluded from the rates
  because that iteration also writes the output mesh; it is still in *total*.

The **first timed run of a session pays for HIP kernel compilation**, so use
`--repeats 2` or more and read the median. Repeats run round-robin across the
steps, so a warming GPU hits all of them evenly.

`final loss` is printed as a sanity check, not a quality metric: steps 1 and 2
should track each other closely, while step 3 draws its isosurface samples from
a different (equivalent-distribution) sampler, so it diverges slightly.

## Other useful invocations

```bash
python3 bench_vns.py run --steps 1,3            # skip a step
python3 bench_vns.py run --grid-res 128         # transfer cost scales with res^3
python3 bench_vns.py run --sketch ../../SampleModels/.../sketch1000.json
python3 bench_vns.py report results/<run>/results.json
python3 bench_vns.py clean                      # remove forks + bench/* branches
```
