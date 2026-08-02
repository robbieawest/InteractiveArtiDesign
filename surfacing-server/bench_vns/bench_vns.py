#!/usr/bin/env python3
"""Benchmark the three VNS performance steps on the excavator sketch.

WHAT IS BEING COMPARED
----------------------
The VNS trainer keeps a res^3 scalar grid (the Poisson solve's output) and,
in the stock code, pushes it across the PCIe bus to the CPU *every iteration*
so that numpy/skimage/open3d can extract and sample the zero-isosurface. At
grid_res 256 that is a 64 MB device->host copy per iteration, and because the
copy is synchronous it also drains the GPU pipeline: the whole graph has to
finish before the readback returns, so nothing overlaps. Removing those
transfers is what the fork's later work is about, and this script measures
whether it actually buys iteration time.

The three steps, each checked out as its own bench fork (a git worktree of
methods/vns, see `setup`):

  step1-upstream   3e9cd19, the paper's code as published.
  step2-fork-main  ec4f30c "Slim output": the DPSR module is built once
                   instead of per iteration (it rebuilt an res^3 spectral
                   filter on the CPU and re-uploaded it every step), ras_p
                   stays on the GPU, and --slim_output stops the periodic
                   checkpoint / grid_values / ras_p / samplings dumps.
  step3-pytorch    the losses_torch branch: marching cubes and the
                   area-weighted surface sampler are reimplemented in torch
                   (utils/mc_torch.py), so the grid never leaves the device;
                   the per-iteration open3d distance mask and the cKDTree
                   k-NN search that fed the unused curve-consistency term are
                   gone, and the loss schedule is parameterised.

PROTOCOL
--------
100 iterations: 50 of initialization (no isosurface extracted, L_smooth off)
then 50 with the isosurface and L_smooth on. Steps 1 and 2 hardcode those
stage boundaries at 200/100, so `setup` backports the two flags that make the
protocol runnable there -- --iso_after/--smooth_after and (step 1 only)
--slim_output -- and nothing else; see backport.py.

Everything else about steps 1 and 2 is left slow on purpose: they keep the
k-NN search, the open3d distance mask, the numpy grid round-trip and the
skimage/open3d isosurface sampler, because removing those *is* step 3.

HARDWARE / ENVIRONMENT
----------------------
Developed on an AMD Radeon RX 7800 XT (gfx1101) under ROCm -- there is no
CUDA here. torch comes from the ROCm wheel index (torch==2.5.1+rocm6.2, see
../requirements-vns.txt) and HIP masquerades as `torch.cuda`, but two env
defines must be set before any torch code runs, or the runtime either refuses
the card or picks the wrong one:

    HSA_OVERRIDE_GFX_VERSION=11.0.0
    HIP_VISIBLE_DEVICES=0

They live in ROCM_ENV below and are injected into every trainer subprocess
(the same pair the VNS adapter sets in adapters/vns.py). Anything already
exported in the calling shell wins. The "hipBLASLt ... unsupported
architecture" warning at startup is benign.

USAGE
-----
    python3 bench_vns.py setup                 # create/refresh the three forks
    python3 bench_vns.py run                   # 100 iters, 50/50, grid_res 256
    python3 bench_vns.py run --repeats 3 --grid-res 128 --steps 1,3
    python3 bench_vns.py report results/<run>/results.json
    python3 bench_vns.py clean                 # drop the forks and branches

The first timed run of a session pays for HIP kernel compilation; use
--repeats 2 or more and read the median.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
SERVER_DIR = BENCH_DIR.parent
VNS_DIR = SERVER_DIR / "methods" / "vns"
FORKS_DIR = BENCH_DIR / "forks"
RESULTS_DIR = BENCH_DIR / "results"
REPO_ROOT = SERVER_DIR.parent

DEFAULT_SKETCH = (
    REPO_ROOT
    / "SampleModels"
    / "p2-c_autonomous_excavator"
    / "Surfaces"
    / "sketchsurf.json"
)

# ROCm (AMD RX 7800 XT / gfx1101): required before torch initialises HIP.
ROCM_ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    "HIP_VISIBLE_DEVICES": "0",
}

# "Epoch: 0 [  40/100 (40%)] Loss: 1.234 = L_Data: ..." — printed every 10
# iterations by the trainer. The companion "Unweighted L_s" line does not
# match (no "Loss:" right after the bracket), so there is exactly one mark
# per reporting point.
PROGRESS_RE = re.compile(r"\[\s*(\d+)/(\d+)\s+\(\s*\d+%\)\]\s+Loss: ([\d.eE+-]+)")


def _load_adapter_vns() -> Any:
    """Import adapters.vns without executing adapters/__init__.py.

    The package __init__ instantiates every adapter, which drags in the
    server env's trimesh; this script only wants the sketch->obj writer and
    the hyperparameters the adapter actually runs with, so bind a bare
    package module and let the submodule import resolve through its
    __path__."""
    if "adapters" not in sys.modules:
        pkg = types.ModuleType("adapters")
        pkg.__path__ = [str(SERVER_DIR / "adapters")]
        sys.modules["adapters"] = pkg
    sys.path.insert(0, str(SERVER_DIR))
    from adapters.common import write_curve_obj  # noqa: E402
    from adapters.vns import DEFAULTS  # noqa: E402

    return DEFAULTS, write_curve_obj


ADAPTER_DEFAULTS, write_curve_obj = _load_adapter_vns()


def load_sketch(path: Path) -> dict[str, Any]:
    """A saved .json document -> the payload the server actually receives.

    Port of buildSurfacingSketch() in src/surfacing/client.ts: the document
    stores each stroke's centerline as a flat local-space float array plus a
    transform, while an adapter is handed world-space [x, y, z] triples tagged
    with part ids. Keep the two in step -- feeding the adapter raw document
    strokes silently benchmarks the wrong geometry."""
    doc = json.loads(path.read_text())
    strokes = []
    for stroke in doc.get("strokes", []):
        t = stroke.get("transform", {})
        pos = t.get("position", {"x": 0, "y": 0, "z": 0})
        quat = t.get("quaternion", {"x": 0, "y": 0, "z": 0, "w": 1})
        scale = t.get("scale", {"x": 1, "y": 1, "z": 1})
        flat = stroke.get("points", [])
        points = []
        for i in range(0, len(flat) - 2, 3):
            x, y, z = flat[i] * scale["x"], flat[i + 1] * scale["y"], flat[i + 2] * scale["z"]
            rx, ry, rz = _rotate(quat, x, y, z)
            points.append([rx + pos["x"], ry + pos["y"], rz + pos["z"]])
        strokes.append({"id": stroke.get("id"), "partId": stroke.get("partId"),
                        "points": points})
    return {
        "strokes": strokes,
        "parts": [{"id": p["id"], "name": p["name"]} for p in doc.get("parts", [])],
        "joints": doc.get("joints", []),
    }


def _rotate(q: dict[str, float], x: float, y: float, z: float) -> tuple[float, float, float]:
    """v + 2 * q_vec x (q_vec x v + w * v) — same as core/rigid.ts rotateVec."""
    qx, qy, qz, qw = q["x"], q["y"], q["z"], q["w"]
    tx = 2 * (qy * z - qz * y)
    ty = 2 * (qz * x - qx * z)
    tz = 2 * (qx * y - qy * x)
    return (x + qw * tx + qy * tz - qz * ty,
            y + qw * ty + qz * tx - qx * tz,
            z + qw * tz + qx * ty - qy * tx)


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    base: str  # commit the fork starts from
    blurb: str
    # trainer flags this fork understands beyond the common set
    extra_flags: tuple[str, ...]
    backport: bool = False  # apply the schedule flags
    slim_backport: bool = False  # also apply --slim_output
    # files copied out of the live submodule working tree (step 3 only)
    snapshot: tuple[str, ...] = field(default_factory=tuple)


STEPS: dict[str, Step] = {
    "1": Step(
        key="step1-upstream",
        label="1. upstream",
        base="3e9cd19",
        blurb="paper code as published",
        extra_flags=("iso_after", "smooth_after"),
        backport=True,
        slim_backport=True,
    ),
    "2": Step(
        key="step2-fork-main",
        label="2. fork main",
        base="ec4f30c",
        blurb="DPSR built once, ras_p on GPU, slim output",
        extra_flags=("iso_after", "smooth_after"),
        backport=True,
    ),
    "3": Step(
        key="step3-pytorch",
        label="3. torch grid",
        base="ec4f30c",
        blurb="torch marching cubes + sampler, no grid readback, no k-NN",
        extra_flags=("iso_after", "smooth_after", "smooth_refine_after", "cc_weight"),
        snapshot=(
            "models/losses.py",
            "surface_reconstruction/recon_dataset.py",
            "surface_reconstruction/surface_recon_args.py",
            "surface_reconstruction/train_surface_reconstruction.py",
            "utils/gridEvaluator.py",
            "utils/mc_torch.py",
        ),
    ),
}


def git(*args: str, cwd: Path = VNS_DIR) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    )
    return out.stdout.strip()


# --------------------------------------------------------------------------
# setup / clean
# --------------------------------------------------------------------------


def setup(force: bool = False) -> None:
    import backport as backport_mod  # local module, same directory

    FORKS_DIR.mkdir(parents=True, exist_ok=True)
    for step in STEPS.values():
        fork = FORKS_DIR / step.key
        branch = f"bench/{step.key}"
        if fork.exists():
            if not force:
                head = git("log", "--oneline", "-1", cwd=fork)
                print(f"{step.key}: exists at {head}")
                continue
            _remove_fork(fork, branch)

        print(f"{step.key}: creating worktree at {step.base}")
        git("worktree", "add", "-b", branch, str(fork), step.base)

        if step.snapshot:
            # step 3 lives as uncommitted work on losses_torch; snapshot it so
            # the fork is a real commit and the user's tree is untouched
            for rel in step.snapshot:
                shutil.copy2(VNS_DIR / rel, fork / rel)
            message = "bench step 3: torch loss/grid path (losses_torch working tree, snapshot)"
        else:
            backport_mod.backport(fork, slim=step.slim_backport)
            message = "bench harness backport (slim_output + iso_after/smooth_after)"

        git("add", "-A", cwd=fork)
        git(
            "-c", "user.name=bench", "-c", "user.email=bench@local",
            "commit", "-q", "-m", message, cwd=fork,
        )
        print(f"  -> {git('log', '--oneline', '-1', cwd=fork)}")


def _remove_fork(fork: Path, branch: str) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(fork)], cwd=VNS_DIR)
    subprocess.run(["git", "branch", "-D", branch], cwd=VNS_DIR, capture_output=True)
    shutil.rmtree(fork, ignore_errors=True)


def clean() -> None:
    for step in STEPS.values():
        _remove_fork(FORKS_DIR / step.key, f"bench/{step.key}")
    git("worktree", "prune")
    print("forks removed (results/ kept)")


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def build_command(step: Step, python: Path, obj: Path, logdir: Path, cfg: dict[str, Any]) -> list[str]:
    """The trainer invocation, built the way the adapter builds it: the
    adapter's DEFAULTS (which are the paper's batch-driver hyperparameters,
    not argparse's) with the benchmark's overrides on top, minus any flag the
    fork does not have."""
    params = dict(ADAPTER_DEFAULTS)
    params.update(
        n_samples=cfg["iters"],
        grid_res=cfg["grid_res"],
        iso_after=cfg["init_iters"],
        smooth_after=cfg["init_iters"],
    )
    # schedule knobs the older forks do not expose. Dropping them is
    # behaviour-neutral for this protocol: smooth_refine_after (600) is past
    # the end of a 100-iteration run either way, and cc_weight only *disables*
    # the curve-consistency work in step 3 -- steps 1 and 2 always do it, which
    # is precisely one of the costs being measured.
    optional = {"iso_after", "smooth_after", "smooth_refine_after", "cc_weight"}
    params = {
        k: v for k, v in params.items()
        if k not in optional or k in step.extra_flags
    }

    # -u is load-bearing: python block-buffers stdout when it is a pipe, so
    # without it the progress lines arrive in one burst at exit and every
    # timestamp collapses into the last block
    cmd = [str(python), "-u", "train_surface_reconstruction.py",
           "--data_path", str(obj), "--logdir", str(logdir)]
    for key, value in params.items():
        values = value if isinstance(value, list) else [value]
        cmd += [f"--{key}", *[str(v) for v in values]]
    cmd += ["--morse_near", "--output_any", "--slim_output"]
    return cmd


def run_once(step: Step, python: Path, obj: Path, out_dir: Path, cfg: dict[str, Any],
             tag: str) -> dict[str, Any]:
    """One timed trainer run. Timing comes from the trainer's own progress
    lines (every 10 iterations), timestamped as they arrive -- no
    instrumentation is added to the forks, so all three run exactly the code
    they ship. Each of those lines calls .item() on the loss, so the GPU is
    synchronised at every mark and the deltas are real elapsed work."""
    fork = FORKS_DIR / step.key
    logdir = out_dir / f"{step.key}-{tag}"
    cmd = build_command(step, python, obj, logdir, cfg)

    # ROCm defines first so anything the caller already exported wins
    env = {**ROCM_ENV, **os.environ}
    log_path = out_dir / f"{step.key}-{tag}.log"
    marks: list[tuple[int, float]] = []
    losses: list[tuple[int, float]] = []
    tail: list[str] = []

    print(f"  running {step.key} [{tag}] ...", end="", flush=True)
    start = time.perf_counter()
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=fork / "surface_reconstruction",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            now = time.perf_counter()
            log_file.write(line)
            tail = (tail + [line])[-40:]
            match = PROGRESS_RE.search(line)
            if match:
                iteration = int(match.group(1))
                marks.append((iteration, now - start))
                losses.append((iteration, float(match.group(3))))
        code = proc.wait()
    total = time.perf_counter() - start

    if code != 0:
        print(" FAILED")
        raise SystemExit(
            f"{step.key} exited with code {code}; last output ({log_path}):\n"
            + "".join(tail)
        )

    mesh_dir = logdir / obj.stem / "mesh"
    meshes = sorted(mesh_dir.glob("mesh_*.ply")) if mesh_dir.exists() else []
    result = {
        "step": step.key,
        "tag": tag,
        "total_s": total,
        "marks": marks,
        "losses": losses,
        "final_loss": losses[-1][1] if losses else None,
        "meshes": [m.name for m in meshes],
        "log": str(log_path.relative_to(BENCH_DIR)),
        "cmd": cmd,
        **_phase_timings(marks, cfg),
    }
    print(f" {total:6.1f}s  init {_fmt(result['init_ms'])}  smooth {_fmt(result['smooth_ms'])}")
    return result


def _phase_timings(marks: list[tuple[int, float]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Split the marks into the initialization phase and the L_smooth phase.

    A mark at iteration i is stamped after iteration i finished, so the
    interval (i -> j) is the cost of iterations i+1..j. An interval counts as
    init when j <= init_iters (the gates are `iter <= iso_after`, so iteration
    init_iters itself is still initialization) and as smooth when i >=
    init_iters. The final partial interval is dropped from the rates because
    the last iteration also writes the output mesh; it is still in total_s."""
    init_iters = cfg["init_iters"]
    startup = marks[0][1] if marks else None
    blocks = []
    for (i0, t0), (i1, t1) in zip(marks, marks[1:]):
        blocks.append({"from": i0, "to": i1, "s": t1 - t0,
                       "ms_per_iter": 1000.0 * (t1 - t0) / max(i1 - i0, 1)})
    init = [b for b in blocks if b["to"] <= init_iters]
    smooth = [b for b in blocks if b["from"] >= init_iters]
    return {
        "startup_s": startup,
        "blocks": blocks,
        "init_ms": _mean([b["ms_per_iter"] for b in init]),
        "smooth_ms": _mean([b["ms_per_iter"] for b in smooth]),
        "init_iters_timed": sum(b["to"] - b["from"] for b in init),
        "smooth_iters_timed": sum(b["to"] - b["from"] for b in smooth),
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _fmt(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.1f}ms"


def run(args: argparse.Namespace) -> None:
    python = Path(args.python)
    if not python.exists():
        raise SystemExit(
            f"VNS python not found at {python} — set up ../requirements-vns.txt "
            "or pass --python"
        )
    steps = [STEPS[s] for s in args.steps.split(",")]
    for step in steps:
        if not (FORKS_DIR / step.key).exists():
            raise SystemExit(f"fork {step.key} missing — run `bench_vns.py setup` first")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = out_dir / "sketch.obj"
    sketch = load_sketch(Path(args.sketch))
    write_curve_obj(sketch, obj)
    n_pts = sum(len(s.get("points", [])) for s in sketch.get("strokes", []))
    print(
        f"input: {Path(args.sketch).name} — {len(sketch.get('strokes', []))} strokes, "
        f"{n_pts} points -> {obj.name}"
    )

    cfg = {
        "iters": args.iters,
        "init_iters": args.init_iters,
        "grid_res": args.grid_res,
        "sketch": str(args.sketch),
        "repeats": args.repeats,
    }
    print(
        f"protocol: {cfg['iters']} iterations, {cfg['init_iters']} init then "
        f"{cfg['iters'] - cfg['init_iters']} with L_smooth, grid_res {cfg['grid_res']}, "
        f"{args.repeats} repeat(s)\n"
    )

    runs: list[dict[str, Any]] = []
    # round-robin over the steps so a warming GPU or a thermal drift hits all
    # of them evenly instead of penalising whichever ran last
    for repeat in range(args.repeats):
        print(f"repeat {repeat + 1}/{args.repeats}")
        for step in steps:
            runs.append(run_once(step, python, obj, out_dir, cfg, f"r{repeat + 1}"))

    payload = {"config": cfg, "rocm_env": ROCM_ENV, "runs": runs,
               "forks": {s.key: git("log", "--oneline", "-1", cwd=FORKS_DIR / s.key)
                         for s in steps}}
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2))
    print()
    print(format_report(payload))
    (out_dir / "report.md").write_text(format_report(payload))
    print(f"\nwritten to {results_path.parent}")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def format_report(payload: dict[str, Any]) -> str:
    cfg = payload["config"]
    by_step: dict[str, list[dict[str, Any]]] = {}
    for r in payload["runs"]:
        by_step.setdefault(r["step"], []).append(r)

    def med(step_runs: list[dict[str, Any]], key: str) -> float | None:
        values = [r[key] for r in step_runs if r.get(key) is not None]
        return statistics.median(values) if values else None

    def num(value: float | None, unit: str = "", digits: int = 1) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}{unit}"

    order = [s.key for s in STEPS.values() if s.key in by_step]
    baseline = by_step[order[0]] if order else []
    base_total = med(baseline, "total_s")

    lines = [
        f"# VNS performance steps — {cfg['iters']} iterations "
        f"({cfg['init_iters']} init + {cfg['iters'] - cfg['init_iters']} with L_smooth), "
        f"grid_res {cfg['grid_res']}",
        "",
        f"Input: `{Path(cfg['sketch']).name}` · repeats: {cfg['repeats']} (median shown) · "
        f"ROCm on Radeon RX 7800 XT ({', '.join(f'{k}={v}' for k, v in payload['rocm_env'].items())})",
        "",
        "| step | total | startup | init ms/iter | L_smooth ms/iter | speedup | final loss |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in order:
        step = next(s for s in STEPS.values() if s.key == key)
        rs = by_step[key]
        total = med(rs, "total_s")
        speedup = f"{base_total / total:.2f}x" if total and base_total else "—"
        lines.append(
            f"| {step.label} — {step.blurb} | {num(total, 's')} | "
            f"{num(med(rs, 'startup_s'), 's')} | "
            f"{num(med(rs, 'init_ms'), 'ms')} | {num(med(rs, 'smooth_ms'), 'ms')} | "
            f"{speedup} | {num(med(rs, 'final_loss'), digits=4)} |"
        )
    lines += [
        "",
        "startup = process launch through the first completed iteration (imports, "
        "curve network, HIP init). Rates come from the trainer's own every-10-iteration "
        "log lines; the final block is excluded from the rates because it also writes "
        "the mesh, but it is included in total.",
        "",
        "Forks: " + ", ".join(f"`{k}` {v}" for k, v in payload["forks"].items()),
    ]
    return "\n".join(lines)


def report(path: Path) -> None:
    print(format_report(json.loads(path.read_text())))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="create the three bench forks")
    p_setup.add_argument("--force", action="store_true",
                         help="delete and recreate forks that already exist")

    p_run = sub.add_parser("run", help="run the benchmark")
    p_run.add_argument("--steps", default="1,2,3", help="comma-separated step numbers")
    p_run.add_argument("--iters", type=int, default=100)
    p_run.add_argument("--init-iters", type=int, default=50,
                       help="length of the initialization stage (iso_after/smooth_after)")
    p_run.add_argument("--grid-res", type=int, default=ADAPTER_DEFAULTS["grid_res"])
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--sketch", default=str(DEFAULT_SKETCH))
    p_run.add_argument("--python", default=str(SERVER_DIR / ".venv-vns" / "bin" / "python"))

    p_report = sub.add_parser("report", help="re-print a results.json")
    p_report.add_argument("results")

    sub.add_parser("clean", help="remove the forks and their branches")

    args = parser.parse_args()
    if args.command == "setup":
        setup(force=args.force)
    elif args.command == "run":
        run(args)
    elif args.command == "report":
        report(Path(args.results))
    elif args.command == "clean":
        clean()


if __name__ == "__main__":
    main()
