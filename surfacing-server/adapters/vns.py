import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable

from .base import EmitFn, LogFn, ProgressFn, SurfacingAdapter
from .common import (
    JOBS_DIR,
    METHODS_DIR,
    SERVER_DIR,
    combine_meshes,
    group_strokes_by_part,
    method_env,
    spawn,
    write_curve_obj,
)

VNS_DIR = METHODS_DIR / "vns"
# override with the VNS_PYTHON env var if the env lives elsewhere
VNS_PYTHON = Path(os.environ.get("VNS_PYTHON", SERVER_DIR / ".venv-vns" / "bin" / "python"))

# hyperparameters as run by the paper's batch driver (run_sdf_recon.py),
# which differ from the argparse defaults; all overridable per job via
# `options`
DEFAULTS: dict[str, Any] = {
    "n_samples": 1000,
    "n_points": 15000,
    "grid_res": 256,
    "grid_size": 1,
    "lr": 5e-5,
    "grad_clip_norm": 10.0,
    "init_type": "siren",
    "decoder_hidden_dim": 256,
    "decoder_n_hidden_layers": 4,
    "loss_type": "siren_supervised",
    "loss_weights": [3e4, 7e3, 0, 7e3, 1e3, 10],
    "morse_type": "l1",
    "morse_decay": "none",
    "decay_params": [3, 0.2, 3, 0.4, 0.001, 0.0001],
    # staged loss schedule; absolute iteration counts, so they do NOT scale
    # with n_samples — shorten a run and these need shortening too
    "iso_after": 200,
    "smooth_refine_after": 600,
    "cc_weight": 0.0,
}

# "Epoch: 0 [ 400/10000 (4%)] Loss: ..." — printed every 10 iterations
PROGRESS_RE = re.compile(r"\[\s*(\d+)/(\d+)\s+\(\s*\d+%\)\]\s+Loss: ([\d.eE+-]+)")


class VnsAdapter(SurfacingAdapter):
    """Variational Neural Surfacing of 3D Sketches (methods/vns submodule).

    Whole-object (default): fits one SDF to all strokes and ignores part
    membership. Part-based (opt-in): fits an independent SDF per part and
    boolean-unions the resulting meshes, so each part gets its own surface
    and iteration budget. Runs train_surface_reconstruction.py in the
    .venv-vns environment; the input is the sketch as an .obj curve network
    (v + l elements). VNS normalizes internally and rescales the output mesh
    back, so results are already in sketch world coordinates."""

    name = "vns"

    # exposed subset of DEFAULTS; names must match the trainer's CLI args
    # since options merge straight over DEFAULTS. part_based / iters_per_part
    # are adapter-only knobs (stripped before building the trainer command).
    params = [
        {
            "name": "part_based",
            "label": "Part-based",
            "type": "bool",
            "default": False,
            "help": "Surface each part separately, then boolean-union the "
            "meshes, instead of fitting one surface to the whole sketch. "
            "Unassigned strokes are ignored in this mode.",
        },
        {
            "name": "n_samples",
            "label": "Iterations",
            "type": "int",
            "default": DEFAULTS["n_samples"],
            "min": 100,
            "max": 20000,
            "step": 100,
            "help": "Whole-object optimization iterations; a mesh snapshot is "
            "saved every 100",
            "enabledWhen": {"param": "part_based", "equals": False},
        },
        {
            "name": "iters_per_part",
            "label": "Iterations per part",
            "type": "int",
            "default": DEFAULTS["n_samples"],
            "min": 100,
            "max": 20000,
            "step": 100,
            "help": "Optimization iterations run for each part in part-based "
            "mode (replaces the whole-object iteration count)",
            "enabledWhen": {"param": "part_based", "equals": True},
        },
        {
            "name": "grid_res",
            "label": "Grid resolution",
            "type": "int",
            "default": DEFAULTS["grid_res"],
            "min": 32,
            "max": 512,
            "step": 32,
            "help": "Marching-cubes / supervision grid resolution (memory & time grow cubically)",
        },
        {
            "name": "n_points",
            "label": "Curve samples",
            "type": "int",
            "default": DEFAULTS["n_points"],
            "min": 1000,
            "max": 100000,
            "step": 1000,
            "help": "Points sampled along the sketch curves per iteration",
        },
        {
            "name": "lr",
            "label": "Learning rate",
            "type": "float",
            "default": DEFAULTS["lr"],
            "min": 0.0,
            "max": 0.01,
            "step": 0.00001,
            "help": "Adam learning rate for the SIREN network",
        },
        {
            "name": "iso_after",
            "label": "Init iterations",
            "type": "int",
            "default": DEFAULTS["iso_after"],
            "min": 0,
            "max": 20000,
            "step": 50,
            "help": "Length of the initialization stage. Until this many "
            "iterations have passed no isosurface is extracted and the "
            "smoothness term is off — the surface is driven by the strokes "
            "alone. Must be well below the iteration count.",
        },
        {
            "name": "smooth_refine_after",
            "label": "Smoothness refine after",
            "type": "int",
            "default": DEFAULTS["smooth_refine_after"],
            "min": 0,
            "max": 20000,
            "step": 50,
            "help": "Iteration after which the smoothness term stops "
            "penalising curvature magnitude and penalises only its variation. "
            "Set beyond the iteration count to never refine.",
        },
    ]

    def run(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        report: ProgressFn,
        log: LogFn,
        emit: EmitFn,
    ) -> bytes:
        if not VNS_PYTHON.exists():
            raise RuntimeError(
                f"VNS environment not found at {VNS_PYTHON} — set it up per "
                "requirements-vns.txt (or point VNS_PYTHON at it)"
            )

        # adapter-only knobs, not trainer CLI args — pull them out so they
        # don't get forwarded to train_surface_reconstruction.py
        options = dict(options)
        part_based = bool(options.pop("part_based", False))
        iters_per_part = int(options.pop("iters_per_part", DEFAULTS["n_samples"]))

        job_dir = JOBS_DIR / f"vns-{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True)

        if part_based:
            return self._run_part_based(
                sketch, options, iters_per_part, job_dir, report, log, emit
            )
        return self._run_whole(sketch, options, job_dir, report, log, emit)

    def _run_whole(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        job_dir: Path,
        report: ProgressFn,
        log: LogFn,
        emit: EmitFn,
    ) -> bytes:
        import trimesh  # server env

        input_path = job_dir / "sketch.obj"
        write_curve_obj(sketch, input_path)
        report(0.0, "starting VNS")
        log(f"input: {len(sketch.get('strokes', []))} strokes -> {input_path}")

        logdir = job_dir / "log"
        # the trainer drops a mesh snapshot every 100 iterations; publish the
        # newest as it appears so the surface can be watched converging. Every
        # snapshot carries the same name, which the client reads as "replace
        # what you have" rather than "here is another part".
        mesh_dir = logdir / input_path.stem / "mesh"
        published: set[str] = set()

        def publish_latest() -> None:
            if not mesh_dir.is_dir():
                return
            snapshots = sorted(
                mesh_dir.glob("mesh_*.ply"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            if not snapshots or snapshots[-1].name in published:
                return
            try:
                emit("surface", trimesh.load(snapshots[-1]).export(file_type="glb"))
            except Exception:
                return  # still being written; the next sweep picks it up
            published.add(snapshots[-1].name)

        def on_iter(frac: float, msg: str) -> None:
            report(0.02 + 0.93 * frac, msg)
            publish_latest()

        mesh_path = self._run_vns(
            input_path,
            logdir,
            {**DEFAULTS, **options},
            log,
            on_iter,
        )
        report(0.97, "converting result to glb")
        data = trimesh.load(mesh_path).export(file_type="glb")
        report(1.0, f"done ({mesh_path.name})")
        return data

    def _run_part_based(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        iters_per_part: int,
        job_dir: Path,
        report: ProgressFn,
        log: LogFn,
        emit: EmitFn,
    ) -> bytes:
        import trimesh  # server env

        groups, part_names = group_strokes_by_part(sketch, log)

        logdir = job_dir / "log"
        params = {**DEFAULTS, **options, "n_samples": iters_per_part}
        # reserve [0, 0.02] for setup and [0.9, 1.0] for the union/export
        span = 0.88 / len(groups)
        meshes: list[Any] = []
        for i, (part_id, strokes) in enumerate(groups.items()):
            name = part_names.get(part_id, part_id)
            lo = 0.02 + span * i
            input_path = job_dir / f"part_{i}.obj"
            report(lo, f"surfacing part '{name}' ({i + 1}/{len(groups)})")
            log(f"--- part '{name}': {len(strokes)} strokes -> {input_path.name}")
            try:
                write_curve_obj({"strokes": strokes}, input_path)
            except ValueError as exc:
                log(f"skipping part '{name}': {exc}")
                continue
            mesh_path = self._run_vns(
                input_path,
                logdir,
                params,
                log,
                lambda frac, msg, lo=lo: report(lo + span * frac, msg),
            )
            mesh = trimesh.load(mesh_path)
            meshes.append(mesh)
            # this part is final even though the job isn't — publish it so the
            # benchmark window can show geometry as it lands
            emit(str(name), mesh.export(file_type="glb"))

        if not meshes:
            raise RuntimeError("part-based VNS produced no surfaces to combine")

        report(0.9, f"combining {len(meshes)} part(s)")
        combined = combine_meshes(meshes, log)
        data = combined.export(file_type="glb")
        report(1.0, f"done ({len(meshes)} parts combined)")
        return data

    def _run_vns(
        self,
        input_path: Path,
        logdir: Path,
        params: dict[str, Any],
        log: LogFn,
        on_iter: Callable[[float, str], None],
    ) -> Path:
        """Run the trainer once on `input_path`, streaming progress to
        `on_iter(fraction 0..1, message)`, and return the final mesh path."""
        cmd = [str(VNS_PYTHON), "train_surface_reconstruction.py",
               "--data_path", str(input_path), "--logdir", str(logdir)]
        for key, value in params.items():
            values = value if isinstance(value, list) else [value]
            cmd += [f"--{key}", *[str(v) for v in values]]
        # --slim_output is our fork's flag: meshes only, no checkpoints/
        # grid dumps (they cost ~70 MB per checkpoint at grid_res 256)
        cmd += ["--morse_near", "--output_any", "--slim_output"]

        env = method_env()

        log_path = input_path.with_suffix(".log")
        tail: list[str] = []
        with open(log_path, "w") as log_file:
            proc = spawn(
                cmd,
                # cwd matters: the script copies ../models/* for backup and
                # resolves its own imports relative to the repo
                cwd=VNS_DIR / "surface_reconstruction",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                # keep only the final state of \r-refreshing progress bars
                clean = line.rstrip("\n").split("\r")[-1].rstrip()
                if clean:
                    log(clean)
                tail = (tail + [line])[-30:]
                match = PROGRESS_RE.search(line)
                if match:
                    done, total, loss = match.groups()
                    fraction = int(done) / max(int(total), 1)
                    on_iter(fraction, f"iter {done}/{total}, loss {loss}")
            code = proc.wait()
        if code != 0:
            raise RuntimeError(
                f"VNS exited with code {code}; last output (full log: {log_path}):\n"
                + "".join(tail)
            )

        # meshes land in <logdir>/<input stem>/mesh/mesh_<iter>.ply every
        # 100 iterations; the highest iteration is the final surface
        mesh_dir = logdir / input_path.stem / "mesh"
        meshes = sorted(
            mesh_dir.glob("mesh_*.ply"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not meshes:
            raise RuntimeError(
                f"VNS finished but produced no mesh in {mesh_dir} "
                f"(full log: {log_path})"
            )
        return meshes[-1]
