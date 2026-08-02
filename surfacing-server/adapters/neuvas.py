import math
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
)

NEUVAS_DIR = METHODS_DIR / "NeuVAS"
# override with the NEUVAS_PYTHON env var if the env lives elsewhere
NEUVAS_PYTHON = Path(
    os.environ.get("NEUVAS_PYTHON", SERVER_DIR / ".venv-neuvas" / "bin" / "python")
)
TRAINER = "code/reconstruction/k1_k2_square_sum.py"

# the loss weights the paper's driver (run_abc_recon.py) uses, as
# decay_params: (data, eikonal, thin-plate, unused, unused)
DEFAULT_DATA_WEIGHT = 130.0
DEFAULT_EIKONAL_WEIGHT = 0.1
DEFAULT_SMOOTHNESS = 5e-5

# the trainer prints "epoch <n>" once per epoch
PROGRESS_RE = re.compile(r"^epoch (\d+)\s*$")

# NeuVAS does not normalize its input: the network's `scale` and the sampler's
# `global_sigma` assume a shape roughly this size, and the marching-cubes grid
# only covers [-1.2, 1.2]. Its own sample data is scaled to a half-extent of
# exactly 0.5, so we match that and map the mesh back afterwards.
TARGET_HALF_EXTENT = 0.5


def resample_strokes(
    strokes: list[dict[str, Any]], target: int
) -> list[tuple[float, float, float]]:
    """The strokes as a point cloud of roughly `target` points, spaced evenly
    along arc length. NeuVAS takes a point cloud rather than a curve network,
    and spacing it evenly matters: the optimization weights every point the
    same, so leaving the raw polyline vertices in would let slowly-drawn
    stretches of a stroke dominate."""
    polylines: list[list[tuple[float, float, float]]] = []
    total_length = 0.0
    for stroke in strokes:
        points = [tuple(p) for p in stroke.get("points", [])]
        if len(points) < 2:
            continue
        length = sum(
            math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)
        )
        if length <= 0:
            continue
        polylines.append(points)
        total_length += length
    if not polylines:
        raise ValueError("sketch has too few stroke points to surface")

    spacing = total_length / max(target, 1)
    out: list[tuple[float, float, float]] = []
    for points in polylines:
        # walk the polyline, dropping a point every `spacing` of arc length
        out.append(points[0])
        carry = 0.0
        for a, b in zip(points, points[1:]):
            seg = math.dist(a, b)
            if seg <= 0:
                continue
            travelled = spacing - carry
            while travelled <= seg:
                t = travelled / seg
                out.append(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))
                travelled += spacing
            carry = (carry + seg) % spacing
        out.append(points[-1])
    return out


class NeuvasAdapter(SurfacingAdapter):
    """NeuVAS: Neural Implicit Surfaces for Variational Shape Modeling
    (methods/NeuVAS submodule).

    Fits a neural SDF to the sketch, like VNS, but with a different smoothness
    term: it minimizes a thin-plate energy (k1² + k2², from the Hessian of the
    field) over the zero level set, and reweights that energy by distance to
    the *feature* curves so sharp G0 creases can survive along the strokes.
    Optimization is per sketch — there is no trained model and nothing to
    download.

    It is the slowest adapter by a wide margin: roughly 0.27 s/epoch on top of
    a marching-cubes snapshot every `snapshot_every` epochs, so the paper's
    10000 epochs is around 45 minutes for one sketch. Budget accordingly
    before starting a sweep, and note that a few thousand epochs already gives
    a recognisable surface.

    Unlike the other methods NeuVAS does not normalize its input, so the
    adapter scales the sketch into the range the network expects and maps the
    result back to sketch world coordinates itself."""

    name = "neuvas"

    params = [
        {
            "name": "part_based",
            "label": "Part-based",
            "type": "bool",
            "default": False,
            "help": "Fit each part separately and merge the meshes, instead of "
            "one surface for the whole sketch. Each part is a full "
            "optimization run, so this multiplies an already long runtime by "
            "the number of parts. Unassigned strokes are ignored in this mode.",
        },
        {
            "name": "epochs",
            "label": "Epochs",
            "type": "int",
            "default": 10000,
            "min": 100,
            "max": 20000,
            "step": 100,
            "help": "Optimization epochs. The paper uses 10000 (~45 min); a "
            "few thousand is usually enough to judge the shape. The thin-plate "
            "term is modulated by a cosine with a 1000-epoch period, so "
            "stopping mid-period lands on a different smoothness than "
            "stopping on a multiple of 1000.",
        },
        {
            "name": "smoothness",
            "label": "Smoothness",
            "type": "float",
            "default": DEFAULT_SMOOTHNESS,
            "min": 0.0,
            "max": 0.001,
            "step": 0.000005,
            "help": "Weight of the thin-plate energy (k1² + k2²) on the zero "
            "level set — the paper's variational term and the main quality "
            "knob. Raising it flattens and simplifies the surface between "
            "strokes; 0 removes the term entirely and leaves a plain "
            "eikonal fit.",
        },
        {
            "name": "sharp_features",
            "label": "Sharp features",
            "type": "bool",
            "default": True,
            "help": "Treat the sketch curves as feature curves: the smoothness "
            "term is scaled down near them, letting creases form along the "
            "strokes instead of rounding them off. Off, the surface is "
            "smoothed uniformly.",
        },
        {
            "name": "fidelity",
            "label": "Fidelity",
            "type": "float",
            "default": DEFAULT_DATA_WEIGHT,
            "min": 1.0,
            "max": 500.0,
            "step": 10.0,
            "help": "Weight pulling the zero level set onto the sketch points. "
            "Large relative to the other terms by design — lower it to let the "
            "surface drift off the strokes in favour of smoothness.",
        },
        {
            "name": "eikonal",
            "label": "Eikonal weight",
            "type": "float",
            "default": DEFAULT_EIKONAL_WEIGHT,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Weight of the |∇f| = 1 penalty that keeps the field a "
            "true signed distance. Too low and the level set degenerates; too "
            "high and it resists fitting the strokes.",
        },
        {
            "name": "samples",
            "label": "Curve samples",
            "type": "int",
            "default": 15000,
            "min": 1000,
            "max": 50000,
            "step": 1000,
            "help": "Points sampled evenly along the strokes to form the input "
            "cloud. Every epoch evaluates all of them, so this scales runtime "
            "directly.",
        },
        {
            "name": "resolution",
            "label": "Grid resolution",
            "type": "int",
            "default": 256,
            "min": 64,
            "max": 512,
            "step": 64,
            "help": "Marching-cubes resolution for the output meshes (cost "
            "grows cubically — 512 costs about 35 s per snapshot).",
        },
        {
            "name": "snapshot_every",
            "label": "Snapshot every",
            "type": "int",
            "default": 1000,
            "min": 100,
            "max": 10000,
            "step": 100,
            "help": "Epochs between mesh snapshots. Each one is published as "
            "it lands so the surface can be watched converging, and the last "
            "is the result — so this must divide the epoch count, or the final "
            "mesh will be from an earlier epoch.",
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
        import trimesh  # server env

        if not NEUVAS_PYTHON.exists():
            raise RuntimeError(
                f"NeuVAS environment not found at {NEUVAS_PYTHON} — set it up "
                "per requirements-neuvas.txt (or point NEUVAS_PYTHON at it)"
            )

        options = dict(options)
        part_based = bool(options.pop("part_based", False))
        params = {
            "epochs": int(options.pop("epochs", 10000)),
            "smoothness": float(options.pop("smoothness", DEFAULT_SMOOTHNESS)),
            "sharp_features": bool(options.pop("sharp_features", True)),
            "fidelity": float(options.pop("fidelity", DEFAULT_DATA_WEIGHT)),
            "eikonal": float(options.pop("eikonal", DEFAULT_EIKONAL_WEIGHT)),
            "samples": int(options.pop("samples", 15000)),
            "resolution": int(options.pop("resolution", 256)),
            "snapshot_every": int(options.pop("snapshot_every", 1000)),
        }
        for leftover in options:
            log(f"ignoring unknown option {leftover!r}")

        if params["epochs"] % params["snapshot_every"] != 0:
            log(
                f"warning: {params['epochs']} epochs is not a multiple of the "
                f"{params['snapshot_every']}-epoch snapshot interval — the "
                "result will be the last snapshot before the end"
            )

        job_dir = JOBS_DIR / f"neuvas-{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True)

        units: list[tuple[str, list[dict[str, Any]]]] = []
        if part_based:
            groups, part_names = group_strokes_by_part(sketch, log)
            for part_id, strokes in groups.items():
                units.append((str(part_names.get(part_id, part_id)), strokes))
        else:
            units.append(("sketch", sketch.get("strokes", [])))

        meshes: list[Any] = []
        span = 0.95 / len(units)
        for index, (label, strokes) in enumerate(units):
            base = 0.02 + span * index
            report(base, f"surfacing '{label}' ({index + 1}/{len(units)})")
            log(f"--- '{label}': {len(strokes)} strokes")
            try:
                mesh = self._run_one(
                    label, strokes, params, job_dir, index, log,
                    lambda frac, msg, base=base: report(base + span * frac, msg),
                    # every snapshot of one unit carries the same name, which
                    # the client reads as "replace what you have" rather than
                    # "here is another part" — so a part converges in place
                    lambda glb, label=label: emit(
                        label if part_based else "surface", glb
                    ),
                )
            except ValueError as exc:
                log(f"skipping '{label}': {exc}")
                continue
            meshes.append(mesh)

        if not meshes:
            raise RuntimeError("NeuVAS produced no surfaces")

        report(0.97, "converting result to glb")
        combined = combine_meshes(meshes, log) if len(meshes) > 1 else meshes[0]
        data = combined.export(file_type="glb")
        report(1.0, f"done ({len(meshes)} surface(s))")
        return data

    def _run_one(
        self,
        label: str,
        strokes: list[dict[str, Any]],
        params: dict[str, Any],
        job_dir: Path,
        index: int,
        log: LogFn,
        on_epoch: Callable[[float, str], None],
        publish: Callable[[bytes], None],
    ) -> Any:
        """One optimization run. Returns the final mesh, in sketch world
        coordinates."""
        import numpy as np  # server env
        import trimesh  # server env

        points = np.asarray(
            resample_strokes(strokes, params["samples"]), dtype=np.float64
        )
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        half_extent = float(np.max(points.max(axis=0) - points.min(axis=0))) / 2
        scale = TARGET_HALF_EXTENT / max(half_extent, 1e-9)
        normalized = (points - center) * scale
        log(f"'{label}': {len(points)} sample points, scaled by {scale:.4g}")

        stem = f"part_{index}"
        input_path = job_dir / f"{stem}.ply"
        trimesh.PointCloud(normalized).export(input_path)
        # the paper's own example passes the input cloud as its own feature
        # set, which is what makes creases form along the sketch curves
        feature_path = str(input_path) if params["sharp_features"] else "none"

        exps_dir = job_dir / "exps"
        plots_dir = exps_dir / stem / "result"
        published: set[str] = set()

        def snapshots() -> list[Path]:
            if not plots_dir.is_dir():
                return []
            return sorted(
                plots_dir.glob(f"igr_*_{stem}.ply"),
                key=lambda p: int(p.name.split("_")[1]),
            )

        def publish_latest() -> None:
            found = snapshots()
            if not found or found[-1].name in published:
                return
            try:
                mesh = trimesh.load(found[-1])
                mesh.vertices = mesh.vertices / scale + center
                publish(mesh.export(file_type="glb"))
            except Exception:
                return  # still being written; the next sweep picks it up
            published.add(found[-1].name)

        cmd = [
            str(NEUVAS_PYTHON), "-u", TRAINER,
            "--input_path", str(input_path),
            "--input_path_feature", feature_path,
            "--points_batch", str(min(10000, max(len(points) - 1, 1))),
            "--nepoch", str(params["epochs"]),
            "--expname", stem,
            "--sub_exp_name", "result",
            # our fork's flags: keep the checkout clean and let the caller
            # choose the snapshot cadence and resolution
            "--exps_dir", str(exps_dir),
            "--plot_frequency", str(params["snapshot_every"]),
            "--plot_resolution", str(params["resolution"]),
            "--decay_params",
            str(params["fidelity"]), str(params["eikonal"]),
            str(params["smoothness"]), "0.001", "0",
            "--use_decay_devlop_lambda",
        ]

        log_path = job_dir / f"{stem}.log"
        tail: list[str] = []
        with open(log_path, "w") as log_file:
            proc = spawn(
                cmd,
                cwd=NEUVAS_DIR,
                env=method_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                clean = line.rstrip("\n").split("\r")[-1].rstrip()
                match = PROGRESS_RE.match(clean)
                if match:
                    epoch = int(match.group(1))
                    on_epoch(
                        epoch / max(params["epochs"], 1),
                        f"{label}: epoch {epoch}/{params['epochs']}",
                    )
                    publish_latest()
                elif clean:
                    log(clean)
                tail = (tail + [line])[-30:]
            code = proc.wait()
        if code != 0:
            raise RuntimeError(
                f"NeuVAS exited with code {code}; last output "
                f"(full log: {log_path}):\n" + "".join(tail)
            )

        publish_latest()
        found = snapshots()
        if not found:
            raise RuntimeError(
                f"NeuVAS finished but produced no mesh in {plots_dir} "
                f"(full log: {log_path}) — the field may never have crossed "
                "zero, which usually means the fidelity weight is too low"
            )
        mesh = trimesh.load(found[-1])
        mesh.vertices = mesh.vertices / scale + center
        return mesh
