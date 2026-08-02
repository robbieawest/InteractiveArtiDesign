import io
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable

from .base import EmitFn, LogFn, ProgressFn, SurfacingAdapter
from .bbox import BBoxAdapter
from .neuvas import NeuvasAdapter
from .ns2s import Ns2sAdapter
from .vns import VnsAdapter
from .common import (
    JOBS_DIR,
    METHODS_DIR,
    SERVER_DIR,
    combine_meshes,
    group_strokes_by_part,
    method_env,
    release_other_workers,
    spawn,
    write_curve_obj,
)

SF3D_DIR = METHODS_DIR / os.environ.get(
    "SF3D_REPO", "surface-fitting-3d-sketches"
)
# override with the SF3D_PYTHON env var if the env lives elsewhere
SF3D_PYTHON = Path(
    os.environ.get("SF3D_PYTHON", SERVER_DIR / ".venv-sf3d" / "bin" / "python")
)
WORKER = Path(__file__).resolve().parent / "sf3d_worker.py"

# The adapters that can build a proxy. Imported by class rather than read out
# of ADAPTERS because __init__ imports *this* module while building it — the
# names it needs do not exist yet at class-definition time.
PROXY_METHODS: dict[str, SurfacingAdapter] = {
    adapter.name: adapter
    for adapter in (Ns2sAdapter(), VnsAdapter(), NeuvasAdapter(), BBoxAdapter())
}


def proxy_params() -> list[dict[str, Any]]:
    """Every proxy adapter's parameters, copied in with a `<method>_` prefix
    and greyed out unless that method is the selected one.

    Mechanical on purpose: a proxy knob the panel doesn't show is a knob
    nobody can reach, and `ns2s`'s marching-cubes threshold in particular
    decides the proxy's *topology* — the one thing this method cannot fix
    downstream. The cost is a long, mostly-dimmed parameter list.

    Two things are dropped. `part_based` is meaningless here: the proxy method
    is handed exactly one unit's strokes and must surface them whole, since
    this adapter has already done any splitting. And with it goes anything
    gated on `part_based` being true, while the whole-object side of such a
    gate — `vns`'s `n_samples`, say — is exactly the branch we do run, so it
    stays.
    """
    out: list[dict[str, Any]] = []
    for method, adapter in PROXY_METHODS.items():
        for param in adapter.params:
            if param["name"] == "part_based":
                continue
            gate = param.get("enabledWhen") or {}
            if gate.get("param") == "part_based" and gate.get("equals") is True:
                continue
            copied = dict(param)
            copied["name"] = f"{method}_{param['name']}"
            copied["label"] = f"{method}: {param['label']}"
            copied["enabledWhen"] = {"param": "proxy_method", "equals": method}
            out.append(copied)
    return out

# the paper's own defaults (scripts/run/run_segment_and_fit.py). The two
# scale-dependent ones — proxy_resolution and sketch_error_dist — are stated
# there for sketches normalized to a unit bounding-box diagonal, so the
# adapter treats them as fractions of the diagonal and multiplies by the real
# one before handing them over.
DEFAULTS: dict[str, Any] = {
    "L0": 50,
    "w_all": 100,
    "w_unary": 1.0,
    "w_smooth": 10.0,
    "w_labels": 0.01,
    "default_edge_weight": 100.0,
    "stroke_edge_weight": 1.0,
    "edge_length_factor_power": 1.0,
    "max_model_degree": 4,
    "lambda_regularization": 1.0,
    "w_normals_fit": 0.1,
    "random_seed": 123,
    "max_iterations": 30,
}

# how much of a unit's progress bar each stage owns. The proxy method is a
# whole other surfacing run, and the projection is the paper's slowest stage
# (Table 1: 15–68s, against <10s for the segmentation on most sketches).
STAGE_SPANS: dict[str, tuple[float, float]] = {
    "proxy_method": (0.00, 0.30),
    "proxy": (0.30, 0.36),
    "init": (0.36, 0.50),
    "segment": (0.50, 0.68),
    "project": (0.68, 0.98),
}


class Sf3dAdapter(SurfacingAdapter):
    """Piecewise-Smooth Surface Fitting onto Unstructured 3D Sketches
    (Yu et al., SIGGRAPH 2022; methods/surface-fitting-3d-sketches submodule).

    The odd one out among these adapters: it does not surface a sketch from
    nothing. It starts from a *proxy* — a manifold mesh of about the right
    shape and topology — and reshapes it, by segmenting the proxy into regions
    and fitting a low-degree implicit polynomial patch to the strokes over
    each region, alternating the two until the labelling stops changing. Patch
    boundaries are pushed onto the strokes, which is what makes creases come
    out sharp instead of rounded. Everything runs on the CPU; there is no
    network and nothing to download.

    The paper makes proxies with VIPSS or by hand in Blender. Neither suits a
    server, so the adapter runs one of the *other* adapters first and remeshes
    its output into a uniform manifold — `proxy_method` picks which. That
    makes this method a refinement pass over another one: `ns2s` gets you a
    proxy in seconds and the sharp features come from here.

    Runtime is a couple of minutes for a whole-object run on top of whatever
    the proxy method costs (paper Table 1: 9–146s initialization, 0.8–79s
    segmentation, 15–68s projection).

    Caveats worth knowing before you read a bad result as a bad method:
    the proxy has to have the topology you want — this method reshapes a
    surface, it never changes its genus or closes a gap; non-manifold output
    is unsupported by construction (PyGEL halfedges); and the paper's
    `open_boundary` trimming needs hand-marked border stroke points, so
    surfaces here always come out closed."""

    name = "sf3d"

    # the fitting itself is numpy/scipy/PyGEL — no torch, no card. The only
    # GPU work in an sf3d job belongs to the proxy method, which claims the
    # card for itself in `_build_proxy`; declaring True here would evict the
    # resident worker the proxy is about to need, on every single job.
    uses_gpu = False

    params = [
        {
            "name": "part_based",
            "label": "Part-based",
            "type": "bool",
            "default": False,
            "help": "Fit each part separately and merge the meshes, instead of "
            "one surface for the whole sketch. Each part gets its own proxy "
            "and its own segmentation, so this multiplies the runtime by the "
            "number of parts — but it also stops one part's strokes from "
            "pulling patches across a joint. Unassigned strokes are ignored.",
        },
        {
            "name": "proxy_method",
            "label": "Proxy method",
            "type": "choice",
            "default": "ns2s",
            "choices": ["ns2s", "vns", "neuvas", "bbox"],
            "help": "Which adapter produces the starting surface this method "
            "reshapes. It must have roughly the right shape and exactly the "
            "right topology. 'ns2s' takes seconds and is the sane default; "
            "'vns' and 'neuvas' cost minutes to an hour before this method "
            "even starts; 'bbox' is only useful for checking the pipeline "
            "runs. The chosen method's own parameters appear at the bottom of "
            "this list, prefixed with its name.",
        },
        {
            "name": "proxy_resolution",
            "label": "Proxy resolution",
            "type": "float",
            "default": 0.007,
            "min": 0.001,
            "max": 0.05,
            "step": 0.001,
            "help": "Target edge length of the remeshed proxy, as a fraction "
            "of the sketch's bounding-box diagonal (the paper uses 0.007). "
            "The proxy cannot represent detail finer than its edges, so too "
            "coarse loses sharp features — and too fine makes every stage "
            "slower, the projection quadratically so.",
        },
        {
            "name": "sketch_error_dist",
            "label": "Sketch imprecision",
            "type": "float",
            "default": 0.01,
            "min": 0.0,
            "max": 0.1,
            "step": 0.001,
            "help": "How far a stroke is allowed to sit off the surface "
            "before the fit is charged for it, as a fraction of the "
            "bounding-box diagonal. 1% suits most sketches; raise it for "
            "over-sketched or VR input so the patches stop chasing noise.",
        },
        {
            "name": "L0",
            "label": "Initial patches",
            "type": "int",
            "default": DEFAULTS["L0"],
            "min": 2,
            "max": 200,
            "step": 1,
            "help": "How many candidate surface models the segmentation "
            "starts from. It only ever merges and drops them, never adds — so "
            "this is an upper bound on the patch count, and starting well "
            "above what the shape needs is normal.",
        },
        {
            "name": "max_model_degree",
            "label": "Max patch degree",
            "type": "int",
            "default": DEFAULTS["max_model_degree"],
            "min": 1,
            "max": 4,
            "step": 1,
            "help": "Degree of the implicit polynomial fitted per patch. 1 is "
            "planes only, 2 adds quadrics, 4 is the paper's default. Lower "
            "degrees give simpler, flatter surfaces and need more patches to "
            "cover the same shape.",
        },
        {
            "name": "w_smooth",
            "label": "Segmentation smoothness",
            "type": "float",
            "default": DEFAULTS["w_smooth"],
            "min": 0.0,
            "max": 100.0,
            "step": 0.5,
            "help": "Cost of a patch boundary crossing a proxy edge that no "
            "stroke lies on. Raise it for fewer, larger patches with "
            "boundaries that hug the strokes; drop it and the segmentation "
            "fragments.",
        },
        {
            "name": "lambda_regularization",
            "label": "Patch regularization",
            "type": "float",
            "default": DEFAULTS["lambda_regularization"],
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
            "help": "Ridge term on the polynomial coefficients. Higher keeps "
            "patches tame where few strokes constrain them, at the cost of "
            "fitting the strokes less tightly.",
        },
        {
            "name": "max_iterations",
            "label": "Segmentation iterations",
            "type": "int",
            "default": DEFAULTS["max_iterations"],
            "min": 1,
            "max": 100,
            "step": 1,
            "help": "Cap on the segment/fit alternation. It stops early when "
            "the labelling stops changing, which the paper reports at 2–8 "
            "iterations, so the cap rarely binds.",
        },
        {
            "name": "projection_iterations",
            "label": "Projection iterations",
            "type": "int",
            "default": 300,
            "min": 10,
            "max": 2000,
            "step": 10,
            "help": "Cap on the L-BFGS steps that pull the proxy onto the "
            "fitted patches. This is the stage that costs the most time and "
            "the one that produces the visible surface.",
        },
        {
            "name": "snapshot_every",
            "label": "Snapshot every",
            "type": "int",
            "default": 10,
            "min": 0,
            "max": 200,
            "step": 5,
            "help": "Projection steps between published previews of the "
            "surface, so it can be watched pulling onto the strokes. 0 turns "
            "previews off; the proxy is always published up front regardless.",
        },
        {
            "name": "snap_to_strokes",
            "label": "Snap seams to strokes",
            "type": "bool",
            "default": False,
            "help": "Pull every patch boundary onto the nearest stroke, not "
            "just the ones the segmentation marked as sharp. Sharpens creases "
            "on clean curve networks; on imprecise or over-sketched input it "
            "drags seams onto strokes that were never meant to be edges.",
        },
        {
            "name": "symmetric",
            "label": "Mirror symmetry",
            "type": "bool",
            "default": False,
            "help": "Fit one half against the x = 0 plane and mirror it. Only "
            "correct if the sketch really is symmetric about that plane — "
            "otherwise half the strokes are discarded.",
        },
    ] + proxy_params()

    def run(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        report: ProgressFn,
        log: LogFn,
        emit: EmitFn,
    ) -> bytes:
        if not SF3D_PYTHON.exists():
            raise RuntimeError(
                f"sf3d environment not found at {SF3D_PYTHON} — set it up per "
                "requirements-sf3d.txt (or point SF3D_PYTHON at it)"
            )
        if not (SF3D_DIR / "scripts").is_dir():
            raise RuntimeError(
                f"sf3d method not found at {SF3D_DIR} — check the submodule "
                "is initialized (or point SF3D_REPO at the checkout)"
            )

        options = dict(options)
        part_based = bool(options.pop("part_based", False))
        proxy_method = str(options.pop("proxy_method", "ns2s"))

        # `<method>_<param>` belongs to the proxy adapter: unprefix the chosen
        # method's and drop the rest, which are the panel's dimmed rows for
        # the methods not selected — present in every payload, never a
        # mistake, so they must not reach the unknown-option warning below
        prefix = f"{proxy_method}_"
        proxy_options = {
            key[len(prefix):]: options.pop(key)
            for key in list(options)
            if key.startswith(prefix)
        }
        for key in list(options):
            if any(key.startswith(f"{m}_") for m in PROXY_METHODS):
                options.pop(key)
        params = {
            "proxy_resolution": float(options.pop("proxy_resolution", 0.007)),
            "sketch_error_dist": float(options.pop("sketch_error_dist", 0.01)),
            "L0": int(options.pop("L0", DEFAULTS["L0"])),
            "max_model_degree": int(
                options.pop("max_model_degree", DEFAULTS["max_model_degree"])
            ),
            "w_smooth": float(options.pop("w_smooth", DEFAULTS["w_smooth"])),
            "lambda_regularization": float(
                options.pop("lambda_regularization",
                            DEFAULTS["lambda_regularization"])
            ),
            "max_iterations": int(
                options.pop("max_iterations", DEFAULTS["max_iterations"])
            ),
            "projection_iterations": int(
                options.pop("projection_iterations", 300)
            ),
            "snapshot_every": int(options.pop("snapshot_every", 10)),
            "snap_to_strokes": bool(options.pop("snap_to_strokes", False)),
            "is_symmetric": bool(options.pop("symmetric", False)),
        }
        for leftover in options:
            log(f"ignoring unknown option {leftover!r}")

        # up front, not in _make_proxy: down there a ValueError means "this
        # unit is degenerate, skip it", which would turn a bad setting into
        # "produced no surfaces"
        from . import ADAPTERS  # deferred: __init__ imports this module

        if proxy_method == self.name:
            raise RuntimeError("sf3d cannot use itself as its proxy method")
        if proxy_method not in ADAPTERS:
            raise RuntimeError(
                f"unknown proxy method {proxy_method!r}; "
                f"available: {sorted(n for n in ADAPTERS if n != self.name)}"
            )

        job_dir = JOBS_DIR / f"sf3d-{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True)

        units: list[tuple[str, list[dict[str, Any]]]] = []
        if part_based:
            groups, part_names = group_strokes_by_part(sketch, log)
            for part_id, strokes in groups.items():
                units.append((str(part_names.get(part_id, part_id)), strokes))
        else:
            units.append(("sketch", sketch.get("strokes", [])))

        meshes: list[Any] = []
        span = 0.97 / len(units)
        for index, (label, strokes) in enumerate(units):
            base = 0.01 + span * index
            report(base, f"surfacing '{label}' ({index + 1}/{len(units)})")
            log(f"--- '{label}': {len(strokes)} strokes")
            try:
                mesh = self._run_one(
                    label, strokes, sketch, proxy_method, proxy_options,
                    params, job_dir,
                    index, log,
                    lambda frac, msg, base=base: report(base + span * frac, msg),
                    # every publish for one unit carries the same name, which
                    # the client reads as "replace what you have" rather than
                    # "here is another part" — so a unit refines in place
                    lambda glb, label=label: emit(
                        label if part_based else "surface", glb
                    ),
                )
            except ValueError as exc:
                log(f"skipping '{label}': {exc}")
                continue
            meshes.append(mesh)

        if not meshes:
            raise RuntimeError("sf3d produced no surfaces")

        report(0.98, "converting result to glb")
        combined = combine_meshes(meshes, log) if len(meshes) > 1 else meshes[0]
        data = combined.export(file_type="glb")
        report(1.0, f"done ({len(meshes)} surface(s))")
        return data

    def _run_one(
        self,
        label: str,
        strokes: list[dict[str, Any]],
        sketch: dict[str, Any],
        proxy_method: str,
        proxy_options: dict[str, Any],
        params: dict[str, Any],
        job_dir: Path,
        index: int,
        log: LogFn,
        on_progress: Callable[[float, str], None],
        publish: Callable[[bytes], None],
    ) -> Any:
        """Proxy, segmentation and projection for one unit. Returns the final
        mesh, in sketch world coordinates (nothing here rescales)."""
        import numpy as np  # server env
        import trimesh  # server env

        unit_dir = job_dir / f"unit_{index}"
        unit_dir.mkdir(parents=True, exist_ok=True)

        sketch_obj = unit_dir / "sketch.obj"
        write_curve_obj({"strokes": strokes}, sketch_obj)

        points = np.vstack([
            np.asarray(s["points"], dtype=float)
            for s in strokes if s.get("points")
        ])
        diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if diagonal <= 0:
            raise ValueError("strokes have no spatial extent")
        log(f"'{label}': bounding-box diagonal {diagonal:.4g}")

        proxy_raw = self._make_proxy(
            label, strokes, sketch, proxy_method, proxy_options, unit_dir, log,
            lambda frac, msg: self._stage("proxy_method", frac, msg, on_progress),
            publish,
        )

        config = {
            "repo": str(SF3D_DIR),
            "sketch_obj": str(sketch_obj),
            "proxy_raw": str(proxy_raw),
            "work_dir": str(unit_dir),
            # the two scale-relative knobs, resolved against this unit's size
            "edge_length": params["proxy_resolution"] * diagonal,
            "sketch_error_dist": params["sketch_error_dist"] * diagonal,
            **{k: v for k, v in DEFAULTS.items() if k not in params},
            **{k: v for k, v in params.items()
               if k not in ("proxy_resolution", "sketch_error_dist")},
        }
        config_path = unit_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))

        final: Path | None = None

        def on_mesh(path: Path, kind: str) -> None:
            nonlocal final
            if kind == "final":
                final = path
                return
            try:
                publish(trimesh.load(path).export(file_type="glb"))
            except Exception:
                pass  # a preview is never worth failing the job over

        self._run_worker(config_path, unit_dir, label, log, on_progress, on_mesh)

        if final is None or not final.is_file():
            raise RuntimeError(
                f"sf3d finished without producing a mesh for '{label}' "
                f"(full log: {unit_dir / 'worker.log'})"
            )
        mesh = trimesh.load(final)
        publish(mesh.export(file_type="glb"))
        return mesh

    def _make_proxy(
        self,
        label: str,
        strokes: list[dict[str, Any]],
        sketch: dict[str, Any],
        proxy_method: str,
        proxy_options: dict[str, Any],
        unit_dir: Path,
        log: LogFn,
        on_progress: Callable[[float, str], None],
        publish: Callable[[bytes], None],
    ) -> Path:
        """Run another adapter over this unit's strokes and write its result
        out as an .obj for the worker to remesh."""
        import trimesh  # server env

        from . import ADAPTERS  # deferred: __init__ imports this module

        proxy = ADAPTERS[proxy_method]
        # the proxy is the GPU half of an sf3d job, so it takes the card the
        # same way a directly-submitted job of that method would — this call
        # does not go through the job runner, which is where that normally
        # happens
        if proxy.uses_gpu:
            release_other_workers(proxy_method, log)
        if proxy_options:
            log(f"'{label}': building proxy with {proxy_method} "
                f"({', '.join(f'{k}={v}' for k, v in sorted(proxy_options.items()))})")
        else:
            log(f"'{label}': building proxy with {proxy_method}")
        glb = proxy.run(
            {"strokes": strokes, "parts": sketch.get("parts", [])},
            # part_based is deliberately absent, not merely unset: the proxy
            # adapter surfaces this unit alone and must not re-split what we
            # already split
            {**proxy_options, "part_based": False},
            lambda frac=0.0, msg="": on_progress(frac, f"proxy: {msg}"),
            lambda line: log(f"[{proxy_method}] {line}"),
            # the proxy method's own partial output is a preview of the proxy,
            # which is exactly what we want on screen while it works
            lambda name, data: publish(data),
        )

        loaded = trimesh.load(io.BytesIO(glb), file_type="glb")
        mesh = (
            trimesh.util.concatenate(loaded.dump())
            if isinstance(loaded, trimesh.Scene)
            else loaded
        )
        if len(mesh.faces) == 0:
            raise ValueError(f"{proxy_method} produced an empty proxy")
        path = unit_dir / "proxy_raw.obj"
        mesh.export(path)
        log(f"'{label}': proxy from {proxy_method}: "
            f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        return path

    def _stage(
        self,
        stage: str,
        frac: float,
        message: str,
        on_progress: Callable[[float, str], None],
    ) -> None:
        low, high = STAGE_SPANS.get(stage, (0.0, 1.0))
        on_progress(low + (high - low) * max(0.0, min(1.0, frac)), message)

    def _run_worker(
        self,
        config_path: Path,
        unit_dir: Path,
        label: str,
        log: LogFn,
        on_progress: Callable[[float, str], None],
        on_mesh: Callable[[Path, str], None],
    ) -> None:
        """Drive one worker subprocess, translating its JSON events into
        progress, log lines and published meshes."""
        cmd = [str(SF3D_PYTHON), "-u", str(WORKER), str(config_path)]
        log_path = unit_dir / "worker.log"
        tail: list[str] = []
        error: str | None = None

        with open(log_path, "w") as log_file:
            proc = spawn(
                cmd,
                cwd=SF3D_DIR,
                env=method_env(),
                stdout=subprocess.PIPE,
                stderr=log_file,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log(line)  # not protocol, but the user may still want it
                    continue
                kind = event.get("event")
                if kind == "log":
                    log(f"{label}: {event['message']}")
                    tail = (tail + [event["message"]])[-30:]
                elif kind == "progress":
                    self._stage(
                        event["stage"], event["frac"],
                        f"{label}: {event.get('message') or event['stage']}",
                        on_progress,
                    )
                elif kind == "mesh":
                    on_mesh(Path(event["path"]), event["kind"])
                elif kind == "error":
                    error = event["message"]
            code = proc.wait()

        if error is not None or code != 0:
            raise RuntimeError(
                f"sf3d failed on '{label}': {error or f'exit code {code}'}"
                f" (full log: {log_path})"
                + ("\n" + "\n".join(tail) if tail else "")
            )
