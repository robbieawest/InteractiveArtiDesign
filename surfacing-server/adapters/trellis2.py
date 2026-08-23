"""TRELLIS.2 (Xiang et al., 2025) as a surfacing method — NVIDIA only.

A 4B image-to-3D model over a sparse voxel representation (O-Voxel) that
handles open surfaces and non-manifold geometry. It is here for one reason:
*stage 1 is the same latent space TRELLIS 1 uses*. Its sparse-structure flow
config names `ss_dec_conv3d_16l8_fp16` — TRELLIS 1's decoder — so the 64^3
constraint grid, the encoder that turns it into a 16^3 latent, and the mix that
puts it into the sampler all carry over unchanged. The sketch inpainting
experiment transfers without redesign; only the pipeline around it differs.

Three things differ, and they are why this is a second adapter rather than a
flag on `trellis`:

*One image.* Conditioning is DINOv3 at 512, and `run()` takes a single image.
There is no multi-image inference path and nothing was trained on one, so this
sends a single three-quarter view — raised, looking down — rather than
reinventing the unposed-view stacking that produced Janus artifacts.

*Geometry only.* Texture is a separate SLat with its own flow model; nothing
downstream reads materials, so it is never sampled.

*The flow is ours.* `trellis` patches `sample_once` onto the sampler instance
for the length of a run. Nothing here is patched: the loop is written out in
`methods/TRELLIS.2/sketchflow/`, which lives in the fork because it imports
torch and `trellis2` and can only run in that checkout's venv. This adapter
prepares geometry and parameters; the worker carries the protocol; neither
touches the maths.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from .base import EmitFn, LogFn, ProgressFn, SurfacingAdapter
from .common import (
    JOBS_DIR,
    backend_method,
    backend_name,
    group_strokes_by_part,
    prune_job_dirs,
    release_other_workers,
    resolve_path,
)
from .trellis_common import (
    OCCUPANCY_GRID,
    ClientViewsConditioner,
    Conditioner,
    Unit,
    UnitResult,
    _voxelize_strokes,
    bundle_capture,
    conditioner_params,
    debug_views_dir,
    drop_small_components,
    fit_to_strokes,
    run_worker,
)

METHOD_NAME = "trellis2"
WORKER = Path(__file__).resolve().parent / "trellis2_worker.py"

# The weights: ~4B parameters, pulled from the hub on first use.
DEFAULT_MODEL = "microsoft/TRELLIS.2-4B"


class SingleViewConditioner(ClientViewsConditioner):
    """One raised three-quarter render of the strokes.

    Same client renderer as TRELLIS 1, different numbers, and the differences
    are all facts about this model rather than preferences:

      * `count` is 1 because TRELLIS.2 conditions on one image. More views
        would either be discarded or need an inference-time stacking hack
        invented for them, and that hack is exactly the confound this method is
        meant to be measured without. So there is no view-count knob: it is not
        an experiment, it is what the model takes.
      * `pitch` and `yaw` put the camera above and off-axis. A single view has
        to carry the whole shape, and a dead-on front view (`count: 1` alone
        gives yaw 0) hides depth entirely — a three-quarter view from above is
        the standard reading angle for exactly that reason.
      * `size` is 1024 because `preprocess_image` downscales only above that,
        so one render feeds the 512 embedding without a resample on the way.

    The render *style* — colour, thickness, margin — is inherited unchanged.
    It follows from what alpha premultiplication and a patch embedder do to
    line art, which both models do alike.
    """

    name = "views"
    label = "Client stroke render"
    help = (
        "A single raised three-quarter render of the strokes, rasterized by "
        "the editor and sent with the job."
    )
    params: list[dict[str, Any]] = []
    view_spec = {
        **ClientViewsConditioner.view_spec,
        "size": 1024,
        "count": 1,
        # radians: about 32 degrees up, 40 degrees round
        "pitch": 0.55,
        "yaw": 0.7,
        "overrides": {},
    }


CONDITIONERS: dict[str, Conditioner] = {
    conditioner.name: conditioner
    for conditioner in [SingleViewConditioner()]
}


class Trellis2Adapter(SurfacingAdapter):
    """Sketch-constrained generation on TRELLIS.2."""

    name = METHOD_NAME
    uses_gpu = True

    # No selector: there is one conditioning strategy, so "*" applies to every
    # option set. A selector exists in `trellis` because several strategies
    # want different renders; inventing one here would be a knob with a single
    # position.
    view_spec = {
        "specs": {"*": CONDITIONERS["views"].view_spec},
    }

    _ALL_PARAMS = [
        {
            "name": "part_based",
            "label": "Part-based",
            "type": "bool",
            "default": False,
            "help": "Generate each part separately and merge the meshes, "
            "instead of one object for the whole sketch. Needs a view per "
            "part from the client. Unassigned strokes are ignored.",
        },
        {
            "name": "pipeline_type",
            "label": "TRELLIS.2: resolution",
            "type": "choice",
            "default": "512",
            "choices": ["512"],
            "help": "Which shape pipeline runs. Only 512 is offered: it is "
            "the ~3s configuration, and the 1024 and cascade paths need the "
            "memory work first (they upsample the latent and raise the token "
            "count several fold). Stage 1 works at 32^3 here — TRELLIS.2 "
            "decodes 64^3 and max-pools down — while the constraint still "
            "acts on the 16^3 latent, as it does on either path.",
        },
        {
            "name": "no_image_cond",
            "label": "Experiment: no image conditioning",
            "type": "bool",
            "default": False,
            "help": "Zero the DINOv3 features so both flow stages run off the "
            "unconditional branch. The view is still rendered and sent, it "
            "just reaches the model as zeros, and guidance stops meaning "
            "anything — cond and neg_cond are both zero, so CFG cancels. "
            "Expect an arbitrary object: the question this answers is whether "
            "the prior alone produces a coherent shell or mush.",
        },
        {
            "name": "sketch_inpaint",
            "label": "Experiment: sketch inpainting",
            "type": "bool",
            "default": True,
            "help": "Constrain the structure stage with the strokes "
            "themselves. The sketch is voxelized to 64^3, encoded to the 16^3 "
            "latent grid with TRELLIS 1's sparse-structure encoder (loaded on "
            "demand and released before sampling), and mixed into the running "
            "latent at every step. This is the only channel that carries "
            "actual 3D geometry into the model — the image branch cannot, it "
            "reconstructs what it is shown and flattens the rest. On by "
            "default here: an unconstrained run is the control, not the "
            "method.",
        },
        {
            "name": "constraint_mix",
            "label": "Experiment: constraint mix",
            "type": "choice",
            "default": "x0",
            "choices": ["x0", "repaint"],
            "help": "Which space the constraint is mixed in. 'x0' blends the "
            "constraint against the model's own clean estimate and re-noises "
            "once with the model's residual. 'repaint' is TRELLIS 1's "
            "behaviour: noise the constraint to the step's time with a fixed "
            "draw and lerp the state toward it. The two are the same method "
            "apart from one term — 'repaint' also blends the two noise "
            "fields, and two unit-scale noises mixed at strength s have scale "
            "sqrt((1-s)^2+s^2), about 0.71 at s=0.5. That leaves the state "
            "carrying less noise than its time claims, over the whole grid, "
            "which is off-distribution. Hence the default.",
        },
        {
            "name": "constraint_strength",
            "label": "Experiment: inpainting strength",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "How much of the prior survives in a constrained cell. "
            "1.0 is a full overwrite — with the unmasked mode below that "
            "means the object IS the sketch, which is not an experiment — and "
            "0 is no inpainting at all. Without a mask this weight is the "
            "whole of the constraint, which is why it is the knob that "
            "matters.",
        },
        {
            "name": "sketch_mask",
            "label": "Experiment: constraint region",
            "type": "choice",
            "default": "none",
            "choices": ["none", "touched", "dilated"],
            "help": "Which latent cells the constraint is written into. "
            "'none' — unmasked mixing over the whole grid, at a strength "
            "below 1 — is the one that works. 'touched' (only cells whose "
            "4^3 voxel block a stroke passes through) is the obvious RePaint "
            "translation and it fails: the encoder's receptive field is far "
            "wider than one cell, so a cell's 8 numbers are not a local fact "
            "about its own block, and pasting them in leaves every neighbour "
            "the decoder reads them beside describing a different object. "
            "'dilated' widens the patch and has the same cause.",
        },
        {
            "name": "sketch_release",
            "label": "Experiment: release the sketch below t",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "Stop applying the constraint once the sampler's time "
            "drops below this. Time runs 1 -> 0, so releasing frees the END "
            "of sampling: hold the strokes while topology is being decided, "
            "then let the prior finish the geometry. Note that rescale_t "
            "bunches the steps up at high t, so a release of 0.5 frees fewer "
            "steps than half of them, and 1.0 turns the constraint off.",
        },
        {
            "name": "seed",
            "label": "TRELLIS.2: seed",
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 1_000_000,
            "step": 1,
            "help": "Sampling seed. The method is generative, so the same "
            "sketch and seed reproduce a result and a different seed is a "
            "different plausible object — worth sweeping before concluding "
            "anything from one run.",
        },
        {
            "name": "ss_steps",
            "label": "TRELLIS.2: structure steps",
            "type": "int",
            "default": 12,
            "min": 1,
            "max": 100,
            "step": 1,
            "help": "Flow steps for the first stage, which samples the "
            "occupancy grid — this decides the silhouette and the topology, "
            "and it is the stage the constraint acts on.",
        },
        {
            "name": "ss_cfg",
            "label": "TRELLIS.2: structure guidance",
            "type": "float",
            "default": 7.5,
            "min": 0.0,
            "max": 20.0,
            "step": 0.5,
            "help": "Classifier-free guidance for the structure stage. "
            "Higher follows the view more literally at the cost of variety. "
            "Left unset it uses the pipeline's own default.",
        },
        {
            "name": "slat_steps",
            "label": "TRELLIS.2: latent steps",
            "type": "int",
            "default": 12,
            "min": 1,
            "max": 100,
            "step": 1,
            "help": "Flow steps for the second stage, which samples shape "
            "features on the voxels stage 1 fixed — surface detail, not "
            "silhouette. No constraint acts here: the voxel set is already "
            "decided.",
        },
        {
            "name": "slat_cfg",
            "label": "TRELLIS.2: latent guidance",
            "type": "float",
            "default": 3.0,
            "min": 0.0,
            "max": 20.0,
            "step": 0.5,
            "help": "Classifier-free guidance for the latent stage. Lower "
            "than the structure stage on purpose.",
        },
        {
            "name": "preprocess",
            "label": "TRELLIS.2: preprocess the view",
            "type": "bool",
            "default": True,
            "help": "Run TRELLIS.2's own crop-and-rescale: crop to the alpha "
            "bounding box, premultiply alpha onto black. Turn it off only if "
            "the render is already exactly that — without an alpha channel it "
            "falls back to background removal, which mangles line art.",
        },
        {
            "name": "fit_to_sketch",
            "label": "Adapter: fit result to the sketch",
            "type": "bool",
            "default": True,
            "help": "NOT part of TRELLIS.2 — the model builds in a normalized "
            "cube with no relation to the drawing's coordinates. A "
            "constrained run needs no search: voxelizing the sketch into the "
            "cube is what makes the frame known, so the exact inverse "
            "normalization is used. Only an unconstrained run falls back to "
            "the yaw sweep and similarity solve.",
        },
        {
            "name": "min_component",
            "label": "Adapter: drop small pieces",
            "type": "float",
            "default": 0.01,
            "min": 0.0,
            "max": 0.5,
            "step": 0.005,
            "help": "NOT part of TRELLIS.2 — added by this adapter. Drop "
            "connected components whose area is below this fraction of the "
            "largest one. 0 keeps everything.",
        },
        {
            "name": "interactive",
            "label": "Adapter: capture the flow",
            "type": "bool",
            "default": False,
            "help": "Keep one occupancy volume per sampling step and open the "
            "flow view when the run finishes. Owning the sampling loop makes "
            "this nearly free to record — the clean estimate it draws is "
            "computed every step anyway — but the capture is held in memory "
            "and is large. Whole-object runs only.",
        },
    ] + conditioner_params(CONDITIONERS)

    @property
    def params(self) -> list[dict[str, Any]]:
        return self._ALL_PARAMS

    @staticmethod
    def available() -> bool:
        """Only where `backends.json` has an entry for this method.

        TRELLIS.2 is CUDA throughout — flash-attn, cumesh, o-voxel, flexgemm —
        with no AMD counterpart of the kind TRELLIS-AMD is for TRELLIS 1, and
        the weights want more VRAM than a consumer card has. Absent rather
        than present-and-failing: a method in the panel that cannot run on this
        machine is worse than one that is not there.
        """
        return bool(backend_method(METHOD_NAME))

    def _backend(self) -> tuple[Path, Path, dict[str, str]]:
        """This machine's checkout, interpreter and extra defines."""
        import os

        table = backend_method(METHOD_NAME)
        if not table:
            raise RuntimeError(
                f"TRELLIS.2 is not available on the {backend_name()!r} "
                "backend: it is NVIDIA-only (flash-attn, cumesh, o-voxel and "
                "flexgemm are all CUDA, and the weights want >=24GB), so "
                f"backends.json has no '{METHOD_NAME}' entry for it"
            )
        repo = resolve_path(os.environ.get("TRELLIS2_REPO") or table["repo"])
        python = resolve_path(
            os.environ.get("TRELLIS2_PYTHON") or table["python"]
        )
        return repo, python, table.get("env", {})

    def run(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        report: ProgressFn,
        log: LogFn,
        emit: EmitFn,
    ) -> bytes:
        repo, python, defines = self._backend()
        if not (repo / "trellis2").is_dir():
            raise RuntimeError(
                f"TRELLIS.2 checkout not found at {repo} — initialize the "
                "submodule there or set TRELLIS2_REPO"
            )
        if not (repo / "sketchflow").is_dir():
            raise RuntimeError(
                f"{repo} has no sketchflow/ package — that is where this "
                "method's sampling loop lives, and it is committed to the "
                "fork; the checkout is probably pinned to a commit before it"
            )
        if not python.exists():
            raise RuntimeError(
                f"TRELLIS.2 environment not found at {python} — create the "
                "venv inside that checkout or set TRELLIS2_PYTHON"
            )
        log(f"backend {backend_name()}: {repo.name} via {python}")

        options = dict(options)
        part_based = bool(options.pop("part_based", False))
        conditioner = CONDITIONERS["views"]

        prefix = "views_"
        conditioner_options = {
            key[len(prefix):]: options.pop(key)
            for key in list(options) if key.startswith(prefix)
        }
        views = options.pop("views", None)

        # One timeline or none: a part-based run is several independent
        # generations, each with its own occupancy grid, and there is nothing
        # for a single scrubber to mean across them.
        interactive = bool(options.pop("interactive", False))
        if interactive and part_based:
            log("flow capture off: not meaningful for a part-based run (each "
                "part is a separate generation)")
            interactive = False

        config = {
            "model": options.pop("model", DEFAULT_MODEL),
            "pipeline_type": str(options.pop("pipeline_type", "512")),
            "seed": int(options.pop("seed", 1)),
            "ss_steps": int(options.pop("ss_steps", 12)),
            "ss_cfg": float(options.pop("ss_cfg", 7.5)),
            "slat_steps": int(options.pop("slat_steps", 12)),
            "slat_cfg": float(options.pop("slat_cfg", 3.0)),
            "preprocess": bool(options.pop("preprocess", True)),
            "no_image_cond": bool(options.pop("no_image_cond", False)),
            "sketch_inpaint": bool(options.pop("sketch_inpaint", True)),
            "constraint_mix": str(options.pop("constraint_mix", "x0")),
            "constraint_strength": float(
                options.pop("constraint_strength", 0.5)
            ),
            "sketch_mask": str(options.pop("sketch_mask", "none")),
            "sketch_weight": float(options.pop("sketch_weight", 1.0)),
            "sketch_release": float(options.pop("sketch_release", 0.0)),
            "interactive": interactive,
        }
        fit = bool(options.pop("fit_to_sketch", True))
        min_component = float(options.pop("min_component", 0.01))
        for leftover in options:
            log(f"ignoring unknown option {leftover!r}")

        units = self._units(sketch, part_based, log)

        prune_job_dirs()
        job_dir = JOBS_DIR / f"trellis2-{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True)

        # this method owns the card for the length of the run: 4B parameters,
        # and the flow transformers peak well above the weights
        release_other_workers(self.name, log)
        debug_dir = debug_views_dir(log)

        meshes: list[Any] = []
        span = 0.98 / len(units)
        for index, unit in enumerate(units):
            base = 0.01 + span * index
            report(base, f"generating '{unit.label}' ({index + 1}/{len(units)})")
            log(f"--- '{unit.label}': {len(unit.strokes)} strokes")
            unit_dir = job_dir / f"unit_{index}"
            unit_dir.mkdir(parents=True, exist_ok=True)
            try:
                images = conditioner.prepare(
                    unit,
                    {**conditioner_options, "views": views,
                     "debug_dir": debug_dir},
                    unit_dir,
                    log,
                )
                result = self._run_one(
                    unit, images, config, fit, min_component, unit_dir, repo,
                    python, defines, log,
                    lambda frac, msg, base=base: report(base + span * frac, msg),
                )
            except ValueError as exc:
                log(f"skipping '{unit.label}': {exc}")
                continue
            emit(unit.label if part_based else "surface",
                 result.mesh.export(file_type="glb"))
            if result.frames is not None:
                emit("flow", bundle_capture(result, log), "trellis-frames")
            meshes.append(result.mesh)

        if not meshes:
            raise RuntimeError("trellis2 produced no surfaces")

        report(0.99, "converting result to glb")
        if len(meshes) == 1:
            combined = meshes[0]
        else:
            # deliberately not combine_meshes(): its boolean union wants closed
            # volumes, and this decoder is free to return open surfaces (that
            # is the point of O-Voxel), so the union would fail into a
            # concatenation anyway after paying for the repair attempt
            import trimesh  # server env

            combined = trimesh.util.concatenate(meshes)
        data = combined.export(file_type="glb")
        report(1.0, f"done ({len(meshes)} mesh(es))")
        return data

    def _units(
        self, sketch: dict[str, Any], part_based: bool, log: LogFn
    ) -> list[Unit]:
        if not part_based:
            return [Unit("sketch", sketch.get("strokes", []))]
        groups, part_names = group_strokes_by_part(sketch, log)
        return [
            Unit(str(part_names.get(part_id, part_id)), strokes, key=part_id)
            for part_id, strokes in groups.items()
        ]

    def _run_one(
        self,
        unit: Unit,
        images: list[Path],
        config: dict[str, Any],
        fit: bool,
        min_component: float,
        unit_dir: Path,
        repo: Path,
        python: Path,
        defines: dict[str, str],
        log: LogFn,
        on_progress: Any,
    ) -> UnitResult:
        """Generate one unit and return its mesh in sketch world coordinates."""
        import numpy as np  # server env
        import trimesh  # server env

        extra: dict[str, Any] = {}
        # The frame the constraint is written in is the frame the result comes
        # back in, so a constrained run has nothing left for the orientation
        # search to find.
        sketch_align: Optional[dict[str, Any]] = None
        if config.get("sketch_inpaint"):
            grid, sketch_align = _voxelize_strokes(unit, log)
            if int(grid.sum()) == 0:
                raise ValueError(
                    "the sketch voxelized to an empty grid — nothing to "
                    "inpaint with"
                )
            log(f"'{unit.label}': constraining with sketch "
                f"({int(grid.sum())} of {OCCUPANCY_GRID ** 3} cells)")
            path = unit_dir / "sketch.npy"
            np.save(path, grid)
            extra["sketch"] = str(path)

        config_path = unit_dir / "config.json"
        config_path.write_text(json.dumps({
            **config,
            **extra,
            "views": [str(p) for p in images],
            "out": str(unit_dir / "mesh.glb"),
        }, indent=2))

        written, manifest = run_worker(
            WORKER, config_path, unit_dir, unit.label, repo, python, defines,
            log, on_progress,
        )
        loaded = trimesh.load(written["final"], force="mesh")
        log(f"'{unit.label}': {len(loaded.vertices)} verts, "
            f"{len(loaded.faces)} faces")
        # before the fit, not after: stray fragments are noise in the
        # stroke-to-surface score the registration minimizes
        loaded = drop_small_components(loaded, min_component, unit, log)

        if not fit:
            align = None
        elif sketch_align is not None:
            align = sketch_align
            loaded.vertices = (
                loaded.vertices * align["scale"] + align["translation"]
            )
            log(f"'{unit.label}': placed by the sketch constraint's own frame "
                f"(scale {align['scale']:.4g}); orientation search skipped")
        else:
            align = fit_to_strokes(loaded, unit, log)

        return UnitResult(
            mesh=loaded,
            raw=None,
            frames=written.get("frames"),
            manifest=manifest,
            align=align,
        )
