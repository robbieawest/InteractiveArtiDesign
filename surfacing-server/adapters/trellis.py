"""TRELLIS (Structured 3D Latents, Xiang et al. 2024) as a surfacing method.

Two things make this adapter unlike the others.

*Two codebases.* Upstream TRELLIS is CUDA-only — custom kernels, xformers,
flash-attn — so an AMD machine runs the TRELLIS-AMD fork instead, from its own
checkout with its own venv and its own defines (torchsparse rather than spconv,
sdpa rather than xformers). Which one this machine uses is not decided here: it
comes from `backends.json` under the active SURFACING_GPU_BACKEND, the same
table that already decides every other vendor-dependent answer. The adapter,
its parameters, and the worker protocol are identical either way, so results
from the two are comparable.

*The sketch is not the input.* Every other method here consumes the strokes as
geometry. TRELLIS consumes *images*: DINOv2 patch tokens are the only channel
through which anything about the sketch reaches the model, and it has no
notion of a curve in space. So a conditioning step has to turn strokes into
something image-shaped first, and there is more than one defensible way to do
that. That step is pluggable (`CONDITIONERS`) rather than inlined, because
picking the right one is the open experimental question — multi-view renders
of the strokes is the first thing to try, not the answer.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

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
    SURFACE_METHOD_BLUR,
    SURFACE_METHOD_MARGIN,
    SURFACE_METHOD_THRESHOLD,
    ClientViewsConditioner,
    Conditioner,
    Unit,
    UnitResult,
    _align_rotation,
    _voxelize_strokes,
    _voxelize_surface,
    bundle_capture,
    conditioner_params,
    debug_views_dir,
    drop_small_components,
    fit_to_strokes,
    run_worker,
)

METHOD_NAME = "trellis"
WORKER = Path(__file__).resolve().parent / "trellis_worker.py"

# The weights. Pulled from the HF hub into ~/.cache/huggingface on first use
# (~2.9GB, plus ~1.2GB of DINOv2 into the torch hub cache) — nothing here
# downloads them explicitly, `from_pretrained` does it.
DEFAULT_MODEL = "microsoft/TRELLIS-image-large"


# The strategies this adapter offers. TRELLIS.2 keeps its own registry — see
# `conditioner_params`.
CONDITIONERS: dict[str, Conditioner] = {
    conditioner.name: conditioner
    for conditioner in [ClientViewsConditioner()]
}



class TrellisAdapter(SurfacingAdapter):
    """Image-to-3D generation conditioned on renders of the sketch.

    Unlike every other adapter here, this one *generates* rather than fits:
    nothing constrains the result to pass through the strokes. Two flow-
    matching stages run in sequence — the first samples a 64^3 occupancy grid
    from noise, the second samples an 8-channel latent on the voxels that
    survived — and a sparse-transformer decoder turns that into a FlexiCubes
    mesh at 256^3. The strokes reach the model only as DINOv2 tokens from the
    conditioning images, so the output resembles the sketch about as well as
    the images do, and no better.

    Geometry only: `formats=['mesh']` skips the Gaussian and radiance-field
    decoders entirely, which is also what keeps the dependency surface small
    enough to run on AMD (neither nvdiffrast nor diff-gaussian-rasterization
    is imported on this path — trellis.renderers is lazy).

    Two caveats worth knowing before reading a bad result as a bad method.
    Raw FlexiCubes output is not watertight and typically arrives in many
    connected components; upstream cleans that in `postprocess_mesh`, which
    needs nvdiffrast and so is not run here. And `run_multi_image` takes no
    camera poses — it reconciles views purely from their tokens — so more
    views is not monotonically better, and three or four well-spread ones beat
    a dozen.
    """

    name = METHOD_NAME
    uses_gpu = True

    # keyed by the conditioner param, so switching strategy changes what the
    # client renders without the client knowing anything about TRELLIS. A
    # strategy that builds its input some other way contributes no entry, and
    # the client then sends no views for it.
    view_spec = {
        "selector": "conditioner",
        "specs": {
            name: conditioner.view_spec
            for name, conditioner in CONDITIONERS.items()
            if conditioner.view_spec
        },
    }

    # Built once; `params` below filters it for what this machine can run.
    _ALL_PARAMS = [
        {
            "name": "part_based",
            "label": "Part-based",
            "type": "bool",
            "default": False,
            "help": "Generate each part separately and merge the meshes, "
            "instead of one object for the whole sketch. Needs a view set per "
            "part from the client. Unassigned strokes are ignored.",
        },
        {
            "name": "conditioner",
            "label": "Adapter: conditioning",
            "type": "choice",
            "default": "views",
            "choices": sorted(CONDITIONERS),
            "help": "How the strokes are turned into the images the model "
            "conditions on. This is the experimental axis of the method: "
            "TRELLIS sees nothing of the sketch except through here.",
        },
        {
            "name": "no_image_cond",
            "label": "Experiment: no image conditioning",
            "type": "bool",
            "default": False,
            "help": "Zero the DINOv2 features so both flow stages run off the "
            "unconditional branch. The views are still rendered and sent, "
            "they just reach the model as zeros. Guidance stops meaning "
            "anything — cond and neg_cond are both zero, so CFG cancels — and "
            "the multi-view injection is skipped. Expect an arbitrary object: "
            "the question this answers is whether the prior alone produces a "
            "coherent shell or mush.",
        },
        {
            "name": "sketch_inpaint",
            "label": "Experiment: sketch inpainting",
            "type": "bool",
            "default": False,
            "help": "Constrain the structure stage with the strokes "
            "themselves. The sketch is voxelized to 64^3, encoded to the 16^3 "
            "latent grid with the sparse-structure encoder (not part of the "
            "pipeline — downloaded and loaded on demand, then released), and "
            "mixed back into the running latent at every step, noised to that "
            "step's time. This is the only channel that carries actual 3D "
            "geometry into the model. Independent of the image switch above; "
            "the two combine.",
        },
        {
            "name": "surface_inpaint",
            "label": "Experiment: surface inpainting",
            "type": "bool",
            "default": False,
            "help": "Add a predicted surface to the constraint grid. The "
            "strokes are surfaced with NeuralSketch2Surf first (hardcoded for "
            "now: probability field, threshold 0.6, blur 1.6 voxels), the "
            "shell of that prediction is voxelized into the same 64^3 cube, "
            "and the two grids are unioned — so the constraint carries a "
            "closed surface rather than a wireframe the prior has to infer "
            "one from. The shell, not the solid: stage 1 voxelizes surfaces, "
            "and a filled grid is both outside what it was trained on and "
            "several times the voxel count stage 2 then pays for. "
            "The two grids stay separate all the way into the sampler, each "
            "with its own weight and release below, and are mixed in turn — "
            "surface first, strokes second, so the strokes win where they "
            "overlap. With the sketch box off this runs on the surface "
            "alone.",
        },
        {
            "name": "surfaced_condition",
            "label": "Experiment: surfaced image condition",
            "type": "bool",
            "default": False,
            "help": "Condition on renders of a predicted surface instead of "
            "renders of the strokes. The client surfaces the sketch with "
            "NeuralSketch2Surf first and renders that solid, opaque and shaded "
            "like any other geometry, from the same camera ring the stroke "
            "views use. This is the other half of the wireframe problem: the "
            "image branch reconstructs what it is shown, and what it has been "
            "shown so far is line art. Costs one ns2s solve before the run "
            "starts — shared with surface inpainting when that is also on, "
            "not paid twice. Whole-object only: a part-based run keeps the "
            "stroke renders.",
        },
        {
            "name": "surface_smooth",
            "label": "Experiment: smooth the surfaced condition",
            "type": "bool",
            "default": False,
            "help": "Run NeuralSketch2Surf's own post-process on the surface "
            "before it is rendered: Laplacian smoothing balanced against the "
            "sketch curves, hole filling, normal repair, Taubin smoothing, at "
            "the method's default balance. Marching cubes on a 112³ field "
            "leaves stair-stepping that has nothing to do with the shape, and "
            "the image branch has no way to know that. Only affects the "
            "renders — the inpainting constraint is built from the field, "
            "which no mesh post-process touches.",
            "enabledWhen": {"param": "surfaced_condition", "equals": True},
        },
        {
            "name": "surface_threshold",
            "label": "Experiment: surface threshold",
            "type": "float",
            "default": SURFACE_METHOD_THRESHOLD,
            "min": 0.05,
            "max": 0.95,
            "step": 0.05,
            "help": "Probability the predicted surface is read at — the level "
            "its shell is taken at for the constraint, and the level marching "
            "cubes runs at for the surfaced image condition, so both boxes "
            "read it. Lower keeps more of what the surfacer was unsure about, "
            "thin features especially. Note that on the constraint side it is "
            "coupled to the blur below: smoothing moves the level set, and "
            "anything above 0.5 moves it inward, so a heavy blur at a high "
            "threshold shrinks the shell and can delete thin parts of it "
            "outright (a 2-voxel sheet peaks around 0.46 after a 1.6-voxel "
            "blur, and vanishes at 0.6). The rendered surface is not subject "
            "to that — it is marched on the unblurred field — so the same "
            "threshold can mean a fatter mesh than shell.",
        },
        {
            "name": "surface_blur",
            "label": "Experiment: surface blur",
            "type": "float",
            "default": SURFACE_METHOD_BLUR,
            "min": 0.0,
            "max": 4.0,
            "step": 0.1,
            "help": "Gaussian smoothing of the probability field, in voxels, "
            "applied on the server before the constraint's shell is taken from "
            "it. 0 uses the prediction as it comes: sharper, and pitted where "
            "the network was uncertain, which a shell then traces every hole "
            "of. The constraint only — marching cubes for the surfaced image "
            "condition runs on the raw probabilities, because closing pinholes "
            "in a signal the sampler reads is no reason to smooth geometry.",
        },
        {
            "name": "sketch_mask",
            "label": "Experiment: constraint region",
            "type": "choice",
            "default": "touched",
            "choices": ["touched", "dilated", "none"],
            "help": "Which latent cells the constraint is written into, for "
            "every source. KNOWN NOT TO WORK: 'touched' — the cells whose 4^3 "
            "voxel block the grid sets. The sparse-structure VAE is a conv "
            "stack with a receptive field far wider than one cell, so a latent "
            "cell is not a local fact about its own block: its value was "
            "computed from a whole neighbourhood, most of which is empty in a "
            "sketch. Writing only the touched cells therefore pastes in values "
            "that encode 'strokes here and nothing around them' while leaving "
            "the neighbours the decoder reads them with untouched, and the two "
            "disagree. 'dilated' grows the region by one cell in each "
            "direction, which softens the seam but does not fix the cause. "
            "'none' is unmasked — the mix is applied to the whole grid, which "
            "is the version that does not pretend the 4x downsampling leaves a "
            "clean per-cell fact to overwrite, and is the one to use.",
        },
        {
            "name": "constraint_strength",
            "label": "Experiment: inpainting strength",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "How much of the prior survives in a constrained cell. The "
            "mix is a lerp — the cell becomes (1-s)*current + s*constraint — "
            "and this is the s, the only thing that trades the constraint "
            "against the model. It does not depend on what is in the grid or "
            "on the weights below: 1.0 is a full overwrite of every covered "
            "cell however the sources divide it, and 0 disables inpainting "
            "entirely without changing anything else about the run.",
        },
        {
            "name": "sketch_weight",
            "label": "Experiment: sketch constraint weight",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "sketch_inpaint", "equals": True},
            "help": "The strokes' share of a cell that both sources cover, "
            "against the surface's weight below — the two are normalized, so "
            "only their ratio matters. It does nothing where the strokes are "
            "the only source there, and nothing at all with surface "
            "inpainting off: how hard the constraint is applied is the "
            "strength above, not this.",
        },
        {
            "name": "surface_weight",
            "label": "Experiment: surface constraint weight",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "surface_inpaint", "equals": True},
            "help": "The surface's share where it overlaps the strokes, "
            "against the sketch weight above. Equal weights average the two "
            "targets; 0.25 against 1.0 gives the surface a fifth of the "
            "overlap and the strokes the rest, which is the sensible "
            "direction — the strokes are what the user drew, this is a guess "
            "about everything between them. Outside the overlap the surface "
            "gets the cell to itself whatever this says. If one source "
            "releases, its share goes to the ones still applying, not back to "
            "the prior.",
        },
        {
            "name": "sketch_release",
            "label": "Experiment: release sketch below t",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "sketch_inpaint", "equals": True},
            "help": "Stop applying the stroke constraint once the flow time "
            "drops below this, leaving the last steps free to reconcile the "
            "constrained cells with their neighbours. Time runs 1 -> 0, so "
            "this frees the END of sampling; 0 constrains all the way down, "
            "which makes the final latent hold the sketch exactly wherever the "
            "mix was full. Note the sampler's rescale_t=3.0 bunches steps up "
            "at high t, so a value frees fewer steps than it looks: at 12 "
            "steps, 0.5 leaves only the last 3 free and 1.0 disables the "
            "constraint outright.",
        },
        {
            "name": "surface_release",
            "label": "Experiment: release surface below t",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "surface_inpaint", "equals": True},
            "help": "The same, for the predicted surface. Releasing it earlier "
            "than the strokes is the interesting run: the surface says what "
            "shape to head toward while the topology is still being decided, "
            "then gets out of the way and lets the prior and the strokes "
            "finish the geometry. The flow view's inpainting signal region "
            "shows exactly when each source drops out.",
        },
        {
            "name": "seed",
            "label": "TRELLIS: seed",
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
            "name": "mode",
            "label": "TRELLIS: multi-view mode",
            "type": "choice",
            "default": "stochastic",
            "choices": ["stochastic", "multidiffusion"],
            "help": "How several unposed views are reconciled. 'stochastic' "
            "resamples which view guides each step; 'multidiffusion' averages "
            "their predictions, which is steadier but blurs disagreement "
            "between views into a compromise shape.",
        },
        {
            "name": "ss_steps",
            "label": "TRELLIS: structure steps",
            "type": "int",
            "default": 12,
            "min": 1,
            "max": 100,
            "step": 1,
            "help": "Flow steps for the first stage, which samples the "
            "occupancy grid — this decides the silhouette and the topology. "
            "Upstream's default is 25; 12 is usually indistinguishable and "
            "twice as fast.",
        },
        {
            "name": "ss_cfg",
            "label": "TRELLIS: structure guidance",
            "type": "float",
            "default": 7.5,
            "min": 0.0,
            "max": 20.0,
            "step": 0.5,
            "help": "Classifier-free guidance for the structure stage. Higher "
            "follows the views more literally at the cost of variety, and past "
            "about 10 tends to produce thin, spiky geometry.",
        },
        {
            "name": "slat_steps",
            "label": "TRELLIS: latent steps",
            "type": "int",
            "default": 12,
            "min": 1,
            "max": 100,
            "step": 1,
            "help": "Flow steps for the second stage, which samples the "
            "structured latent on the occupied voxels — this decides surface "
            "detail, not silhouette.",
        },
        {
            "name": "slat_cfg",
            "label": "TRELLIS: latent guidance",
            "type": "float",
            "default": 3.0,
            "min": 0.0,
            "max": 20.0,
            "step": 0.5,
            "help": "Classifier-free guidance for the latent stage. Lower "
            "than the structure stage on purpose — upstream uses 3.",
        },
        {
            "name": "preprocess",
            "label": "TRELLIS: preprocess views",
            "type": "bool",
            "default": True,
            "help": "Run TRELLIS's own crop-and-rescale over each view: crop "
            "to the alpha bounding box at 1.2x, resize to 518px, premultiply "
            "alpha onto black. Turn it off only if the views are already "
            "exactly that, since without an alpha channel it falls back to "
            "background removal that mangles line art.",
        },
        {
            "name": "simplify_ratio",
            "label": "TRELLIS: simplify (simplify_ratio)",
            "type": "float",
            "default": 0.9,
            "min": 0.0,
            "max": 0.98,
            "step": 0.02,
            "help": "Upstream's `postprocess_mesh(simplify_ratio=...)`: "
            "fraction of faces removed by quadric edge collapse. CPU "
            "(pyvista), so unlike hidden-face removal it runs on either "
            "backend. Raw output is ~250k faces of mostly "
            "redundant detail; 0.9 is upstream's default. 0 keeps the raw "
            "mesh. CPU, so it costs seconds and works on either backend.",
        },
        {
            "name": "fill_holes",
            "label": "TRELLIS: remove hidden faces (fill_holes)",
            "type": "choice",
            "default": "auto",
            "choices": ["auto", "on", "off"],
            "help": "Upstream's `postprocess_mesh(fill_holes=...)`. The name "
            "is TRELLIS's and is a misnomer: it does not fill holes, it "
            "rasterizes the mesh from 100 views and REMOVES hidden geometry "
            "— components that are rarely visible, and interior shells "
            "reachable through a small hole. It is what clears the loose "
            "fragments around the surface. Only offered on backends that can "
            "run it.",
        },
        {
            "name": "min_component",
            "label": "Adapter: drop small pieces",
            "type": "float",
            "default": 0.01,
            "min": 0.0,
            "max": 0.5,
            "step": 0.005,
            "help": "NOT part of TRELLIS — added by this adapter as the "
            "stand-in for hidden-face removal on backends that cannot run "
            "it. Removes connected components whose surface area is below "
            "this fraction of the largest one. FlexiCubes leaves a shell of detached "
            "fragments hugging the surface, which read as shimmer at the "
            "silhouette. Crude by comparison — it judges by size, not "
            "visibility, so raising it too far eats genuinely small parts. "
            "0 keeps everything.",
        },
        {
            "name": "fit_to_sketch",
            "label": "Adapter: fit to sketch",
            "type": "bool",
            "default": True,
            "help": "Scale and centre the result into the strokes' bounding "
            "box. The model works in a normalized cube and knows nothing of "
            "the sketch's units, so without this the mesh lands nowhere near "
            "the drawing. Uniform scaling only — fitting each axis "
            "independently would shear the shape the model inferred.",
        },
        {
            "name": "interactive",
            "label": "Adapter: interactive flow view",
            "type": "bool",
            "default": False,
            "lockedWhileSurfaced": True,
            "help": "Record what the two flow stages did, step by step, and "
            "show it in the viewport instead of just the finished mesh: the "
            "conditioning views, the sketch, the occupancy grid the structure "
            "stage samples, and where the latent stage is still moving. Costs "
            "a decoder pass per structure step and a few MB per run, so it is "
            "off by default and cannot be changed while a surface is on "
            "screen — the run has to record it. Ignored for part-based runs, "
            "which are several independent generations with no single "
            "timeline.",
        },
        {
            "name": "keep_raw",
            "label": "Adapter: keep unprocessed mesh",
            "type": "bool",
            "default": False,
            "lockedWhileSurfaced": True,
            "help": "Also return the mesh as FlexiCubes produced it, before "
            "simplification, hole filling and fragment removal, so the viewer "
            "can switch between the two. Worth it when the question is "
            "whether cleanup deleted something real — it is the stage that "
            "removes geometry. Roughly ten times the size of the delivered "
            "mesh.",
        },
    ] + conditioner_params(CONDITIONERS)

    @property
    def params(self) -> list[dict[str, Any]]:
        """The parameters this machine can actually act on.

        `fill_holes` is dropped where the backend cannot run it, rather than
        shown and ignored: the panel's `enabledWhen` only gates on another
        param's value, so there is no way to grey a row out on a fact about
        the machine, and a knob that silently does nothing is worse than an
        absent one. Nothing is lost for cross-backend comparison — the setting
        that makes that fair is forcing it OFF, which is already the only
        behaviour available here.
        """
        if backend_method(METHOD_NAME).get("fill_holes", False):
            return self._ALL_PARAMS
        return [p for p in self._ALL_PARAMS if p["name"] != "fill_holes"]

    # --- repo / interpreter resolution ---

    def _backend(self) -> tuple[Path, Path, dict[str, str]]:
        """This machine's checkout, interpreter and extra defines. Env vars
        win over the table so a machine can keep either checkout elsewhere
        without editing a file that is shared with the cluster."""
        import os

        table = backend_method(METHOD_NAME)
        if not table:
            raise RuntimeError(
                f"backends.json has no '{METHOD_NAME}' entry for backend "
                f"{backend_name()!r} — TRELLIS needs one, because the CUDA and "
                "AMD builds are different checkouts"
            )
        repo = resolve_path(os.environ.get("TRELLIS_REPO") or table["repo"])
        python = resolve_path(
            os.environ.get("TRELLIS_PYTHON") or table["python"]
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
        if not (repo / "trellis").is_dir():
            raise RuntimeError(
                f"TRELLIS checkout not found at {repo} (backend "
                f"{backend_name()!r}) — clone it there or set TRELLIS_REPO"
            )
        if not python.exists():
            raise RuntimeError(
                f"TRELLIS environment not found at {python} — create the venv "
                "inside that checkout or set TRELLIS_PYTHON"
            )
        log(f"backend {backend_name()}: {repo.name} via {python}")

        options = dict(options)
        part_based = bool(options.pop("part_based", False))
        conditioner_name = str(options.pop("conditioner", "views"))
        if conditioner_name not in CONDITIONERS:
            raise RuntimeError(
                f"unknown conditioner {conditioner_name!r}; "
                f"available: {sorted(CONDITIONERS)}"
            )
        conditioner = CONDITIONERS[conditioner_name]

        # `<conditioner>_<param>` rows for the strategies the user did not
        # pick are in every payload and are not mistakes — strip them before
        # the unknown-option warning, and unprefix the selected one's
        prefix = f"{conditioner_name}_"
        conditioner_options = {
            key[len(prefix):]: options.pop(key)
            for key in list(options)
            if key.startswith(prefix)
        }
        for key in list(options):
            if any(key.startswith(f"{n}_") for n in CONDITIONERS):
                options.pop(key)

        # the images themselves are not a `params` row — they are payload, not
        # a knob, and the panel never renders them
        views = options.pop("views", None)

        # One timeline or none: a part-based run is several independent
        # generations, each with its own occupancy grid and its own fit, and
        # there is nothing for a single scrubber to mean across them.
        interactive = bool(options.pop("interactive", False))
        if interactive and part_based:
            log("interactive flow view off: not meaningful for a part-based "
                "run (each part is a separate generation)")
            interactive = False

        config = {
            "model": options.pop("model", DEFAULT_MODEL),
            "seed": int(options.pop("seed", 1)),
            "mode": str(options.pop("mode", "stochastic")),
            "ss_steps": int(options.pop("ss_steps", 12)),
            "ss_cfg": float(options.pop("ss_cfg", 7.5)),
            "slat_steps": int(options.pop("slat_steps", 12)),
            "slat_cfg": float(options.pop("slat_cfg", 3.0)),
            "preprocess": bool(options.pop("preprocess", True)),
            "no_image_cond": bool(options.pop("no_image_cond", False)),
            "sketch_inpaint": bool(options.pop("sketch_inpaint", False)),
            "surface_inpaint": bool(options.pop("surface_inpaint", False)),
            # Acted on by the client, which has to surface and render before
            # it can submit at all; recorded here so the run's config says
            # what its conditioning images actually were.
            "surfaced_condition": bool(options.pop("surfaced_condition", False)),
            "surface_smooth": bool(options.pop("surface_smooth", False)),
            "surface_threshold": float(
                options.pop("surface_threshold", SURFACE_METHOD_THRESHOLD)
            ),
            "surface_blur": float(
                options.pop("surface_blur", SURFACE_METHOD_BLUR)
            ),
            "sketch_mask": str(options.pop("sketch_mask", "touched")),
            "constraint_strength": float(
                options.pop("constraint_strength", 1.0)
            ),
            "sketch_weight": float(options.pop("sketch_weight", 1.0)),
            "surface_weight": float(options.pop("surface_weight", 1.0)),
            "sketch_release": float(options.pop("sketch_release", 0.0)),
            "surface_release": float(options.pop("surface_release", 0.0)),
            "interactive": interactive,
            "keep_raw": bool(options.pop("keep_raw", False)),
            "simplify_ratio": float(options.pop("simplify_ratio", 0.9)),
            "fill_holes": self._fill_holes(
                str(options.pop("fill_holes", "auto")), log
            ),
        }
        fit = bool(options.pop("fit_to_sketch", True))
        min_component = float(options.pop("min_component", 0.01))
        for leftover in options:
            log(f"ignoring unknown option {leftover!r}")

        units = self._units(sketch, part_based, log)

        prune_job_dirs()
        job_dir = JOBS_DIR / f"trellis-{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True)

        # Before the card is claimed, not after: surfacing runs on the GPU
        # too, and its worker sits on ~13GB that the release below is there to
        # reclaim. Every unit is predicted in one call so the model loads once.
        surfaces: list[Optional[tuple[Any, dict[str, Any]]]] = [None] * len(units)
        if config["surface_inpaint"]:
            report(0.005, "predicting the surface for the constraint")
            surfaces = self._surface_grids(
                units, job_dir, config["surface_blur"], log
            )

        # this method owns the card for the length of the run: the pipeline is
        # ~4GB of weights and the flow transformers peak well above that
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
                    python, defines, surfaces[index], log,
                    lambda frac, msg, base=base: report(base + span * frac, msg),
                )
            except ValueError as exc:
                log(f"skipping '{unit.label}': {exc}")
                continue
            emit(unit.label if part_based else "surface",
                 result.mesh.export(file_type="glb"))
            if result.raw is not None:
                emit("surface (unprocessed)",
                     result.raw.export(file_type="glb"), "raw")
            if result.frames is not None:
                emit("flow", self._bundle(result, log), "trellis-frames")
            meshes.append(result.mesh)

        if not meshes:
            raise RuntimeError("trellis produced no surfaces")

        report(0.99, "converting result to glb")
        if len(meshes) == 1:
            combined = meshes[0]
        else:
            # deliberately not combine_meshes(): its boolean union wants
            # closed volumes, and raw FlexiCubes output is neither watertight
            # nor single-component, so the union would fail into a
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

    @staticmethod
    def _surface_grids(
        units: list[Unit], job_dir: Path, blur: float, log: LogFn
    ) -> list[Optional[tuple[Any, dict[str, Any]]]]:
        """Predicted occupancy fields, one slot per unit, None where there is
        none.

        The surfacing method is another adapter, called as a component: it
        owns how its own network is run and what frame its output is in, and
        this only needs the field and the way back to world. A failure here is
        not fatal — the run falls back to whatever the strokes give, which is
        the experiment's own control condition.

        Positional, not keyed by label: two parts may carry the same name, and
        a dict would quietly hand one part's surface to another.

        Both sides of the card are handled here. Surfacing is a GPU job too,
        so it evicts the other methods before it loads — the same rule every
        GPU method follows, and skipping it would mean loading ~13GB on top of
        whatever was already resident. Then its own worker is stopped as soon
        as the grids are in hand, rather than left for the general sweep: the
        process holds its model until it is reaped, and the pipeline that
        loads next needs the whole card.
        """
        from . import ns2s

        keys = [f"{index}: {unit.label}" for index, unit in enumerate(units)]
        # Set only if the method actually has to predict something. With the
        # surfaced image condition on, the client's own ns2s job has already
        # made these fields and they come back from its cache, in which case
        # nothing is loaded and there is nothing to release either.
        predicted = False

        def before_predict() -> None:
            nonlocal predicted
            predicted = True
            release_other_workers(ns2s.METHOD_NAME, log)

        try:
            grids = ns2s.probability_grids(
                {
                    key: {"strokes": unit.strokes}
                    for key, unit in zip(keys, units)
                },
                job_dir,
                SURFACE_METHOD_MARGIN,
                blur,
                log,
                before_predict,
            )
        except Exception as exc:
            log(f"surface inpainting: prediction failed ({type(exc).__name__}: "
                f"{exc}); continuing without a surface")
            return [None] * len(units)
        finally:
            # Logged either way: "was the surfacing model still resident when
            # TRELLIS started" is the first question an out-of-memory run
            # raises, and a silent success answers it as poorly as a silent
            # failure.
            if not predicted:
                log("surface inpainting: reused an existing prediction, so "
                    "nothing was loaded onto the card")
            try:
                stopped = ns2s.WORKER.stop()
                log("released the ns2s worker after the prediction"
                    if stopped else
                    "ns2s worker was already gone after the prediction")
            except Exception as exc:
                log(f"WARNING: could not release the ns2s worker ({exc}) — it "
                    "is still holding the GPU and TRELLIS may run out of memory")
        return [grids.get(key) for key in keys]

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
        surface: Optional[tuple[Any, dict[str, Any]]],
        log: LogFn,
        on_progress: Callable[[float, str], None],
    ) -> "UnitResult":
        """Generate one unit and return its mesh in sketch world coordinates,
        together with whatever extras this run was asked to record."""
        import numpy as np  # server env
        import trimesh  # server env

        extra: dict[str, Any] = {}
        # The frame the constraint is written in is the frame the result comes
        # back in, so the orientation search below has nothing left to find:
        # `_fit` sweeps yaw because TRELLIS normally builds in an arbitrary
        # canonical frame, and voxelizing the sketch into the cube is exactly
        # what stops that being true. Keep the inverse normalization instead —
        # it is exact, and a 72-candidate ICP against a shape the strokes
        # already pin can only move it off.
        sketch_align: Optional[dict[str, Any]] = None
        if config.get("sketch_inpaint") or config.get("surface_inpaint"):
            # The strokes are voxelized either way: they define the cube the
            # constraint is written in (and so the frame the result is read
            # in), whether or not they end up in the grid themselves.
            strokes, sketch_align = _voxelize_strokes(unit, log)
            # One file per source rather than a union: the worker mixes each at
            # its own weight and releases each at its own time, which a single
            # grid cannot express. Sharing a cube is what makes them
            # composable — both are written in the frame the strokes define.
            grids: dict[str, Any] = {}
            if config.get("sketch_inpaint"):
                grids["sketch"] = strokes
            if config.get("surface_inpaint"):
                if surface is None:
                    log(f"'{unit.label}': no predicted surface for this unit; "
                        "constraining with "
                        + ("the strokes alone" if config.get("sketch_inpaint")
                           else "nothing"))
                else:
                    field, header = surface
                    grids["surface"] = _voxelize_surface(
                        field, header, sketch_align,
                        float(config.get(
                            "surface_threshold", SURFACE_METHOD_THRESHOLD
                        )),
                        log,
                    )
            for name in list(grids):
                if int(grids[name].sum()) == 0:
                    log(f"'{unit.label}': the {name} constraint came out "
                        "empty — dropping it")
                    del grids[name]
            if not grids:
                raise ValueError(
                    "the constraint grid came out empty — nothing to inpaint "
                    "with"
                )
            log(f"'{unit.label}': constraining with " + ", ".join(
                f"{name} ({int(grid.sum())} of {OCCUPANCY_GRID ** 3} cells)"
                for name, grid in grids.items()
            ))
            for name, grid in grids.items():
                path = unit_dir / f"{name}.npy"
                np.save(path, grid)
                extra[name] = str(path)

        config_path = unit_dir / "config.json"
        config_path.write_text(json.dumps({
            **config,
            **extra,
            "views": [str(p) for p in images],
            "out": str(unit_dir / "mesh.glb"),
        }, indent=2))

        written, manifest = self._run_worker(
            config_path, unit_dir, unit.label, repo, python, defines, log,
            on_progress,
        )
        loaded = trimesh.load(written["final"], force="mesh")
        log(f"'{unit.label}': {len(loaded.vertices)} verts, "
            f"{len(loaded.faces)} faces")
        # before the fit, not after: stray fragments are noise in the
        # stroke-to-surface score the registration minimizes
        loaded = self._drop_small_components(loaded, min_component, unit, log)
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
            align = self._fit(loaded, unit, log)

        # The raw mesh is a second view of the same object, not a second
        # object: it takes the transform solved on the processed one rather
        # than its own, or the two would not sit on top of each other — and
        # the fragments it still has are exactly what would drag its own fit.
        raw = None
        if "raw" in written:
            raw = trimesh.load(written["raw"], force="mesh")
            if align is not None:
                raw.vertices = (
                    raw.vertices @ _align_rotation(align).T
                ) * align["scale"] + align["translation"]

        return UnitResult(
            mesh=loaded,
            raw=raw,
            frames=written.get("frames"),
            manifest=manifest,
            align=align,
        )

    def _bundle(self, result: "UnitResult", log: LogFn) -> bytes:
        return bundle_capture(result, log)

    def _fill_holes(self, choice: str, log: LogFn) -> bool:
        """Resolve the `auto` setting against this backend's capability.

        Explicit rather than silently automatic because the two backends
        otherwise produce differently-cleaned meshes, and this adapter exists
        partly so CUDA and AMD results can be compared — forcing `off` on both
        is how you make that comparison fair.
        """
        if choice == "on":
            return True
        if choice == "off":
            return False
        capable = bool(backend_method(METHOD_NAME).get("fill_holes", False))
        if not capable:
            log(f"fill_holes off: not usable on the {backend_name()} backend "
                "(see backends.json)")
        return capable

    def _drop_small_components(
        self, mesh: Any, fraction: float, unit: Unit, log: LogFn
    ) -> Any:
        return drop_small_components(mesh, fraction, unit, log)

    def _fit(
        self, mesh: Any, unit: Unit, log: LogFn
    ) -> Optional[dict[str, Any]]:
        return fit_to_strokes(mesh, unit, log)

    def _run_worker(
        self,
        config_path: Path,
        unit_dir: Path,
        label: str,
        repo: Path,
        python: Path,
        defines: dict[str, str],
        log: LogFn,
        on_progress: Callable[[float, str], None],
    ) -> tuple[dict[str, Path], Optional[dict[str, Any]]]:
        return run_worker(
            WORKER, config_path, unit_dir, label, repo, python, defines, log,
            on_progress,
        )
