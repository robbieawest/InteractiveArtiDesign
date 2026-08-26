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
    SURFACE_METHOD_MARGIN,
    ClientViewsConditioner,
    Conditioner,
    Unit,
    UnitResult,
    _voxelize_strokes,
    _voxelize_surface,
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

# Defaults for the two numbers that decide what is read off the prediction.
# Both are exposed as options rather than pinned here, and not because either
# default is in doubt: the editor has to surface the sketch itself for the
# image condition, ns2s caches a field on `(sketch, resolution, margin, blur)`,
# and a client asking for one blur while this adapter asks for another solves
# the same prediction twice. An option is how the two stay in step.
#
# 0.6 is NS2S's own level — `process_and_save` marches there — so it is the
# surface the network was trained to produce, and moving it is second-guessing
# the surfacing method rather than measuring it.
SURFACE_THRESHOLD = 0.6

# No pre-blur, and this one is worth stating because TRELLIS 1 defaults to
# 1.6 and it is wrong there too. Smoothing the probability field *before* thresholding moves the level
# set inward and deletes thin features outright rather than shrinking them: on
# a synthetic field with a 2-voxel plate, the shell at 0.6 falls from 3252
# cells to 1274 under a 1.6-voxel blur, and the plate keeps 3% of itself.
# Whatever the blur was buying (closing pinholes the network was unsure
# about), it was paying for it in geometry. Thickness is expressed by dilating
# the voxelized shell instead, where it is controllable and reversible.
SURFACE_BLUR = 0.0

# How far the shell is ever dilated, whatever multiple is asked for. A
# multiple that is not reached by here is asking for a solid rather than a
# thicker surface, and a solid is what shelling the prediction existed to
# avoid — stage 1 voxelizes surfaces, and the mesh decoder downstream pays per
# unit of surface area.
MAX_DILATION = 12


def _optional_float(value: Any) -> Optional[float]:
    """A float, or `None` for "leave the checkpoint's own value alone".

    `None` and any negative number both mean unset. The negative is there
    because the UI sends every declared option on every run, so there is no way
    for a float slider to express absence — -1 is the sentinel it uses, and the
    real range starts at 0.
    """
    if value is None or float(value) < 0:
        return None
    return float(value)


def _dilate(grid: Any, radius: int) -> Any:
    """Grow a binary grid by `radius` voxels under the 6-neighbour metric.

    Thickness rather than blur, because blurring a binary grid only means
    thresholding it again on the way into an encoder that was trained on
    binary grids — and where the level is cut then decides whether a thin
    sheet thickens or vanishes, which is the failure this whole path is
    avoiding. A dilation has no level: the radius is the thickness, in voxels,
    uniformly, and radius 0 is the shell itself.

    Six-neighbour (Manhattan) rather than the 3^3 max-pool used for the sketch
    mask, which would grow diagonally by sqrt(3) per step and make "6 voxels"
    mean something different along an edge than along an axis.
    """
    import numpy as np  # server env

    grown = grid.astype(bool)
    for _ in range(radius):
        step = grown.copy()
        for axis in range(3):
            for direction in (1, -1):
                shifted = np.roll(grown, direction, axis=axis)
                # roll wraps; the wrapped-in plane is outside the grid
                plane = [slice(None)] * 3
                plane[axis] = 0 if direction == 1 else -1
                shifted[tuple(plane)] = False
                step |= shifted
        grown = step
    return grown.astype(np.uint8)


class StrokeViewsConditioner(ClientViewsConditioner):
    """Raised three-quarter renders of the strokes.

    Same client renderer as TRELLIS 1, different numbers, and the differences
    are all facts about this model rather than preferences:

      * `count` defaults to 1 because TRELLIS.2 was trained on a single image
        and one view is what it is honestly measured on. It is a knob rather
        than a constant because the model's *interface* takes several without
        being taught to: the DINOv3 extractor already stacks a list and the
        flow model cross-attends to whatever cond it is handed. Above 1 the
        views are dealt one per model call — upstream TRELLIS 1's own
        'stochastic' algorithm, ported into `sketchflow/flow.py` rather than
        invented here.
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
        "Raised three-quarter renders of the strokes, rasterized by the "
        "editor and sent with the job."
    )
    # Declared on the adapter rather than here: `conditioner_params` gates a
    # conditioner's own parameters on a `conditioner` selector param, and this
    # adapter has no selector (there is one strategy), so a knob declared here
    # would be permanently disabled in the UI.
    params: list[dict[str, Any]] = []
    view_spec = {
        **ClientViewsConditioner.view_spec,
        "size": 1024,
        "count": 1,
        # radians: about 32 degrees up, 40 degrees round
        "pitch": 0.55,
        "yaw": 0.7,
        "overrides": {
            "views_count": "count",
            "views_layout": "layout",
        },
    }


CONDITIONERS: dict[str, Conditioner] = {
    conditioner.name: conditioner
    for conditioner in [StrokeViewsConditioner()]
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
            "name": "views_count",
            "label": "views",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 8,
            "step": 1,
            "help": "How many renders of the sketch the model conditions on. "
            "1 is what TRELLIS.2 was trained on. Above 1 the views are dealt "
            "out one per model call — TRELLIS 1's 'stochastic' algorithm, "
            "ported rather than invented — so no single view owns a step. "
            "That is aimed at the pose problem: one fixed image lets the "
            "model commit to a frame the sketch constraint cannot then argue "
            "with, because interpolating latents cannot rotate an object. "
            "The views are embedded once up front, so this costs nothing per "
            "step. Under a jumping schedule each cycle of a block sees a "
            "different view, which is what gives the resampling new evidence "
            "rather than the same image again.",
        },
        {
            "name": "views_layout",
            "label": "view layout",
            "type": "choice",
            "default": "ring",
            "choices": ["ring", "helix"],
            "help": "Where the cameras go when there is more than one. "
            "'ring' orbits at one elevation, so the views differ in yaw "
            "alone. 'helix' also climbs as it orbits — the views are less "
            "alike, which is worth more to a model that gets no camera poses "
            "than another view from the same height would be. Inert at one "
            "view, which has nowhere to climb to.",
        },
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
            "default": False,
            "help": "Constrain the structure stage with the strokes "
            "themselves. The sketch is voxelized to 64^3, encoded to the 16^3 "
            "latent grid with TRELLIS 1's sparse-structure encoder (loaded on "
            "demand and released before sampling), and mixed into the running "
            "latent at every step. This is the only channel that carries "
            "actual 3D geometry into the model — the image branch cannot, it "
            "reconstructs what it is shown and flattens the rest. Off by "
            "default, along with the surface below, so that the unconstrained "
            "run is what you get without asking and each source is switched "
            "on deliberately: the two constraints are separate experiments "
            "and the interesting comparisons are between them, not against a "
            "default that already has one of them on.",
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
            "name": "jump_delta",
            "label": "Experiment: jump length in t",
            "type": "float",
            "default": 0.1,
            "min": 0.0,
            "max": 0.5,
            "step": 0.05,
            "help": "RePaint resampling. How far back up the trajectory a "
            "jump goes, measured in t. The run is tiled into blocks: descend "
            "until this much t has been covered, redo that same span the "
            "number of times below, then move on. 0 turns jumping off. "
            "Measured in t rather than in steps because rescale_t makes the "
            "steps wildly non-uniform — 'back three steps' is a different "
            "amount of trajectory at every point in the run — and because in "
            "t it adapts on its own: at rescale_t 5 the nodes are dense at "
            "high t, so 0.1 spans four steps at the top and one at the "
            "bottom, putting the resampling where structure is being decided "
            "without anything having to place it there. Jumps only happen on "
            "a constrained run; with nothing pushing the state off the "
            "marginal there is nothing to reconcile.",
        },
        {
            "name": "jump_cycles",
            "label": "Experiment: jump cycles per block",
            "type": "int",
            "default": 5,
            "min": 0,
            "max": 20,
            "step": 1,
            "help": "How many times each block is redone — RePaint's "
            "jump_n_sample. This is PER BLOCK, not across the run: 5 means "
            "every block is traversed six times, and how many blocks there "
            "are is decided by the jump length above. Why any of this is "
            "needed: the model supplies a bounded amount of correction per "
            "unit of t, so a constraint that re-injects a disagreement on "
            "every step is fought with one step's worth of correction each "
            "time and never converges — it tracks the disagreement instead "
            "of resolving it. Extra passes at the same t buy correction "
            "budget without moving the time axis the checkpoint was tuned "
            "on, which is the only other way to get it. Costs a full "
            "guided pass each, so the run time scales with it.",
        },
        {
            "name": "jump_release",
            "label": "Experiment: stop jumping below t",
            "type": "float",
            "default": 0.6,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "Blocks whose top is below this do not jump. The tail of "
            "the run has no disagreement left to reconcile, so cycles spent "
            "there are cycles wasted. 0.6 is where the constraint's own ramp "
            "settles and where the guidance interval ends, so the "
            "constrained, guided and resampled windows coincide. Note that "
            "the schedule leaves almost no resolution down here: the only "
            "times below 0.5 are 0.3125 and 0, so anything in (0, 0.31] is "
            "indistinguishable from 0.",
        },
        {
            "name": "sketch_mask",
            "label": "Experiment: constraint region",
            "type": "choice",
            "default": "none",
            "choices": ["none", "touched", "dilated"],
            "enabledWhen": {"param": "sketch_inpaint", "equals": True},
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
            "name": "sketch_fraction",
            "label": "Experiment: sketch mix fraction at t=1",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "sketch_inpaint", "equals": True},
            "help": "How much of each constrained cell comes from the sketch "
            "rather than from the model's own clean estimate, at the start of "
            "sampling. It is the 'a' in mixed = (1-a)*estimate + a*sketch, so "
            "it is a share of an interpolation and has no meaning outside "
            "[0, 1]: at 1.0 with the unmasked mode the object simply IS the "
            "sketch, which is not an experiment, and at 0 there is no "
            "inpainting. Without a mask this is the whole of the constraint, "
            "which is why it is the knob that matters. It is not a strength "
            "in any free-scale sense and is deliberately not named one. "
            "Replaces the old global inpainting strength, which is still "
            "read as the default for both sources so older configs "
            "reproduce.",
        },
        {
            "name": "sketch_fraction_end",
            "label": "Experiment: sketch mix fraction at t=0.6",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "sketch_inpaint", "equals": True},
            "help": "The same fraction at t=0.6. The two are the endpoints of "
            "a line in t, followed below 0.6 as well rather than stopping "
            "there, and clamped to [0, 1]. Equal values hold the constraint "
            "flat, which is what it did before this existed. Lowering the "
            "end is the interesting direction: the sketch is a wireframe, so "
            "holding it at full fraction while geometry is being finalized "
            "is what makes the result look like one. A line that ends below "
            "where it starts crosses zero on its own some way below the "
            "anchor, which is a release with resolution at every step — the "
            "hard release below has none under 0.5, where the only times are "
            "0.3125 and 0.",
        },
        {
            "name": "sketch_onset",
            "label": "Experiment: hold the sketch off above t",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.005,
            "enabledWhen": {"param": "sketch_inpaint", "equals": True},
            "help": "Do not apply the sketch while t is still above this — "
            "the mirror of the release below, and 1.0 is off. The case for "
            "it: the image branch has no pose input, so its coarse structure "
            "comes out at whatever orientation it infers, while the sketch "
            "grid is fixed in the strokes' own frame. If the two disagree, "
            "mixing them at the very top averages two poses and gives "
            "neither. Letting the image settle a pose first, for a step or "
            "two only, is the whole of the intent. The first two times on "
            "the shipped schedule are 0.9821 and 0.9615, so it takes a value "
            "below 0.9615 to skip both — hence the fine step.",
        },
        {
            "name": "sketch_release",
            "label": "Experiment: release the sketch below t",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "sketch_inpaint", "equals": True},
            "help": "Stop applying the constraint once the sampler's time "
            "drops below this. Time runs 1 -> 0, so releasing frees the END "
            "of sampling: hold the strokes while topology is being decided, "
            "then let the prior finish the geometry. Note that rescale_t "
            "bunches the steps up at high t, so a release of 0.5 frees fewer "
            "steps than half of them, and 1.0 turns the constraint off.",
        },
        {
            "name": "surface_inpaint",
            "label": "Experiment: surface inpainting",
            "type": "bool",
            "default": False,
            "help": "Add a predicted surface to the constraint. The strokes "
            "are surfaced with NeuralSketch2Surf, the shell of that "
            "prediction is voxelized into the same 64^3 cube the strokes "
            "define, and it is mixed in alongside them — surface first, "
            "strokes second, so the strokes win where the two overlap. This "
            "is the thing the sketch constraint cannot be: a wireframe is "
            "off-distribution as a shape whatever space it is mixed in, and a "
            "closed surface is not. Unlike the strokes it is applied THICK "
            "and thinned as sampling proceeds — see below — because a "
            "predicted surface is a guess about where the boundary is, and a "
            "one-voxel shell states that guess far more precisely than the "
            "surfacer meant it.",
        },
        {
            "name": "surface_thickness",
            "label": "Experiment: surface thickness at t=1",
            "type": "float",
            "default": 6.0,
            "min": 0.0,
            "max": 12.0,
            "step": 0.5,
            "enabledWhen": {"param": "surface_inpaint", "equals": True},
            "help": "Voxels the predicted shell is dilated by at the start of "
            "sampling. The constraint then says 'the boundary is somewhere in "
            "this slab' instead of 'the boundary is exactly here', which is "
            "the honest reading of a prediction the surfacer itself is unsure "
            "about. Thickness is the ONLY way to say that: the "
            "sparse-structure VAE was trained on binary grids and its decoder "
            "saturates, so a soft occupancy is not representable at all — a "
            "blurred grid only gets thresholded again on the way in.",
        },
        {
            "name": "surface_thickness_end",
            "label": "Experiment: surface thickness at t=0.6",
            "type": "float",
            "default": 2.0,
            "min": 0.0,
            "max": 12.0,
            "step": 0.5,
            "enabledWhen": {"param": "surface_inpaint", "equals": True},
            "help": "Thickness at t=0.6. Together with the value above this "
            "fixes a line in t, which is then followed below 0.6 as well "
            "rather than stopping there — it flattens out at one voxel, which "
            "it reaches at t=0.5. Note the 0.6 here is a fixed anchor and is "
            "NOT the release below: they share a default so that the "
            "constraint ends where the ramp does, but moving the release "
            "leaves the ramp exactly where it was, so the two can be swept "
            "independently.",
        },
        {
            "name": "surface_fraction",
            "label": "Experiment: surface mix fraction at t=1",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "surface_inpaint", "equals": True},
            "help": "The surface's own share of the mix at the start of "
            "sampling, independent of the sketch's. Separate because the two "
            "are separate experiments and want opposite schedules: the "
            "surface is a closed shape and is the thing that can safely be "
            "held, while the sketch is a wireframe and is the thing that "
            "should fade. Note this is absolute, unlike the weight below, "
            "which only decides the ratio between the two where they "
            "overlap.",
        },
        {
            "name": "surface_fraction_end",
            "label": "Experiment: surface mix fraction at t=0.6",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "surface_inpaint", "equals": True},
            "help": "The same fraction at t=0.6, the other end of a line in "
            "t clamped to [0, 1]. Equal values hold it flat. This ramp is "
            "independent of the thickness ramp even though both are anchored "
            "at 0.6: thickness says how precisely the surface states where "
            "the boundary is, this says how hard that statement is applied.",
        },
        {
            "name": "surface_onset",
            "label": "Experiment: hold the surface off above t",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.005,
            "enabledWhen": {"param": "surface_inpaint", "equals": True},
            "help": "Do not apply the surface while t is still above this; "
            "1.0 is off. Same argument as the sketch's, and the same "
            "resolution caveat — the first two times are 0.9821 and 0.9615.",
        },
        {
            "name": "surface_release",
            "label": "Experiment: release the surface below t",
            "type": "float",
            "default": 0.6,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "enabledWhen": {"param": "surface_inpaint", "equals": True},
            "help": "Stop applying the surface once the sampler's time drops "
            "below this — the surface says what shape to head toward while "
            "topology is being decided, then gets out of the way and lets the "
            "prior and the strokes finish the geometry. Its share goes to the "
            "sources still applying rather than back to the prior. Defaults "
            "to 0.6 rather than 0 because that is where the thickness ramp "
            "lands at one voxel's slab and the guess has said what it has to "
            "say; 0 holds it all the way down.",
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
            "against the sketch weight. Only the ratio matters and it does "
            "nothing where the surface is the only source in a cell — how "
            "hard the constraint is applied is the strength, not this. "
            "Lowering it is the sensible direction: the strokes are what the "
            "user drew, this is a guess about everything between them.",
        },
        {
            "name": "surfaced_condition",
            "label": "Experiment: surfaced image condition",
            "type": "bool",
            "default": False,
            "help": "Condition on a render of a predicted surface instead of "
            "a render of the strokes. The editor surfaces the sketch with "
            "NeuralSketch2Surf first and renders that solid, opaque and "
            "shaded like any other geometry, from the same camera the stroke "
            "render uses. This is the other half of the wireframe problem: "
            "the image branch reconstructs what it is shown, and what it has "
            "been shown so far is line art, which it faithfully turns into "
            "thin geometry. Independent of surface inpainting — that puts the "
            "prediction into the sampler, this puts it in front of the "
            "camera — and when both are on the prediction is solved once and "
            "shared. Whole-object only: a part-based run keeps its stroke "
            "renders.",
        },
        {
            "name": "surface_smooth",
            "label": "Experiment: smooth the surfaced condition",
            "type": "bool",
            "default": False,
            "enabledWhen": {"param": "surfaced_condition", "equals": True},
            "help": "Run NeuralSketch2Surf's own post-process before the "
            "surface is rendered: Laplacian smoothing balanced against the "
            "sketch curves, hole filling, normal repair, Taubin smoothing. "
            "Marching cubes on a 112^3 field leaves stair-stepping that has "
            "nothing to do with the shape and the image branch has no way to "
            "know that. Affects the render only — the inpainting constraint "
            "is built from the field, which no mesh post-process touches.",
        },
        {
            "name": "surface_threshold",
            "label": "Experiment: surface threshold",
            "type": "float",
            "default": SURFACE_THRESHOLD,
            "min": 0.05,
            "max": 0.95,
            "step": 0.05,
            "help": "The probability the prediction is read at — the level "
            "its shell is taken at for inpainting, and the level marching "
            "cubes runs at for the surfaced condition, so both boxes read it. "
            "The default is NS2S's own: `process_and_save` marches at 0.6, so "
            "this is the surface the network was trained to produce, and "
            "moving it is second-guessing the surfacing method rather than "
            "measuring it.",
        },
        {
            "name": "surface_blur",
            "label": "Experiment: surface blur",
            "type": "float",
            "default": SURFACE_BLUR,
            "min": 0.0,
            "max": 4.0,
            "step": 0.1,
            "help": "Gaussian smoothing of the probability field, in voxels, "
            "applied before anything is read off it. TRELLIS 1 defaults this "
            "to 1.6 and that is a mistake worth not repeating: blurring "
            "before a threshold moves the level set inward and deletes thin "
            "features rather than shrinking them. Measured on a synthetic "
            "field, the shell at 0.6 falls from 3252 cells to 1274 under a "
            "1.6-voxel blur, and a 2-voxel plate keeps 3% of itself. Hence 0. "
            "It is still a knob because the editor has to ask for the same "
            "blur this adapter does or the prediction is solved twice — ns2s "
            "caches on it.",
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
            "name": "ss_guidance_rescale",
            "label": "TRELLIS.2: structure guidance rescale",
            "type": "float",
            "default": -1.0,
            "min": -1.0,
            "max": 1.0,
            "step": 0.05,
            "help": "CFG rescale for the structure stage — new in TRELLIS.2, "
            "TRELLIS 1 has no equivalent. Guidance at 7.5 extrapolates far "
            "outside the two predictions and inflates the spread of the clean "
            "estimate it implies; this divides that back out, blending the "
            "corrected estimate in at this weight. 0 is off, 1 is fully "
            "renormalized, and -1 means leave the checkpoint's own value "
            "alone (upstream's demo uses 0.7 here).",
        },
        {
            "name": "ss_rescale_t",
            "label": "TRELLIS.2: structure step schedule",
            "type": "float",
            "default": -1.0,
            "min": -1.0,
            "max": 6.0,
            "step": 0.1,
            "help": "How the flow steps are distributed over t. 1 is uniform; "
            "higher bunches them toward t=1, where structure is decided, so "
            "the last step covers a longer stretch of the trajectory on its "
            "own. TRELLIS.2 ships 5 for this stage against TRELLIS 1's 3, "
            "which is why its coarse phase runs so much longer before the "
            "shape resolves. -1 leaves the checkpoint's own value alone.",
        },
        {
            "name": "slat_rescale_t",
            "label": "TRELLIS.2: latent step schedule",
            "type": "float",
            "default": -1.0,
            "min": -1.0,
            "max": 6.0,
            "step": 0.1,
            "help": "The same distribution for the latent stage, where "
            "upstream uses 3. -1 leaves the checkpoint's own value alone.",
        },
        {
            "name": "slat_guidance_rescale",
            "label": "TRELLIS.2: latent guidance rescale",
            "type": "float",
            "default": -1.0,
            "min": -1.0,
            "max": 1.0,
            "step": 0.05,
            "help": "The same term for the latent stage, where upstream's "
            "demo uses 0.5. -1 leaves the checkpoint's own value alone.",
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
            # None means "leave the checkpoint's own value alone" — the
            # schedule only overrides what is not None. TRELLIS.2 ships a CFG
            # rescale term that TRELLIS 1 has no equivalent for, so there is no
            # sensible default to hardcode here.
            "ss_guidance_rescale": _optional_float(
                options.pop("ss_guidance_rescale", None)
            ),
            "ss_rescale_t": _optional_float(options.pop("ss_rescale_t", None)),
            "slat_rescale_t": _optional_float(
                options.pop("slat_rescale_t", None)
            ),
            "slat_guidance_rescale": _optional_float(
                options.pop("slat_guidance_rescale", None)
            ),
            "preprocess": bool(options.pop("preprocess", True)),
            "no_image_cond": bool(options.pop("no_image_cond", False)),
            "sketch_inpaint": bool(options.pop("sketch_inpaint", False)),
            "constraint_mix": str(options.pop("constraint_mix", "x0")),
            # Kept as the fallback the per-source fractions default to, so a
            # config written before they existed still reproduces. Nothing
            # declares it any more; only an older saved config carries it.
            "constraint_strength": float(
                options.pop("constraint_strength", 0.5)
            ),
            "jump_delta": float(options.pop("jump_delta", 0.1)),
            "jump_cycles": int(options.pop("jump_cycles", 5)),
            "jump_release": float(options.pop("jump_release", 0.6)),
            "sketch_mask": str(options.pop("sketch_mask", "none")),
            "sketch_weight": float(options.pop("sketch_weight", 1.0)),
            "sketch_fraction": float(options.pop("sketch_fraction", 0.5)),
            "sketch_fraction_end": float(
                options.pop("sketch_fraction_end", 0.5)
            ),
            "sketch_onset": float(options.pop("sketch_onset", 1.0)),
            "sketch_release": float(options.pop("sketch_release", 0.0)),
            "surface_inpaint": bool(options.pop("surface_inpaint", False)),
            "surface_thickness": float(options.pop("surface_thickness", 6.0)),
            "surface_thickness_end": float(
                options.pop("surface_thickness_end", 2.0)
            ),
            "surface_weight": float(options.pop("surface_weight", 1.0)),
            "surface_fraction": float(options.pop("surface_fraction", 0.5)),
            "surface_fraction_end": float(
                options.pop("surface_fraction_end", 0.5)
            ),
            "surface_onset": float(options.pop("surface_onset", 1.0)),
            "surface_release": float(options.pop("surface_release", 0.6)),
            "surface_threshold": float(
                options.pop("surface_threshold", SURFACE_THRESHOLD)
            ),
            "surface_blur": float(options.pop("surface_blur", SURFACE_BLUR)),
            # Acted on by the editor, which has to surface and render before
            # it can submit at all — the job protocol is one-way, so an
            # adapter cannot hand the client geometry mid-run and wait for
            # renders of it. Recorded here so a run's config says what its
            # conditioning image actually was.
            "surfaced_condition": bool(
                options.pop("surfaced_condition", False)
            ),
            "surface_smooth": bool(options.pop("surface_smooth", False)),
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

        # Before the card is claimed, not after: surfacing runs on the GPU
        # too and its worker sits on ~13GB that the release below reclaims.
        # Every unit is predicted in one call so the model loads once.
        surfaces: list[Optional[tuple[Any, dict[str, Any]]]] = [None] * len(units)
        if config["surface_inpaint"]:
            report(0.005, "predicting the surface for the constraint")
            surfaces = self._surface_grids(
                units, job_dir, float(config["surface_blur"]), log
            )

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
            # Whatever the worker hands over before it finishes. The mesh
            # decode is the last step and the one that fails, and the capture
            # is complete long before it, so a failed run still has a flow
            # view to show — which is the one thing that makes the failure
            # readable rather than just fatal.
            partial: dict[str, Any] = {}
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
                    python, defines, surfaces[index], partial, log,
                    lambda frac, msg, base=base: report(base + span * frac, msg),
                )
            except ValueError as exc:
                self._emit_partial(partial, emit, log)
                log(f"skipping '{unit.label}': {exc}")
                continue
            except Exception:
                self._emit_partial(partial, emit, log)
                raise
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

    @staticmethod
    def _emit_partial(
        partial: dict[str, Any], emit: EmitFn, log: LogFn
    ) -> None:
        """Emit the capture of a run that did not finish, if there is one.

        `align` is whatever was known before the worker started, which for a
        constrained run is the exact answer — the constraint's own frame — and
        for an unconstrained one is nothing, since that alignment is fitted
        from a mesh this run never produced. The client leaves the lattice at
        the origin in that case rather than inventing one.
        """
        if "frames" not in partial:
            return
        from types import SimpleNamespace

        log("the run did not finish; emitting the flow capture it had already "
            "completed")
        emit("flow", bundle_capture(SimpleNamespace(
            frames=partial["frames"],
            manifest=partial.get("manifest"),
            align=partial.get("align"),
        ), log), "trellis-frames")

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

        The surfacing method is another adapter called as a component: it owns
        how its network is run and what frame its output is in, and this only
        needs the field and the way back to world. A failure is not fatal —
        the run falls back to the strokes, which is the control condition
        anyway.

        Positional rather than keyed by label: two parts may carry the same
        name, and a dict would quietly hand one part's surface to another.

        Both sides of the card are handled here. Surfacing is a GPU job too,
        so it evicts the other methods before it loads, and its own worker is
        stopped as soon as the grids are in hand rather than left for the
        general sweep — the process holds its model until it is reaped, and a
        4B pipeline loads next.

        `blur` has to be the value the editor asked for when it surfaced for
        the image condition, or this misses that cache and solves again:
        ns2s keys its field cache on `(sketch, resolution, margin, blur)`.
        `SURFACE_METHOD_MARGIN` is pinned to the client's own constant for the
        same reason. At the default blur of 0 what comes back is the raw
        prediction — see `SURFACE_BLUR`.
        """
        from . import ns2s

        keys = [f"{index}: {unit.label}" for index, unit in enumerate(units)]
        # Set only if the method actually has to predict something: a field
        # already in ns2s's cache costs nothing and releases nothing.
        predicted = False

        def before_predict() -> None:
            nonlocal predicted
            predicted = True
            release_other_workers(ns2s.METHOD_NAME, log)

        try:
            grids = ns2s.probability_grids(
                {key: {"strokes": unit.strokes}
                 for key, unit in zip(keys, units)},
                job_dir, SURFACE_METHOD_MARGIN, blur, log,
                before_predict,
            )
        except Exception as exc:
            log(f"surface inpainting: prediction failed ({type(exc).__name__}: "
                f"{exc}); continuing without a surface")
            return [None] * len(units)
        finally:
            # Logged either way: "was the surfacing model still resident when
            # TRELLIS.2 started" is the first question an out-of-memory run
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
                    "is still holding the GPU and TRELLIS.2 may run out of "
                    "memory")
        return [grids.get(key) for key in keys]

    @staticmethod
    def _surface_stack(
        field: Any,
        header: dict[str, Any],
        align: dict[str, Any],
        config: dict[str, Any],
        log: LogFn,
    ) -> Any:
        """The predicted shell and every dilation of it the schedule can want.

        One file holding `[radius, 64, 64, 64]`, because the thickness the
        sampler applies changes with `t` and encoding is done once before
        sampling starts. Which radius is used when is the sampler's business
        (`sketchflow/constraint.py`); which radii *exist* is this side's, and
        it is cheap to be generous — a radius is 256KB and the whole stack
        encodes in one batched pass of a small VAE.

        Grown until the thickest multiple the ramp asks for is *reached*, plus
        one, since a fractional thickness is a blend of the two radii
        bracketing it. How many radii that takes is a fact about the shape,
        not a constant: thickness is expressed as a multiple of the shell's
        own cell count, and how fast a dilation grows depends on the surface
        it grows from. A compact blob doubles in a single step; a sparse,
        spindly one takes several.
        """
        import numpy as np  # server env

        shell = _voxelize_surface(
            field, header, align,
            float(config.get("surface_threshold", SURFACE_THRESHOLD)), log
        )
        cells = int(shell.sum())
        if cells == 0:
            return shell[None]
        thickest = max(
            float(config.get("surface_thickness", 6.0)),
            float(config.get("surface_thickness_end", 2.0)),
        )

        stack = [shell]
        while len(stack) <= MAX_DILATION:
            if stack[-1].sum() / cells >= thickest:
                break
            stack.append(_dilate(shell, len(stack)))
        # one past, for the blend; capped, because a multiple large enough to
        # need more than this is asking for a solid, not a thicker surface
        if len(stack) <= MAX_DILATION:
            stack.append(_dilate(shell, len(stack)))
        else:
            log(f"surface constraint: stopped at radius {MAX_DILATION} "
                f"without reaching {thickest:g}x the shell — the ramp will "
                "clamp to the thickest dilation written")

        log(f"surface constraint: {len(stack)} dilation(s) of a "
            f"{cells}-cell shell, " + ", ".join(
                f"r{radius}={int(grid.sum())} ({grid.sum() / cells:.1f}x)"
                for radius, grid in enumerate(stack)))
        return np.stack(stack)

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
        partial: dict[str, Any],
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
        if config.get("sketch_inpaint") or config.get("surface_inpaint"):
            # The strokes are voxelized either way: they define the cube the
            # constraint is written in — and so the frame the result is read
            # in — whether or not they end up in the grid themselves. That is
            # also the answer for the surface: it is mapped through the
            # strokes' own normalization, so adding it does not move the frame
            # and the placement shortcut below still holds exactly.
            strokes, sketch_align = _voxelize_strokes(unit, log)
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
                    grids["surface"] = self._surface_stack(
                        field, header, sketch_align, config, log
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
                # the surface is a stack of dilations, so report its shell
                f"{name} ({int((grid[0] if grid.ndim == 4 else grid).sum())} "
                f"of {OCCUPANCY_GRID ** 3} cells)"
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

        # Known before the worker starts, so a capture emitted from a failed
        # run still lands on the drawing.
        partial["align"] = sketch_align if fit else None

        written, manifest = run_worker(
            WORKER, config_path, unit_dir, unit.label, repo, python, defines,
            log, on_progress,
            on_frames=lambda path, got: partial.update(
                frames=path, manifest=got
            ),
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
