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

import base64
import json
import re
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .base import EmitFn, LogFn, ProgressFn, SurfacingAdapter
from .common import (
    JOBS_DIR,
    SERVER_DIR,
    backend_method,
    backend_name,
    group_strokes_by_part,
    method_env,
    prune_job_dirs,
    release_other_workers,
    resolve_path,
    spawn,
)

METHOD_NAME = "trellis"
WORKER = Path(__file__).resolve().parent / "trellis_worker.py"

# The weights. Pulled from the HF hub into ~/.cache/huggingface on first use
# (~2.9GB, plus ~1.2GB of DINOv2 into the torch hub cache) — nothing here
# downloads them explicitly, `from_pretrained` does it.
DEFAULT_MODEL = "microsoft/TRELLIS-image-large"


# --- conditioning ---------------------------------------------------------


@dataclass
class Unit:
    """One thing to surface: the whole sketch, or a single part."""

    label: str
    strokes: list[dict[str, Any]]
    # How the client keys this unit's conditioning views: the part id, not
    # `label`. Part names are user-typed and not unique, so two parts called
    # "wheel" would collide into one render set and both units would surface
    # the same geometry. The label stays the name, for logs and for the
    # partial-result names the viewer shows.
    key: str = "sketch"


@dataclass
class UnitResult:
    """What one generation produced: always a mesh in sketch coordinates,
    plus whatever the interactive options asked to be kept."""

    mesh: Any
    raw: Optional[Any] = None
    frames: Optional[Path] = None
    manifest: Optional[dict[str, Any]] = None
    # the similarity that put the mesh where it is, so the occupancy lattice
    # (which is the unit cube the mesh was generated in) can follow it
    align: Optional[dict[str, Any]] = None


def _align_rotation(align: dict[str, Any]) -> Any:
    """The 3x3 rotation out of a flattened align record."""
    import numpy as np  # server env

    return np.asarray(align["rotation"], dtype=float).reshape(3, 3)


class Conditioner(ABC):
    """Turns a unit's strokes into the conditioning images TRELLIS consumes.

    Deliberately an interface with one implementation so far. The multi-view
    stroke render below is a guess at what a 2D foundation model can make of
    line art — it is off-distribution for a model trained on rendered solids,
    and the alternatives (silhouettes, inflated proxy renders, depth from a
    cheap surfacer, posed views paired with a structure prior) are different
    experiments rather than tweaks to this one. Adding one means a new class
    and a line in CONDITIONERS; nothing else here changes.
    """

    name: str
    label: str
    help: str = ""
    # extra parameters, surfaced by the adapter with a `<name>_` prefix and
    # greyed out unless this conditioner is the selected one
    params: list[dict[str, Any]] = []
    # what the client should render, if this strategy wants images at all
    # (see SurfacingAdapter.view_spec). None means it builds its input some
    # other way and the client sends nothing.
    view_spec: Optional[dict[str, Any]] = None

    @abstractmethod
    def prepare(
        self,
        unit: Unit,
        options: dict[str, Any],
        work_dir: Path,
        log: LogFn,
    ) -> list[Path]:
        """Write this unit's conditioning images into `work_dir` and return
        them in view order. Raise ValueError if this unit cannot be
        conditioned (the adapter skips it and carries on with the rest)."""


# --- debug mirror ---------------------------------------------------------

# The conditioning renders are the only thing the model actually sees of the
# sketch, so when a run comes back wrong they are the first thing to look at —
# but they live in a per-job scratch directory named after a uuid, which is
# pruned, and on the AMD fork a crash in the pipeline is common enough that
# "find the job dir of the run that died" was the slow part of every debug
# loop. So they are also copied to one fixed path, written the moment each PNG
# is decoded (before the worker is spawned at all, hence before anything can
# fail), and holding exactly one run: the directory is emptied when a run
# starts, so whatever is in there is always the latest attempt.
#
# AMD only, deliberately: this is a debugging aid for the fork, and on CUDA it
# would be an unexplained directory appearing in the server tree.
DEBUG_VIEWS_DIR = SERVER_DIR / "debug" / "latest"
DEBUG_BACKEND = "rocm"


def debug_views_dir(log: LogFn) -> Optional[Path]:
    """Empty and return the debug mirror, or None if this run should not
    write one. Never raises: a debugging aid must not be able to fail a run."""
    if backend_name() != DEBUG_BACKEND:
        return None
    try:
        shutil.rmtree(DEBUG_VIEWS_DIR, ignore_errors=True)
        DEBUG_VIEWS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log(f"warning: could not prepare {DEBUG_VIEWS_DIR}: {exc}")
        return None
    log(f"conditioning renders mirrored to {DEBUG_VIEWS_DIR}")
    return DEBUG_VIEWS_DIR


def _mirror_view(debug_dir: Optional[Path], unit: "Unit", name: str,
                 data: bytes, log: LogFn) -> None:
    if debug_dir is None:
        return
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", unit.label).strip("_") or "unit"
    try:
        target = debug_dir / slug
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(data)
    except OSError as exc:
        log(f"warning: could not mirror {name} to {debug_dir}: {exc}")


def _decode_image(value: str) -> bytes:
    """A PNG from the client: a `data:image/png;base64,...` URL (what a canvas
    toDataURL produces) or bare base64."""
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    return base64.b64decode(value)


class ClientViewsConditioner(Conditioner):
    """Multi-view renders of the strokes, rasterized by the browser.

    The renders are made client-side (src/engine/strokeViews.ts) rather than
    here because the app already owns a three.js view of the document, and a
    second renderer in python would be a second thing to keep honest about
    stroke width, framing and pose. They arrive as PNG data URLs in
    `options["views"]`, either

        [url, url, ...]                     one set, for the whole sketch
        {"<unit label>": [url, ...], ...}   a set per part, for part-based runs

    The style those renders need is not arbitrary, and it is a fact about
    TRELLIS rather than about drawing — so it is declared here in `view_spec`
    and the client's renderer stays generic:

      * `strokeColor` is light because preprocess_image ends with
        `output[:, :, :3] * output[:, :, 3:4]`, premultiplying alpha onto
        BLACK. Dark strokes composite to nothing.
      * `strokeThickness` is a fraction of the sketch's bounding radius, sized
        for roughly 6px at 518. Views are cropped to the alpha bbox and
        resampled with LANCZOS, which a hairline barely survives; and DINOv2
        sees 518/14 = 37x37 patches, so a thin wireframe leaves almost every
        patch empty. Tubes also carry shading, closer to the solid objects the
        model was trained on.
      * `size` is 518 because that is what preprocess_image resizes to;
        rendering there avoids a resample.
      * `margin` is loose because preprocess_image re-crops to the alpha bbox
        at 1.2x anyway — this only has to keep the subject off the edge.
      * `count` is small because run_multi_image takes no camera poses and
        reconciles views from their tokens alone, so extra unposed views add
        ambiguity faster than evidence.
      * `layout` is how the views are spread. Both count and layout are
        exposed as parameters (`overrides` below) because they are the
        experiment here, not a property of the model: whether a fifth view
        helps, and whether looking down beats going round, are the questions
        this conditioner exists to ask. The render *style* stays fixed —
        colour, thickness and size follow from what preprocess_image and
        DINOv2 do, and are not a matter of taste.
    """

    name = "views"
    label = "Client stroke renders"
    help = (
        "Multi-view renders of the strokes, rasterized by the editor and sent "
        "with the job. This is the plain experiment: show the model the "
        "sketch from a few angles and let it infer a solid."
    )
    view_spec = {
        "size": 518,
        "count": 4,
        "pitch": 0.35,
        "layout": "ring",
        "pitchMax": 1.2,
        "strokeColor": "#dcdcdc",
        "strokeThickness": 0.012,
        "margin": 1.15,
        # param name (as the adapter prefixes it) -> ViewSpec field. The
        # client resolves these against the job's options, so the knobs reach
        # the renderer without it knowing anything about TRELLIS.
        "overrides": {
            "views_count": "count",
            "views_layout": "layout",
        },
    }
    params = [
        {
            "name": "count",
            "label": "views",
            "type": "int",
            "default": 4,
            "min": 1,
            "max": 12,
            "step": 1,
            "help": "How many renders the model conditions on. Nothing in "
            "TRELLIS caps this, but in 'stochastic' mode the views are dealt "
            "out one per sampling step, so more views than steps are never "
            "used at all and fewer steps each is the price of more views. "
            "'multidiffusion' uses every view every step, at a cost linear in "
            "this number.",
        },
        {
            "name": "layout",
            "label": "view layout",
            "type": "choice",
            "default": "ring",
            "choices": ["ring", "helix"],
            "help": "Where the cameras go. 'ring' orbits at one elevation, so "
            "the views differ in yaw alone. 'helix' also climbs as it orbits, "
            "ending looking down at the sketch — the views are less alike, "
            "which is worth more to a model that gets no camera poses than "
            "another view from the same height would be.",
        },
    ]

    def prepare(
        self,
        unit: Unit,
        options: dict[str, Any],
        work_dir: Path,
        log: LogFn,
    ) -> list[Path]:
        views = options.get("views")
        if not views:
            raise RuntimeError(
                "the 'views' conditioner needs rendered views, but the job "
                "carried none — the client must render the sketch (see "
                "src/engine/strokeViews.ts) and pass them as options.views"
            )
        if isinstance(views, dict):
            images = views.get(unit.key)
            if not images:
                raise ValueError(
                    f"no views were sent for '{unit.label}' (key "
                    f"{unit.key!r}; got: {', '.join(sorted(views)) or 'nothing'})"
                )
        else:
            images = views

        debug_dir = options.get("debug_dir")
        paths: list[Path] = []
        for index, value in enumerate(images):
            name = f"view_{index:02d}.png"
            path = work_dir / name
            try:
                data = _decode_image(value)
            except Exception as exc:
                raise ValueError(f"view {index} is not decodable: {exc}")
            path.write_bytes(data)
            # mirrored here rather than after the loop, so a later view that
            # fails to decode still leaves the good ones to look at
            _mirror_view(debug_dir, unit, name, data, log)
            paths.append(path)
        if not paths:
            raise ValueError("no views to condition on")
        log(f"'{unit.label}': {len(paths)} conditioning view(s)")
        return paths


CONDITIONERS: dict[str, Conditioner] = {
    conditioner.name: conditioner
    for conditioner in [ClientViewsConditioner()]
}


def conditioner_params() -> list[dict[str, Any]]:
    """Every conditioner's own parameters, prefixed and gated on it being the
    selected one — the same trick sf3d uses for its proxy methods, so a knob
    belonging to an unselected strategy is visible but inert rather than
    missing."""
    out: list[dict[str, Any]] = []
    for name, conditioner in CONDITIONERS.items():
        for param in conditioner.params:
            copied = dict(param)
            copied["name"] = f"{name}_{param['name']}"
            copied["label"] = f"{name}: {param['label']}"
            copied["enabledWhen"] = {"param": "conditioner", "equals": name}
            out.append(copied)
    return out


# --- sketch voxelization --------------------------------------------------

# Edge of the occupancy grid the sparse-structure VAE works at. The encoder
# wants [B, 1, 64, 64, 64]; the decoder produces the same.
OCCUPANCY_GRID = 64
# Spacing, in voxels, at which a polyline is resampled before rasterizing. Below
# 1 no cell along the segment can be skipped; 0.4 leaves margin for the
# diagonal case without making the point count silly.
STROKE_SAMPLE_SPACING = 0.4
# How much of the cube's width the sketch's longest axis is scaled to. TRELLIS
# objects are normalized to fill the unit cube, so the sketch should too — but
# not to the very edge, since the model routinely adds volume the drawing never
# had (a thicker back, a base) and that volume needs somewhere to go.
CUBE_FILL = 0.9

# Where the surfacing run that feeds the constraint starts from. The threshold
# is NS2S's own default; the blur is what turns a shell full of pinholes into
# one a level set can be taken of. The two interact — smoothing moves the
# level set, and above 0.5 it moves inward — so they are one pair of knobs
# rather than two independent ones.
SURFACE_METHOD_THRESHOLD = 0.6
SURFACE_METHOD_BLUR = 1.6
# NS2S's default grid margin — the surface is allowed to bulge outside the
# strokes, and this is the room it has to do it in.
SURFACE_METHOD_MARGIN = 1.2


def _voxelize_strokes(unit: "Unit", log: LogFn) -> tuple[Any, dict[str, Any]]:
    """The unit's strokes as a 64^3 binary grid in TRELLIS's frame, plus the
    similarity that maps that cube back onto the drawing.

    Two changes of basis are folded together here, and both have to match what
    the rest of the pipeline already does or the constraint lands rotated
    inside the volume it is meant to constrain.

    *Normalization.* The sketch is centred on its bounding box and scaled
    uniformly so its longest axis spans `CUBE_FILL` of the cube. Uniform, not
    per-axis: anisotropic scaling would hand the model a sheared object and the
    prior would complete a sheared one back.

    *Handedness.* The document is y-up, TRELLIS is z-up. `main` in the worker
    exports meshes through `axes = [[1,0,0],[0,0,-1],[0,1,0]]`, i.e. world
    (x, y, z) = (x_t, z_t, -y_t). This is that map inverted — x_t = x_w,
    y_t = -z_w, z_t = y_w — so the grid written here is in the same frame the
    decoder's output is read in.

    The returned align is the exact inverse of the normalization (identity
    rotation, since the strokes define the frame rather than being searched
    for in it), in the same form `_fit` returns: world = scale * R @ v + t,
    for a mesh already in the y-up unit cube.
    """
    import numpy as np  # server env

    points = [
        np.asarray(s["points"], dtype=float)
        for s in unit.strokes if len(s.get("points") or []) > 0
    ]
    if not points:
        raise ValueError("no stroke points to build a sketch constraint from")

    every = np.vstack(points)
    low, high = every.min(axis=0), every.max(axis=0)
    centre = (low + high) / 2
    extent = float(np.max(high - low))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("the strokes have no extent to normalize")
    scale = extent / CUBE_FILL  # world units per unit of the cube

    grid = np.zeros((OCCUPANCY_GRID,) * 3, dtype=np.uint8)
    for stroke in points:
        cube = (stroke - centre) / scale
        # y-up world -> z-up TRELLIS, then cube coords -> voxel coords
        trellis = np.column_stack([cube[:, 0], -cube[:, 2], cube[:, 1]])
        voxel = (trellis + 0.5) * OCCUPANCY_GRID

        if len(voxel) == 1:
            dense = voxel
        else:
            # resample along arc length so no cell on a long segment is
            # stepped over; a stroke is a polyline, not a point cloud
            steps = np.linalg.norm(np.diff(voxel, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(steps)])
            if arc[-1] <= 0:
                dense = voxel[:1]
            else:
                count = max(2, int(np.ceil(arc[-1] / STROKE_SAMPLE_SPACING)) + 1)
                where = np.linspace(0.0, arc[-1], count)
                dense = np.column_stack(
                    [np.interp(where, arc, voxel[:, axis]) for axis in range(3)]
                )

        index = np.clip(
            np.floor(dense).astype(int), 0, OCCUPANCY_GRID - 1
        )
        grid[index[:, 0], index[:, 1], index[:, 2]] = 1

    occupied = int(grid.sum())
    log(f"'{unit.label}': sketch voxelized to {occupied} of "
        f"{OCCUPANCY_GRID ** 3} cells "
        f"({100 * occupied / OCCUPANCY_GRID ** 3:.2f}%)")
    align = {
        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "scale": float(scale),
        "translation": [float(v) for v in centre],
    }
    return grid, align


def _surface_shell(field: Any, cut: int, log: LogFn) -> Any:
    """The boundary of `field >= cut`, as indices into the field.

    NS2S predicts *occupancy* — the solid, interior included — and TRELLIS's
    stage-1 grid is a voxelization of a *surface*: about twenty thousand cells
    for a typical object, all of them on the shell. Handing it a filled block
    asks for something its stage-1 latents were never trained on, and the
    voxel set that comes back is what stage 2 then allocates per cell, so a
    solid constraint is also the expensive kind of wrong.

    So the constraint is the shell: an occupied voxel keeping at least one of
    its six face neighbours outside the threshold. Anything off the edge of
    the grid counts as outside, which only matters if the prediction is
    clipped by its own margin.
    """
    import numpy as np  # server env

    solid = field >= cut
    # a voxel is interior when all six neighbours are solid too; shifting the
    # array is that test, with False shifted in at the faces
    interior = solid.copy()
    for axis in range(3):
        for step in (1, -1):
            shifted = np.roll(solid, step, axis=axis)
            # roll wraps; the wrapped-in plane is outside the grid, so clear it
            plane = [slice(None)] * 3
            plane[axis] = 0 if step == 1 else -1
            shifted[tuple(plane)] = False
            interior &= shifted
    shell = solid & ~interior
    log(f"surface constraint: {int(solid.sum())} voxel(s) above the "
        f"threshold, {int(shell.sum())} on the shell")
    return np.argwhere(shell)


def _voxelize_surface(
    field: Any,
    header: dict[str, Any],
    align: dict[str, Any],
    threshold: float,
    log: LogFn,
) -> Any:
    """A predicted occupancy field as a 64^3 binary grid in TRELLIS's frame.

    The field arrives in NS2S's own normalized cube and `header["align"]` says
    where that cube sits in the world; `align` is the sketch normalization
    `_voxelize_strokes` chose, which is the frame the constraint is written
    in. So this is two changes of basis in a row — NS2S cube -> world ->
    TRELLIS cube — with the same y-up to z-up flip the strokes get.

    The *shell* of the prediction, not its interior — see `_surface_shell`.

    Mapped forwards, voxel by voxel, rather than sampled backwards: the field
    is 112^3 across roughly the same extent as a 64^3 cube, so every target
    cell a surface passes through is hit by some source voxel and nothing
    needs interpolating. It also makes the clipped count exact — the surface
    is allowed to bulge outside the strokes, and `CUBE_FILL` only leaves it so
    much room before the cube's wall cuts it off.
    """
    import numpy as np  # server env

    cut = int(round(threshold * 255))
    index = _surface_shell(field, cut, log)
    if index.size == 0:
        log("surface constraint: the prediction is empty at threshold "
            f"{threshold:g} — nothing to add")
        return np.zeros((OCCUPANCY_GRID,) * 3, dtype=np.uint8)

    resolution = int(header.get("grid", field.shape[0]))
    source = header["align"]
    # voxel centre -> the method's unit cube -> world
    cube = (index + 0.5) / resolution - 0.5
    world = cube * float(source["scale"]) + np.asarray(
        source["translation"], dtype=float
    )

    # world -> the sketch's unit cube -> TRELLIS's z-up frame -> voxel coords
    unit = (world - np.asarray(align["translation"], dtype=float)) / float(
        align["scale"]
    )
    trellis = np.column_stack([unit[:, 0], -unit[:, 2], unit[:, 1]])
    voxel = np.floor((trellis + 0.5) * OCCUPANCY_GRID).astype(int)

    inside = np.all((voxel >= 0) & (voxel < OCCUPANCY_GRID), axis=1)
    clipped = int((~inside).sum())
    voxel = voxel[inside]

    grid = np.zeros((OCCUPANCY_GRID,) * 3, dtype=np.uint8)
    if len(voxel):
        grid[voxel[:, 0], voxel[:, 1], voxel[:, 2]] = 1
    occupied = int(grid.sum())
    log(f"surface constraint: {len(index)} shell voxel(s) at p >= "
        f"{threshold:g} -> {occupied} of {OCCUPANCY_GRID ** 3} "
        f"cells ({100 * occupied / OCCUPANCY_GRID ** 3:.2f}%)"
        + (f", {clipped} outside the cube" if clipped else ""))
    return grid


# --- fitting --------------------------------------------------------------

# Yaw candidates for the orientation search, over the full turn. Not tied to
# the view count: the number of views is a per-conditioner setting that can
# change, and an uneven view layout puts the error somewhere other than a
# clean fraction of a turn anyway. 5-degree steps, then the similarity solve
# below refines position and scale within the winning candidate.
YAW_CANDIDATES = 72
# points drawn from the surface to score against. Enough that a stroke's
# nearest sample is a fair stand-in for its nearest surface point.
SURFACE_SAMPLES = 20000
# alternations of (find correspondences, solve scale+translation). It
# converges in two or three from a bounding-box start; more is wasted on 72
# candidates.
FIT_ITERATIONS = 3


def _yaw_matrix(angle: float) -> Any:
    """Rotation about +Y, the up axis in both frames."""
    import numpy as np

    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _yaw_degrees(rotation: Any) -> float:
    import numpy as np

    return float(np.degrees(np.arctan2(rotation[0, 2], rotation[0, 0])) % 360)


def _solve_similarity(
    target: Any, surface: Any, tree: Any, rotation: Any
) -> Any:
    """Best uniform scale and translation for one candidate rotation.

    Alternates like ICP but with the rotation held fixed, so each step is a
    closed form rather than a solve: map the strokes into the mesh's frame,
    take each one's nearest surface sample, then fit the scale and offset that
    carry those samples onto the strokes. Returns
    (mean residual, scale, translation, rotation), or None if degenerate.
    """
    import numpy as np

    # bounding boxes are a poor final answer but a fine starting guess
    rotated = surface @ rotation.T
    mesh_extent = float(np.ptp(rotated, axis=0).max())
    target_extent = float(np.ptp(target, axis=0).max())
    if mesh_extent <= 0 or target_extent <= 0:
        return None
    scale = target_extent / mesh_extent
    translation = target.mean(axis=0) - scale * rotated.mean(axis=0)

    residual = float("inf")
    for _ in range(FIT_ITERATIONS):
        # strokes into the mesh's own frame, where the sample tree lives
        probe = (target - translation) @ rotation / scale
        distances, indices = tree.query(probe)
        residual = float(distances.mean() * scale)

        source = surface[indices] @ rotation.T
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        centred_source = source - source_mean
        denominator = float((centred_source * centred_source).sum())
        if denominator <= 0:
            return None
        scale = float((centred_source * (target - target_mean)).sum() / denominator)
        if not np.isfinite(scale) or scale <= 0:
            return None
        translation = target_mean - scale * source_mean

    return residual, scale, translation, rotation


# --- progress -------------------------------------------------------------

# how much of a unit's bar each worker stage owns. The model load dominates a
# cold run (two flow transformers plus DINOv2) and is free on every run after;
# sampling itself is seconds on this hardware.
STAGE_SPANS: dict[str, tuple[float, float]] = {
    "load": (0.00, 0.35),
    "condition": (0.35, 0.40),
    "sparse_structure": (0.40, 0.62),
    "slat": (0.62, 0.84),
    "capture": (0.84, 0.88),
    "decode": (0.88, 1.00),
}

# Wire container for a captured run: a fixed magic, a JSON header describing
# the frames, then the frames back to back. Binary because the payload is
# whole u8 volumes — base64 in the job's JSON would be a third again as many
# bytes through the same pipe that carries the log.
BUNDLE_MAGIC = b"TRLZ"
BUNDLE_VERSION = 1


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
            "read it. Lower keeps more of what the "
            "surfacer was unsure about — thin features especially, which the "
            "blur below pushes down. Note that the two are coupled: smoothing "
            "moves the level set, and anything above 0.5 moves it inward, so "
            "a heavy blur at a high threshold shrinks the surface and can "
            "delete thin parts of it outright (a 2-voxel sheet peaks around "
            "0.46 after a 1.6-voxel blur, and vanishes at 0.6).",
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
            "applied on the server before anything is thresholded — so it "
            "moves the constraint's shell and the rendered surface alike. 0 uses the "
            "prediction as it comes: sharper, and pitted where the network "
            "was uncertain, which a shell then traces every hole of.",
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
    ] + conditioner_params()

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
        """Pack a captured run for the client: magic, JSON header, frames.

        The header carries the alignment because the client cannot derive it.
        `run_multi_image` takes no camera poses, so TRELLIS builds in whatever
        frame it picks and nothing maps the sketch into it — the similarity
        `_fit` recovered after the fact is the only thing that says where the
        unit cube (and so the lattice) sits against the drawing. Without a
        fit there is no answer, and the client leaves the lattice at the
        origin rather than inventing one.
        """
        assert result.frames is not None
        payload = result.frames.read_bytes()
        manifest = dict(result.manifest or {})
        grid = int(manifest.get("grid", 64))
        header = json.dumps({
            **manifest,
            "frameBytes": grid ** 3,
            "align": result.align,
        }).encode()

        stages = ", ".join(
            f"{s['name']} x{s['steps']}" for s in manifest.get("stages", [])
        )
        log(f"flow capture: {stages} at {grid}^3 "
            f"({len(payload) / 1e6:.1f} MB)")
        return b"".join([
            BUNDLE_MAGIC,
            BUNDLE_VERSION.to_bytes(4, "little"),
            len(header).to_bytes(4, "little"),
            header,
            payload,
        ])

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
        """Remove connected components far smaller than the main body.

        FlexiCubes leaves a shell of detached fragments hugging the surface —
        on a measured run, 265 of them holding 3.7% of the area, all within
        2.5% of the body — which read as shimmer along the silhouette and
        z-fight where they overlap. TRELLIS removes these by visibility; this
        is the size-based stand-in for backends where that cannot run.
        """
        if fraction <= 0 or len(mesh.faces) == 0:
            return mesh
        import trimesh  # server env

        pieces = mesh.split(only_watertight=False)
        if len(pieces) <= 1:
            return mesh
        largest = max(piece.area for piece in pieces)
        kept = [piece for piece in pieces if piece.area >= largest * fraction]
        if not kept or len(kept) == len(pieces):
            return mesh
        merged = trimesh.util.concatenate(kept)
        log(f"'{unit.label}': dropped {len(pieces) - len(kept)} small "
            f"component(s) of {len(pieces)}, keeping {len(merged.faces)} of "
            f"{len(mesh.faces)} faces")
        return merged

    def _fit(
        self, mesh: Any, unit: Unit, log: LogFn
    ) -> Optional[dict[str, Any]]:
        """Register the generated mesh onto the unit's strokes, in place.

        Returns the similarity it applied (world = scale * rotation @ v +
        translation), or None if it could not solve one. The interactive
        viewer needs that transform for the occupancy lattice: the lattice is
        the unit cube the mesh was generated in, and this is the only thing
        that knows where that cube belongs relative to the drawing.

        Bounding boxes are not enough, for two reasons that show up as a mesh
        sitting visibly wrong against the drawing.

        *Orientation.* `run_multi_image` takes no camera poses, so TRELLIS
        reconciles the views from their DINOv2 tokens alone and builds its
        result in whatever canonical frame it picks. Nothing ties that frame
        to the sketch's, and with evenly spaced views the model cannot tell
        which one was the front — so the error tends to land on a multiple of
        360/len(views) degrees. There is no transform to record and invert
        here: the pipeline never maps the sketch into a canonical space, so
        the rotation has to be recovered after the fact.

        *Position.* Aligning bbox centres assumes the generated solid and the
        stroke network have the same centre, and they do not — the model adds
        volume the drawing never had (a filled base, a thicker back), which
        drags the box centre off and shifts the mesh bodily.

        So: sweep yaw about the up axis, and at each candidate solve the
        uniform scale and translation that actually minimize the distance from
        the strokes to the surface — the strokes should lie *on* it. Keep the
        best. Rotation stays a 1-DoF search rather than a full pose solve
        because up is not in question: both frames are y-up by construction.
        """
        import numpy as np  # server env
        from scipy.spatial import cKDTree  # server env

        points = [
            np.asarray(s["points"], dtype=float)
            for s in unit.strokes if s.get("points")
        ]
        if not points:
            log(f"'{unit.label}': no stroke points to fit against; leaving "
                "the mesh in its generated frame")
            return
        target = np.vstack(points)
        if len(mesh.faces) == 0:
            return

        # correspondences come from a surface sample, not the vertices: a
        # FlexiCubes mesh is densest where the surface is most curved, so
        # nearest-vertex would bias every fit toward the fiddly bits
        surface = mesh.sample(SURFACE_SAMPLES)
        tree = cKDTree(surface)

        best = None
        for index in range(YAW_CANDIDATES):
            yaw = 2 * np.pi * index / YAW_CANDIDATES
            rotation = _yaw_matrix(yaw)
            fit = _solve_similarity(target, surface, tree, rotation)
            if fit is not None and (best is None or fit[0] < best[0]):
                best = fit

        if best is None:  # degenerate strokes or a zero-extent mesh
            return None
        residual, scale, translation, rotation = best
        mesh.vertices = (mesh.vertices @ rotation.T) * scale + translation

        extent = float(np.ptp(target, axis=0).max()) or 1.0
        log(f"'{unit.label}': fitted to sketch (scale {scale:.4g}, yaw "
            f"{_yaw_degrees(rotation):.0f}deg, mean stroke-to-surface "
            f"distance {100 * residual / extent:.2f}% of the sketch)")
        return {
            "rotation": [float(v) for v in rotation.reshape(-1)],
            "scale": float(scale),
            "translation": [float(v) for v in translation],
        }

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
        """Drive one worker subprocess, translating its JSON events into
        progress and log lines.

        Returns the files it wrote, keyed by role ("final" always; "raw" and
        "frames" only when the run was asked for them), and the frame
        manifest that goes with the capture."""
        cmd = [str(python), "-u", str(WORKER), str(config_path)]
        log_path = unit_dir / "worker.log"
        tail: list[str] = []
        error: Optional[str] = None
        written: dict[str, Path] = {}
        manifest: Optional[dict[str, Any]] = None

        with open(log_path, "w") as log_file:
            proc = spawn(
                cmd,
                cwd=repo,  # trellis imports itself relative to the checkout
                env=method_env(defines),
                stdout=subprocess.PIPE,
                stderr=log_file,  # tqdm bars and torch warnings live here
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
                    low, high = STAGE_SPANS.get(event["stage"], (0.0, 1.0))
                    frac = max(0.0, min(1.0, event["frac"]))
                    on_progress(
                        low + (high - low) * frac,
                        f"{label}: {event.get('message') or event['stage']}",
                    )
                elif kind == "mesh":
                    written[event.get("kind", "final")] = Path(event["path"])
                elif kind == "frames":
                    written["frames"] = Path(event["path"])
                    manifest = event.get("manifest")
                elif kind == "error":
                    error = event["message"]
            code = proc.wait()

        if error is not None or code != 0:
            raise RuntimeError(
                f"trellis failed on '{label}': {error or f'exit code {code}'}"
                f" (full log: {log_path})"
                + ("\n" + "\n".join(tail) if tail else "")
            )
        final = written.get("final")
        if final is None or not final.is_file():
            raise RuntimeError(
                f"trellis finished without producing a mesh for '{label}' "
                f"(full log: {log_path})"
            )
        return written, manifest
