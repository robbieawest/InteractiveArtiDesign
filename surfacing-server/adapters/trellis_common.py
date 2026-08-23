"""Everything the two TRELLIS adapters share.

`trellis` (TRELLIS 1, either GPU backend) and `trellis2` (TRELLIS.2, NVIDIA
only) are different pipelines with different parameters, but the work *around*
the pipeline is the same work — and not by coincidence. TRELLIS.2 reuses
TRELLIS 1's sparse-structure VAE unchanged (`configs/gen/
ss_flow_img_dit_1_3B_64_bf16.json` names `ss_dec_conv3d_16l8_fp16` as its
decoder), so the 64^3 constraint grid, the cube the sketch is normalized into,
and the latent it encodes to are literally the same objects in both.

So this module holds: the unit split, the conditioner interface and the client
render strategy, the sketch and surface voxelization, the similarity fit back
onto the strokes, and the capture bundle format. What it does not hold is
anything about how a pipeline is driven — that is what differs, and it stays in
the adapters.

Server environment: no torch, and numpy/trimesh/scipy imported inside the
functions that need them.
"""

import base64
import json
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .base import LogFn
from .common import SERVER_DIR, backend_name, method_env, spawn

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


def conditioner_params(
    conditioners: dict[str, "Conditioner"]
) -> list[dict[str, Any]]:
    """Every conditioner's own parameters, prefixed and gated on it being the
    selected one — the same trick sf3d uses for its proxy methods, so a knob
    belonging to an unselected strategy is visible but inert rather than
    missing.

    Takes the registry rather than reading a module-level one: each adapter
    owns which strategies it offers. TRELLIS.2 conditions on a single image and
    shares none of TRELLIS 1's view knobs, so a shared registry would mean each
    adapter advertising the other's parameters.
    """
    out: list[dict[str, Any]] = []
    for name, conditioner in conditioners.items():
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


def bundle_capture(result: "UnitResult", log: LogFn) -> bytes:
    """Pack a captured run for the client: magic, JSON header, frames.

    The header carries the alignment because the client cannot derive it.
    Neither pipeline maps the sketch into its canonical frame, so the
    similarity recovered afterwards — by `fit_to_strokes`, or exactly by the
    constraint's own normalization when there is one — is the only thing that
    says where the unit cube (and so the lattice) sits against the drawing.
    Without a fit there is no answer, and the client leaves the lattice at the
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
    log(f"flow capture: {stages} at {grid}^3 ({len(payload) / 1e6:.1f} MB)")
    return b"".join([
        BUNDLE_MAGIC,
        BUNDLE_VERSION.to_bytes(4, "little"),
        len(header).to_bytes(4, "little"),
        header,
        payload,
    ])


def drop_small_components(
    mesh: Any, fraction: float, unit: "Unit", log: LogFn
) -> Any:
    """Remove connected components far smaller than the main body.

    Both decoders leave a shell of detached fragments hugging the surface — on
    a measured TRELLIS 1 run, 265 of them holding 3.7% of the area, all within
    2.5% of the body — which read as shimmer along the silhouette and z-fight
    where they overlap. Size is a cruder rule than visibility: it cannot tell a
    small part from a stray shell, only small from large.
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


def fit_to_strokes(
    mesh: Any, unit: "Unit", log: LogFn
) -> Optional[dict[str, Any]]:
    """Register a generated mesh onto the unit's strokes, in place.

    Returns the similarity it applied (world = scale * rotation @ v +
    translation), or None if it could not solve one. The interactive viewer
    needs that transform for the occupancy lattice: the lattice is the unit
    cube the mesh was generated in, and this is the only thing that knows where
    that cube belongs relative to the drawing.

    Bounding boxes are not enough, for two reasons that show up as a mesh
    sitting visibly wrong against the drawing.

    *Orientation.* Neither pipeline takes camera poses, so the model builds its
    result in whatever canonical frame it picks and nothing ties that frame to
    the sketch's. There is no transform to record and invert — the pipeline
    never maps the sketch into a canonical space, so the rotation has to be
    recovered after the fact.

    *Position.* Aligning bbox centres assumes the generated solid and the
    stroke network have the same centre, and they do not — the model adds
    volume the drawing never had (a filled base, a thicker back), which drags
    the box centre off and shifts the mesh bodily.

    So: sweep yaw about the up axis, and at each candidate solve the uniform
    scale and translation that actually minimize the distance from the strokes
    to the surface — the strokes should lie *on* it. Keep the best. Rotation
    stays a 1-DoF search rather than a full pose solve because up is not in
    question: both frames are y-up by construction.

    A constrained run skips this entirely: voxelizing the sketch into the cube
    is exactly what makes the generated frame known, and the exact inverse
    normalization beats a 72-candidate ICP against a shape the strokes already
    pin.
    """
    import numpy as np  # server env
    from scipy.spatial import cKDTree  # server env

    points = [
        np.asarray(s["points"], dtype=float)
        for s in unit.strokes if s.get("points")
    ]
    if not points:
        log(f"'{unit.label}': no stroke points to fit against; leaving the "
            "mesh in its generated frame")
        return None
    target = np.vstack(points)
    if len(mesh.faces) == 0:
        return None

    # correspondences come from a surface sample, not the vertices: a decoded
    # mesh is densest where the surface is most curved, so nearest-vertex
    # would bias every fit toward the fiddly bits
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
        f"{_yaw_degrees(rotation):.0f}deg, mean stroke-to-surface distance "
        f"{100 * residual / extent:.2f}% of the sketch)")
    return {
        "rotation": [float(v) for v in rotation.reshape(-1)],
        "scale": float(scale),
        "translation": [float(v) for v in translation],
    }


def run_worker(
    worker: Path,
    config_path: Path,
    unit_dir: Path,
    label: str,
    repo: Path,
    python: Path,
    defines: dict[str, str],
    log: LogFn,
    on_progress: Callable[[float, str], None],
) -> tuple[dict[str, Path], Optional[dict[str, Any]]]:
    """Drive one worker subprocess, translating its JSON events into progress
    and log lines.

    Returns the files it wrote, keyed by role ("final" always; "raw" and
    "frames" only when the run was asked for them), and the frame manifest that
    goes with the capture.

    Shared by both TRELLIS adapters because the protocol is the same protocol:
    a worker in a foreign environment, one JSON object per line, files handed
    over by path rather than down the pipe.
    """
    cmd = [str(python), "-u", str(worker), str(config_path)]
    log_path = unit_dir / "worker.log"
    tail: list[str] = []
    error: Optional[str] = None
    written: dict[str, Path] = {}
    manifest: Optional[dict[str, Any]] = None

    with open(log_path, "w") as log_file:
        proc = spawn(
            cmd,
            cwd=repo,  # the method imports itself relative to the checkout
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
            f"{worker.stem} failed on '{label}': "
            f"{error or f'exit code {code}'} (full log: {log_path})"
            + ("\n" + "\n".join(tail) if tail else "")
        )
    final = written.get("final")
    if final is None or not final.is_file():
        raise RuntimeError(
            f"{worker.stem} finished without producing a mesh for '{label}' "
            f"(full log: {log_path})"
        )
    return written, manifest

