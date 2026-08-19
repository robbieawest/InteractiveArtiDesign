"""Runs inside a TRELLIS checkout's venv (upstream on CUDA, TRELLIS-AMD on
ROCm), driven by adapters/trellis.py. Reads a config json, writes a glb, and
reports progress as JSON lines on stdout.

Nothing here imports from the server package: this process has TRELLIS's
dependency set, not the server's, and the two environments never meet.
"""

import json
import os
import sys
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

# This file sits in the server's adapters/ directory, so python puts that
# directory at the front of sys.path — where `import trellis` finds
# adapters/trellis.py (this worker's own adapter, which then fails on its
# relative imports) instead of the TRELLIS package. Drop it, and put the
# checkout on instead: the adapter spawns us with cwd there, and python does
# not add the cwd for a script invoked by path.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, os.getcwd())

# Every one of these has to be set before `import torch` and before anything
# under `trellis` is imported. The adapter exports them too (from
# backends.json), so these are the fallback for running this file by hand.
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_BACKEND", "torchsparse")

# The protocol stream, claimed before any library can print to it. TRELLIS and
# its dependencies write to stdout freely (hub download notices, spconv
# banners); left alone they would interleave with the JSON the adapter parses.
# stdout is redirected to stderr for the rest of the process, where the adapter
# tees it into the job's worker.log.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def _emit(event: str, **fields: object) -> None:
    _PROTOCOL.write(json.dumps({"event": event, **fields}) + "\n")
    _PROTOCOL.flush()


def log(message: str) -> None:
    _emit("log", message=message)


def progress(stage: str, frac: float, message: str = "") -> None:
    _emit("progress", stage=stage, frac=frac, message=message)


def configure_torchsparse_for_hip() -> None:
    """torchsparse's default ImplicitGEMM dataflow is inline PTX, so it cannot
    run on HIP at all; GatherScatter is the portable path. No-op on CUDA,
    where spconv is the backend anyway."""
    import torch

    if os.environ.get("SPARSE_BACKEND") != "torchsparse":
        return
    if not getattr(torch.version, "hip", None):
        return
    try:
        from torchsparse.nn.functional.conv.conv_config import (
            Dataflow,
            _default_conv_config,
            set_global_conv_config,
        )
        from torchsparse.nn.functional.conv.conv_mode import set_conv_mode

        config = _default_conv_config.copy()
        config["dataflow"] = Dataflow.GatherScatter
        set_global_conv_config(config)
        set_conv_mode(0)
        log("torchsparse configured for HIP (GatherScatter, mode 0)")
    except Exception as exc:
        log(f"warning: could not configure torchsparse for HIP: {exc}")


def load_views(paths: list[str]) -> list[object]:
    """Conditioning images, kept RGBA.

    preprocess_image uses an existing alpha channel directly and only falls
    back to rembg without one — and rembg is photo-trained, so on sparse line
    art it removes the drawing. A view arriving without alpha is a bug in
    whatever produced it, so say so rather than let it through quietly.
    """
    from PIL import Image

    images = []
    for path in paths:
        image = Image.open(path)
        if image.mode != "RGBA":
            log(f"warning: {Path(path).name} is {image.mode}, not RGBA — it "
                "will go through background removal, which is unreliable for "
                "strokes")
        images.append(image)
    return images


GRID = 64  # edge of the occupancy grid the sparse-structure decoder produces
# Edge of the latent grid the structure flow model samples. The VAE is three
# resolution levels with two stride-2 blocks between them (decoder channels
# [512, 128, 32]), so 16 -> 32 -> 64: one latent cell per 4^3 voxel block.
LATENT_GRID = 16
LATENT_STRIDE = GRID // LATENT_GRID
# Where the sparse-structure *encoder* comes from. The pipeline does not carry
# it — generation only ever needs the decoder — so it is fetched and loaded on
# demand, and dropped again once the sketch is encoded.
SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"

# The constraint sources, in the order their mixes are applied. Sketch last on
# purpose: where two sources cover the same latent cell the later mix wins, and
# the strokes are the thing the user actually drew — the surface is a guess
# about everything between them.
CONSTRAINT_SOURCES = ("surface", "sketch")

# What each source is drawn at in the captured signal volume, so a glance at
# the flow view says which one is still being applied. Densities, not colours:
# the region renders occupancy, and the sketch sits on top at full strength.
CONSTRAINT_DENSITY = {"surface": 150, "sketch": 255}


def encode_constraints(config, device, log):
    """Every constraint grid this run carries, encoded to latents with masks.

    Returns a list of `{name, latent, mask, weight, release}` — the
    [1, 8, 16, 16, 16] posterior mean and a [1, 1, 16, 16, 16] float field
    saying where that source applies — empty when the run is unconstrained.

    Kept as separate entries rather than one unioned grid so each source can be
    mixed at its own weight and released at its own time. That is the only
    reason they are separate: a single grid cannot express "hold the strokes to
    the end but let the surface go early", which is the experiment.

    The encoder is loaded here rather than with the pipeline and released
    immediately after — once for all sources. It is a few hundred MB that is
    useless to generation, it is only needed once per unit, and the flow
    transformers peak high enough on this card that holding it for the whole
    run is not free.

    `sample_posterior=False` because training consumed the posterior mean, not
    a draw from it: the flow model's data distribution is the means. Drawing
    here would add variance the model never saw. Stage-1 latents are also not
    normalized — unlike SLat there are no mean/std stats — so the encoder's
    output is the flow input directly.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    present = [
        (name, config.get(name)) for name in CONSTRAINT_SOURCES
        if config.get(name)
    ]
    if not present:
        return []

    volumes = []
    for name, path in present:
        grid = np.load(path)
        occupied = int(grid.sum())
        if occupied == 0:
            raise RuntimeError(
                f"the {name} constraint voxelized to an empty grid — nothing "
                "to constrain with"
            )
        volumes.append(
            (name, torch.from_numpy(grid).to(device).float()[None, None],
             occupied)
        )

    import trellis.models as models

    log(f"loading the sparse-structure encoder ({SS_ENCODER.split('/')[-1]})")
    encoder = models.from_pretrained(SS_ENCODER).eval().to(device)
    try:
        with torch.no_grad():
            latents = [
                encoder(volume, sample_posterior=False)
                for _, volume, _ in volumes
            ]
    finally:
        del encoder
        torch.cuda.empty_cache()

    mode = str(config.get("sketch_mask", "touched"))

    constraints = []
    for (name, volume, occupied), latent in zip(volumes, latents):
        if not torch.isfinite(latent).all():
            raise RuntimeError(
                f"the {name} constraint encoded to a non-finite latent"
            )
        # A cell is "touched" if any voxel in its 4^3 block is set. Max pooling
        # is that, exactly, and matches the stride the VAE downsamples at.
        if mode == "none":
            mask = torch.ones(
                1, 1, LATENT_GRID, LATENT_GRID, LATENT_GRID, device=device
            )
        else:
            mask = F.max_pool3d(volume, LATENT_STRIDE)
            if mode == "dilated":
                # one cell of cushion in every direction, so the prior meets
                # the constraint across a soft edge rather than a hard step
                mask = F.max_pool3d(mask, 3, stride=1, padding=1)

        # The raw weight: it is a share against the other sources, not a mix
        # against the prior, so the strength is applied later and separately.
        weight = float(config.get(f"{name}_weight", 1.0))
        release = float(config.get(f"{name}_release", 0.0))
        touched = int((mask > 0).sum().item())
        log(f"{name} constraint: {occupied} voxel(s) of {GRID ** 3} -> mask "
            f"'{mode}' over {touched} of {LATENT_GRID ** 3} latent cell(s) "
            f"({100 * touched / LATENT_GRID ** 3:.1f}%), share {weight:g}"
            + ("" if release <= 0 else f", released below t={release:g}"))
        constraints.append({
            "name": name,
            "latent": latent.float(),
            "mask": mask,
            "weight": weight,
            "release": release,
        })
    return constraints


def to_texture_bytes(volume):
    """A [x, y, z] TRELLIS-frame volume as bytes for a WebGL 3D texture.

    Two reorderings at once, both of which have to match the client exactly:

    * TRELLIS works z-up, the document is y-up, and the mesh already gets the
      same change of basis before export. World (x, y, z) = (x_t, z_t, -y_t),
      so the third axis is transposed into place and then reversed.
    * WebGL wants x varying fastest, which is the opposite of C order, so the
      axes are reversed again on the way out.

    The result addresses voxel (a, b, c) at world position
    ((a + 0.5)/GRID - 0.5, ...) — a unit cube centred on the origin, which is
    the frame the FlexiCubes mesh comes out in.
    """
    import numpy as np

    world = np.transpose(volume, (0, 2, 1))[:, :, ::-1]
    return np.ascontiguousarray(np.transpose(world, (2, 1, 0))).tobytes()


def capture_structure_frames(pipeline, samples, log):
    """One u8 occupancy field per sparse-structure step.

    The stored value is `sigmoid(logit) * 255`, not a bit: the pipeline's own
    threshold is `logit > 0`, which lands exactly on code 128, so the binary
    grid is recoverable while the near-threshold voxels — the ones an
    inpainting prior would be fighting for — survive the trip. A linear
    quantization of the raw logits would spend most of its codes on the
    saturated tails and none where it matters.
    """
    import torch

    decoder = pipeline.models["sparse_structure_decoder"]
    frames = []
    for latent in samples.pred_x_0:
        with torch.no_grad():
            field = torch.sigmoid(decoder(latent).float())[0, 0]
        volume = (field.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        frames.append(to_texture_bytes(volume))
        del field
    log(f"captured {len(frames)} structure frame(s) at {GRID}^3")
    return frames


def capture_slat_frames(samples, coords, log):
    """One u8 field per SLAT step, holding distance from the final latent.

    SLAT never touches occupancy — the voxel set is fixed by stage one — so
    what is worth watching is *where* the latent is still moving. Each frame
    stores per-voxel ||x0_i - x0_final||, normalized across the whole
    timeline so the colour scale is comparable between steps: bright early,
    dark once that voxel has settled.

    Costs nothing to compute. The sampler already retained every pred_x_0;
    upstream just drops them.
    """
    import numpy as np
    import torch

    final = samples.pred_x_0[-1].feats
    distances = [
        (step.feats - final).norm(dim=1).float().cpu().numpy()
        for step in samples.pred_x_0
    ]
    peak = max((float(d.max()) for d in distances), default=0.0) or 1.0

    index = coords[:, 1:].long().cpu().numpy()
    frames = []
    for distance in distances:
        volume = np.zeros((GRID, GRID, GRID), dtype=np.uint8)
        volume[index[:, 0], index[:, 1], index[:, 2]] = np.clip(
            distance / peak * 255.0, 0, 255
        ).astype(np.uint8)
        frames.append(to_texture_bytes(volume))
    log(f"captured {len(frames)} latent frame(s) over {len(index)} voxel(s)")
    return frames


@contextmanager
def inject_sketch_constraint(sampler, constraints, config, log):
    """Mix the noised constraint latents into the structure sampler each step.

    The flow's forward marginal is `x_t = (1-t) * x_0 + (sigma_min +
    (1-sigma_min) * t) * eps` — read straight off `FlowEulerSampler`'s
    `_eps_to_xstart`. So the sketch's state at any time is computable in closed
    form, and the constraint is: after each Euler step lands on `t_prev`, pull
    the latent toward where the sketch would be at `t_prev`.

    Applied to `pred_x_prev` rather than to the model input, which is the same
    place but the honest one — the next step's model call reads it, and the
    step's own prediction is left as the model made it so the captured
    `pred_x_0` still shows what the prior wanted rather than what it was told.

    `eps` is drawn once and reused for every step, and shared by every source.
    Redrawing (what RePaint does, and what its resampling loop needs) makes the
    constraint jitter between steps; with a fixed draw the sketch follows one
    coherent trajectory down to `x_0`, which is what a constraint should look
    like. Sharing it across sources buys two things: a cell covered by both
    sees the same noise either way, so one source releasing is a change of
    target and not also a change of noise — and the RNG stream does not depend
    on how many sources are on, so seeds stay comparable across runs.

    Two stages, deliberately separated. First the sources are combined into one
    target — a convex combination, each source's per-cell share being its
    weight normalized against the weights of the other sources covering that
    cell. Then the latent is lerped toward that target by the strength, over
    every cell any source covers:

        target = sum_i (w_i * mask_i / sum_j w_j * mask_j) * known_i
        x     <- x + strength * covered * (target - x)

    So the weights say only how the sources divide the constrained cells
    between themselves, and the strength alone says how much of the prior
    survives there — 1.0 is a full overwrite of every covered cell whatever the
    weights are. Folding the weights into the mix against `x` instead (the
    obvious thing) entangles the two: two sources at 0.5 would leave the prior
    a quarter share at full strength, which is not what a strength knob means.

    The normalization is recomputed per live set, so a source releasing hands
    its share to the ones still applying rather than leaking it to the prior.
    The corollary is that a source's weight does nothing when it is the only
    one covering a cell — it is a ratio, and there is nothing to take a ratio
    against. That is the strength's job.

    Nothing here is a patched sampler in the checkout: `sample_once` is
    swapped on the instance for the duration and put back after, the same
    shape as upstream's own `inject_sampler_multi_image` (which patches
    `_inference_model`, so the two compose).
    """
    import torch

    # One entry per sampler step: which sources were applied on it. Recorded
    # rather than recomputed because this closure is the only place that knows
    # `t_prev`; the capture reads it afterwards to draw the signal the way the
    # run actually saw it. Appending changes nothing about the sampling — it
    # draws nothing and touches no tensor.
    applied: list[tuple[str, ...]] = []

    if not constraints:
        yield applied
        return

    sigma_min = sampler.sigma_min
    strength = float(config.get("constraint_strength", 1.0))
    eps = torch.randn_like(constraints[0]["latent"])
    prepared = [
        (
            entry["name"],
            entry["latent"],
            (entry["weight"] * entry["mask"]).to(entry["latent"].dtype),
            entry["release"],
        )
        for entry in constraints
    ]
    log(f"structure constraint: strength {strength:g} over " + ", ".join(
        f"{name} (share {entry['weight']:g}"
        + ("" if release <= 0 else f", until t={release:g}") + ")"
        for entry, (name, _, _, release) in zip(constraints, prepared)
    ))

    if str(config.get("sketch_mask")) == "none" and strength > 0.75:
        log("warning: unmasked at this strength replaces the whole object "
            "with the constraint — the mask covers every cell, so nothing of "
            "the prior survives")

    # Per live set of sources: `(latent, share)` for each one and the cells any
    # of them cover. `share` is that source's weight over the summed weights of
    # everything live covering the cell, so the shares add to 1 wherever
    # anything is constrained — which is what makes the signal below a weighted
    # average of the sources and nothing more. Cached because the sets repeat:
    # there are only as many as there are distinct release times.
    splits: dict[tuple[str, ...], tuple[list, "torch.Tensor"]] = {}

    def split(live):
        if live not in splits:
            entries = [entry for entry in prepared if entry[0] in live]
            total = sum((entry[2] for entry in entries[1:]), entries[0][2])
            covered = (total > 0).to(total.dtype)
            splits[live] = (
                # clamp only to keep the division finite; `covered` zeroes
                # those cells out anyway
                [(latent, cov / total.clamp_min(1e-12))
                 for _, latent, cov, _ in entries],
                covered,
            )
        return splits[live]

    original = sampler.sample_once

    def sample_once(model, x_t, t, t_prev, cond=None, **kwargs):
        out = original(model, x_t, t, t_prev, cond, **kwargs)
        live = tuple(
            name for name, _, _, release in prepared if t_prev >= release
        )
        applied.append(live)
        if live and strength > 0:
            scale = sigma_min + (1 - sigma_min) * t_prev
            dtype = out.pred_x_prev.dtype
            sources, covered = split(live)

            # The inpainting signal for this step. Each live source says where
            # its own grid would be at t_prev; `share` splits the cells they
            # both cover between them (sketch 1.0 against surface 0.25 -> four
            # fifths sketch there) and is 1 where only one of them is present.
            signal = torch.zeros_like(out.pred_x_prev)
            for latent, share in sources:
                known = ((1 - t_prev) * latent + scale * eps).to(dtype)
                signal = signal + share.to(dtype) * known

            # The only mix against the model's own prediction: lerp the
            # previous latent toward that signal by the strength. `covered` is
            # 0 outside the constraint region, pinning those cells to the
            # previous latent.
            mix = (strength * covered).to(dtype)
            out.pred_x_prev = (1 - mix) * out.pred_x_prev + mix * signal
        return out

    sampler.sample_once = sample_once
    try:
        yield applied
    finally:
        # delete rather than reassign: `sample_once` lives on the class, and
        # putting the bound method back as an instance attribute would leave
        # the sampler permanently shadowed by a closure over this run
        del sampler.sample_once


def capture_constraint_frames(config, applied, log):
    """The constraint grids themselves, one frame per structure step.

    No single grid moves while it is on — each is fixed before sampling starts
    and every step mixes the same one in, noised to that step's time — so a
    frame changes only when a source is released. With one source that is the
    same volume repeated and then empty; with two released at different times
    it is the sketch outliving the surface, or the reverse. What the model is
    told does not evolve, only when it stops being told, and this draws exactly
    that. Sources are distinguishable by density (see `CONSTRAINT_DENSITY`),
    the sketch drawn last so it wins where they overlap.

    The 64^3 grids, not the 16^3 latents the mix actually happens in. The
    latent is what the sampler sees, but it is an encoding of this, and this is
    the thing a person can compare against the strokes on screen.
    """
    import numpy as np

    if not applied:
        return None
    grids = {
        name: np.load(config[name]).astype(np.uint8) > 0
        for name in CONSTRAINT_SOURCES if config.get(name)
    }
    if not grids:
        return None

    shape = next(iter(grids.values())).shape
    cache: dict[tuple[str, ...], bytes] = {}
    frames = []
    for live in applied:
        key = tuple(name for name in CONSTRAINT_SOURCES if name in live)
        if key not in cache:
            volume = np.zeros(shape, dtype=np.uint8)
            for name in key:
                volume[grids[name]] = CONSTRAINT_DENSITY[name]
            cache[key] = to_texture_bytes(volume)
        frames.append(cache[key])

    held = {
        name: sum(1 for live in applied if name in live) for name in grids
    }
    log("constraint capture: " + ", ".join(
        f"{name} applied on {count} of {len(applied)} step(s)"
        for name, count in held.items()
    ))
    return frames


def report_sketch_coverage(config, coords, log):
    """How much of each constraint the finished occupancy grid contains.

    The one number that says whether the constraint took. It is not the
    experiment's score — that is IoU against a known object, measured away
    from the strokes — but a low number here means the run failed mechanically
    rather than the prior failing to complete anything.
    """
    import numpy as np

    paths = [
        (name, config[name]) for name in CONSTRAINT_SOURCES
        if config.get(name)
    ]
    if not paths:
        return
    index = coords[:, 1:].long().cpu().numpy()
    occupied = np.zeros((GRID, GRID, GRID), dtype=bool)
    occupied[index[:, 0], index[:, 1], index[:, 2]] = True
    for name, path in paths:
        wanted = np.load(path).astype(bool)
        hit = int((occupied & wanted).sum())
        total = int(wanted.sum())
        log(f"{name} coverage: {hit}/{total} constrained voxel(s) occupied "
            f"({100 * hit / max(total, 1):.1f}%)")


def run_pipeline(pipeline, images, config, capture, log):
    """`run_multi_image`, inlined so the per-step latents survive.

    Nothing here is a hook or a patched sampler: `FlowEulerSampler.sample`
    already returns `pred_x_0` for every step and upstream's convenience
    method reads only `.samples` off it. This is the same calls in the same
    order, keeping the return value.

    That order is load-bearing. `torch.manual_seed` runs once for *both*
    stages and each sampler draws its own noise from the stream, so moving a
    call — or adding one that draws — changes the noise and the same seed
    stops reproducing earlier runs. Capture is read-only for exactly that
    reason: it decodes latents that already exist and never touches the RNG.

    Returns the mesh, and the captured frames when `capture` is on.
    """
    import torch

    from trellis.modules import sparse as sp

    if config["preprocess"]:
        images = [pipeline.preprocess_image(image) for image in images]
    cond = pipeline.get_cond(images)
    cond["neg_cond"] = cond["neg_cond"][:1]

    # The unconditional branch. Zeroing after get_cond rather than skipping it
    # keeps the token shape the flow models expect — they take a cond tensor,
    # not an optional one, and the branch this selects is the one training
    # taught by dropping the image on 10% of steps. cond and neg_cond are then
    # identical, so CFG cancels whatever its strength; and the multi-view
    # injection has nothing left to cycle through, so it is skipped (in
    # 'stochastic' it would deal out one identical row per step, and its
    # index list is sized to the step count).
    mode = config["mode"]
    unconditional = bool(config.get("no_image_cond"))
    if unconditional:
        before = float(cond["cond"].abs().max())
        cond = {
            "cond": torch.zeros_like(cond["cond"][:1]),
            "neg_cond": torch.zeros_like(cond["cond"][:1]),
        }
        after = float(cond["cond"].abs().max())
        log(f"image conditioning off: DINOv2 tokens {tuple(cond['cond'].shape)} "
            f"max|x| {before:.4g} -> {after:.4g}, guidance inert (cond == "
            "neg_cond), multi-view injection skipped")

    def multi_image(sampler_name, steps):
        if unconditional:
            return nullcontext()
        return pipeline.inject_sampler_multi_image(
            sampler_name, len(images), steps, mode=mode
        )

    torch.manual_seed(config["seed"])

    # Draws nothing from the RNG (`sample_posterior=False` takes the posterior
    # mean), so it is safe here between the seed and the noise below. The
    # constraint's own noise is drawn later, when the context manager opens,
    # which is after the sampler's starting noise — so a constrained run and an
    # unconstrained one at the same seed begin from the same x_1 and differ
    # only by the constraint.
    constraints = encode_constraints(config, pipeline.device, log)

    progress("sparse_structure", 0.0, "sampling structure")
    flow_model = pipeline.models["sparse_structure_flow_model"]
    reso = flow_model.resolution
    noise = torch.randn(
        1, flow_model.in_channels, reso, reso, reso
    ).to(pipeline.device)
    ss_params = {
        **pipeline.sparse_structure_sampler_params,
        "steps": config["ss_steps"],
        "cfg_strength": config["ss_cfg"],
    }
    sampler = pipeline.sparse_structure_sampler
    with multi_image("sparse_structure_sampler", ss_params["steps"]), \
            inject_sketch_constraint(
                sampler, constraints, config, log) as applied:
        structure = sampler.sample(
            flow_model, noise, **cond, **ss_params, verbose=True
        )
    decoder = pipeline.models["sparse_structure_decoder"]
    coords = torch.argwhere(decoder(structure.samples) > 0)[:, [0, 2, 3, 4]].int()
    log(f"structure: {coords.shape[0]} occupied voxel(s) of {GRID ** 3}")
    if coords.shape[0] == 0:
        raise RuntimeError(
            "the structure stage produced an empty occupancy grid — usually "
            "means the views had no foreground left after the alpha crop"
        )
    report_sketch_coverage(config, coords, log)
    progress("sparse_structure", 1.0, "sampled")

    progress("slat", 0.0, "sampling latent")
    flow_model = pipeline.models["slat_flow_model"]
    noise = sp.SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels).to(
            pipeline.device
        ),
        coords=coords,
    )
    slat_params = {
        **pipeline.slat_sampler_params,
        "steps": config["slat_steps"],
        "cfg_strength": config["slat_cfg"],
    }
    with multi_image("slat_sampler", slat_params["steps"]):
        latent = pipeline.slat_sampler.sample(
            flow_model, noise, **cond, **slat_params, verbose=True
        )
    std = torch.tensor(pipeline.slat_normalization["std"])[None].to(
        pipeline.device
    )
    mean = torch.tensor(pipeline.slat_normalization["mean"])[None].to(
        pipeline.device
    )
    slat = latent.samples * std + mean
    progress("slat", 1.0, "sampled")

    frames = None
    if capture:
        progress("capture", 0.0, "decoding flow frames")
        frames = {
            "structure": capture_structure_frames(pipeline, structure, log),
            "latent": capture_slat_frames(latent, coords, log),
        }
        constraint = capture_constraint_frames(config, applied, log)
        if constraint:
            frames["constraint"] = constraint
        progress("capture", 1.0, "captured")

    progress("decode", 0.0, "decoding mesh")
    return pipeline.decode_slat(slat, ["mesh"])["mesh"][0], frames


def write_frames(frames, directory):
    """Frames to disk as one flat file plus a manifest.

    Down the pipe would mean base64 through a line-delimited JSON protocol
    that also carries the log; a job-directory file costs one read in the
    adapter and keeps the two channels apart. It is deleted with the rest of
    the job directory (`prune_job_dirs`) — nothing about the capture is meant
    to outlive the run.
    """
    blob = directory / "frames.bin"
    manifest = {"grid": GRID, "stages": []}
    with open(blob, "wb") as handle:
        # "constraint" is optional and is not a stage of the flow: it runs
        # alongside the structure steps rather than after them, and the client
        # reads stages by name, so its place in the file carries no meaning.
        for stage in ("structure", "latent", "constraint"):
            if not frames.get(stage):
                continue
            offset = handle.tell()
            for frame in frames[stage]:
                handle.write(frame)
            manifest["stages"].append({
                "name": stage,
                "offset": offset,
                "steps": len(frames[stage]),
            })
    (directory / "frames.json").write_text(json.dumps(manifest))
    _emit("frames", path=str(blob), manifest=manifest)


def postprocess(vertices, faces, config):
    """TRELLIS's own mesh cleanup, as far as this machine can run it.

    Two independent stages, and only one of them is portable:

    * `simplify` is quadric edge collapse through pyvista — CPU, no
      rasterizer, works everywhere.
    * `fill_holes` rasterizes the mesh from 100 views to score per-face
      visibility, drops components that are rarely visible, and mincuts away
      interior shells reachable through small holes. It is the stage that
      clears the loose fragments FlexiCubes leaves around the surface, and it
      needs a working rasterizer. On AMD the fork's simplified coarse
      rasterizer returns empty visibility and it deletes the entire mesh
      (AMD_GPU_GUIDE.md section 3.3, "Disable fill_holes (Critical!)"), so the
      adapter turns it off there and clears fragments by area instead.

    Failure here is never fatal: cleanup improves a result, it does not make
    one, and losing the whole generation to a postprocessing bug would be a
    poor trade.
    """
    ratio = float(config.get("simplify_ratio", 0.0))
    fill = bool(config.get("fill_holes", False))
    if ratio <= 0 and not fill:
        return vertices, faces

    before = len(vertices), len(faces)
    try:
        from trellis.utils.postprocessing_utils import postprocess_mesh

        vertices, faces = postprocess_mesh(
            vertices, faces,
            simplify=ratio > 0,
            simplify_ratio=ratio,
            fill_holes=fill,
            verbose=False,
        )
    except Exception as exc:
        log(f"warning: mesh postprocessing failed ({type(exc).__name__}: "
            f"{exc}); keeping the raw mesh")
        return vertices, faces

    if len(faces) == 0:
        log("warning: postprocessing removed every face; keeping the raw mesh")
        return vertices, faces
    log(f"postprocess: {before[0]} -> {len(vertices)} verts, "
        f"{before[1]} -> {len(faces)} faces "
        f"(simplify={ratio if ratio > 0 else 'off'}, fill_holes={fill})")
    return vertices, faces


def stub_kaolin_if_absent() -> None:
    """Upstream FlexiCubes imports one shape-assertion helper from kaolin —
    `kaolin.utils.testing.check_tensor`, called only inside its input
    validation — and nothing else in the mesh path touches the library. Kaolin
    is a large source build tied to an exact torch version, so paying for it to
    satisfy six asserts is a bad trade, and on a machine without a CUDA toolkit
    it is not payable at all.

    So a module object carrying that one function goes into sys.modules before
    `trellis` is imported. Done here rather than by editing the checkout
    because FlexiCubes is a *nested* submodule of upstream TRELLIS: a patch
    there would not survive a re-clone and would have to be reapplied on every
    machine. The AMD fork vendors FlexiCubes and has already replaced the
    import with a local copy of the same helper — this is that fix, made to
    survive.

    A real *working* kaolin, if one is present, always wins — which is why the
    check imports it rather than asking whether it is installed. A kaolin whose
    compiled extension never built is installed, importable by `find_spec`, and
    raises on use: treating that as "present" hands FlexiCubes a broken library
    and steps aside from the fix.
    """
    import types

    try:
        import kaolin.utils.testing  # noqa: F401
        return
    except Exception as exc:
        if not isinstance(exc, ModuleNotFoundError):
            log(f"kaolin is installed but does not import ({exc}); stubbing it")

    def check_tensor(tensor, shape, throw=True):
        import torch

        ok = torch.is_tensor(tensor) and len(tensor.shape) == len(shape) and all(
            expected is None or actual == expected
            for actual, expected in zip(tensor.shape, shape)
        )
        if not ok and throw:
            raise ValueError(f"expected shape {shape}, got {getattr(tensor, 'shape', type(tensor))}")
        return ok

    kaolin = types.ModuleType("kaolin")
    utils = types.ModuleType("kaolin.utils")
    testing = types.ModuleType("kaolin.utils.testing")
    testing.check_tensor = check_tensor
    utils.testing = testing
    kaolin.utils = utils
    sys.modules.update({
        "kaolin": kaolin,
        "kaolin.utils": utils,
        "kaolin.utils.testing": testing,
    })
    log("using the built-in check_tensor stub in place of kaolin")


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())

    progress("load", 0.0, "loading TRELLIS")
    configure_torchsparse_for_hip()
    stub_kaolin_if_absent()

    import numpy as np
    import torch
    import trimesh

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError(
            "no GPU visible to torch — TRELLIS is not usable on CPU "
            "(check HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES)"
        )
    log(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    # What the card looks like before this process allocates anything. The
    # first question an out-of-memory run raises is whether something else was
    # still resident when the pipeline loaded — an experiment that surfaces
    # the sketch first has another method's model on the card minutes earlier
    # — and this is the line that answers it without guesswork.
    free, total = torch.cuda.mem_get_info()
    log(f"GPU memory before loading: {free / 2**30:.1f} GiB free of "
        f"{total / 2**30:.1f} GiB")

    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(config["model"])
    pipeline.cuda()
    progress("load", 1.0, "pipeline ready")

    progress("condition", 0.0, "loading views")
    images = load_views(config["views"])
    log(f"{len(images)} conditioning view(s), mode={config['mode']}")
    progress("condition", 1.0, "")

    mesh, frames = run_pipeline(
        pipeline, images, config, bool(config.get("interactive")), log
    )
    if not mesh.success:
        raise RuntimeError(
            "the decoder produced an empty mesh — usually means the views had "
            "no foreground left after the alpha crop (are the strokes light "
            "and opaque?)"
        )

    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    # TRELLIS works z-up in a normalized cube; the document (and three.js) is
    # y-up. Same rotation postprocessing_utils.to_glb applies before writing.
    axes = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    out = Path(config["out"])

    # The unprocessed mesh is written first and separately, so the viewer can
    # show what simplification and hole filling actually removed. It is an
    # order of magnitude larger than the delivered one (simplify_ratio is the
    # fraction of faces *removed*), which is why it is opt-in.
    if config.get("keep_raw"):
        raw = out.with_name(f"{out.stem}_raw{out.suffix}")
        trimesh.Trimesh(vertices @ axes, faces).export(raw)
        log(f"raw mesh: {len(vertices)} vertices, {len(faces)} faces")
        _emit("mesh", path=str(raw), kind="raw")

    vertices, faces = postprocess(vertices, faces, config)
    vertices = vertices @ axes
    log(f"mesh: {len(vertices)} vertices, {len(faces)} faces")

    trimesh.Trimesh(vertices, faces).export(out)
    progress("decode", 1.0, "done")
    _emit("mesh", path=str(out), kind="final")

    if frames is not None:
        write_frames(frames, out.parent)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit("error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
