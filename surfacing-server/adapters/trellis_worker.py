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

    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(config["model"])
    pipeline.cuda()
    progress("load", 1.0, "pipeline ready")

    progress("condition", 0.0, "loading views")
    images = load_views(config["views"])
    log(f"{len(images)} conditioning view(s), mode={config['mode']}")
    progress("condition", 1.0, "")

    # The seam for the interactive work: both samplers take their parameters
    # here, and the occupancy grid the first stage produces is decoded inside
    # `run_multi_image`. Watching either evolve means stepping the samplers
    # from this file instead of calling the convenience method, and emitting a
    # frame per step — the protocol above already carries arbitrary events.
    progress("sparse_structure", 0.0, "sampling structure")
    outputs = pipeline.run_multi_image(
        images,
        seed=config["seed"],
        formats=["mesh"],  # skips the gaussian and radiance-field decoders
        preprocess_image=config["preprocess"],
        mode=config["mode"],
        sparse_structure_sampler_params={
            "steps": config["ss_steps"],
            "cfg_strength": config["ss_cfg"],
        },
        slat_sampler_params={
            "steps": config["slat_steps"],
            "cfg_strength": config["slat_cfg"],
        },
    )
    progress("slat", 1.0, "sampled")

    progress("decode", 0.15, "extracting mesh")
    mesh = outputs["mesh"][0]
    if not mesh.success:
        raise RuntimeError(
            "the decoder produced an empty mesh — usually means the views had "
            "no foreground left after the alpha crop (are the strokes light "
            "and opaque?)"
        )

    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertices, faces = postprocess(vertices, faces, config)
    # TRELLIS works z-up in a normalized cube; the document (and three.js) is
    # y-up. Same rotation postprocessing_utils.to_glb applies before writing.
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    log(f"mesh: {len(vertices)} vertices, {len(faces)} faces")

    out = Path(config["out"])
    trimesh.Trimesh(vertices, faces).export(out)
    progress("decode", 1.0, "done")
    _emit("mesh", path=str(out), kind="final")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit("error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
