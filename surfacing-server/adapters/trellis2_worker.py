"""Runs inside the TRELLIS.2 checkout's venv, driven by adapters/trellis2.py.

Protocol only. Read a config json, call `sketchflow.generate`, write a glb and
report progress as JSON lines on stdout. Every decision about *what* is
computed lives in `methods/TRELLIS.2/sketchflow/`, in that checkout's
environment; this file exists because the server and that environment can only
talk over a pipe.

Contrast with `trellis_worker.py`, which holds TRELLIS 1's sampling hacks
because there was nowhere else to put them: nothing here patches, wraps or
monkeypatches anything.
"""

import json
import os
import sys
import traceback
from pathlib import Path

# This file sits in the server's adapters/ directory, so python puts that
# directory at the front of sys.path — where `import trellis2` would be fine
# but `import sketchflow` would not resolve at all. Drop it and put the
# checkout on instead: the adapter spawns us with cwd there, and python does
# not add the cwd for a script invoked by path.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, os.getcwd())

# Before torch and before anything under trellis2 is imported. The adapter
# exports these too (from backends.json); these are the fallback for running
# this file by hand.
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The protocol stream, claimed before any library can print to it. TRELLIS.2
# and its dependencies write to stdout freely (hub download notices, the
# attention backend banner); left alone they would interleave with the JSON the
# adapter parses. stdout goes to stderr for the rest of the process, where the
# adapter tees it into the job's worker.log.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def _emit(event: str, **fields: object) -> None:
    _PROTOCOL.write(json.dumps({"event": event, **fields}) + "\n")
    _PROTOCOL.flush()


def log(message: str) -> None:
    _emit("log", message=message)


def progress(stage: str, frac: float, message: str = "") -> None:
    _emit("progress", stage=stage, frac=frac, message=message)


def write_frames(frames: dict, times: dict, directory: Path) -> None:
    """Frames to disk as one flat file plus a manifest.

    Down the pipe would mean base64 through a line-delimited JSON protocol that
    also carries the log; a job-directory file costs one read in the adapter
    and keeps the two channels apart. It is deleted with the rest of the job
    directory — nothing about a capture is meant to outlive its run.
    """
    from sketchflow import GRID

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
            entry = {
                "name": stage, "offset": offset, "steps": len(frames[stage]),
            }
            # Optional, and the client treats it as such: TRELLIS 1 bundles
            # carry no times and still scrub. That is why this is an added
            # field rather than a bundle version bump — a capture never
            # outlives its job directory, so there is nothing old to read.
            if times.get(stage):
                entry["times"] = [round(float(t), 6) for t in times[stage]]
            manifest["stages"].append(entry)
    (directory / "frames.json").write_text(json.dumps(manifest))
    _emit("frames", path=str(blob), manifest=manifest)


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())

    progress("load", 0.0, "starting")
    import numpy as np
    import torch
    import trimesh

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no GPU visible to torch — TRELLIS.2 is not usable on CPU "
            "(check CUDA_VISIBLE_DEVICES)"
        )

    import sketchflow

    result = sketchflow.generate(config, log, progress)

    # TRELLIS works z-up in a normalized cube; the document (and three.js) is
    # y-up. The capture's `to_texture_bytes` applies the same change of basis,
    # so the mesh and the volumes land in one frame.
    axes = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    out = Path(config["out"])
    trimesh.Trimesh(result.vertices @ axes, result.faces).export(out)
    progress("decode", 1.0, "done")
    _emit("mesh", path=str(out), kind="final")

    if result.frames:
        write_frames(result.frames, result.times, out.parent)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit("error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
