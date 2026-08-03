"""Helpers shared by the subprocess-backed method adapters (vns, ns2s).

Everything here runs in the *server* environment, so it must stay torch-free;
the methods themselves are invoked as subprocesses in their own venvs.
"""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional, Protocol

from .base import LogFn

SERVER_DIR = Path(__file__).resolve().parent.parent
METHODS_DIR = SERVER_DIR / "methods"
# scratch root for method intermediates. Overridable because on a cluster each
# task needs its own directory on node-local disk (/disk/scratch/...): two
# array tasks sharing one root would prune each other's working files, and the
# traffic — 275MB for a part-based sf3d run — has no business on a network
# filesystem.
JOBS_DIR = Path(os.environ.get("SURFACING_JOBS_DIR", SERVER_DIR / "jobs"))

# AMD GPU (ROCm): the user's defines, which must be set before any torch code
# runs. Anything already exported in the calling environment wins — spread
# os.environ after this dict, not before.
ROCM_ENV: dict[str, str] = {
    "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    "HIP_VISIBLE_DEVICES": "0",
}


def method_env() -> dict[str, str]:
    """Environment for a method subprocess: the ROCm defines, overridable by
    whatever the server itself was started with.

    SURFACING_GPU_BACKEND=cuda drops the ROCm defines entirely — on an NVIDIA
    machine HSA_OVERRIDE_GFX_VERSION is meaningless and HIP_VISIBLE_DEVICES
    is actively misleading next to the CUDA_VISIBLE_DEVICES Slurm sets."""
    base = {} if os.environ.get("SURFACING_GPU_BACKEND") == "cuda" else ROCM_ENV
    return {**base, **os.environ}


# how long to wait for an evicted worker to exit before killing it. It only
# has to notice SIGTERM between requests, so this is generous already.
WORKER_STOP_TIMEOUT = 20.0


class ResidentWorker(Protocol):
    """A method process kept alive between jobs so successive runs of the same
    adapter share one model load."""

    def stop(self) -> bool:
        """Terminate the process and wait for it to actually exit. True if one
        was running (i.e. VRAM was just released)."""

    def kill(self) -> bool:
        """Same, but without waiting for the worker's own lock — the lock is
        held for the length of a job, so this is the only way to interrupt one
        that is mid-inference. The call being cancelled sees its worker vanish
        and fails, which is the point."""


_RESIDENT: dict[str, ResidentWorker] = {}
_RESIDENT_LOCK = threading.Lock()


def register_resident_worker(method: str, worker: ResidentWorker) -> None:
    """Declare `method`'s resident worker so other methods can evict it."""
    with _RESIDENT_LOCK:
        _RESIDENT[method] = worker


def release_other_workers(method: str, log: LogFn) -> None:
    """Stop every resident worker except `method`'s, before `method` touches
    the GPU.

    A resident worker holds its whole model in VRAM for its lifetime — the
    ns2s one sits on ~13GB — which on a single card leaves later jobs of other
    methods to die with an out-of-memory error partway through (VNS OOMs in
    `loss.backward()` once the training schedule's demand peaks). So the
    running method wins the card: it evicts the others, and they reload on
    their next job. Back-to-back runs of one method — the benchmark case the
    residency exists for — never evict anything.
    """
    with _RESIDENT_LOCK:
        others = [(n, w) for n, w in _RESIDENT.items() if n != method]
    for name, worker in others:
        try:
            if worker.stop():
                log(f"released the resident {name} worker to free the GPU")
        except Exception as exc:  # never let cleanup fail the job
            log(f"could not release the resident {name} worker: {exc}")


# --- cancellation ---------------------------------------------------------
#
# A job owns whatever it spawned, and a stopped job has to actually stop: the
# methods here run for minutes to hours on the GPU, so a client that walks
# away while the work grinds on is a machine held hostage. Adapters spawn
# through `spawn()` instead of subprocess.Popen so every process is attached
# to the job on whose thread it was started, and `cancel_job` can end it.

_JOB_PROCESSES: dict[str, list[subprocess.Popen]] = {}
_JOB_LOCK = threading.Lock()
# the job running on this thread, set by the job runner. A thread-local rather
# than a parameter so adapters need not thread a job id through every call.
_CURRENT = threading.local()


def set_current_job(job_id: Optional[str]) -> None:
    _CURRENT.job_id = job_id


def spawn(*args: Any, **kwargs: Any) -> subprocess.Popen:
    """subprocess.Popen, registered against the current job so it can be
    killed from outside. Adapters should use this for every method process."""
    proc = subprocess.Popen(*args, **kwargs)
    job_id = getattr(_CURRENT, "job_id", None)
    if job_id:
        with _JOB_LOCK:
            _JOB_PROCESSES.setdefault(job_id, []).append(proc)
    return proc


def forget_job_processes(job_id: str) -> None:
    """Drop a finished job's process list. The processes are gone by then —
    this just stops the table growing for the life of the server."""
    with _JOB_LOCK:
        _JOB_PROCESSES.pop(job_id, None)


def cancel_job(job_id: str, log: LogFn) -> int:
    """End everything a job is running: its own subprocesses, and every
    resident worker.

    All the workers, not just the cancelled method's: a job's GPU work is not
    always its own — sf3d's proxy step runs another adapter's worker — and a
    stop is the one moment where a cold start next time is clearly the lesser
    cost. Returns how many processes were ended.

    Killed, not asked politely: these methods do not poll for cancellation,
    and the adapter thread finds out the ordinary way, by its subprocess
    ending or its worker's pipe closing."""
    with _JOB_LOCK:
        procs = list(_JOB_PROCESSES.get(job_id, []))
    ended = 0
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            proc.terminate()
            try:
                proc.wait(timeout=WORKER_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            ended += 1
        except Exception as exc:  # a race with the process exiting is fine
            log(f"could not stop a method process: {exc}")

    with _RESIDENT_LOCK:
        residents = list(_RESIDENT.items())
    for name, worker in residents:
        try:
            if worker.kill():
                log(f"killed the resident {name} worker")
                ended += 1
        except Exception as exc:
            log(f"could not kill the resident {name} worker: {exc}")
    return ended


# how many job scratch directories to keep around, newest first. Everything
# under jobs/ is intermediate — input .obj files, the per-iteration mesh
# snapshots the trainers drop, optimizer checkpoints, tensorboard events, the
# probability volumes ns2s writes beside each reconstruction — and a
# part-based ns2s run leaves ~160MB of it. The only output that outlives a job
# is the glb the adapter returns (plus, for a benchmark, the copy the server
# writes into benchmarks/). Keeping a short tail rather than deleting on
# success means a job that crashed still has its scratch to inspect, and it
# stays bounded even when the server is killed mid-run.
KEEP_JOB_DIRS = int(os.environ.get("SURFACING_KEEP_JOBS", "3"))

# never delete a directory whose tree was touched this recently, however many
# newer ones there are. Every long-running method writes as it goes — VNS and
# NeuVAS drop a mesh snapshot every few minutes, ns2s and vrs2s finish in
# seconds — so anything quiet for this long is not a job still in flight.
JOB_DIR_GRACE_SECONDS = 15 * 60


def _tree_mtime(directory: Path) -> float:
    """Newest mtime anywhere under `directory`.

    The directory's own mtime is not enough: NeuVAS and VNS write into nested
    subdirectories, so a job that has been running for half an hour can have a
    top-level mtime from when it started and look like the stalest thing here.
    """
    newest = directory.stat().st_mtime
    for path in directory.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:  # vanished mid-walk; nothing to weigh
            continue
    return newest


def prune_job_dirs(keep: int = KEEP_JOB_DIRS) -> None:
    """Delete all but the `keep` most recently written job scratch
    directories. Called as a job starts, so the one about to be created is
    never a candidate — and anything still being written is protected by the
    grace period regardless of rank."""
    if keep < 0 or not JOBS_DIR.is_dir():
        return
    now = time.time()
    dirs = sorted(
        ((d, _tree_mtime(d)) for d in JOBS_DIR.iterdir() if d.is_dir()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for stale, mtime in dirs[keep:]:
        if now - mtime < JOB_DIR_GRACE_SECONDS:
            continue
        shutil.rmtree(stale, ignore_errors=True)


def write_curve_obj(sketch: dict[str, Any], path: Path) -> None:
    """The sketch strokes as an .obj curve network: v records plus an
    `l i j` segment per consecutive point pair (what VNS's CurveNetwork and
    NeuralSketch2Surf's parse_obj_robust both parse)."""
    verts: list[str] = []
    segs: list[str] = []
    for stroke in sketch.get("strokes", []):
        points = stroke.get("points", [])
        start = len(verts)
        for x, y, z in points:
            verts.append(f"v {x} {y} {z}")
        # obj indices are 1-based; VNS drops degenerate edges itself, skip
        # the exactly-coincident ones here to keep its warnings quiet
        for i in range(start, len(verts) - 1):
            if points[i - start] != points[i - start + 1]:
                segs.append(f"l {i + 1} {i + 2}")
    if not segs:
        raise ValueError("sketch has too few stroke points to surface")
    path.write_text("\n".join(verts + segs) + "\n")


def group_strokes_by_part(
    sketch: dict[str, Any], log: LogFn
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Strokes grouped by part id, plus the id → name table. Unassigned
    strokes (partId None) are dropped — part-based modes have nothing
    meaningful to do with them."""
    part_names = {p["id"]: p["name"] for p in sketch.get("parts", [])}
    groups: dict[str, list[dict[str, Any]]] = {}
    dropped = 0
    for stroke in sketch.get("strokes", []):
        part_id = stroke.get("partId")
        if part_id is None:
            dropped += 1
            continue
        groups.setdefault(part_id, []).append(stroke)
    if not groups:
        raise ValueError(
            "part-based surfacing needs strokes assigned to parts, but none of "
            f"the {dropped} stroke(s) are — this sketch was never segmented "
            "(a SketchLab export with no Part_* nodes imports this way). "
            "Segment it, or run this method with part-based off."
        )
    if dropped:
        log(f"ignoring {dropped} unassigned stroke(s) (part-based mode)")
    return groups, part_names


def combine_meshes(meshes: list[Any], log: LogFn) -> Any:
    """Merge per-part meshes into one. A boolean union needs every part to be
    a closed volume; a sketch is often surfaced as an open sheet, so we repair
    what we can, union when all parts are watertight, and otherwise fall back
    to a plain concatenation (still one mesh, just no CSG)."""
    import trimesh  # server env

    for mesh in meshes:
        # tidy up marching-cubes output so is_volume has a fair chance
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    if len(meshes) == 1:
        return meshes[0]

    open_count = sum(1 for m in meshes if not m.is_volume)
    if open_count == 0:
        try:
            return trimesh.boolean.union(meshes)
        except Exception as exc:  # engine missing or numerics blew up
            log(f"boolean union failed ({exc}); merging without booleans")
    else:
        log(
            f"{open_count}/{len(meshes)} part surface(s) are not closed "
            "volumes — merging without a boolean union (concatenation)"
        )
    return trimesh.util.concatenate(meshes)
