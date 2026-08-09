"""Surfacing job server — see README.md for the protocol.

Jobs run in daemon threads so status polling stays responsive while a
method grinds through its optimization. State is in-memory only: this is a
single-user localhost sidecar, restarting it just forgets old jobs.
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import benchmarks
from adapters import ADAPTERS
from adapters.common import (
    cancel_job,
    forget_job_processes,
    prune_job_dirs,
    release_other_workers,
    set_current_job,
)

# A job's meshes are freed as soon as the client takes the result
# (_release_geometry), so this bound only covers jobs nobody collected — a
# failure, or a run the client abandoned. Those still hold their result bytes
# and every partial they published, which for a part-based run is tens of
# megabytes, so only a short tail is kept.
MAX_FINISHED_JOBS = 4


@dataclass
class Partial:
    """A finished piece of a still-running job (typically one part)."""
    name: str
    glb: bytes


@dataclass
class Job:
    id: str
    method: str
    status: str = "pending"  # pending | running | done | error
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Optional[bytes] = None
    # free-form text lines emitted by the adapter (fetched incrementally by
    # the client via /log; append-only so line indices are stable cursors)
    log: list[str] = field(default_factory=list)
    # same append-only cursor idea for geometry published mid-run
    partials: list[Partial] = field(default_factory=list)
    # partials ever published, kept separately so the client's cursor still
    # makes sense after the bytes have been released
    partial_count: int = 0
    # set once the meshes have been handed over and freed; a second fetch is
    # a client bug, not something to keep hundreds of megabytes alive for
    released: bool = False
    # asked to stop: the processes were killed under it, so whatever error the
    # adapter reports on the way out is a consequence, not the cause
    cancelled: bool = False
    # where a finished result is written on disk, if this job belongs to a
    # benchmark run; None for ordinary interactive jobs
    save: Optional["SaveTarget"] = None
    saved_path: Optional[str] = None


app = FastAPI(title="Surfacing job server")
jobs: dict[str, Job] = {}


class SaveTarget(BaseModel):
    """Destination for a benchmark job's result: benchmarks/<id>/<adapter>/
    <run>/<sketch>.glb, written by the server so results survive the tab."""
    benchmarkId: str
    adapter: str
    run: str
    sketch: str


class JobRequest(BaseModel):
    method: str
    sketch: dict[str, Any]
    options: dict[str, Any] = {}
    save: Optional[SaveTarget] = None


def _run_job(job: Job, sketch: dict[str, Any], options: dict[str, Any]) -> None:
    def report(progress: float, message: str = "") -> None:
        job.progress = max(0.0, min(1.0, progress))
        if message:
            job.message = message

    def emit(name: str, glb: bytes) -> None:
        job.partials.append(Partial(name=name, glb=glb))
        job.partial_count += 1

    job.status = "running"
    set_current_job(job.id)
    try:
        adapter = ADAPTERS[job.method]
        # the running method owns the GPU: another method's resident worker is
        # still holding its model in VRAM, which is what made VNS OOM after an
        # ns2s run. Same-method runs keep their warm worker.
        if adapter.uses_gpu:
            release_other_workers(job.method, job.log.append)
        job.result = adapter.run(sketch, options, report, job.log.append, emit)
        if job.save is not None:
            job.saved_path = str(
                benchmarks.save_result(
                    job.save.benchmarkId, job.save.adapter, job.save.run,
                    job.save.sketch, job.result,
                )
            )
            job.log.append(f"saved to {job.saved_path}")
        job.progress = 1.0
        job.status = "done"
    except Exception as exc:  # surface the failure to the client, don't die
        # a cancelled job dies of its processes being killed; report the ask,
        # not the symptom (an empty output file, a closed pipe)
        job.error = "cancelled" if job.cancelled else f"{type(exc).__name__}: {exc}"
        job.status = "error"
    finally:
        set_current_job(None)
        forget_job_processes(job.id)
        _evict_finished()


def _evict_finished() -> None:
    """Forget all but the newest few terminal jobs. `jobs` is insertion
    ordered, so this drops the oldest results and partials first."""
    finished = [j for j in jobs.values() if j.status in ("done", "error")]
    for job in finished[:-MAX_FINISHED_JOBS]:
        jobs.pop(job.id, None)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "methods": [
            {
                "name": name,
                "params": ADAPTERS[name].params,
                # absent for every method that consumes strokes as geometry;
                # the client only renders views when a method asks for them
                "viewSpec": ADAPTERS[name].view_spec,
            }
            for name in sorted(ADAPTERS)
        ],
    }


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict[str, str]:
    if req.method not in ADAPTERS:
        raise HTTPException(
            404, f"unknown method {req.method!r}; available: {sorted(ADAPTERS)}"
        )
    prune_job_dirs()
    job = Job(id=uuid.uuid4().hex, method=req.method, save=req.save)
    jobs[job.id] = job
    threading.Thread(
        target=_run_job, args=(job, req.sketch, req.options), daemon=True
    ).start()
    return {"jobId": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return {
        "id": job.id,
        "method": job.method,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        # cursor for /partials — the client fetches the ones it hasn't seen
        "partialCount": job.partial_count,
        "savedPath": job.saved_path,
    }


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict[str, Any]:
    """Stop a job for real: kill its method processes and every resident
    worker, freeing the GPU now rather than whenever the method would have
    finished.

    There is no cooperative cancellation to fall back on — none of these
    methods check for it — so this is a kill, and the job lands in `error`
    with "cancelled". Safe to call on a job that has already finished (it
    reports 0 ended) so a client racing the last poll needs no special
    case."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status in ("done", "error"):
        return {"cancelled": False, "ended": 0, "status": job.status}
    job.cancelled = True
    job.message = "cancelled"
    job.log.append("cancelled by the client")
    ended = cancel_job(job_id, job.log.append)
    return {"cancelled": True, "ended": ended, "status": job.status}


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str, after: int = 0) -> dict[str, Any]:
    """Log lines from index `after` on, plus the cursor for the next call."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    lines = job.log[after:]
    return {"lines": lines, "next": after + len(lines)}


@app.get("/api/jobs/{job_id}/partials")
def job_partials(job_id: str, after: int = 0) -> dict[str, Any]:
    """Names of the pieces published from index `after` on. The geometry
    itself is fetched one at a time from /partials/{index}, so a client that
    only wants the count never pays for the bytes."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    names = [p.name for p in job.partials[after:]]
    return {"names": names, "next": after + len(names)}


@app.get("/api/jobs/{job_id}/partials/{index}")
def job_partial(job_id: str, index: int) -> Response:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if not 0 <= index < len(job.partials):
        raise HTTPException(404, "no such partial")
    return Response(job.partials[index].glb, media_type="model/gltf-binary")


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.released:
        where = f"; it is on disk at {job.saved_path}" if job.saved_path else ""
        raise HTTPException(410, f"result already delivered{where}")
    if job.status != "done" or job.result is None:
        raise HTTPException(409, f"job is {job.status}, not done")
    return Response(_release_geometry(job), media_type="model/gltf-binary")


def _release_geometry(job: Job) -> bytes:
    """Hand over a finished job's result and free every mesh it is holding.

    The protocol is one-shot in both directions: a client pulls the partials
    it has not seen *before* it asks for the result, and asks for the result
    exactly once. So by the time this runs nothing is reachable any more —
    keeping it would mean MAX_FINISHED_JOBS whole surfaces plus every partial
    resident with no reader, which for a part-based sweep is hundreds of
    megabytes. Benchmark results are on disk regardless (`saved_path`), and
    the job entry itself stays so /jobs/{id} still reports how it ended.

    Failed jobs are left alone: the client pulls their partials on the same
    poll that first sees the failure, so there is no safe moment before
    eviction."""
    result = job.result or b""
    job.partials = []
    job.result = None
    job.released = True
    return result


# --- benchmarks -----------------------------------------------------------
#
# The browser cannot read a folder of sketches or write results next to the
# repo, so the benchmark window drives these. Preprocessing (glTF -> sketch
# document) still happens client-side, because the importer is three.js.


class SaveSketchRequest(BaseModel):
    name: str
    document: Any


class CopyRequest(BaseModel):
    """Where a clean copy should land. The client picks the id, the same way
    it does for a fresh benchmark."""
    target: str


def _bench(fn, *args):
    """Run a benchmarks.py call, turning its errors into 400s."""
    try:
        return fn(*args)
    except benchmarks.BenchmarkError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/benchmark/scan")
def benchmark_scan(dir: str) -> dict[str, Any]:
    """Surfaceable inputs in a source folder: loose sketch .json files and
    subfolders containing a .gltf."""
    return _bench(benchmarks.scan_source, dir)


@app.get("/api/benchmark/file")
def benchmark_file(path: str) -> Response:
    """Raw bytes of a source file, so the client can run the glTF importer."""
    return Response(
        _bench(benchmarks.read_source_file, path),
        media_type="application/octet-stream",
    )


@app.get("/api/benchmark")
def benchmark_list() -> dict[str, Any]:
    return {"benchmarks": benchmarks.list_benchmarks()}


@app.post("/api/benchmark/{benchmark_id}/sketches")
def benchmark_save_sketch(benchmark_id: str, req: SaveSketchRequest) -> dict[str, str]:
    path = _bench(benchmarks.save_sketch, benchmark_id, req.name, req.document)
    return {"path": str(path)}


@app.post("/api/benchmark/{benchmark_id}/copy")
def benchmark_copy(benchmark_id: str, req: CopyRequest) -> dict[str, str]:
    """Copy one benchmark's sketches into a new folder, leaving its results
    behind — a clean slate over the same inputs."""
    path = _bench(benchmarks.copy_sketches, benchmark_id, req.target)
    return {"id": req.target, "path": str(path)}


@app.get("/api/benchmark/{benchmark_id}/sketches")
def benchmark_list_sketches(benchmark_id: str) -> dict[str, Any]:
    return {"sketches": _bench(benchmarks.list_sketches, benchmark_id)}


@app.get("/api/benchmark/{benchmark_id}/sketches/{name}")
def benchmark_read_sketch(benchmark_id: str, name: str) -> Any:
    return _bench(benchmarks.read_sketch, benchmark_id, name)


@app.get("/api/benchmark/{benchmark_id}/progress")
def benchmark_get_progress(benchmark_id: str) -> dict[str, Any]:
    return {"progress": _bench(benchmarks.read_progress, benchmark_id)}


@app.put("/api/benchmark/{benchmark_id}/progress")
def benchmark_put_progress(
    # Body(...) is load-bearing: a bare `Any` is not a model, so FastAPI binds
    # it as a *query* parameter and never reads the request body — which is
    # how this quietly 422'd every save until it was noticed
    benchmark_id: str, progress: Any = Body(...)
) -> dict[str, str]:
    path = _bench(benchmarks.write_progress, benchmark_id, progress)
    return {"path": str(path)}


@app.get("/api/benchmark/{benchmark_id}/results/{adapter}/{run}/{sketch}")
def benchmark_result(
    benchmark_id: str, adapter: str, run: str, sketch: str
) -> Response:
    """A stored result, so reopening a finished benchmark shows its meshes."""
    data = _bench(benchmarks.read_result, benchmark_id, adapter, run, sketch)
    return Response(data, media_type="model/gltf-binary")
