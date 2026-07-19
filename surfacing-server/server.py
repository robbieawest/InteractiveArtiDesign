"""Surfacing job server — see README.md for the protocol.

Jobs run in daemon threads so status polling stays responsive while a
method grinds through its optimization. State is in-memory only: this is a
single-user localhost sidecar, restarting it just forgets old jobs.
"""

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from adapters import ADAPTERS


@dataclass
class Job:
    id: str
    method: str
    status: str = "pending"  # pending | running | done | error
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Optional[bytes] = None


app = FastAPI(title="Surfacing job server")
jobs: dict[str, Job] = {}


class JobRequest(BaseModel):
    method: str
    sketch: dict[str, Any]
    options: dict[str, Any] = {}


def _run_job(job: Job, sketch: dict[str, Any], options: dict[str, Any]) -> None:
    def report(progress: float, message: str = "") -> None:
        job.progress = max(0.0, min(1.0, progress))
        if message:
            job.message = message

    job.status = "running"
    try:
        job.result = ADAPTERS[job.method].run(sketch, options, report)
        job.progress = 1.0
        job.status = "done"
    except Exception as exc:  # surface the failure to the client, don't die
        job.error = f"{type(exc).__name__}: {exc}"
        job.status = "error"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "methods": sorted(ADAPTERS)}


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict[str, str]:
    if req.method not in ADAPTERS:
        raise HTTPException(
            404, f"unknown method {req.method!r}; available: {sorted(ADAPTERS)}"
        )
    job = Job(id=uuid.uuid4().hex, method=req.method)
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
    }


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status != "done" or job.result is None:
        raise HTTPException(409, f"job is {job.status}, not done")
    return Response(job.result, media_type="model/gltf-binary")
