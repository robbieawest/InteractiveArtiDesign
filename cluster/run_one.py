#!/usr/bin/env python3
"""Run exactly one cell and write its surface.

Split out of run_cell.py so a cell can be executed two ways from one body:

  * one process per cell — run_cell.py, one Slurm array task;
  * many cells inside one long-lived process — local/worker.py, one per GPU.

The second is the whole point of the split. `WORKER` in adapters/ns2s.py is a
module-level singleton that only respawns when its process has died, so
consecutive cells of one adapter in one interpreter share a single model load
(~13GB, ~15s) instead of paying it per cell. A process that exits after one
cell can never benefit from that, which is exactly the cluster's situation and
exactly what the local runner is built to avoid.

This is server.py's `_run_job` with the job bookkeeping removed: no threads, no
polling, no in-memory result to hand over. The adapter contract is unchanged —
`report`, `log` and `emit` all become stdout.

progress.json is deliberately never written here. It is a single file shared by
the whole sweep, and hundreds of tasks appending to it over NFS would corrupt
it; finalize.py rebuilds it once at the end from what is actually on disk.
"""

import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import NONE, SERVER_DIR, Cell, load_progress, load_sketch, options_for_part, sub_sketch  # noqa: E402

sys.path.insert(0, str(SERVER_DIR))
import benchmarks  # noqa: E402
from adapters import ADAPTERS  # noqa: E402

# `combine_meshes` catches a failed boolean union, logs, and concatenates
# instead — a valid glb and a zero exit code. That is the right behaviour for
# an interactive server and a silent quality regression across a sweep, so the
# cell records it. Distinct from the benign "not closed volumes" message, which
# is ordinary geometry: an open sheet is a normal surfacing result.
UNION_FAILED = "boolean union failed"


def part_file_name(index: int, name: str) -> str:
    """`03_left_wheel.glb` for an emitted piece called "left wheel".

    The label is the part's user-authored name, so it can hold spaces, slashes
    or a leading dash; the ordinal makes the file sort and be unique whatever
    the slug collapses to. Deliberately not `part_NN.glb`: that name belongs to
    a cell split across tasks, and finalize.py merges exactly that glob.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "part"
    return f"{index:02d}_{slug}.glb"


def meta_path(bench: str, adapter: str, run: str, sketch: str, part: str) -> Path:
    if part == NONE:
        return (
            benchmarks.bench_dir(bench) / adapter / run / f"{sketch}.meta.json"
        )
    return benchmarks.part_dir(bench, adapter, run, sketch) / f"{part}.meta.json"


def run_one(bench: str, cell: Cell, force: bool = False) -> int:
    """Surface one cell. Returns a process-style status: 0 done or already
    done, 1 failed.

    Never raises for an ordinary method failure — the traceback is printed and
    the meta sidecar records the error, because in both callers a failed cell
    must not take the rest of the sweep with it.
    """
    is_part = cell.is_part
    label = f"{cell.adapter}/{cell.run}/{cell.sketch}" + (f"/{cell.part}" if is_part else "")

    if cell.adapter not in ADAPTERS:
        print(f"unknown adapter {cell.adapter!r}; have {sorted(ADAPTERS)}", file=sys.stderr)
        return 1

    # --- resume: a finished cell is left alone ---------------------------
    if is_part:
        done = benchmarks.part_dir(
            bench, cell.adapter, cell.run, cell.sketch
        ) / f"{cell.part}.glb"
    else:
        done = (
            benchmarks.bench_dir(bench) / cell.adapter / cell.run
            / f"{cell.sketch}.glb"
        )
    if done.is_file() and not force:
        print(f"skip {label}: already done ({done})")
        return 0

    # --- what to run -----------------------------------------------------
    progress = load_progress(bench)
    runs = progress.get("runs", {}).get(cell.adapter, [])
    matched = [r for r in runs if r["id"] == cell.run]
    if not matched:
        print(
            f"no run {cell.run!r} for adapter {cell.adapter!r} in progress.json",
            file=sys.stderr,
        )
        return 1
    options = dict(matched[0].get("options", {}))

    sketch = load_sketch(bench, cell.sketch)
    if is_part:
        # one part surfaced standalone, exactly as sf3d's _make_proxy runs
        # another adapter over one unit's strokes
        sketch = sub_sketch(sketch, cell.part_id)
        options = options_for_part(cell.adapter, options)

    # --- run --------------------------------------------------------------
    log_lines: list[str] = []
    emitted: list[str] = []

    # A part-based run that was *not* split across tasks (ns2s, bbox, and
    # anything with "split": false in profiles.json) surfaces every part inside
    # one task and returns only the combined mesh. Its per-part meshes exist
    # only as emit() calls — the server keeps them for the browser, and here
    # they used to be counted and thrown away, leaving the individual parts
    # nowhere but the task's scratch directory, which is deleted on success.
    # So write them beside the combined result. Nothing else reads these
    # (finalize.py merges `part_*.glb`, which these deliberately are not);
    # they are the per-part evidence a split run gets for free.
    #
    # The gate is the cell shape, not the run's part_based flag: an adapter is
    # free to emit per part without exposing that option (bbox does). A whole
    # cell that turns out to emit a single piece is just the result under
    # another name, and its file is removed below rather than kept as a copy.
    persist_parts = not is_part
    part_files: dict[str, str] = {}
    parts_out = benchmarks.part_dir(bench, cell.adapter, cell.run, cell.sketch)

    def report(progress_frac: float = 0.0, message: str = "") -> None:
        if message:
            print(f"[{progress_frac:5.1%}] {message}", flush=True)

    def log(line: str) -> None:
        log_lines.append(line)
        print(line, flush=True)

    def emit(name: str, glb: bytes) -> None:
        # the names are kept because progress.json records them per cell and
        # finalize.py restores that
        if name not in emitted:
            emitted.append(name)
        if not persist_parts:
            return
        # repeated emits of one name are a refinement of that same piece (a
        # VNS snapshot, an ns2s preview before smoothing), so the file is
        # overwritten rather than accumulated — last write is the final one
        path = parts_out / part_file_name(emitted.index(name), name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(glb)
            part_files[name] = path.name
        except OSError as exc:
            # never fail the cell over the extra copy: the combined result is
            # the deliverable and is written by the caller regardless
            print(f"WARNING could not write part {name!r}: {exc}", file=sys.stderr)

    print(f"=== {label}")
    print(f"    options: {json.dumps(options, sort_keys=True)}")
    started = time.time()
    meta = {
        "adapter": cell.adapter, "run": cell.run, "sketch": cell.sketch,
        "part": cell.part, "partId": cell.part_id, "options": options,
    }

    try:
        glb = ADAPTERS[cell.adapter].run(sketch, options, report, log, emit)
    except Exception as exc:
        elapsed = time.time() - started
        traceback.print_exc()
        meta |= {
            "state": "error",
            "seconds": round(elapsed, 1),
            "error": f"{type(exc).__name__}: {exc}",
            # parts finished before the failure are already on disk, and are
            # the salvage from a cell that died three parts from the end
            "parts": emitted,
            "partFiles": part_files,
        }
        path = meta_path(bench, cell.adapter, cell.run, cell.sketch, cell.part)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=1))
        print(f"FAILED {label} after {elapsed:.1f}s", file=sys.stderr)
        return 1

    elapsed = time.time() - started

    # one emitted piece is the whole surface under another name (vns snapshots
    # its progress, ns2s previews the unsmoothed mesh); keeping it would just
    # store the result twice
    if len(part_files) == 1:
        only = parts_out / next(iter(part_files.values()))
        only.unlink(missing_ok=True)
        part_files.clear()
        try:
            parts_out.rmdir()  # only if we created it and it is now empty
        except OSError:
            pass

    if is_part:
        out = benchmarks.save_part_result(
            bench, cell.adapter, cell.run, cell.sketch, cell.part, glb
        )
    else:
        out = benchmarks.save_result(
            bench, cell.adapter, cell.run, cell.sketch, glb
        )

    degraded = any(UNION_FAILED in line for line in log_lines)
    meta |= {
        "state": "done",
        "seconds": round(elapsed, 1),
        "parts": emitted,
        # part name -> the .glb written for it beside the combined result;
        # the names cannot be recovered from the filenames alone, since they
        # are slugged
        "partFiles": part_files,
        "bytes": len(glb),
        "unionFailed": degraded,
    }
    path = meta_path(bench, cell.adapter, cell.run, cell.sketch, cell.part)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1))

    if degraded:
        # not fatal — the mesh is usable — but it means manifold3d/networkx are
        # missing from .venv-server, which would otherwise go unnoticed across
        # hundreds of separate task logs
        print(
            f"WARNING {label}: boolean union failed, parts were concatenated. "
            "Check manifold3d and networkx in .venv-server.",
            file=sys.stderr,
        )
    print(f"OK {label} in {elapsed:.1f}s -> {out}")
    if part_files:
        print(f"   {len(part_files)} per-part surface(s) -> {parts_out}")
    return 0
