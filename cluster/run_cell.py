#!/usr/bin/env python3
"""Run exactly one cell and write its surface. One Slurm array task.

This is server.py's `_run_job` with the job bookkeeping removed: no threads, no
polling, no in-memory result to hand over. The adapter contract is unchanged —
`report`, `log` and `emit` all become stdout, which Slurm captures per task.

progress.json is deliberately never written here. It is a single file shared by
the whole sweep, and hundreds of tasks appending to it over NFS would corrupt
it; finalize.py rebuilds it once at the end from what is actually on disk.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import NONE, SERVER_DIR, load_progress, load_sketch, options_for_part, sub_sketch  # noqa: E402

sys.path.insert(0, str(SERVER_DIR))
import benchmarks  # noqa: E402
from adapters import ADAPTERS  # noqa: E402

# `combine_meshes` catches a failed boolean union, logs, and concatenates
# instead — a valid glb and a zero exit code. That is the right behaviour for
# an interactive server and a silent quality regression across a sweep, so the
# cell records it. Distinct from the benign "not closed volumes" message, which
# is ordinary geometry: an open sheet is a normal surfacing result.
UNION_FAILED = "boolean union failed"


def meta_path(bench: str, adapter: str, run: str, sketch: str, part: str) -> Path:
    if part == NONE:
        return (
            benchmarks.bench_dir(bench) / adapter / run / f"{sketch}.meta.json"
        )
    return benchmarks.part_dir(bench, adapter, run, sketch) / f"{part}.meta.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sketch", required=True)
    ap.add_argument("--part", default=NONE, help="ordinal like part_07, or -")
    ap.add_argument("--part-id", default=NONE, help="the sketch's id for that part")
    ap.add_argument("--force", action="store_true", help="redo a finished cell")
    args = ap.parse_args()

    is_part = args.part != NONE
    label = f"{args.adapter}/{args.run}/{args.sketch}" + (f"/{args.part}" if is_part else "")

    if args.adapter not in ADAPTERS:
        print(f"unknown adapter {args.adapter!r}; have {sorted(ADAPTERS)}", file=sys.stderr)
        return 1

    # --- resume: a finished cell is left alone --------------------------
    if is_part:
        done = benchmarks.part_dir(
            args.bench, args.adapter, args.run, args.sketch
        ) / f"{args.part}.glb"
    else:
        done = (
            benchmarks.bench_dir(args.bench) / args.adapter / args.run
            / f"{args.sketch}.glb"
        )
    if done.is_file() and not args.force:
        print(f"skip {label}: already done ({done})")
        return 0

    # --- what to run -----------------------------------------------------
    progress = load_progress(args.bench)
    runs = progress.get("runs", {}).get(args.adapter, [])
    matched = [r for r in runs if r["id"] == args.run]
    if not matched:
        print(
            f"no run {args.run!r} for adapter {args.adapter!r} in progress.json",
            file=sys.stderr,
        )
        return 1
    options = dict(matched[0].get("options", {}))

    sketch = load_sketch(args.bench, args.sketch)
    if is_part:
        # one part surfaced standalone, exactly as sf3d's _make_proxy runs
        # another adapter over one unit's strokes
        sketch = sub_sketch(sketch, args.part_id)
        options = options_for_part(args.adapter, options)

    # --- run --------------------------------------------------------------
    log_lines: list[str] = []
    emitted: list[str] = []

    def report(progress_frac: float = 0.0, message: str = "") -> None:
        if message:
            print(f"[{progress_frac:5.1%}] {message}", flush=True)

    def log(line: str) -> None:
        log_lines.append(line)
        print(line, flush=True)

    def emit(name: str, glb: bytes) -> None:
        # nothing polls for partials here; the names are kept because
        # progress.json records them per cell and finalize.py restores that
        if name not in emitted:
            emitted.append(name)

    print(f"=== {label}")
    print(f"    options: {json.dumps(options, sort_keys=True)}")
    started = time.time()
    meta = {
        "adapter": args.adapter, "run": args.run, "sketch": args.sketch,
        "part": args.part, "partId": args.part_id, "options": options,
    }

    try:
        glb = ADAPTERS[args.adapter].run(sketch, options, report, log, emit)
    except Exception as exc:
        elapsed = time.time() - started
        traceback.print_exc()
        meta |= {
            "state": "error",
            "seconds": round(elapsed, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }
        path = meta_path(args.bench, args.adapter, args.run, args.sketch, args.part)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=1))
        print(f"FAILED {label} after {elapsed:.1f}s", file=sys.stderr)
        return 1

    elapsed = time.time() - started

    if is_part:
        out = benchmarks.save_part_result(
            args.bench, args.adapter, args.run, args.sketch, args.part, glb
        )
    else:
        out = benchmarks.save_result(
            args.bench, args.adapter, args.run, args.sketch, glb
        )

    degraded = any(UNION_FAILED in line for line in log_lines)
    meta |= {
        "state": "done",
        "seconds": round(elapsed, 1),
        "parts": emitted,
        "bytes": len(glb),
        "unionFailed": degraded,
    }
    path = meta_path(args.bench, args.adapter, args.run, args.sketch, args.part)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
