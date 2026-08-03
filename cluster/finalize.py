#!/usr/bin/env python3
"""After a sweep: merge per-part surfaces, then rebuild progress.json.

Two jobs that both have to happen once, after everything else, in the server
environment (trimesh + manifold3d) and without a GPU.

  merge      a split cell's parts are separate .glb files; the client reads a
             single <sketch>.glb. This runs the same `combine_meshes` the
             adapters use for an in-process part-based run, so a split cell and
             an unsplit one differ in scheduling and not in output.

  reconcile  progress.json is the client's file — the server treats it as
             opaque and the runner never writes it, because hundreds of tasks
             sharing one file on NFS would corrupt it. Its `runs` config
             survived the trip; only its per-cell `status` map is stale. This
             rebuilds that map from what is on disk, so reopening the benchmark
             in the browser shows the grid populated.

Parts that failed are reported and skipped: a cell merges whatever succeeded
rather than being lost entirely, and is marked degraded so the gaps are
findable afterwards.
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import (  # noqa: E402
    NONE,
    SERVER_DIR,
    bench_dir,
    enumerate_cells,
    load_progress,
)

sys.path.insert(0, str(SERVER_DIR))
import benchmarks  # noqa: E402
from adapters.common import combine_meshes  # noqa: E402

# progress.json keys its status map on a NUL-joined triple
SEP = "\0"


def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(io.BytesIO(path.read_bytes()), file_type="glb")
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(loaded.dump())
    return loaded


def merge_cell(bench: str, adapter: str, run: str, sketch: str, force: bool) -> str:
    """Combine one cell's per-part surfaces into <sketch>.glb. Returns a short
    status word for the report."""
    merged = bench_dir(bench) / adapter / run / f"{sketch}.glb"
    parts = benchmarks.list_part_results(bench, adapter, run, sketch)
    if not parts:
        return "no-parts"
    if merged.is_file() and not force:
        return "already-merged"

    meshes = []
    for path in parts:
        try:
            meshes.append(_load_mesh(path))
        except Exception as exc:
            print(f"  ! unreadable {path.name}: {exc}", file=sys.stderr)
    if not meshes:
        return "all-parts-unreadable"

    combined = (
        combine_meshes(meshes, lambda line: print(f"  {line}"))
        if len(meshes) > 1
        else meshes[0]
    )
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_bytes(combined.export(file_type="glb"))
    return f"merged {len(meshes)}/{len(parts)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmark")
    ap.add_argument("--profiles", default=str(Path(__file__).parent / "profiles.json"))
    ap.add_argument("--force", action="store_true", help="re-merge finished cells")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--no-reconcile", action="store_true")
    args = ap.parse_args()

    bench = args.benchmark
    profiles = json.loads(Path(args.profiles).read_text())
    grouped = enumerate_cells(bench, profiles)

    # the cells collapse to one status entry per (adapter, run, sketch): the
    # split is a scheduling detail and progress.json has never known about it
    triples: dict[tuple[str, str, str], list[str]] = {}
    for cells in grouped.values():
        for cell in cells:
            triples.setdefault((cell.adapter, cell.run, cell.sketch), [])
            if cell.is_part:
                triples[(cell.adapter, cell.run, cell.sketch)].append(cell.part)

    # --- merge -----------------------------------------------------------
    if not args.no_merge:
        print("=== merging per-part cells")
        for (adapter, run, sketch), parts in sorted(triples.items()):
            if not parts:
                continue  # never split; the adapter already wrote one glb
            status = merge_cell(bench, adapter, run, sketch, args.force)
            print(f"  {adapter}/{run}/{sketch}: {status}")

    # --- reconcile --------------------------------------------------------
    if args.no_reconcile:
        return 0

    print("\n=== rebuilding progress.json")
    progress = load_progress(bench)
    status: dict[str, dict] = {}
    counts = {"done": 0, "error": 0, "pending": 0}
    degraded: list[str] = []

    for (adapter, run, sketch), parts in sorted(triples.items()):
        merged = bench_dir(bench) / adapter / run / f"{sketch}.glb"
        cell_dir = benchmarks.part_dir(bench, adapter, run, sketch)

        # every sidecar this cell produced, split or not
        metas = []
        whole_meta = bench_dir(bench) / adapter / run / f"{sketch}.meta.json"
        if whole_meta.is_file():
            metas.append(json.loads(whole_meta.read_text()))
        if cell_dir.is_dir():
            for path in sorted(cell_dir.glob("part_*.meta.json")):
                metas.append(json.loads(path.read_text()))

        errors = [m for m in metas if m.get("state") == "error"]
        if any(m.get("unionFailed") for m in metas):
            degraded.append(f"{adapter}/{run}/{sketch}")

        if merged.is_file():
            state = "done"
            if errors:
                message = f"completed with {len(errors)} failed part(s)"
            else:
                message = "completed"
        elif errors:
            state = "error"
            message = errors[0].get("error", "failed")
        else:
            state = "pending"
            message = ""

        counts[state] += 1
        # part names as the client records them: for an unsplit cell that is
        # whatever the adapter emitted; for a split one, the ordinals merged
        names: list[str] = []
        for meta in metas:
            if meta.get("state") != "done":
                continue
            names.extend(meta.get("parts") or ([meta["part"]] if meta.get("part", NONE) != NONE else []))

        status[SEP.join((adapter, run, sketch))] = {
            "state": state,
            "progress": 1 if state == "done" else 0,
            "message": message,
            "parts": names,
        }

    progress["status"] = status
    (bench_dir(bench) / "progress.json").write_text(json.dumps(progress))

    print(
        f"  {counts['done']} done, {counts['error']} error, "
        f"{counts['pending']} pending  ({len(status)} cells)"
    )
    if degraded:
        print(
            f"\n  WARNING: {len(degraded)} cell(s) fell back to concatenation "
            "because the boolean union failed — check manifold3d and networkx "
            "in .venv-server:"
        )
        for name in degraded[:10]:
            print(f"    {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
