"""Benchmark storage: source-folder scanning and the on-disk result tree.

A benchmark is one timestamped folder under <repo>/benchmarks:

    benchmarks/2026-07-28T15-04-22/
      sketches/            preprocessed inputs (sketch .json, articulation
                           included, in the pose they are to be surfaced in) —
                           selectable as a source folder for a rerun, which
                           then needs no preprocessing at all
      progress.json        bench state, so a reload or a trip back to the
                           editor doesn't lose where the run got to
      <adapter>/<run>/<sketch>.glb

Only the server writes here: the browser cannot, and results should outlive
the tab. glTF sources are converted to sketch documents client-side (the
importer is three.js), so this module only ever reads raw bytes and writes
whatever JSON the client hands back.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
BENCH_ROOT = REPO_ROOT / "benchmarks"

# folder/file names we generate or accept; anything else is rejected rather
# than sanitized, so a surprising name fails loudly instead of writing
# somewhere unexpected
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class BenchmarkError(ValueError):
    """Bad path or name — surfaced to the client as a 400."""


def _safe(name: str, what: str) -> str:
    if not SAFE_NAME.match(name) or ".." in name:
        raise BenchmarkError(f"unsafe {what}: {name!r}")
    return name


def resolve_source(path: str) -> Path:
    """A user-chosen source folder. Absolute paths are honoured (this is a
    single-user localhost sidecar and sketches live wherever the user keeps
    them); relative ones resolve against the repo root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise BenchmarkError(f"not a directory: {candidate}")
    return candidate


def scan_source(path: str) -> dict[str, Any]:
    """Find the surfaceable inputs in a folder.

    Two shapes are recognised, matching how sketches actually arrive:
      - a loose `.json` sketch document, and
      - a subfolder holding a `.gltf` (its `.bin` sits alongside and is
        fetched by the loader on demand), i.e. the SampleModels layout.
    The glTF entries still need client-side preprocessing; the json ones are
    ready to run, which is what makes a previous benchmark's `sketches/`
    folder directly reusable as a source."""
    root = resolve_source(path)
    entries: list[dict[str, Any]] = []

    for child in sorted(root.iterdir()):
        if child.is_file() and child.suffix.lower() == ".json":
            entries.append({
                "name": child.stem,
                "kind": "json",
                "path": str(child),
            })
        elif child.is_dir():
            gltfs = sorted(child.glob("*.gltf")) + sorted(child.glob("*.glb"))
            if gltfs:
                entries.append({
                    "name": child.name,
                    "kind": "gltf",
                    "path": str(gltfs[0]),
                })

    return {"dir": str(root), "entries": entries}


def read_source_file(path: str) -> bytes:
    """Raw bytes of one file inside a scanned source folder — the client
    fetches the .gltf and its .bin through this to run the importer."""
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise BenchmarkError(f"not a file: {target}")
    return target.read_bytes()


def bench_dir(benchmark_id: str) -> Path:
    return BENCH_ROOT / _safe(benchmark_id, "benchmark id")


def sketches_dir(benchmark_id: str) -> Path:
    return bench_dir(benchmark_id) / "sketches"


def save_sketch(benchmark_id: str, name: str, document: Any) -> Path:
    """Store one preprocessed sketch document. Called once per input before
    any run starts; reruns of the same benchmark id overwrite in place."""
    path = sketches_dir(benchmark_id) / f"{_safe(name, 'sketch name')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def list_sketches(benchmark_id: str) -> list[str]:
    directory = sketches_dir(benchmark_id)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def read_sketch(benchmark_id: str, name: str) -> Any:
    path = sketches_dir(benchmark_id) / f"{_safe(name, 'sketch name')}.json"
    if not path.is_file():
        raise BenchmarkError(f"no such sketch: {name}")
    return json.loads(path.read_text())


def copy_sketches(source_id: str, target_id: str) -> Path:
    """Start a new benchmark from another one's inputs and nothing else.

    The point is a clean rerun: same sketches, no results, no per-cell status,
    so the sweep starts from zero without disturbing the folder it came from
    (whose surfaces stay exactly as they are). The run configuration is
    carried over by the client, which writes the new progress.json itself."""
    source = sketches_dir(source_id)
    if not source.is_dir():
        raise BenchmarkError(f"benchmark has no sketches: {source_id}")
    target = bench_dir(target_id)
    if target.exists():
        raise BenchmarkError(f"benchmark already exists: {target_id}")

    destination = sketches_dir(target_id)
    destination.mkdir(parents=True)
    for path in sorted(source.glob("*.json")):
        shutil.copyfile(path, destination / path.name)
    return target


def save_result(
    benchmark_id: str, adapter: str, run: str, sketch: str, glb: bytes
) -> Path:
    """Write one finished surface into <bench>/<adapter>/<run>/<sketch>.glb."""
    path = (
        bench_dir(benchmark_id)
        / _safe(adapter, "adapter name")
        / _safe(run, "run name")
        / f"{_safe(sketch, 'sketch name')}.glb"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb)
    return path


def part_dir(benchmark_id: str, adapter: str, run: str, sketch: str) -> Path:
    """Where one cell's per-part surfaces live, when the cell was split across
    cluster tasks: <bench>/<adapter>/<run>/<sketch>/ — a sibling of the merged
    <sketch>.glb rather than a replacement for it, so the client keeps reading
    the same path it always has."""
    return (
        bench_dir(benchmark_id)
        / _safe(adapter, "adapter name")
        / _safe(run, "run name")
        / _safe(sketch, "sketch name")
    )


def save_part_result(
    benchmark_id: str, adapter: str, run: str, sketch: str, part: str, glb: bytes
) -> Path:
    """Write one part's surface. `part` is an ordinal like `part_07`, not the
    part's name: names are user-authored ("Part 1" — a space) and ids are
    nanoids that may lead with `-`, so neither survives `_safe`. The ordinal to
    id/name mapping is recorded in the sidecar written beside it."""
    path = part_dir(benchmark_id, adapter, run, sketch) / f"{_safe(part, 'part name')}.glb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb)
    return path


def list_part_results(
    benchmark_id: str, adapter: str, run: str, sketch: str
) -> list[Path]:
    """This cell's finished parts, in ordinal order. Empty when the cell was
    never split, or split but nothing succeeded."""
    directory = part_dir(benchmark_id, adapter, run, sketch)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("part_*.glb"))


def read_result(benchmark_id: str, adapter: str, run: str, sketch: str) -> bytes:
    path = (
        bench_dir(benchmark_id)
        / _safe(adapter, "adapter name")
        / _safe(run, "run name")
        / f"{_safe(sketch, 'sketch name')}.glb"
    )
    if not path.is_file():
        raise BenchmarkError(f"no result for {adapter}/{run}/{sketch}")
    return path.read_bytes()


def write_progress(benchmark_id: str, progress: Any) -> Path:
    """Persist the whole bench state as the client models it. Deliberately
    opaque to the server: the client owns the shape, this just has to survive
    a reload."""
    path = bench_dir(benchmark_id) / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress))
    return path


def read_progress(benchmark_id: str) -> Optional[Any]:
    path = bench_dir(benchmark_id) / "progress.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def list_benchmarks() -> list[dict[str, Any]]:
    """Existing benchmark folders, newest first — everything the reopen
    picker needs to describe one without loading it. Ids are timestamps, so
    sorting the names descending is newest-first."""
    if not BENCH_ROOT.is_dir():
        return []
    out = []
    for child in sorted(BENCH_ROOT.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        sketches = child / "sketches"
        out.append({
            "id": child.name,
            "sketches": len(list(sketches.glob("*.json"))) if sketches.is_dir() else 0,
            # finished surfaces on disk; a benchmark with none was prepared
            # but never started
            "results": sum(1 for _ in child.rglob("*.glb")),
            # without this there is no run configuration to restore, only
            # sketches — such a folder reopens as a fresh, unstarted bench
            "hasProgress": (child / "progress.json").is_file(),
        })
    return out
