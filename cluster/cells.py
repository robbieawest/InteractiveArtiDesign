"""Shared vocabulary for the cluster runner: what a cell is, and how one is
derived from a benchmark.

A *cell* is one unit of Slurm work. Either

    (adapter, run, sketch)              — one task surfaces the whole sketch
    (adapter, run, sketch, part)        — one task surfaces one part

Which of the two a benchmark produces is never decided here by adapter name.
It comes from that run's own `part_based` option in progress.json, and from
whether cluster/profiles.json prefers splitting for the adapter. A run with
part_based:false is always whole, whatever the profile says; a run with
part_based:true splits if the profile allows it and stays whole if not.

Both plan_sweep.py (which enumerates cells) and run_cell.py (which executes
one) import this, so the two can never disagree about what cell 37 means.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "surfacing-server"

# stands in for "no part" in the manifest, which is a TSV and so cannot carry
# an empty field unambiguously
NONE = "-"


class Cell(NamedTuple):
    adapter: str
    run: str
    sketch: str
    # ordinal used for the filename (`part_07`), or NONE for a whole cell
    part: str
    # the sketch's own id for that part, used to select strokes. Kept separate
    # from the ordinal so nothing depends on two scripts iterating in the same
    # order — the id is authoritative, the ordinal is just a safe filename.
    part_id: str

    @property
    def is_part(self) -> bool:
        return self.part != NONE

    def to_row(self) -> str:
        return "\t".join(self)

    @classmethod
    def from_row(cls, row: str) -> "Cell":
        fields = row.rstrip("\n").split("\t")
        if len(fields) != 5:
            raise ValueError(f"malformed manifest row: {row!r}")
        return cls(*fields)


def bench_dir(benchmark_id: str) -> Path:
    return REPO_ROOT / "benchmarks" / benchmark_id


def load_progress(benchmark_id: str) -> dict[str, Any]:
    path = bench_dir(benchmark_id) / "progress.json"
    if not path.is_file():
        raise SystemExit(
            f"no progress.json in {bench_dir(benchmark_id)} — a benchmark "
            "needs its run configuration, which is authored in the browser. "
            "Prepare the bench locally and copy the folder across."
        )
    return json.loads(path.read_text())


def load_sketch(benchmark_id: str, name: str) -> dict[str, Any]:
    return json.loads(
        (bench_dir(benchmark_id) / "sketches" / f"{name}.json").read_text()
    )


def list_sketches(benchmark_id: str) -> list[str]:
    directory = bench_dir(benchmark_id) / "sketches"
    if not directory.is_dir():
        raise SystemExit(f"no sketches/ in {bench_dir(benchmark_id)}")
    return sorted(p.stem for p in directory.glob("*.json"))


def parts_with_strokes(sketch: dict[str, Any]) -> list[tuple[str, str]]:
    """The parts a part-based run will actually surface, as (id, name).

    Declared parts and populated parts are not the same set — an empty part is
    ordinary (p3-a_puma declares 19 and only 14 carry strokes), and the
    adapters' own `group_strokes_by_part` silently drops the empty ones. If we
    enumerated `sketch["parts"]` instead we would submit tasks that die on
    "part-based surfacing needs strokes assigned to parts".

    Ordered by the document's own part list so the ordinals are stable across
    reruns; any id found on a stroke but never declared is appended rather than
    dropped, so a malformed document loses nothing silently.
    """
    on_strokes = [
        s["partId"] for s in sketch.get("strokes", []) if s.get("partId")
    ]
    present = set(on_strokes)
    names = {p["id"]: p.get("name", p["id"]) for p in sketch.get("parts", [])}

    ordered = [(pid, names[pid]) for pid in names if pid in present]
    seen = {pid for pid, _ in ordered}
    for pid in on_strokes:  # undeclared, keep document order of first sighting
        if pid not in seen:
            ordered.append((pid, pid))
            seen.add(pid)
    return ordered


def profile_for(profiles: dict[str, Any], adapter: str) -> dict[str, Any]:
    """Scheduling policy for an adapter, falling back to `default` so a
    benchmark naming an adapter this file has never heard of still runs."""
    merged = dict(profiles.get("default", {}))
    merged.update(profiles.get(adapter, {}))
    return merged


# --- part_based-gated parameters -------------------------------------------
#
# Some adapters expose a parameter that only applies in part-based mode and
# replaces a whole-object one (VNS: `iters_per_part` "replaces the whole-object
# iteration count" `n_samples`). A split task runs one part with
# part_based:false, which would silently pick up the whole-object value and
# quietly not reproduce the unsplit run. So the value has to be carried across.
#
# The mapping is read off the adapters' own `enabledWhen` metadata rather than
# tabulated here, so adding an adapter needs no change to this file.


def _gated_params(adapter_name: str) -> tuple[list[str], list[str]]:
    """(params active only when part_based, params active only when not)."""
    import sys

    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    from adapters import ADAPTERS  # noqa: E402  (needs the path above)

    on_true, on_false = [], []
    for param in ADAPTERS[adapter_name].params:
        gate = param.get("enabledWhen") or {}
        if gate.get("param") != "part_based":
            continue
        (on_true if gate.get("equals") is True else on_false).append(param["name"])
    return on_true, on_false


def options_for_part(adapter_name: str, options: dict[str, Any]) -> dict[str, Any]:
    """A run's options rewritten to surface a single part as a standalone
    whole-object job — which is exactly what sf3d's `_make_proxy` already does
    to run another adapter over one unit's strokes.

    Raises rather than guessing when an adapter's gated parameters are not the
    one-for-one shape this can translate: running with the wrong iteration
    count produces a plausible mesh that quietly is not what was configured,
    and that is far worse than a failed submission.
    """
    out = dict(options)
    out["part_based"] = False

    on_true, on_false = _gated_params(adapter_name)
    if not on_true and not on_false:
        return out
    if len(on_true) == 1 and len(on_false) == 1:
        if on_true[0] in out:
            out[on_false[0]] = out[on_true[0]]
        out.pop(on_true[0], None)
        return out

    raise SystemExit(
        f"cannot split {adapter_name!r} into per-part tasks: it has "
        f"{len(on_true)} parameter(s) gated on part_based=true "
        f"({', '.join(on_true) or 'none'}) and {len(on_false)} gated on false "
        f"({', '.join(on_false) or 'none'}). Only a one-for-one replacement "
        "can be translated automatically. Set \"split\": false for this "
        "adapter in cluster/profiles.json to run it whole instead."
    )


def sub_sketch(sketch: dict[str, Any], part_id: str) -> dict[str, Any]:
    """One part's strokes, shaped as a standalone sketch document."""
    strokes = [s for s in sketch.get("strokes", []) if s.get("partId") == part_id]
    if not strokes:
        raise SystemExit(f"part {part_id!r} has no strokes")
    return {"strokes": strokes, "parts": sketch.get("parts", [])}


def enumerate_cells(
    benchmark_id: str,
    profiles: dict[str, Any],
    only_adapter: Optional[str] = None,
) -> dict[str, list[Cell]]:
    """Every cell the benchmark implies, grouped by adapter.

    Nothing here is specific to any particular benchmark: the adapters come
    from progress.json's own keys, the runs from its run lists, the sketches
    from the folder, and the parts from each sketch's strokes.
    """
    progress = load_progress(benchmark_id)
    sketches = list_sketches(benchmark_id)
    parts_cache: dict[str, list[tuple[str, str]]] = {}

    out: dict[str, list[Cell]] = {}
    for adapter, runs in sorted(progress.get("runs", {}).items()):
        if only_adapter and adapter != only_adapter:
            continue
        prefers_split = bool(profile_for(profiles, adapter).get("split", True))
        cells: list[Cell] = []
        for run in runs:
            run_id = run["id"]
            part_based = bool(run.get("options", {}).get("part_based", False))
            split = prefers_split and part_based
            for sketch in sketches:
                if not split:
                    cells.append(Cell(adapter, run_id, sketch, NONE, NONE))
                    continue
                if sketch not in parts_cache:
                    parts_cache[sketch] = parts_with_strokes(
                        load_sketch(benchmark_id, sketch)
                    )
                parts = parts_cache[sketch]
                if not parts:
                    # segmented nowhere: the adapter would raise on this, so
                    # fall back to the whole sketch rather than submit a task
                    # that cannot succeed
                    cells.append(Cell(adapter, run_id, sketch, NONE, NONE))
                    continue
                for index, (part_id, _name) in enumerate(parts):
                    cells.append(
                        Cell(adapter, run_id, sketch, f"part_{index:02d}", part_id)
                    )
        if cells:
            out[adapter] = cells
    return out
