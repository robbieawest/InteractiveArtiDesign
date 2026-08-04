#!/usr/bin/env python3
"""Turn a benchmark into manifests and the sbatch lines that run them.

Submits nothing. It writes one manifest per adapter and prints the commands,
because deciding when to occupy a partition is a person's call, not a script's.

    cluster/.venv-server/bin/python cluster/plan_sweep.py 2026-08-02T02-09-46

Everything about *what* runs comes from the benchmark: the adapters are
progress.json's keys, the runs are its run lists (including each run's own
part_based flag), the sketches are whatever is in sketches/, and the parts are
whatever carries strokes. profiles.json contributes only resource requests.

Re-running is safe and is the intended way to resume: the manifests are
regenerated whole, and run_cell.py skips any cell whose result already exists,
so resubmitting the same arrays picks up exactly what is missing.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import (  # noqa: E402
    BENCH_ROOT,
    REPO_ROOT,
    Cell,
    enumerate_cells,
    load_progress,
    profile_for,
    seed_inputs,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmark", help="benchmark id (the folder under benchmarks/)")
    ap.add_argument("-p", "--partition", default="Teaching")
    ap.add_argument("--profiles", default=str(Path(__file__).parent / "profiles.json"))
    ap.add_argument("--adapter", help="plan only this adapter")
    ap.add_argument(
        "--out",
        default=None,
        help="manifest directory (default cluster/manifests/<benchmark>)",
    )
    args = ap.parse_args()

    profiles = json.loads(Path(args.profiles).read_text())

    # done here, once, rather than in each of several hundred array tasks
    seeded = seed_inputs(args.benchmark)
    if seeded:
        print(seeded)

    grouped = enumerate_cells(args.benchmark, profiles, args.adapter)
    if not grouped:
        print("nothing to plan: progress.json declares no runs", file=sys.stderr)
        return 1

    out_dir = Path(args.out or Path(__file__).parent / "manifests" / args.benchmark)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(args.benchmark)
    submits: list[str] = []

    print(f"benchmark {args.benchmark}")
    print(f"manifests -> {out_dir}")
    print(f"results   -> {BENCH_ROOT / args.benchmark}\n")
    print(f"{'adapter':10} {'runs':>5} {'tasks':>7}  {'split':<6} {'gres':<26} time")
    print("-" * 78)

    for adapter, cells in grouped.items():
        profile = profile_for(profiles, adapter)
        manifest = out_dir / f"{adapter}.tsv"
        manifest.write_text("".join(c.to_row() + "\n" for c in cells))

        n_runs = len(progress["runs"][adapter])
        split_cells = sum(1 for c in cells if c.is_part)
        # a run's part_based decides per run, so an adapter can legitimately be
        # mixed: some runs split, others whole
        shape = (
            "all" if split_cells == len(cells)
            else "none" if split_cells == 0
            else "mixed"
        )
        print(
            f"{adapter:10} {n_runs:>5} {len(cells):>7}  {shape:<6} "
            f"{profile['gres']:<26} {profile['time']}"
        )

        throttle = profile.get("throttle", 8)
        # --parsable prints the bare job id and nothing else, so the finalize
        # dependency populates itself. Reading the id off the "Submitted batch
        # job N" line by hand is how you end up with an empty --dependency=
        # afterany: and a "Job dependency problem" rejection.
        submits.append(
            f"{adapter.upper()}_ID=$(sbatch --parsable "
            f"-p {args.partition} --gres=gpu:{profile['gres']} "
            f"-t {profile['time']} --array=0-{len(cells) - 1}%{throttle} "
            f"cluster/job.sh {manifest.relative_to(REPO_ROOT)} {args.benchmark})"
        )

    total = sum(len(c) for c in grouped.values())
    print("-" * 78)
    print(f"{'':10} {'':>5} {total:>7}  tasks total\n")

    print("# paste these as-is — each captures its job id for the line below:")
    for line in submits:
        print(line)
    print("echo " + " ".join(f"{a.upper()}_ID=${a.upper()}_ID" for a in grouped))

    ids = ":".join(f"${a.upper()}_ID" for a in grouped)
    print(
        "\n# then merge per-part results and rebuild progress.json "
        "(CPU only, no --gres). afterany, not afterok, so a few failed cells\n"
        "# do not block merging everything that worked:"
    )
    print(
        f"sbatch -p {args.partition} -t 00:30:00 "
        f"--dependency=afterany:{ids} cluster/finalize.sh {args.benchmark}"
    )
    print(
        "\n# if the arrays have already finished, or you lost the ids, drop\n"
        "# the dependency entirely — an empty afterany: is what Slurm rejects\n"
        "# with 'Job dependency problem':"
    )
    print(
        f"sbatch -p {args.partition} -t 00:30:00 "
        f"cluster/finalize.sh {args.benchmark}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
