#!/usr/bin/env python3
"""Run exactly one cell and write its surface. One Slurm array task.

An argv wrapper around `run_one`, which holds the work itself. The split is so
the local multi-GPU runner can call the same body many times inside one
long-lived process (see run_one.py's module docstring for why that matters);
job.sh's command line is unchanged by it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import NONE, Cell  # noqa: E402
from run_one import run_one  # noqa: E402


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

    return run_one(
        args.bench,
        Cell(args.adapter, args.run, args.sketch, args.part, args.part_id),
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
