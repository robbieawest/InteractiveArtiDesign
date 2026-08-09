#!/usr/bin/env python3
"""One long-lived worker, pinned to one GPU by the environment its parent sets.

Reads cell rows on stdin, runs each, writes one status line per cell on stdout.
Spawned by run_sweep.py, which owns the pinning; nothing here knows or cares
which card it has.

Runs under the *server* interpreter, exactly as cluster/run_cell.py does.
`.venv-server` holds no torch: every adapter resolves its own interpreter from
`<METHOD>_PYTHON` and invokes the method as a subprocess, so one worker handles
every adapter and switching between them costs nothing at this level.

The loop is the point. adapters/ns2s.py's `WORKER` is a module-level singleton
that respawns only when its process has died, so consecutive cells of one
adapter share a single model load. Given the parent drains one adapter before
starting the next, `release_other_workers` never fires either, and nothing is
ever evicted mid-sweep.

Protocol, one line each way:

    <- READY                       once, at startup
    -> <cell row>                  a Cell.to_row(), tab separated
    <- DONE\t<status>\t<cell row>  status 0 ok, 1 failed
    -> STOP                        or EOF, to finish

Exactly one cell is ever outstanding: the parent sends the next only after
reading a DONE. That invariant is what lets it drive several of these with
`selectors` — two protocol lines can never sit in one read buffer, so a
readline() after a select() cannot swallow the next worker's turn.
"""

import argparse
import contextlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cluster"))
from cells import Cell  # noqa: E402
from run_one import run_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--force", action="store_true", help="redo finished cells")
    args = ap.parse_args()

    print("READY", flush=True)

    for line in sys.stdin:
        row = line.strip()
        if not row or row == "STOP":
            break
        try:
            cell = Cell.from_row(row)
        except ValueError as exc:
            print(f"malformed row from parent: {exc}", file=sys.stderr, flush=True)
            print(f"DONE\t1\t{row}", flush=True)
            continue

        # run_one writes the adapter's report/log/emit stream to stdout, which
        # here is the protocol channel. Send it to stderr — the parent keeps a
        # per-GPU log file — or the first progress line desynchronises the
        # parent's readline loop.
        with contextlib.redirect_stdout(sys.stderr):
            try:
                status = run_one(args.bench, cell, force=args.force)
            except Exception as exc:  # noqa: BLE001 — one bad cell must not end the sweep
                import traceback
                traceback.print_exc()
                print(f"worker error on {row}: {exc}", file=sys.stderr, flush=True)
                status = 1

        print(f"DONE\t{status}\t{row}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
