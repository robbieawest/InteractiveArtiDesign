#!/usr/bin/env python3
"""Run a whole benchmark on this machine, across its GPUs.

The local counterpart of cluster/plan_sweep.py + job.sh. What runs comes from
the same place and by the same rules — cells.py reads the benchmark's own
progress.json, so a run's `part_based` flag and profiles.json's `split` decide
cell shape here exactly as they do on Slurm.

Two scheduling decisions, both deliberate:

  adapters sequential.  release_other_workers() (adapters/common.py) evicts
      every other method's resident worker whenever a method starts, because
      one card cannot hold two models. Interleaving adapters would therefore
      reload ~13GB on every switch. Draining one adapter completely before
      starting the next means nothing is ever evicted, and the residency the
      adapters already implement finally pays off.

  cells within an adapter parallel, handed out dynamically.  Cell cost spans
      seconds (ns2s) to hours (vns), so a static split of the queue would leave
      one card idle through the whole tail. Each worker gets its next cell only
      when it reports the last one done.

One worker per GPU, never more: the ns2s resident worker is ~13GB and VNS
measures 12.4GB, so two workers on one 24GB card would OOM both.

Usage (source cluster/env.sh first — the workers need every <METHOD>_PYTHON,
not just their own, because sf3d runs a proxy adapter in-process):

    source cluster/env.sh
    "$SERVER_PYTHON" local/run_sweep.py <benchmark-id>
    "$SERVER_PYTHON" local/run_sweep.py <benchmark-id> --gpus 0 --adapter vns
"""

import argparse
import json
import os
import selectors
import subprocess
import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "local"
sys.path.insert(0, str(REPO / "cluster"))
from cells import Cell, enumerate_cells, seed_inputs  # noqa: E402


class GpuWorker:
    """One worker subprocess, pinned to one GPU."""

    def __init__(self, gpu: int, bench: str, scratch: Path, logs: Path,
                 python: str, force: bool):
        env = dict(os.environ)

        # Both variables, unconditionally. method_env() (adapters/common.py)
        # returns {**backend_defaults, **os.environ} — os.environ spread last —
        # so whichever one the active backend reads wins over the table's
        # default, and the other is simply inert.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["HIP_VISIBLE_DEVICES"] = str(gpu)

        # Per worker, never shared. prune_job_dirs() deletes all but the newest
        # few directories under JOBS_DIR at every job start, ranking across the
        # whole root; two workers sharing one would rank each other's live
        # scratch and delete it. Results are a different tree entirely
        # (SURFACING_BENCH_ROOT) and are never pruned by anything.
        jobs = scratch / f"gpu{gpu}"
        jobs.mkdir(parents=True, exist_ok=True)
        env["SURFACING_JOBS_DIR"] = str(jobs)
        # SURFACING_KEEP_JOBS is deliberately left alone. The cluster sets -1
        # because a task owns its root and dies after one cell; these workers
        # live for a whole sweep, so the default keep-3 rotation is what stops
        # scratch growing without bound (a part-based sf3d cell leaves ~275MB).

        self.gpu = gpu
        self.log_path = logs / f"gpu{gpu}.log"
        self._log = self.log_path.open("a", buffering=1)

        cmd = [python, str(HERE / "worker.py"), "--bench", bench]
        if force:
            cmd.append("--force")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            env=env, cwd=REPO, text=True, bufsize=1,
        )
        greeting = self.proc.stdout.readline().strip()
        if greeting != "READY":
            raise SystemExit(
                f"worker on gpu{gpu} did not start (said {greeting!r}); "
                f"see {self.log_path}"
            )

    def send(self, cell: Cell) -> None:
        self.proc.stdin.write(cell.to_row() + "\n")
        self.proc.stdin.flush()

    def stop(self) -> None:
        try:
            self.proc.stdin.write("STOP\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass  # already gone
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._log.close()


def drain(workers: list[GpuWorker], cells: list[Cell],
          sel: selectors.BaseSelector) -> list[str]:
    """Run every cell of one adapter across the workers. Returns failed rows."""
    queue = deque(cells)
    inflight = 0
    failures: list[str] = []

    for worker in workers:
        if queue:
            worker.send(queue.popleft())
            inflight += 1

    while inflight:
        for key, _ in sel.select():
            worker: GpuWorker = key.data
            line = worker.proc.stdout.readline()
            if not line:
                raise SystemExit(
                    f"worker on gpu{worker.gpu} died; see {worker.log_path}"
                )
            if not line.startswith("DONE\t"):
                continue  # not protocol; ignore rather than desynchronise
            _, status, row = line.rstrip("\n").split("\t", 2)
            inflight -= 1
            if status != "0":
                failures.append(row)
            mark = "ok  " if status == "0" else "FAIL"
            print(f"[gpu{worker.gpu}] {mark} {row}", flush=True)
            if queue:
                worker.send(queue.popleft())
                inflight += 1

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", help="benchmark id (the folder under benchmarks/)")
    ap.add_argument("--gpus", default=os.environ.get("SURFACING_GPUS", "0,1"),
                    help="comma-separated GPU indices, one worker each (default 0,1)")
    ap.add_argument("--adapter", help="run only this adapter")
    ap.add_argument("--profiles", default=str(REPO / "cluster" / "profiles.json"))
    ap.add_argument("--scratch", default=os.environ.get("SURFACING_SCRATCH"),
                    help="root for per-worker method scratch "
                         "(default surfacing-server/jobs, where it already goes)")
    ap.add_argument("--python", default=os.environ.get("SERVER_PYTHON", sys.executable),
                    help="interpreter for the workers (default: this one)")
    ap.add_argument("--force", action="store_true", help="redo finished cells")
    ap.add_argument("--dry-run", action="store_true", help="list cells and stop")
    ap.add_argument("--finalize", action="store_true",
                    help="run cluster/finalize.py afterwards")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        raise SystemExit("--gpus named no devices")

    profiles = json.loads(Path(args.profiles).read_text())
    if (note := seed_inputs(args.benchmark)):
        print(note)
    cells = enumerate_cells(args.benchmark, profiles, only_adapter=args.adapter)
    if not cells:
        raise SystemExit(
            f"no cells for {args.benchmark}"
            + (f" and adapter {args.adapter!r}" if args.adapter else "")
        )

    total = sum(len(v) for v in cells.values())
    for adapter in sorted(cells):
        print(f"{adapter:>8}: {len(cells[adapter])} cells")
    print(f"{'total':>8}: {total} cells over {len(gpus)} GPU(s)")
    if args.dry_run:
        for adapter in sorted(cells):
            for cell in cells[adapter]:
                print(cell.to_row())
        return 0

    # Under surfacing-server/ by default, matching JOBS_DIR's own default in
    # adapters/common.py — so gpu0/ and gpu1/ sit exactly where method scratch
    # already went, under the .gitignore rule that already covers it.
    scratch = Path(args.scratch) if args.scratch else REPO / "surfacing-server" / "jobs"
    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    workers = [GpuWorker(g, args.benchmark, scratch, logs, args.python, args.force)
               for g in gpus]
    sel = selectors.DefaultSelector()
    for worker in workers:
        sel.register(worker.proc.stdout, selectors.EVENT_READ, worker)

    failures: list[str] = []
    try:
        for adapter in sorted(cells):
            print(f"=== {adapter}: {len(cells[adapter])} cells", flush=True)
            failures += drain(workers, cells[adapter], sel)
    finally:
        sel.close()
        for worker in workers:
            worker.stop()

    if failures:
        print(f"\n{len(failures)} failed cell(s):")
        for row in failures:
            print(f"  {row}")
        print("per-GPU logs in", logs)
    else:
        print(f"\nall {total} cells ok")

    finalize = REPO / "cluster" / "finalize.py"
    if args.finalize:
        print(f"\n=== finalize {args.benchmark}", flush=True)
        subprocess.run([args.python, str(finalize), args.benchmark], cwd=REPO, check=False)
    else:
        print(f"\nnow run:  {args.python} {finalize} {args.benchmark}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
