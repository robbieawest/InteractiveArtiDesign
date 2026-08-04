#!/bin/bash
#SBATCH --job-name=surface
#SBATCH --output=cluster/logs/%A_%a.out
#SBATCH --error=cluster/logs/%A_%a.out
#
# One array task = one cell. Submit from the repo root; partition, --gres,
# --time and --array come from the command line (plan_sweep.py prints them),
# so nothing benchmark-specific is baked in here.
#
#   sbatch -p Teaching --gres=gpu:h200_1g.18gb:1 -t 04:00:00 --array=0-227%16 \
#          cluster/job.sh cluster/manifests/<bench>/vns.tsv <bench>

set -uo pipefail

MANIFEST="${1:?usage: job.sh <manifest.tsv> <benchmark-id>}"
BENCH="${2:?usage: job.sh <manifest.tsv> <benchmark-id>}"

# Not $(dirname $BASH_SOURCE): sbatch copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script on the compute node and runs it from
# there, so this file's own path says nothing about where the repo is. Under
# `set -u` that failure is silent and total — env.sh does not load, $REPO is
# unbound, and the first command substitution to mention it dies taking any
# `||` fallback inside it with it. Every task exits 1 in under a second.
#
# SLURM_SUBMIT_DIR is the directory sbatch was called from, which is the repo
# root (the sbatch lines plan_sweep.py prints use paths relative to it, as do
# the --output paths above). Outside Slurm, fall back to this script's dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -r "$SLURM_SUBMIT_DIR/cluster/env.sh" ]]; then
    ENV_SH="$SLURM_SUBMIT_DIR/cluster/env.sh"
else
    ENV_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
fi
if [[ ! -r "$ENV_SH" ]]; then
    echo "cannot find cluster/env.sh (looked at $ENV_SH)." >&2
    echo "Submit from the repo root, or export REPO and re-run." >&2
    exit 1
fi
source "$ENV_SH"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
# absolute first, then as given (relative to the submit dir, which is cwd).
# Spelled out rather than `a || b` in one substitution: under `set -u` an
# unbound variable aborts the whole subshell, so the fallback would never run.
MANIFEST_PATH="$REPO/$MANIFEST"
[[ -f "$MANIFEST_PATH" ]] || MANIFEST_PATH="$MANIFEST"
if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "no manifest at $REPO/$MANIFEST or $MANIFEST" >&2
    echo "Run plan_sweep.py to regenerate it, and submit from the repo root." >&2
    exit 1
fi
ROW=$(sed -n "$((TASK + 1))p" "$MANIFEST_PATH")
if [[ -z "$ROW" ]]; then
    echo "no row $TASK in $MANIFEST_PATH ($(wc -l < "$MANIFEST_PATH") rows)" >&2
    exit 1
fi
IFS=$'\t' read -r ADAPTER RUN SKETCH PART PART_ID <<< "$ROW"

echo "host=$(hostname)  task=$TASK  gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Results must not land on node-local scratch: it is wiped, so an eight-hour
# VNS cell that finishes at 04:00 would be gone before anyone looked at it.
# Checked before the work, not after — the whole point is to fail in seconds
# rather than at the end of a GPU allocation.
mkdir -p "$RESULTS_ROOT" || { echo "cannot create $RESULTS_ROOT" >&2; exit 1; }
# pwd -P, so a symlink into scratch is caught too
RESULTS_REAL=$(cd "$RESULTS_ROOT" && pwd -P)
case "$RESULTS_REAL/" in
    "$SCRATCH_ROOT"/*|"$SCRATCH_ROOT"/|/disk/scratch/*|/tmp/*)
        echo "refusing to run: results root $RESULTS_REAL is on node-local" >&2
        echo "scratch, which is wiped. Set RESULTS_ROOT to a path under" >&2
        echo "\$HOME (see cluster/env.sh) and resubmit." >&2
        exit 1 ;;
esac
[[ -w "$RESULTS_REAL" ]] || { echo "results root not writable: $RESULTS_REAL" >&2; exit 1; }
echo "results=$RESULTS_REAL"

# Node-local scratch, unique per task. Two array tasks do land on one node, so
# a shared directory would have them pruning each other's working files.
export SURFACING_JOBS_DIR="$SCRATCH_ROOT/${SLURM_JOB_ID:-local}_${TASK}"
mkdir -p "$SURFACING_JOBS_DIR" || {
    echo "cannot create scratch $SURFACING_JOBS_DIR" >&2; exit 1; }
echo "scratch=$SURFACING_JOBS_DIR"

"$SERVER_PYTHON" "$REPO/cluster/run_cell.py" \
    --bench "$BENCH" --adapter "$ADAPTER" --run "$RUN" \
    --sketch "$SKETCH" --part "$PART" --part-id "$PART_ID"
STATUS=$?

# Clean up after a success, keep the evidence after a failure. A part-based
# sf3d cell leaves ~275MB, so this cannot simply accumulate — but deleting the
# scratch of the run you need to debug is the one thing worse than that.
if [[ $STATUS -eq 0 ]]; then
    rm -rf "$SURFACING_JOBS_DIR"
else
    echo "kept scratch for inspection: $SURFACING_JOBS_DIR" >&2
fi
exit $STATUS
