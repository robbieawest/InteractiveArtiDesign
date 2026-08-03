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

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
ROW=$(sed -n "$((TASK + 1))p" "$REPO/$MANIFEST" 2>/dev/null || sed -n "$((TASK + 1))p" "$MANIFEST")
if [[ -z "$ROW" ]]; then
    echo "no row $TASK in $MANIFEST" >&2
    exit 1
fi
IFS=$'\t' read -r ADAPTER RUN SKETCH PART PART_ID <<< "$ROW"

echo "host=$(hostname)  task=$TASK  gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Node-local scratch, unique per task. Two array tasks do land on one node, so
# a shared directory would have them pruning each other's working files.
export SURFACING_JOBS_DIR="$SCRATCH_ROOT/${SLURM_JOB_ID:-local}_${TASK}"
mkdir -p "$SURFACING_JOBS_DIR"
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
