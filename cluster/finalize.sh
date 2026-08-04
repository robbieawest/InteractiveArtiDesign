#!/bin/bash
#SBATCH --job-name=surface-finalize
#SBATCH --output=cluster/logs/finalize_%j.out
#SBATCH --error=cluster/logs/finalize_%j.out
#
# Merge per-part surfaces and rebuild progress.json. CPU only — no --gres.
# Gate it on every array finishing, with afterany rather than afterok so a few
# failed cells do not block merging everything that worked:
#
#   sbatch -p Teaching -t 00:30:00 \
#          --dependency=afterany:$VNS_ID:$NS2S_ID cluster/finalize.sh <bench>

set -euo pipefail

BENCH="${1:?usage: finalize.sh <benchmark-id>}"

# See the note in job.sh: this runs from Slurm's spool copy, so it has to find
# the repo through SLURM_SUBMIT_DIR rather than through its own path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -r "$SLURM_SUBMIT_DIR/cluster/env.sh" ]]; then
    source "$SLURM_SUBMIT_DIR/cluster/env.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
fi

"$SERVER_PYTHON" "$REPO/cluster/finalize.py" "$BENCH"
