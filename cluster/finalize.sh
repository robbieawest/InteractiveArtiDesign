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
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

"$SERVER_PYTHON" "$REPO/cluster/finalize.py" "$BENCH"
