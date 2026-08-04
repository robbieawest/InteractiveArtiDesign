#!/bin/bash
# Zip a finished benchmark for the manual copy to AFS. Run on the head node:
# AFS is reachable there, and a batch job's Kerberos token may not be, so
# writing to /afs from inside a job can fail hours in.
#
#   cluster/archive.sh 2026-08-02T02-09-46
#   cp cluster/archives/<bench>.tar.gz /afs/inf.ed.ac.uk/user/<p>/<UUN>/...

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

BENCH="${1:?usage: archive.sh <benchmark-id>}"
SRC="$RESULTS_ROOT/$BENCH"
[[ -d "$SRC" ]] || { echo "no such benchmark: $SRC" >&2; exit 1; }

OUT_DIR="$REPO/cluster/archives"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$BENCH.tar.gz"

tar czf "$OUT" -C "$RESULTS_ROOT" "$BENCH"

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "  $(find "$SRC" -name '*.glb' | wc -l) surfaces, $(ls "$SRC/sketches" | wc -l) sketches"
echo
echo "copy to AFS from the head node, then unpack locally:"
echo "  cp $OUT /afs/inf.ed.ac.uk/user/<prefix>/$UUN/"
