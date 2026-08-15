#!/bin/bash
# Zip a benchmark's merged surfaces — every adapter, every run, one glb per
# sketch — together with the inputs the viewer needs to read them: sketches/
# and progress.json. The per-part cells under <adapter>/<run>/<sketch>/ are
# left out: they are inputs to finalize.py, not results, and they dominate the
# size.
#
# Run on the head node, like archive.sh:
#
#   cluster/archive_glb.sh 2026-08-13T02-14-41
#   cp cluster/archives/<bench>-glb.tar.gz /afs/inf.ed.ac.uk/user/<p>/<UUN>/...
#
# Run cluster/finalize.py first. A split cell has no <sketch>.glb until it is
# merged, so an unfinalized benchmark archives as an empty (or partial) tarball
# — which is why this prints the count of what it found before writing.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

BENCH="${1:?usage: archive_glb.sh <benchmark-id>}"
SRC="$RESULTS_ROOT/$BENCH"
[[ -d "$SRC" ]] || { echo "no such benchmark: $SRC" >&2; exit 1; }

OUT_DIR="$REPO/cluster/archives"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$BENCH-glb.tar.gz"

# Exactly <bench>/<adapter>/<run>/<sketch>.glb: the depth is what separates a
# merged surface from a per-part one, since part cells live one level deeper.
GLBS="$(mktemp)"
LIST="$(mktemp)"
trap 'rm -f "$GLBS" "$LIST"' EXIT
find "$SRC" -mindepth 3 -maxdepth 3 -name '*.glb' -printf "$BENCH/%P\n" \
    | sort > "$GLBS"

COUNT=$(wc -l < "$GLBS")
if [[ "$COUNT" -eq 0 ]]; then
    echo "no merged surfaces in $SRC — run cluster/finalize.py $BENCH first" >&2
    exit 1
fi

echo "=== $COUNT surfaces"
cut -d/ -f2-3 "$GLBS" | uniq -c | sed 's/^/  /'

# The inputs, so the tarball unpacks into a benchmark the viewer can open on
# its own. tar recurses into sketches/; both are skipped if absent rather than
# failing the archive, since the surfaces are the point.
cp "$GLBS" "$LIST"
for input in progress.json sketches; do
    if [[ -e "$SRC/$input" ]]; then
        echo "$BENCH/$input" >> "$LIST"
    else
        echo "  (no $input in $SRC)"
    fi
done

tar czf "$OUT" -C "$RESULTS_ROOT" -T "$LIST"

SKIPPED=$(find "$SRC" -mindepth 4 -name '*.glb' | wc -l)
echo
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "  skipped $SKIPPED per-part glb(s)"
