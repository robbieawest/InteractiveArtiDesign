#!/bin/bash
# Bundle what the cluster needs, from your local checkout. Run locally.
#
#   cluster/pack.sh 2026-08-02T02-09-46
#
# What is deliberately left out, and why:
#   .venv, .venv-*
#               ~60GB locally, and ROCm-built — setup_venvs.sh rebuilds them
#               against CUDA on the far side. Both patterns: the server's own
#               environment is a bare `.venv`, which `.venv-*` does not match.
#
# What is deliberately kept, because it is not in git and cannot be refetched
# without network on the far side:
#   methods/NeuralSketch2Surf/checkpoints/best_model_jit.pt   (54MB)
#   jobs/       stale method scratch
#   bench_vns/  the VNS fork study, a separate concern
#   node_modules, dist, SampleModels
#               frontend and raw glTF; the sketches are already preprocessed
#
# tar, not zip: it preserves the executable bit on these scripts and on
# anything built under methods/.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BENCH="${1:?usage: pack.sh <benchmark-id> [more-benchmark-ids...]}"
OUT="${OUT:-icf-bundle.tar.gz}"

INCLUDE=(cluster surfacing-server)
for bench in "$@"; do
    [[ -d "benchmarks/$bench/sketches" ]] || {
        echo "no benchmarks/$bench/sketches" >&2; exit 1; }
    INCLUDE+=("benchmarks/$bench/sketches")
    [[ -f "benchmarks/$bench/progress.json" ]] &&
        INCLUDE+=("benchmarks/$bench/progress.json")
done

tar czf "$OUT" \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.venv' --exclude='.venv-*' --exclude='jobs' --exclude='bench_vns' \
    --exclude='node_modules' --exclude='logs/*' --exclude='manifests/*' \
    "${INCLUDE[@]}"

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
cat <<EOF

copy it up, then unpack into the path cluster/env.sh expects:

  scp -o ProxyJump=\$UUN@student.ssh.inf.ed.ac.uk $OUT \$UUN@icf:~/
  ssh -J \$UUN@student.ssh.inf.ed.ac.uk \$UUN@icf
  mkdir -p ~/InteractiveArtiDesign && tar xzf ~/$OUT -C ~/InteractiveArtiDesign
EOF
