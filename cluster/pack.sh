#!/bin/bash
# Bundle the code the cluster needs, from your local checkout. Run locally.
#
#   cluster/pack.sh                          code only
#   cluster/pack.sh 2026-08-02T02-09-46      code + that benchmark's inputs
#
# A benchmark is not part of the bundle's job. It is a self-contained folder of
# JSON that changes on a different schedule from the code, so sending one is an
# ordinary tar and needs nothing from this script:
#
#   tar czf bench.tar.gz -C benchmarks --exclude='*.glb' <benchmark-id>
#
# The argument form stays because sending both at once is convenient the first
# time, when neither is on the far side yet.
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
if [[ $# -eq 0 ]]; then
    echo "  code only — no benchmark inputs included"
else
    echo "  with inputs for: $*"
fi
cat <<EOF

copy it up and unpack over the repo. Unpacking on top of an existing checkout
is the normal way to update the far side: everything in the bundle is code,
and nothing under benchmarks/ is overwritten unless you named one.

  scp -o ProxyJump=\$UUN@student.ssh.inf.ed.ac.uk $OUT \$UUN@icf:~/
  ssh -J \$UUN@student.ssh.inf.ed.ac.uk \$UUN@icf
  mkdir -p ~/InteractiveArtiDesign && tar xzf ~/$OUT -C ~/InteractiveArtiDesign
EOF
if [[ $# -eq 0 ]]; then
    cat <<'EOF'

to send a benchmark separately (inputs only — results stay here):

  tar czf bench.tar.gz -C benchmarks --exclude='*.glb' <benchmark-id>
  scp -o ProxyJump=$UUN@student.ssh.inf.ed.ac.uk bench.tar.gz $UUN@icf:~/
  tar xzf ~/bench.tar.gz -C ~/InteractiveArtiDesign/benchmarks
EOF
fi
