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
#   node_modules, dist
#               hundreds of MB, and platform-specific — `npm ci` rebuilds them
#               on the far side from the lockfile, which is why that lockfile is
#               in the bundle.
#   jobs/       stale method scratch
#   bench_vns/  the VNS fork study, a separate concern
#   SampleModels
#               raw glTF; the sketches are already preprocessed
#
# What is deliberately kept, because it is not in git and cannot be refetched
# without network on the far side:
#   methods/NeuralSketch2Surf/checkpoints/best_model_jit.pt   (54MB)
#
# The frontend sources ride along — src/, index.html, vite.config.ts,
# tsconfig.json, package.json, package-lock.json — so `npm run dev` runs on the
# far side and is reached through an SSH tunnel. That is the only way to watch a
# sweep from the machine it runs on: the editor's Surfacer and benchmark panels
# talk to the surfacing server through the /api proxy in vite.config.ts, so the
# browser has to be same-origin with a Vite serving *there*, not here. A few
# hundred KB of TypeScript, against nothing else that shows a running job.
#
# tar, not zip: it preserves the executable bit on these scripts and on
# anything built under methods/.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT="${OUT:-icf-bundle.tar.gz}"

# local/ rides along: the same sweep over the same cells, scheduled on a
# machine's own GPUs instead of Slurm, and it imports cells.py and run_one.py
# out of cluster/. Sending one without the other leaves a half-usable checkout,
# and it is a few KB. The --exclude='logs/*' below already covers local/logs.
INCLUDE=(cluster local surfacing-server
         src index.html vite.config.ts tsconfig.json
         package.json package-lock.json)
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

to run the editor there and reach it from here (node_modules is not in the
bundle, so the first unpack needs one networked install):

  npm ci                                    # on the far side, once per bump
  npm run dev                               # binds 127.0.0.1:5173 there
  ssh -N -L 5173:localhost:5173 \$UUN@icf    # here; then open localhost:5173
EOF
if [[ $# -eq 0 ]]; then
    cat <<'EOF'

to send a benchmark separately (inputs only — results stay here):

  tar czf bench.tar.gz -C benchmarks --exclude='*.glb' <benchmark-id>
  scp -o ProxyJump=$UUN@student.ssh.inf.ed.ac.uk bench.tar.gz $UUN@icf:~/
  tar xzf ~/bench.tar.gz -C ~/InteractiveArtiDesign/benchmarks
EOF
fi
