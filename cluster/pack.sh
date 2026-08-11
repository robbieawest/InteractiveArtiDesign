#!/bin/bash
# Bundle the code the cluster needs, from your local checkout. Run locally.
#
#   cluster/pack.sh                          code only
#   cluster/pack.sh 2026-08-02T02-09-46      code + that benchmark's inputs
#   cluster/pack.sh --changed                only what git says you changed
#   cluster/pack.sh --changed=origin/main    ...since a ref, commits included
#
# --changed packs the tracked files that differ from a base ref (HEAD by
# default, so: your uncommitted edits). It is for the second and later sends,
# where the far side already has a checkout and a full bundle is a few hundred
# MB to move three files. Two things it deliberately does not do:
#
#   * untracked files are not in it. That is the whole meaning of "tracked",
#     and it is the trap — a file you have just created is exactly the kind you
#     want to send. The script lists any it finds and makes you `git add` them
#     (an intent-to-add, `git add -N`, is enough) or send a full bundle.
#   * deletions do not propagate. Unpacking only ever writes files, so a file
#     you deleted here stays on the far side. Delete it there by hand.
#
# A named benchmark is still packed whole under --changed: you asked for it by
# name, and its inputs are not the thing being iterated on.
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

BASE=""          # empty = pack everything; otherwise the ref to diff against
BENCHES=()
for arg in "$@"; do
    case "$arg" in
        --changed)   BASE="HEAD" ;;
        --changed=*) BASE="${arg#--changed=}" ;;
        -*) echo "unknown flag: $arg" >&2; exit 1 ;;
        *)  BENCHES+=("$arg") ;;
    esac
done

# local/ rides along: the same sweep over the same cells, scheduled on a
# machine's own GPUs instead of Slurm, and it imports cells.py and run_one.py
# out of cluster/. Sending one without the other leaves a half-usable checkout,
# and it is a few KB. The --exclude='logs/*' below already covers local/logs.
CODE=(cluster local surfacing-server
      src index.html vite.config.ts tsconfig.json
      package.json package-lock.json)
INCLUDE=("${CODE[@]}")
for bench in ${BENCHES+"${BENCHES[@]}"}; do
    [[ -d "benchmarks/$bench/sketches" ]] || {
        echo "no benchmarks/$bench/sketches" >&2; exit 1; }
    INCLUDE+=("benchmarks/$bench/sketches")
    [[ -f "benchmarks/$bench/progress.json" ]] &&
        INCLUDE+=("benchmarks/$bench/progress.json")
done

EXCLUDES=(--exclude='.git' --exclude='__pycache__' --exclude='*.pyc'
          --exclude='.venv' --exclude='.venv-*' --exclude='jobs' --exclude='bench_vns'
          --exclude='node_modules' --exclude='logs/*' --exclude='manifests/*')

if [[ -z "$BASE" ]]; then
    tar czf "$OUT" "${EXCLUDES[@]}" "${INCLUDE[@]}"
else
    git rev-parse --verify --quiet "$BASE" >/dev/null || {
        echo "not a ref: $BASE" >&2; exit 1; }

    # -d drops deletions: tar cannot pack a file that is gone, and unpacking
    # could not remove it on the far side anyway.
    mapfile -t CHANGED < <(
        git diff --name-only --diff-filter=d "$BASE" -- "${CODE[@]}")

    # The files most likely to be missing from a --changed bundle, called out
    # by name rather than left to fail as an ImportError on the far side.
    mapfile -t UNTRACKED < <(
        git ls-files --others --exclude-standard -- "${CODE[@]}")
    if ((${#UNTRACKED[@]})); then
        printf 'untracked, so NOT in this bundle:\n' >&2
        printf '  %s\n' "${UNTRACKED[@]}" >&2
        printf 'git add -N them to include them, or pack without --changed.\n\n' >&2
    fi

    if ((${#CHANGED[@]} == 0)) && ((${#BENCHES[@]} == 0)); then
        echo "nothing tracked has changed since $BASE" >&2; exit 1
    fi

    # Benchmark inputs stay whole; only the code list is filtered.
    BENCH_PATHS=()
    for path in "${INCLUDE[@]}"; do
        for code in "${CODE[@]}"; do
            [[ "$path" == "$code" ]] && continue 2
        done
        BENCH_PATHS+=("$path")
    done
    tar czf "$OUT" "${EXCLUDES[@]}" \
        ${CHANGED+"${CHANGED[@]}"} ${BENCH_PATHS+"${BENCH_PATHS[@]}"}
fi

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
if [[ -n "$BASE" ]]; then
    echo "  ${#CHANGED[@]} file(s) changed since $BASE:"
    printf '    %s\n' ${CHANGED+"${CHANGED[@]}"}
fi
if ((${#BENCHES[@]} == 0)); then
    echo "  code only — no benchmark inputs included"
else
    echo "  with inputs for: ${BENCHES[*]}"
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
if ((${#BENCHES[@]} == 0)); then
    cat <<'EOF'

to send a benchmark separately (inputs only — results stay here):

  tar czf bench.tar.gz -C benchmarks --exclude='*.glb' <benchmark-id>
  scp -o ProxyJump=$UUN@student.ssh.inf.ed.ac.uk bench.tar.gz $UUN@icf:~/
  tar xzf ~/bench.tar.gz -C ~/InteractiveArtiDesign/benchmarks
EOF
fi
