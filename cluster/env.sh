# Sourced by every cluster script. Edit the two paths at the top once; nothing
# below should need touching.
#
# Not executable on purpose — `source cluster/env.sh`, don't run it.

# --- usually nothing to edit -----------------------------------------------
# REPO locates itself from this file, so an unpacked bundle works wherever it
# landed. Override any of these from the environment (export first — a
# `VAR=x source env.sh` prefix applies only to the source command and is gone
# afterwards).
UUN="${UUN:-$USER}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/disk/scratch/$UUN}"
# ---------------------------------------------------------------------------

SERVER="$REPO/surfacing-server"
export REPO SERVER SCRATCH_ROOT UUN

# Where every surface is written: <root>/<bench>/<adapter>/<run>/<sketch>.glb,
# read by benchmarks.py and cells.py alike (SURFACING_BENCH_ROOT).
#
# Anchored on the home filesystem, NOT on $REPO. /disk/scratch is node-local
# and periodically wiped, so a run that finishes at 04:00 can be gone by the
# time you look at it — and since REPO now locates itself from this file, a
# bundle unpacked on scratch "to keep the NFS traffic down" would silently put
# every result there. This decouples the two: the code can live anywhere, the
# results always land somewhere durable.
#
# Default: $REPO/benchmarks when the repo is already under $HOME (the normal
# case, and what a local checkout does), otherwise ~/InteractiveArtiDesign/
# benchmarks. Override with RESULTS_ROOT if you keep results elsewhere — e.g.
# a group share. It must not be under $SCRATCH_ROOT; job.sh refuses to start
# if it is.
HOME_DIR="${HOME:-/home/$UUN}"
if [[ -z "${RESULTS_ROOT:-}" ]]; then
    case "$REPO/" in
        "$HOME_DIR"/*) RESULTS_ROOT="$REPO/benchmarks" ;;
        *)             RESULTS_ROOT="$HOME_DIR/InteractiveArtiDesign/benchmarks" ;;
    esac
fi
export RESULTS_ROOT
export SURFACING_BENCH_ROOT="$RESULTS_ROOT"

# Interpreter per method. Every adapter already reads its own variable and
# falls back to <server>/.venv-<m>/bin/python, so these are only here to be
# explicit — and to keep working if the venvs are ever moved off NFS.
#
# All of them are exported for every job, not just the job's own method: an
# sf3d task runs its proxy adapter in-process, so it needs that adapter's
# interpreter too (ns2s by default, but the run config chooses).
#
# These are the methods setup_venvs.sh builds. vrs2s is deliberately absent —
# no venv is built for it, so nothing here points at one that cannot exist.
# bbox needs no entry at all: it runs entirely in the server environment.
# To add a method, add a row to setup_venvs.sh's table and a line here.
export NS2S_PYTHON="$SERVER/.venv-ns2s/bin/python"
export VNS_PYTHON="$SERVER/.venv-vns/bin/python"
export NEUVAS_PYTHON="$SERVER/.venv-neuvas/bin/python"
export SF3D_PYTHON="$SERVER/.venv-sf3d/bin/python"

# The environment the runner itself and the mesh booleans run in.
export SERVER_PYTHON="$SERVER/.venv-server/bin/python"

# One userspace prefix, built by setup_native.sh, holding the two things this
# machine cannot otherwise provide:
#
#   python 3.10   ICF ships only 3.12, which this stack cannot use: every
#                 scikit-sparse before 0.4.13 caps at <3.12, and numpy<2 (
#                 pinned in three requirements files) has no 3.13 wheels.
#                 3.10 is what the local venvs and the ROCm setup use.
#   SuiteSparse   CHOLMOD, for the VNS Poisson solve. libsuitesparse-dev needs
#                 root; conda-forge builds target glibc 2.17 and load anywhere.
#
# Both in one prefix on purpose: two conda prefixes on LD_LIBRARY_PATH can
# disagree about libstdc++/libgcc, and the interpreter is the worst place to
# discover that.
#
# Do not delete it after setup. Each venv records this path as its base
# interpreter in pyvenv.cfg, and links CHOLMOD from here at runtime.
#
# If a suitesparse module ever appears, prefer it — `module show <name>` gives
# the prefix, and the load belongs here (not just at build time) so every job
# sees the build scikit-sparse compiled against. `module` is a shell function,
# so source its init first in a non-interactive shell.
export DEPS_PREFIX="$REPO/deps/env"
export SUITESPARSE_PREFIX="$DEPS_PREFIX"
export CLUSTER_PYTHON="$DEPS_PREFIX/bin/python"
export LD_LIBRARY_PATH="$DEPS_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# No ROCm here: drops HSA_OVERRIDE_GFX_VERSION / HIP_VISIBLE_DEVICES, which
# are meaningless on NVIDIA and misleading beside Slurm's CUDA_VISIBLE_DEVICES.
export SURFACING_GPU_BACKEND=cuda

# One task owns its scratch directory outright, so there is nothing to prune —
# and pruning would be actively wrong, since two array tasks can land on the
# same node and would delete each other's working files.
export SURFACING_KEEP_JOBS=-1
