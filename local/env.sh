# Sourced before running a sweep on a workstation — your AMD box, or the
# 2x3090 machine over ssh. The local counterpart of cluster/env.sh.
#
# Not executable on purpose — `source local/env.sh`, don't run it.
#
# Do NOT source cluster/env.sh here instead. It also sets SCRATCH_ROOT to
# /disk/scratch, puts the ICF deps prefix on LD_LIBRARY_PATH, and sets
# SURFACING_KEEP_JOBS=-1 (no scratch rotation, right for a one-cell Slurm task
# and wrong for a worker that lives for a whole sweep).

# --- the one line to edit per machine ---------------------------------------
#
# cuda  — NVIDIA. adapters/common.py drops the ROCm defines entirely.
# rocm  — AMD. HSA_OVERRIDE_GFX_VERSION is applied to every method subprocess.
#
# Deliberately explicit rather than probed: a wrong guess here does not fail,
# it runs on the CPU or picks the wrong device, and you find out an hour in.
# Anything other than "cuda" means ROCm, which is also the default when this
# file is never sourced at all — that is why the AMD box has always worked
# with no env file.
export SURFACING_GPU_BACKEND="${SURFACING_GPU_BACKEND:-rocm}"

# Which cards run_sweep.py puts a worker on, one each. Override per invocation
# with --gpus. One worker per card and no more: the ns2s resident worker is
# ~13GB and VNS measures 12.4GB, so two on one 24GB card OOMs both.
export SURFACING_GPUS="${SURFACING_GPUS:-0,1}"
# ---------------------------------------------------------------------------

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVER="$REPO/surfacing-server"
export REPO SERVER

# Where every surface is written: <root>/<bench>/<adapter>/<run>/<sketch>.glb.
# The same variable benchmarks.py and cells.py both read, so the server, the
# runner and finalize.py always agree. No cluster-style indirection needed —
# there is no node-local scratch to accidentally land on.
export SURFACING_BENCH_ROOT="${SURFACING_BENCH_ROOT:-$REPO/benchmarks}"

# SURFACING_JOBS_DIR is deliberately NOT set. run_sweep.py gives each worker
# its own (surfacing-server/jobs/gpu0, .../gpu1) because prune_job_dirs ranks
# across the whole root and two workers sharing one would delete each other's
# live scratch. Setting it here would override that and reintroduce exactly
# the problem. SURFACING_KEEP_JOBS is likewise left at its default 3, so each
# worker's root self-rotates instead of growing all sweep.

# --- interpreters -----------------------------------------------------------
#
# The server environment runs run_sweep.py, the workers and the mesh booleans;
# it holds no torch. Each method's own venv is invoked as a subprocess by its
# adapter, which is why one worker process can handle every adapter.
#
# The name differs by machine: setup_venvs.sh builds `.venv-server`, but the
# original local checkout has a bare `.venv`. Prefer whichever exists rather
# than renaming a working environment on either side.
if [[ -x "$SERVER/.venv-server/bin/python" ]]; then
    export SERVER_PYTHON="$SERVER/.venv-server/bin/python"
elif [[ -x "$SERVER/.venv/bin/python" ]]; then
    export SERVER_PYTHON="$SERVER/.venv/bin/python"
else
    echo "local/env.sh: no server venv at $SERVER/.venv-server or $SERVER/.venv" >&2
    echo "  run cluster/setup_venvs.sh first (see local/README or cluster/README.md)" >&2
fi

# All of them exported for every run, not just the one being run: sf3d runs its
# proxy adapter in-process, so an sf3d cell needs that adapter's interpreter
# too (ns2s by default, but the run config chooses). Each adapter falls back to
# this same path on its own, so these are here to be explicit and to keep
# working if the venvs are ever moved.
export NS2S_PYTHON="$SERVER/.venv-ns2s/bin/python"
export VNS_PYTHON="$SERVER/.venv-vns/bin/python"
export NEUVAS_PYTHON="$SERVER/.venv-neuvas/bin/python"
export SF3D_PYTHON="$SERVER/.venv-sf3d/bin/python"
export VRS2S_PYTHON="$SERVER/.venv-vrs2s/bin/python"
# bbox needs no entry: it runs entirely in the server environment.

# --- native deps, only where they were built --------------------------------
#
# cluster/setup_native.sh builds a userspace prefix holding python 3.10 and
# SuiteSparse (CHOLMOD, which VNS's scikit-sparse links) for machines where
# libsuitesparse-dev cannot be apt-installed without root — which includes the
# 2x3090 box. A workstation with the system package needs none of this, so the
# prefix is only put on the library path when it actually exists.
if [[ -d "$REPO/deps/env/lib" ]]; then
    export DEPS_PREFIX="$REPO/deps/env"
    export SUITESPARSE_PREFIX="$DEPS_PREFIX"
    export LD_LIBRARY_PATH="$DEPS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

echo "backend=$SURFACING_GPU_BACKEND gpus=$SURFACING_GPUS results=$SURFACING_BENCH_ROOT"
