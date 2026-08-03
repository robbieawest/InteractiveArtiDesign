# Sourced by every cluster script. Edit the two paths at the top once; nothing
# below should need touching.
#
# Not executable on purpose — `source cluster/env.sh`, don't run it.

# --- edit these ------------------------------------------------------------
UUN="${UUN:-$USER}"
REPO="${REPO:-/home/$UUN/InteractiveArtiDesign}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/disk/scratch/$UUN}"
# ---------------------------------------------------------------------------

SERVER="$REPO/surfacing-server"
export REPO SERVER SCRATCH_ROOT UUN

# Interpreter per method. Every adapter already reads its own variable and
# falls back to <server>/.venv-<m>/bin/python, so these are only here to be
# explicit — and to keep working if the venvs are ever moved off NFS.
#
# All of them are exported for every job, not just the job's own method: an
# sf3d task runs its proxy adapter in-process, so it needs that adapter's
# interpreter too (ns2s by default, but the run config chooses).
export NS2S_PYTHON="$SERVER/.venv-ns2s/bin/python"
export VNS_PYTHON="$SERVER/.venv-vns/bin/python"
export NEUVAS_PYTHON="$SERVER/.venv-neuvas/bin/python"
export SF3D_PYTHON="$SERVER/.venv-sf3d/bin/python"
export VRS2S_PYTHON="$SERVER/.venv-vrs2s/bin/python"

# The environment the runner itself and the mesh booleans run in.
export SERVER_PYTHON="$SERVER/.venv-server/bin/python"

# CHOLMOD, for the VNS Poisson solve. Built into a userspace prefix by
# setup_native.sh because libsuitesparse-dev needs root and we have none.
#
# If the cluster ships a suitesparse module, prefer it — uncomment these, set
# the prefix to what `module show` reports, and keep the load here rather than
# only at build time: scikit-sparse is compiled against these headers, so every
# job has to see the same build. `module` is a shell function, so it needs its
# init sourced first in a non-interactive shell.
#
#   for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
#       [[ -r "$init" ]] && { source "$init"; break; }
#   done
#   module load suitesparse
#
export SUITESPARSE_PREFIX="$REPO/deps/suitesparse"
export LD_LIBRARY_PATH="$SUITESPARSE_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# No ROCm here: drops HSA_OVERRIDE_GFX_VERSION / HIP_VISIBLE_DEVICES, which
# are meaningless on NVIDIA and misleading beside Slurm's CUDA_VISIBLE_DEVICES.
export SURFACING_GPU_BACKEND=cuda

# One task owns its scratch directory outright, so there is nothing to prune —
# and pruning would be actively wrong, since two array tasks can land on the
# same node and would delete each other's working files.
export SURFACING_KEEP_JOBS=-1
