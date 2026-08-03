#!/bin/bash
# Build the method environments. Run once, on the head node.
#
# The head node, not a compute node: this needs outbound network for pip (and
# setup_native.sh for micromamba and the pygco sources), and compute nodes are
# commonly airgapped. It is not what the head node's "no compute intensive
# processes" banner is aimed at either — the only compilation here is one
# Cython extension, a few seconds of it.
#
# Model weights are not fetched: the ns2s checkpoint arrives in pack.sh's
# tarball.
#
#   cluster/setup_venvs.sh
#
# Verify afterwards on a compute node, which is the only place there is a GPU:
#
#   srun -p Teaching --gres=gpu:h200_1g.18gb:1 --pty \
#     surfacing-server/.venv-ns2s/bin/python -c \
#     "import torch; print(torch.__version__, torch.cuda.is_available())"
#
# torch is installed FIRST, from its own index, and never appears in the
# requirements files. That ordering is load-bearing: requirements-ns2s.txt
# pulls pytorch-lightning and monai, both of which depend on torch, so a wrong
# order silently fetches a CPU wheel from PyPI over the top.
#
# The ROCm indices are the local-machine defaults; override for CUDA:
#   TORCH_INDEX=cu128 TORCH_INDEX_VNS=cu121 cluster/setup_venvs.sh
#
# Run setup_native.sh first — the vns venv compiles scikit-sparse against the
# SuiteSparse prefix it builds.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SERVER"

TORCH_INDEX="${TORCH_INDEX:-rocm6.4}"
TORCH_INDEX_VNS="${TORCH_INDEX_VNS:-rocm6.2}"
PYTHON="${PYTHON:-python3}"
ONLY="${1:-}"

# name | torch spec (empty = none) | index override (empty = TORCH_INDEX)
VENVS=(
    "server|                     |"
    "vns   |torch==2.5.1         |$TORCH_INDEX_VNS"
    "ns2s  |torch==2.8.0         |"
    "neuvas|torch==2.8.0         |"
    "sf3d  |                     |"
)

for row in "${VENVS[@]}"; do
    IFS='|' read -r name spec index <<< "$row"
    name="${name// /}"; spec="${spec// /}"; index="${index// /}"
    [[ -n "$ONLY" && "$ONLY" != "$name" ]] && continue

    venv=".venv-$name"
    reqs="requirements.txt"
    [[ "$name" != "server" ]] && reqs="requirements-$name.txt"
    [[ -f "$reqs" ]] || { echo "missing $reqs" >&2; exit 1; }

    echo "=== $venv  ($reqs)"
    "$PYTHON" -m venv "$venv"
    "$venv/bin/pip" install --upgrade pip wheel >/dev/null

    if [[ -n "$spec" ]]; then
        url="https://download.pytorch.org/whl/${index:-$TORCH_INDEX}"
        echo "--- torch: $spec from $url"
        "$venv/bin/pip" install "$spec" --index-url "$url"
    fi

    if [[ "$name" == "vns" ]]; then
        # CHOLMOD headers and libs, from the userspace prefix setup_native.sh
        # built. scikit-sparse ships source-only — there are no manylinux
        # wheels — so this compiles a Cython extension against them.
        export CPPFLAGS="-I$SUITESPARSE_PREFIX/include -I$SUITESPARSE_PREFIX/include/suitesparse ${CPPFLAGS:-}"
        export LDFLAGS="-L$SUITESPARSE_PREFIX/lib ${LDFLAGS:-}"
    fi

    "$venv/bin/pip" install -r "$reqs"
    unset CPPFLAGS LDFLAGS
done

# --- preflight: the boolean backend ---------------------------------------
#
# combine_meshes catches a failed union, logs, and concatenates instead — a
# valid glb and a zero exit code. Across hundreds of task logs that is
# invisible, so prove the backend works here, once, loudly.
if [[ -z "$ONLY" || "$ONLY" == "server" ]]; then
    echo "=== preflight: mesh booleans in .venv-server"
    "$SERVER/.venv-server/bin/python" - <<'PY'
import sys
import trimesh
a = trimesh.creation.icosphere(radius=1.0)
b = trimesh.creation.icosphere(radius=1.0)
b.apply_translation([0.5, 0, 0])
try:
    union = trimesh.boolean.union([a, b])
except Exception as exc:
    sys.exit(f"FAIL: boolean union raised ({exc}). Install manifold3d and networkx.")
if not union.is_volume:
    sys.exit("FAIL: union produced a non-volume; the backend is not working.")
if len(union.faces) >= len(a.faces) + len(b.faces):
    sys.exit("FAIL: union looks like a concatenation, not a CSG union.")
print(f"ok: union -> {len(union.faces)} faces, watertight")
PY
fi

echo
echo "done. Sanity check an interpreter:"
echo "  $SERVER/.venv-ns2s/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
