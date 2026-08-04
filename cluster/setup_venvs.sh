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
ONLY="${1:-}"

# --- interpreter ----------------------------------------------------------
#
# 3.10 or 3.11, not whatever `python3` happens to be. This stack targets 3.10
# (every local venv is python3.10, and requirements-sf3d.txt says as much), and
# a newer interpreter breaks it in two places at once:
#
#   scikit-sparse   every release before 0.4.13 caps at <3.12, so pip ignores
#                   them and reports the pin as unsatisfiable
#   numpy<2         pinned in three requirements files for the vendored code.
#                   1.26.x covers 3.12, but there are no numpy 1.x wheels at
#                   all for 3.13, so the venv simply cannot be built
#
# Override with PYTHON=/path/to/python3.11 if the search misses yours.
# The prefix setup_native.sh builds comes first: on ICF /usr/bin has only
# 3.12, so that is normally the only usable interpreter on the machine.
if [[ -z "${PYTHON:-}" ]]; then
    for candidate in "$CLUSTER_PYTHON" python3.11 python3.10 python3; do
        if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
            PYTHON="$(command -v "$candidate" 2>/dev/null || echo "$candidate")"
            break
        fi
    done
fi
read -r PY_MAJOR PY_MINOR < <("$PYTHON" -c 'import sys; print(sys.version_info[0], sys.version_info[1])')
if (( PY_MAJOR != 3 || PY_MINOR < 10 || PY_MINOR > 11 )); then
    cat >&2 <<EOF
$PYTHON is Python $PY_MAJOR.$PY_MINOR; this stack needs 3.10 or 3.11.

On 3.12+ pip reports "Could not find a version that satisfies the requirement
scikit-sparse==0.4.12" — that pin caps at <3.12 — and on 3.13 numpy<2 has no
wheels either. Neither is really about scikit-sparse or numpy.

On ICF /usr/bin has only 3.12, so the interpreter comes from the deps prefix
that setup_native.sh builds. Run that first:

    cluster/setup_native.sh

It creates $DEPS_PREFIX with python and SuiteSparse together, and this script
then picks it up automatically. To use a different one:

    PYTHON=/path/to/python3.10 cluster/setup_venvs.sh ${ONLY:-}
EOF
    exit 1
fi
echo "interpreter: $PYTHON (Python $PY_MAJOR.$PY_MINOR)"

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

        # requirements-vns.txt pins scikit-sparse==0.4.12 because Ubuntu 22.04
        # ships SuiteSparse 5.10 and anything newer wants >= 7. Our prefix is
        # 7.x, so the pin is not merely unnecessary here, it is wrong — and it
        # caps at Python <3.12 besides. Install the rest from the file, then
        # scikit-sparse unpinned, leaving the file correct for the local
        # ROCm/Ubuntu setup it was written for.
        stripped="$(mktemp)"
        grep -v '^[[:space:]]*scikit-sparse' "$reqs" > "$stripped"
        "$venv/bin/pip" install -r "$stripped"
        rm -f "$stripped"
        echo "--- scikit-sparse (unpinned; prefix is SuiteSparse 7.x)"
        "$venv/bin/pip" install scikit-sparse
    else
        "$venv/bin/pip" install -r "$reqs"
    fi
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
