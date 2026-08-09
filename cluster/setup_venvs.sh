#!/bin/bash
# Build the method environments. Run once, on the head node.
#
# The head node, not a compute node: this needs outbound network for the wheels
# (and setup_native.sh for micromamba and the pygco sources), and compute nodes
# are commonly airgapped. It is not what the head node's "no compute intensive
# processes" banner is aimed at either — the only compilation here is one
# Cython extension, a few seconds of it.
#
# Needs uv (see below); it builds the venvs and installs into them. The paths
# are unchanged — .venv-<name>/bin/python — so env.sh and the adapters' own
# <METHOD>_PYTHON fallbacks do not know the difference.
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
# Which index — ROCm or CUDA — follows from SURFACING_GPU_BACKEND alone, via
# surfacing-server/backends.json. Source the env file for the machine first:
#
#   source local/env.sh    # workstation; edit its one backend line
#   cluster/setup_venvs.sh
#
# TORCH_INDEX / TORCH_INDEX_VNS still override, for a one-off (a driver too old
# for the table's default, say), but nothing needs them in the normal case.
#
# Run setup_native.sh first — the vns venv compiles scikit-sparse against the
# SuiteSparse prefix it builds.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SERVER"

ONLY="${1:-}"

# --- uv -------------------------------------------------------------------
#
# uv builds the venvs and installs into them; the layout is unchanged
# (.venv-<name>/bin/python), so env.sh and every adapter's <METHOD>_PYTHON
# fallback keep working untouched.
#
# The reason it is worth requiring: uv hardlinks packages out of one global
# cache, and five venvs here each carry a ~2.5GB torch plus its CUDA/ROCm
# runtime. Under pip that is five copies on a home filesystem — on NFS, five
# copies over the network. Under uv it is one.
#
# Not installed automatically. Fetching and running an installer from the
# network is not something a setup script should do behind you, and on the
# cluster it is a head-node action you want to take deliberately.
UV="${UV:-$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")}"
if [[ ! -x "$UV" ]]; then
    cat >&2 <<'EOF'
uv not found. It needs no root and installs into ~/.local/bin:

    curl -LsSf https://astral.sh/uv/install.sh | sh
    wget -qO- https://astral.sh/uv/install.sh | sh     # if there is no curl

    export PATH="$HOME/.local/bin:$PATH"

Then re-run this script. Set UV=/path/to/uv if it lives elsewhere.
EOF
    exit 1
fi
echo "uv: $UV ($("$UV" --version))"

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
#
# uv is the last resort and the one that always works: `uv python install 3.10`
# fetches a standalone build into ~/.local/share/uv, no root and no compiler.
# That is what makes setup_native.sh's interpreter half optional — it is still
# needed for SuiteSparse, but no longer for having a 3.10 at all.
if [[ -z "${PYTHON:-}" ]]; then
    for candidate in "$CLUSTER_PYTHON" python3.11 python3.10; do
        if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
            PYTHON="$(command -v "$candidate" 2>/dev/null || echo "$candidate")"
            break
        fi
    done
fi
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$("$UV" python find 3.10 2>/dev/null || true)"
fi
if [[ -z "${PYTHON:-}" ]]; then
    echo "no Python 3.10/3.11 found; fetching one with uv (no root needed)" >&2
    "$UV" python install 3.10
    PYTHON="$("$UV" python find 3.10)"
fi
read -r PY_MAJOR PY_MINOR < <("$PYTHON" -c 'import sys; print(sys.version_info[0], sys.version_info[1])')
if (( PY_MAJOR != 3 || PY_MINOR < 10 || PY_MINOR > 11 )); then
    cat >&2 <<EOF
$PYTHON is Python $PY_MAJOR.$PY_MINOR; this stack needs 3.10 or 3.11.

On 3.12+ pip reports "Could not find a version that satisfies the requirement
scikit-sparse==0.4.12" — that pin caps at <3.12 — and on 3.13 numpy<2 has no
wheels either. Neither is really about scikit-sparse or numpy.

You reached this by pointing PYTHON at something explicitly — the search above
falls back to uv, which would have fetched a 3.10 rather than get here. Either
drop the override, or point it somewhere valid:

    uv python install 3.10
    PYTHON="\$(uv python find 3.10)" cluster/setup_venvs.sh ${ONLY:-}

setup_native.sh also builds a 3.10 into $DEPS_PREFIX, alongside the SuiteSparse
that the vns venv still needs; either interpreter works.
EOF
    exit 1
fi
echo "interpreter: $PYTHON (Python $PY_MAJOR.$PY_MINOR)"

# --- backend --------------------------------------------------------------
#
# Printed, not assumed. Installing the wrong vendor's torch does not fail —
# it succeeds and produces venvs where torch.cuda.is_available() is False, and
# the preflight below tests mesh booleans, not torch. So say out loud which
# stack is about to be built, before ~60GB of wheels are fetched.
#
# cluster/env.sh defaults the backend to cuda but does not force it, so
# sourcing local/env.sh first (rocm) still wins.
BACKENDS="$SERVER/backends.json"
backend_index() {  # backend_index <venv-name> -> that venv's index, or default
    "$PYTHON" -c '
import json, sys
name, backend, path = sys.argv[1], sys.argv[2], sys.argv[3]
table = json.load(open(path))
if backend not in table or backend.startswith("_"):
    known = ", ".join(sorted(k for k in table if not k.startswith("_")))
    sys.exit(f"SURFACING_GPU_BACKEND={backend!r} is not in {path}; known: {known}")
index = table[backend]["torch_index"]
print(index.get(name, index["default"]))
' "$1" "$SURFACING_GPU_BACKEND" "$BACKENDS"
}
TORCH_INDEX="${TORCH_INDEX:-$(backend_index default)}"
TORCH_INDEX_VNS="${TORCH_INDEX_VNS:-$(backend_index vns)}"
echo "backend: $SURFACING_GPU_BACKEND (torch from $TORCH_INDEX, vns from $TORCH_INDEX_VNS)"

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
    # --seed puts pip/setuptools/wheel inside the venv. uv does not need them —
    # every install below goes through `uv pip --python` — but a method that
    # shells out to its own pip still finds one, which is what a `python -m
    # venv` environment always gave it.
    "$UV" venv --python "$PYTHON" --seed "$venv"
    PIP=("$UV" pip install --python "$venv/bin/python")

    if [[ -n "$spec" ]]; then
        url="https://download.pytorch.org/whl/${index:-$TORCH_INDEX}"
        echo "--- torch: $spec from $url"
        "${PIP[@]}" "$spec" --index-url "$url"
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
        "${PIP[@]}" -r "$stripped"
        rm -f "$stripped"
        echo "--- scikit-sparse (unpinned; prefix is SuiteSparse 7.x)"
        # scikit-sparse is source-only and its setup.py imports numpy and
        # Cython, so it needs them present rather than isolated away. pip's
        # isolation happened to work via its pyproject build-requires; uv is
        # stricter, so be explicit instead of relying on that.
        "${PIP[@]}" cython "numpy<2"
        "${PIP[@]}" --no-build-isolation scikit-sparse
    else
        "${PIP[@]}" -r "$reqs"
    fi
    unset CPPFLAGS LDFLAGS

    # --- did the right torch survive? ------------------------------------
    #
    # The requirements files pull packages that depend on torch (ns2s takes
    # pytorch-lightning and monai), so a resolver is always one bad step from
    # replacing the vendor wheel installed above with a generic PyPI build.
    # That failure is silent — the venv works, and torch.cuda.is_available()
    # is just False, discovered an hour into a sweep. Check it here instead.
    if [[ -n "$spec" ]]; then
        "$venv/bin/python" - "$SURFACING_GPU_BACKEND" <<'PY'
import sys, torch
backend = sys.argv[1]
want = {"cuda": "cu", "rocm": "rocm"}[backend]
local = (torch.__version__.split("+") + [""])[1]
if want not in local:
    sys.exit(
        f"FAIL: torch {torch.__version__} is not a {backend} build "
        f"(expected a +{want}... local version). A requirements file most "
        f"likely pulled a generic wheel from PyPI over the top of it."
    )
print(f"ok: torch {torch.__version__}")
PY
    fi
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
