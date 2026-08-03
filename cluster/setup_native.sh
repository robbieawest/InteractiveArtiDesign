#!/bin/bash
# The parts pip cannot do. Run before setup_venvs.sh, on the head node —
# every step here reaches the network (micromamba, the gco sources, the
# checkpoint) and compute nodes are commonly airgapped.
#
#   cluster/setup_native.sh
#
# Two things to build, in userspace because there is no root here:
#   1. SuiteSparse   CHOLMOD, for the VNS Poisson solve. libsuitesparse-dev
#                    would need apt. conda-forge builds target glibc 2.17 so
#                    they load anywhere, which copying Ubuntu's .so files
#                    would not.
#   2. pygco         sf3d's graph-cut extension. The gco-v3.0 sources are not
#                    redistributable, hence a download step.
#
# and one thing only checked, not fetched: the ns2s checkpoint ships inside
# pack.sh's bundle (it is not in git, so it cannot be cloned). The check below
# exists for the case where the repo arrived some other way.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- 1. SuiteSparse --------------------------------------------------------
# `module` is a shell function defined by an init script your interactive
# profile sources, so in a plain script it usually does not exist yet. Pull it
# in before asking it anything, or the check below silently finds nothing and
# builds a prefix you did not need.
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
                /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash; do
        [[ -r "$init" ]] && { source "$init"; break; }
    done
fi

if [[ -f "$SUITESPARSE_PREFIX/lib/libcholmod.so" ]]; then
    echo "=== SuiteSparse already at $SUITESPARSE_PREFIX"
elif command -v module >/dev/null 2>&1 &&
     module avail suitesparse 2>&1 | grep -qi suitesparse; then
    # a site build is preferable to ours: it is compiled against this
    # cluster's toolchain and someone else maintains it
    echo "=== a suitesparse module exists — prefer it:"
    module avail suitesparse 2>&1 | grep -i suitesparse
    echo
    echo "    'module show <name>' prints its prefix. Then in cluster/env.sh:"
    echo "      export SUITESPARSE_PREFIX=<that prefix>"
    echo "    and add the matching 'module load' there too, so every job gets"
    echo "    the same build this compiled against. Check it is visible from"
    echo "    compute nodes as well as here — MODULEPATH can differ."
    exit 1
else
    echo "=== SuiteSparse -> $SUITESPARSE_PREFIX"
    mkdir -p "$REPO/deps"
    if ! command -v micromamba >/dev/null; then
        echo "--- fetching micromamba (static binary, no admin needed)"
        mkdir -p "$REPO/deps/bin"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
            | tar -xj -C "$REPO/deps" bin/micromamba
        export PATH="$REPO/deps/bin:$PATH"
    fi
    micromamba create -y -p "$SUITESPARSE_PREFIX" -c conda-forge suitesparse
fi

# conda-forge ships SuiteSparse 7.x. The ==0.4.12 pin in requirements-vns.txt
# exists only because Ubuntu 22.04 ships 5.10, so it does not apply here.
if grep -q '^scikit-sparse==0.4.12' "$SERVER/requirements-vns.txt"; then
    echo
    echo "NOTE: requirements-vns.txt pins scikit-sparse==0.4.12 for Ubuntu's"
    echo "      SuiteSparse 5.10. This prefix is 7.x — if the build fails,"
    echo "      relax that pin to 'scikit-sparse' and retry."
fi

# --- 2. pygco --------------------------------------------------------------
PYGCO="$SERVER/methods/surface-fitting-3d-sketches/external/pygco"
if [[ -d "$PYGCO" ]]; then
    echo "=== pygco graph-cut extension"
    make -C "$PYGCO" download
    make -C "$PYGCO" all
else
    echo "=== pygco: $PYGCO not present, skipping (sf3d will fail without it)"
fi

# --- 3. checkpoints --------------------------------------------------------
NS2S_CKPT="$SERVER/methods/NeuralSketch2Surf/checkpoints/best_model_jit.pt"
if [[ -f "$NS2S_CKPT" ]]; then
    echo "=== ns2s checkpoint present ($(du -h "$NS2S_CKPT" | cut -f1))"
else
    # expected to arrive in the bundle; getting here means it did not
    echo "=== ns2s checkpoint MISSING at $NS2S_CKPT" >&2
    echo "    It ships inside cluster/pack.sh's tarball — if you cloned the" >&2
    echo "    repo instead, it is not in git. Copy it from your local" >&2
    echo "    checkout, or download best_model_jit.pt from" >&2
    echo "    https://huggingface.co/HongshengY/S2V_Net (or set NS2S_CHECKPOINT)." >&2
fi

echo
echo "now: cluster/setup_venvs.sh"
