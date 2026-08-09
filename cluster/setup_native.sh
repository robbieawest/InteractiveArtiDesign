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

PY_VERSION="${PY_VERSION:-3.10}"

# Download to stdout, with whichever fetcher the machine has. Not curl alone:
# it is absent on some managed hosts and unavailable without root there, and
# this is the only thing in the setup that needs the network by hand.
fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -Ls "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$1"
    else
        echo "need curl or wget to fetch $1" >&2
        return 1
    fi
}

if [[ -x "$DEPS_PREFIX/bin/python" && -f "$DEPS_PREFIX/lib/libcholmod.so" ]]; then
    echo "=== deps prefix already at $DEPS_PREFIX"
    "$DEPS_PREFIX/bin/python" --version
else
    echo "=== deps prefix -> $DEPS_PREFIX  (python $PY_VERSION + SuiteSparse)"
    mkdir -p "$REPO/deps"
    if ! command -v micromamba >/dev/null; then
        echo "--- fetching micromamba (one static binary, no admin needed)"
        mkdir -p "$REPO/deps/bin"
        fetch https://micro.mamba.pm/api/micromamba/linux-64/latest \
            | tar -xj -C "$REPO/deps" bin/micromamba
        export PATH="$REPO/deps/bin:$PATH"
    fi
    # one prefix, both things: conda-forge is being used here only as a way to
    # obtain an interpreter and a C library without root. The project's own
    # environments remain pip venvs built from requirements-*.txt.
    micromamba create -y -p "$DEPS_PREFIX" -c conda-forge \
        "python=$PY_VERSION" suitesparse
fi

for probe in "$DEPS_PREFIX/bin/python" "$DEPS_PREFIX/lib/libcholmod.so"; do
    [[ -e "$probe" ]] || { echo "expected $probe after create" >&2; exit 1; }
done

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
