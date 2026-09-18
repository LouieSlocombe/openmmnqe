#!/bin/bash
# Shared handling of the two sibling packages, sourced by conda_install.sh,
# custom_install.sh and custom_install_sol.sh. Both are released on PyPI and
# pyproject.toml asks for them by version, but they get edited alongside
# openmmnqe, so every installer clones them once and installs them in editable
# mode instead -- a `git pull` in the checkout is then all it takes to update
# one, and the release is what everyone else gets.

# name=url pairs, in install order. The name is both the directory the repo is
# cloned into and the module the install is checked against.
#
# geodesic_interpolate and sella used to follow reactiontools here, because it
# declared both as `name @ git+...` dependencies and pip re-fetched them over
# any editable install. reactiontools carries the code itself as of its 1.0.0,
# in reactiontools.tools_geodesic and reactiontools.tools_sella, so cloning
# either would now install a copy that nothing imports.
EDITABLE_REPOS=(
    "forcefill=https://github.com/LouieSlocombe/forcefill.git"
    "reactiontools=https://github.com/LouieSlocombe/reactiontools.git"
)

# clone_repo <url> <path>
# Clones <url> into <path> unless a checkout is already there, which is left
# exactly as it is -- these hold work in progress, so nothing here pulls,
# resets or removes them.
clone_repo() {
    local url="$1"
    local path="$2"

    if [ -d "${path}/.git" ]; then
        echo "=== Using existing checkout: ${path} ==="
    elif [ -e "${path}" ]; then
        echo "${path} exists but is not a git checkout; move it aside and re-run." >&2
        return 1
    else
        echo "=== Cloning $(basename "${path}") into ${path} ==="
        git clone "${url}" "${path}"
    fi
}

# install_editable_repos <src_dir>
# Clones each sibling package into <src_dir> and installs it editable. It still
# runs *after* openmmnqe itself, as it always has: pip resolves both releases
# from PyPI on the way to installing openmmnqe, and the editable installs then
# take their place. Only the cost of that changed with the releases -- while the
# two were `name @ git+...` requirements, pip re-cloned them even when an
# editable install was already there, and this order was the only one that
# worked.
install_editable_repos() {
    local src_dir="$1"
    local entry name url

    mkdir -p "${src_dir}"
    for entry in "${EDITABLE_REPOS[@]}"; do
        name="${entry%%=*}"
        url="${entry#*=}"
        clone_repo "${url}" "${src_dir}/${name}"
        echo "=== Installing ${name} (editable) ==="
        pip install -e "${src_dir}/${name}"
    done
}

# check_editable_repos <src_dir>
# Fails if any of them import from site-packages rather than the checkout.
check_editable_repos() {
    local src_dir="$1"

    python -c "
import importlib, pathlib, sys

src = pathlib.Path('${src_dir}').resolve()
for name in '${EDITABLE_REPOS[*]%%=*}'.split():
    path = pathlib.Path(importlib.import_module(name).__file__).resolve()
    if src not in path.parents:
        sys.exit(f'{name} is not editable: imported from {path.parent}')
    print(f'{name}: {path.parent}')
"
}
