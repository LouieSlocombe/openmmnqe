#!/bin/bash
# Builds OpenMM and its plugins from source into a dedicated conda environment.
#
#   bash custom_install.sh
#
# The environment is recreated from scratch on every run, and sources are cloned
# into ../../openmmnqe_sources (a sibling of the repo).
#
# forcefill, reactiontools, geodesic_interpolate and sella are cloned next to this
# repository and installed editable. Existing checkouts are used as they are, never
# wiped. Set SRC_DIR to keep them somewhere else:
#
#   SRC_DIR="${HOME}/src" bash custom_install.sh

# Exit immediately on error and fail pipelines cleanly, so a broken build does not
# fall through to the later steps and report success.
set -eo pipefail

ENV_NAME="openmmnqe_custom"
OPENMM_VERSION="master"  # 8.5.0 master
OPENMM_ML_VERSION="main" # 1.6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${SCRIPT_DIR}/../../openmmnqe_sources"
# Alongside the repository, so the checkouts survive the wipe WORK_DIR gets.
SRC_DIR="${SRC_DIR:-$(dirname "${REPO_DIR}")}"

# Pulls in build_plumed() and build_py_plumed(), with the PLUMED versions they pin.
source "${SCRIPT_DIR}/build_plumed.sh"
# Pulls in install_editable_repos() and check_editable_repos(), with the git
# dependencies they clone.
source "${SCRIPT_DIR}/editable_repos.sh"

echo "=== Initializing Conda Environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
conda env create -f "${SCRIPT_DIR}/environment_custom.yml"
conda activate "${ENV_NAME}"

echo "=== Preparing Build Directory ==="
if [ -d "${WORK_DIR}" ]; then
    echo "Removing existing sources directory for a clean build..."
    rm -rf "${WORK_DIR}"
fi
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "=== Compiling OpenMM ${OPENMM_VERSION} ==="
git clone --branch "${OPENMM_VERSION}" --depth 1 --filter=blob:none https://github.com/openmm/openmm.git
cd openmm
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(which python)" \
    -DCUDA_TOOLKIT_ROOT_DIR="${CONDA_PREFIX}" \
    -DCUDAToolkit_ROOT="${CONDA_PREFIX}"
make -j"$(nproc)"
make install
make PythonInstall

cd "${WORK_DIR}"

echo "=== Compiling openmm-torch ==="
git clone --depth 1 --filter=blob:none https://github.com/openmm/openmm-torch.git
cd openmm-torch
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENMM_DIR="${CONDA_PREFIX}" \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
make -j"$(nproc)"
make install
make PythonInstall

cd "${WORK_DIR}"

echo "=== Compiling NNPOps ==="
git clone --depth 1 --filter=blob:none https://github.com/openmm/NNPOps.git
cd NNPOps
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
make -j"$(nproc)"
make install

cd "${WORK_DIR}"

echo "=== Compiling OpenMM-ML ${OPENMM_ML_VERSION} ==="
git clone --branch "${OPENMM_ML_VERSION}" --depth 1 --filter=blob:none https://github.com/openmm/openmm-ml.git
cd openmm-ml
pip install .

build_plumed "${WORK_DIR}"
build_py_plumed "${WORK_DIR}"

echo "=== Installing openmmnqe (editable) ==="
pip install -e "${REPO_DIR}"

# After openmmnqe, which drags its own copies of these in from git.
install_editable_repos "${SRC_DIR}"

echo "=== Verifying Installation ==="
check_editable_repos "${SRC_DIR}"
echo "editable dependencies: OK"

echo "=== Build Complete! ==="
echo "Checkouts: ${SRC_DIR}"
