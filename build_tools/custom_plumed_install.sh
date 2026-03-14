#!/bin/bash
# Exit immediately on error, treat unset variables as errors, fail pipelines cleanly
set -eo pipefail
#bash custom_install.sh
#conda remove -n openmmnqe_plumed_custom --all -y

# ==========================================
# Configuration Variables
# ==========================================
OPENMM_VERSION="8.4.0" # master
OPENMM_ML_VERSION="1.5"
PLUMED_VERSION="v2.10.0"
OPENMM_PLUMED_VERSION="master"

# Determine the absolute directory of the script and set the working dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/../../sources_plumed"

echo "=== Initializing Conda Environment ==="
# Source conda.sh directly instead of modifying user's .bashrc with 'conda init'
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f "${SCRIPT_DIR}/environment_plumed_custom.yml"
conda activate openmmnqe_plumed_custom

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

echo "=== Compiling OpenMM-ML ${OPENMM_ML_VERSION} ==="
git clone --branch "${OPENMM_ML_VERSION}" --depth 1 --filter=blob:none https://github.com/openmm/openmm-ml.git
cd openmm-ml
pip install .

cd "${WORK_DIR}"

echo "=== Compiling PLUMED ${PLUMED_VERSION} ==="
git clone --branch "${PLUMED_VERSION}" --depth 1 --filter=blob:none https://github.com/plumed/plumed2.git
cd plumed2
./configure --prefix="${CONDA_PREFIX}" --enable-modules=opes
make -j"$(nproc)"
make install

export PLUMED_INCLUDE_DIR="${CONDA_PREFIX}/include/plumed"
export PLUMED_LIBRARY_DIR="${CONDA_PREFIX}/lib"

cd "${WORK_DIR}"

echo "=== Compiling openmm plumed ==="
git clone --branch "${OPENMM_PLUMED_VERSION}" --depth 1 --filter=blob:none https://github.com/openmm/openmm-plumed.git
cd openmm-plumed
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENMM_DIR="${CONDA_PREFIX}" \
    -DPLUMED_INCLUDE_DIR="${PLUMED_INCLUDE_DIR}" \
    -DPLUMED_LIBRARY_DIR="${PLUMED_LIBRARY_DIR}" \
    -DPYTHON_EXECUTABLE="$(which python)"
make -j"$(nproc)"
make install
make PythonInstall
cd python
pip install . --no-build-isolation

cd "${WORK_DIR}"

echo "=== Installing Downstream Dependencies ==="
pip install openmmforcefields --no-deps
git clone --depth 1 --filter=blob:none https://github.com/openmm/pdbfixer.git
cd pdbfixer
pip install . --no-deps
cd "${WORK_DIR}"

echo "=== Build Complete! ==="