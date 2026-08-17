#!/bin/bash
# Builds the openmmnqe environment on the Sol cluster, using conda-forge packages
# for everything except PLUMED, which has to be compiled with the opes module.
#
#   sbatch sub_sol_install.sh          # batch
#   ./custom_install_sol.sh            # from an interactive session, e.g.
#                                      # interactive -t 60 -p htc -c 12 --mem=128G -G a100:1
#
# The environment is recreated from scratch on every run.

set -eo pipefail

# === Configuration ===
ENV_NAME="openmmnqe"
CUDA_VERSION="12.6"

# Sources are built under $SCRATCH; refuse to run rather than risk rm -rf'ing / below.
WORK_DIR="${SCRATCH:?SCRATCH is not set - run this on a Sol node, or set it manually}/${ENV_NAME}_sources"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pulls in build_plumed() and build_py_plumed(), with the PLUMED versions they pin.
source "${SCRIPT_DIR}/build_plumed.sh"

# === Environment Setup ===
module purge
module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest

echo "=== Cleaning previous installations ==="
rm -rf "${WORK_DIR}"
mamba env remove -n "${ENV_NAME}" -y 2>/dev/null || true

echo "=== Initializing Conda Environment ==="
mamba create -n "${ENV_NAME}" -c conda-forge python=3.12 -y
source activate "${ENV_NAME}"

echo "=== Installing Dependencies ==="
mamba install -c conda-forge -y \
    "cuda-version<=${CUDA_VERSION}" \
    "pytorch=*=cuda*" \
    pymace=0.3.15 \
    ase=3.28.0 \
    openmm=8.5.1 \
    openmm-ml=1.6 \
    openmmforcefields=0.15.1 \
    pdbfixer \
    mdanalysis \
    doxygen \
    swig \
    cython
pip3 install torch-dftd
pip3 install git+https://github.com/LouieSlocombe/geodesic_interpolate.git

echo "=== Preparing Build Directory ==="
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

build_plumed "${WORK_DIR}"
build_py_plumed "${WORK_DIR}"

echo "=== Installing openmmnqe ==="
pip3 install git+ssh://git@github.com/LouieSlocombe/openmmnqe.git

conda deactivate
echo "=== Build Complete! ==="
