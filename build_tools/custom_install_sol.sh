#!/bin/bash
# Builds the openmmnqe environment on the Sol cluster, using conda-forge packages
# for everything except PLUMED, which has to be compiled with the opes module.
#
#   sbatch sub_sol_install.sh          # batch
#   ./custom_install_sol.sh            # from an interactive session, e.g.
#                                      # interactive -t 60 -p htc -c 12 --mem=128G -G a100:1
#
# The environment is recreated from scratch on every run.
#
# openmmnqe and its git dependencies are cloned into $SRC_DIR and installed
# editable, so a `git pull` there is all it takes to update them. Existing checkouts
# are used as they are, never wiped.

set -eo pipefail

# === Configuration ===
ENV_NAME="openmmnqe"
CUDA_VERSION="12.6"

# Sources are built under $SCRATCH; refuse to run rather than risk rm -rf'ing / below.
WORK_DIR="${SCRATCH:?SCRATCH is not set - run this on a Sol node, or set it manually}/${ENV_NAME}_sources"
# Editable checkouts live outside the build area: WORK_DIR is wiped on every run.
SRC_DIR="${SRC_DIR:-${HOME}/${ENV_NAME}_src}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pulls in build_plumed() and build_py_plumed(), with the PLUMED versions they pin.
source "${SCRIPT_DIR}/build_plumed.sh"
# Pulls in clone_repo(), install_editable_repos() and check_editable_repos().
source "${SCRIPT_DIR}/editable_repos.sh"

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
# Mirrors environment.yml, minus nnpops and the compiler toolchain the modules
# above already provide, and with cuda-version pinned to what Sol's driver takes.
mamba install -c conda-forge -y \
    "cuda-version<=${CUDA_VERSION}" \
    "pytorch=*=cuda*" \
    "pymace>=0.3.16" \
    "ase>=3.28.0" \
    "openmm>=8.5.1" \
    "openmm-ml>=1.6" \
    openmmforcefields \
    ambertools \
    pdbfixer \
    mdanalysis \
    mdtraj \
    pandas \
    matplotlib \
    pytest \
    pytest-cov \
    doxygen \
    swig \
    cython
pip3 install torch-dftd

echo "=== Preparing Build Directory ==="
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

build_plumed "${WORK_DIR}"
build_py_plumed "${WORK_DIR}"

echo "=== Installing openmmnqe (editable) ==="
clone_repo "ssh://git@github.com/LouieSlocombe/openmmnqe.git" "${SRC_DIR}/${ENV_NAME}"
pip3 install -e "${SRC_DIR}/${ENV_NAME}"

# After openmmnqe, which drags its own copies of these in from git.
install_editable_repos "${SRC_DIR}"

echo "=== Verifying Installation ==="
plumed --no-mpi config -q module opes
echo "PLUMED opes module: OK"
python3 -c "import plumed; plumed.Plumed()"
echo "py-plumed kernel load: OK"
python3 -c "from openmmplumed import PlumedForce"
echo "openmm-plumed: OK"
check_editable_repos "${SRC_DIR}"
echo "editable dependencies: OK"
python3 -c "import openmmnqe"
echo "openmmnqe: OK"

conda deactivate
echo "=== Build Complete! ==="
echo "Checkouts: ${SRC_DIR}"
