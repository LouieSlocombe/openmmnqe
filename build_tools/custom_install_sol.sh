#!/bin/bash
#bash custom_install.sh
#interactive -t 60 -p htc -c 32 --mem=64G -G 1

OPENMM_VERSION="master"
PLUMED_VERSION="v2.10.0"
OPENMM_PLUMED_VERSION="master"

rm -rf $SCRATCH/openmmnqe_sources
rm -rf $HOME/.conda/envs/openmmnqe_custom

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR=${SCRATCH}
WORK_DIR="${SCRATCH}/openmmnqe_sources"

module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest

echo "=== Initializing Conda Environment ==="
mamba create -n openmmnqe -c conda-forge python=3.12 -y
source activate openmmnqe

pip3 install torch --index-url https://download.pytorch.org/whl/cu126
mamba install -c conda-forge pymace=0.3.15 ase=3.28.0 openmm=8.5 openmm-ml=1.6 openmmforcefields==0.15.1
mamba install -c conda-forge doxygen swig cython -y
pip3 install git+https://github.com/LouieSlocombe/geodesic_interpolate.git

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

source deactivate
echo "=== Build Complete! ==="