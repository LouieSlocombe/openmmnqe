#!/bin/bash
#bash custom_install.sh
#interactive -t 60 -p htc -c 32 --mem=64G -G 1

# === Configuration ===
ENV_NAME="openmmnqe"
OPENMM_VERSION="master"
PLUMED_VERSION="v2.10.0"
OPENMM_PLUMED_VERSION="master"

WORK_DIR="${SCRATCH}/${ENV_NAME}_sources"

# === Environment Setup ===
module purge
module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest

echo "=== Cleaning previous installations ==="
rm -rf "${WORK_DIR}"
rm -rf "${HOME}/.conda/envs/${ENV_NAME}"

echo "=== Initializing Conda Environment ==="
mamba create -n "${ENV_NAME}" -c conda-forge python=3.12 -y
source activate "${ENV_NAME}"

echo "=== Installing Dependencies ==="
mamba install -c conda-forge -y \
    "pytorch=*=cuda*" \
    pymace=0.3.15 \
    ase=3.28.0 \
    openmm=8.5 \
    openmm-ml=1.6 \
    openmmforcefields==0.15.1 \
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

echo "=== Compiling OpenMM-PLUMED ${OPENMM_PLUMED_VERSION} ==="
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