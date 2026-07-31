#!/bin/bash
# Shared PLUMED build steps, sourced by custom_install.sh and custom_install_sol.sh.
# Both installers need an identical PLUMED, so the versions are pinned here in one place.

PLUMED_VERSION="v2.10.0"
OPENMM_PLUMED_VERSION="master"

# build_plumed <work_dir>
# Compiles PLUMED (with the opes module) and the OpenMM-PLUMED plugin into $CONDA_PREFIX,
# cloning the sources into <work_dir>. Leaves the shell in <work_dir>.
build_plumed() {
    local work_dir="$1"

    echo "=== Compiling PLUMED ${PLUMED_VERSION} ==="
    cd "${work_dir}"
    git clone --branch "${PLUMED_VERSION}" --depth 1 --filter=blob:none https://github.com/plumed/plumed2.git
    cd plumed2
    ./configure --prefix="${CONDA_PREFIX}" --enable-modules=opes
    make -j"$(nproc)"
    make install

    export PLUMED_INCLUDE_DIR="${CONDA_PREFIX}/include/plumed"
    export PLUMED_LIBRARY_DIR="${CONDA_PREFIX}/lib"

    echo "=== Compiling OpenMM-PLUMED ${OPENMM_PLUMED_VERSION} ==="
    cd "${work_dir}"
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

    cd "${work_dir}"
}
