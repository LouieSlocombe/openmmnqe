# cd to the build tools directory
cd $HOME/skunkworks/openmmnqe/build_tools
#bash custom_install_2.sh
#conda remove -n openmmnqe_custom --all -y

conda init bash
conda env create -f environment_custom_2.yml
conda activate openmmnqe_custom
cd ../..

# if the sources directory exists remove it to ensure a clean build
if [ -d "sources" ]; then
    rm -rf sources
fi

mkdir -p sources && cd sources

# OpenMM
git clone https://github.com/openmm/openmm.git && cd openmm
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
         -DPYTHON_EXECUTABLE=$(which python)
make -j$(nproc)
make install
make PythonInstall
cd ../..

# OpenMM-ML
git clone https://github.com/openmm/openmm-ml.git && cd openmm-ml
pip install .
cd ..

conda install -c conda-forge zlib mkl mkl-include sysroot_linux-64 -y
conda install -c conda-forge compilers make cmake pkg-config zlib libblas liblapack numpy -y

# PLUMED
git clone --branch v2.10.0 https://github.com/plumed/plumed2.git && cd plumed2
./configure --prefix=$CONDA_PREFIX --enable-modules=opes
make -j$(nproc)
make install
export PLUMED_INCLUDE_DIR=$CONDA_PREFIX/include/plumed
export PLUMED_LIBRARY_DIR=$CONDA_PREFIX/lib
cd ..

# openmm-plumed
git clone https://github.com/openmm/openmm-plumed.git && cd openmm-plumed
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
         -DOPENMM_DIR=$CONDA_PREFIX \
         -DPLUMED_INCLUDE_DIR=$PLUMED_INCLUDE_DIR \
         -DPLUMED_LIBRARY_DIR=$PLUMED_LIBRARY_DIR \
         -DPYTHON_EXECUTABLE=$(which python)
make -j$(nproc)
make install
make PythonInstall
cd python
pip install . --no-build-isolation
cd ../../..

# OpenMM-ForceFields and PDBFixer
conda install -c conda-forge openmmforcefields pdbfixer -y

# Need to rebuild OpenMM
cd openmm/build
make clean
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
         -DPYTHON_EXECUTABLE=$(which python)
make -j$(nproc)
make install
make PythonInstall
cd ../..

# Return to the build tools directory
cd openmm/build_tools