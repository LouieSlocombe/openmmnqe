# cd to the build tools directory
cd $HOME/skunkworks/openmmnqe/build_tools
#bash custom_plumed_install.sh
#conda remove -n openmmnqe_plumed_custom --all -y

conda init bash
source $(conda info --base)/etc/profile.d/conda.sh
conda env create -f environment_plumed_custom.yml
conda activate openmmnqe_plumed_custom
cd ../..

# if the sources directory exists remove it to ensure a clean build
if [ -d "sources_plumed" ]; then
    rm -rf sources_plumed
fi

mkdir -p sources_plumed && cd sources_plumed

# OpenMM
git clone --branch v8.4.0 https://github.com/openmm/openmm.git && cd openmm
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
         -DPYTHON_EXECUTABLE=$(which python)
make -j$(nproc)
make install
make PythonInstall
cd ../..

# OpenMM-ML
git clone --branch v1.5 https://github.com/openmm/openmm-ml.git && cd openmm-ml
pip install .
cd ..

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