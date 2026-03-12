conda env create -f environment_custom.yml
conda activate openmmnqe_custom
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

conda install -c conda-forge openmm-plumed=2.1 openmmforcefields openmmtools pdbfixer -y