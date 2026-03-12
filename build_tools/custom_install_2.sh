#bash build_tools/custom_install_2.sh
#conda remove -n openmmnqe_custom --all

conda env create -f environment_custom_2.yml
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

#conda install -c conda-forge openmm-plumed openmmforcefields pdbfixer -y