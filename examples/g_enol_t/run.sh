#!/bin/bash

module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest
source activate openmmnqe

python3 1_minimise.py
python3 2_heating.py
python3 3_npt.py
python3 4_production.py
#python3 5_analysis.py
