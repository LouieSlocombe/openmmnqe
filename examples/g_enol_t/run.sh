#!/bin/bash

module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest
source activate openmmnqe

# Standard MD
python3 1_minimise.py
python3 2_heating.py
python3 3_npt.py
python3 4_production.py

# RPMD
python3 5_rpmd_eq.py
python3 6_rpmd_production.py

# adQTB
python3 7_adqtb_eq.py
python3 8_adqtb_prod.py