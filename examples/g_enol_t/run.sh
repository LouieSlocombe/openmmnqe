#!/bin/bash
#SBATCH --job-name=run
#SBATCH -N 1
#SBATCH -p htc
#SBATCH -c 12
#SBATCH -q public
#SBATCH --time=0-04:00:00
#SBATCH --mem=128G
#SBATCH -G a100:1
#SBATCH -o run.out
#SBATCH -e run.out
#SBATCH --export=NONE

# interactive -t 60 -p htc -c 12 --mem=128G -G a100:1
#set -e

#cd /scratch/lslocomb/gt_enol_1/
module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest
source activate openmmnqe

# Standard MD
python3 1_minimise.py >> py.out 2>&1
python3 2_heating.py >> py.out 2>&1
python3 3_npt.py >> py.out 2>&1
python3 4_production.py >> py.out 2>&1

## RPMD
#python3 5_rpmd_eq.py >> py.out 2>&1
#python3 6_rpmd_production.py >> py.out 2>&1

# adQTB
python3 7_adqtb_eq.py >> py.out 2>&1
python3 8_adqtb_prod.py >> py.out 2>&1