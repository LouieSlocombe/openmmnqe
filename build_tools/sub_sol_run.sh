#!/bin/bash
#SBATCH --job-name=run
#SBATCH -N 1
#SBATCH -p htc
#SBATCH -c 128
#SBATCH -q public
#SBATCH --time=0-04:00:00
#SBATCH --mem=0
#SBATCH -G 1
#SBATCH -o run.out
#SBATCH -e run.out
#SBATCH --export=NONE

ENV_NAME="openmmnqe_custom"

module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest
source activate $ENV_NAME

$HOME/.conda/envs/$ENV_NAME/bin/python3 analysis_fold.py ./plots/ >> py.out 2>&1