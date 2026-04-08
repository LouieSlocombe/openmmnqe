#!/bin/bash
#SBATCH --job-name=run
#SBATCH -N 1
#SBATCH -p htc
#SBATCH -c 128
#SBATCH -q public
#SBATCH --time=0-04:00:00
#SBATCH --mem=0
#SBATCH -o analysis_fold.out
#SBATCH -e analysis_fold.out
#SBATCH --export=NONE

ENV_NAME="openmmnqe_custom"

module load mamba/latest
source activate $ENV_NAME

# Run the cleaner
# /packages/envs/$ENV_NAME/bin/python3 cleaner.py

# Run the analysis script
$HOME/.conda/envs/$ENV_NAME/bin/python3 analysis_fold.py ./data/ ./plots/ >> analysis_fold.out 2>&1