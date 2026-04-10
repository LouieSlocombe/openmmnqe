#!/bin/bash
#SBATCH --job-name=install
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

module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest

./custom_install_sol.sh >> bash.out 2>&1