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

# Runs a single simulation script in the openmmnqe environment:
#   sbatch sub_sol_run.sh [script.py]

set -eo pipefail

ENV_NAME="openmmnqe"
PY_SCRIPT="${1:-g_enol_t.py}"

module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest
source activate "${ENV_NAME}"

python3 "${PY_SCRIPT}" >> py.out 2>&1
