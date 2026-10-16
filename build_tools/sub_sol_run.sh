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

# Runs one simulation script in the openmmnqe environment, from the directory
# the job was submitted in. Submit from the repository root, so the examples
# can find their inputs under tests/data/:
#   sbatch build_tools/sub_sol_run.sh examples/workflows.py gc_pt_solvated
# Without arguments it runs exactly that workflow.

set -eo pipefail

ENV_NAME="openmmnqe"
if [ "$#" -eq 0 ]; then
    set -- examples/workflows.py gc_pt_solvated
fi

module load cuda-13.0.1-gcc-13.2.0
module load mamba/latest
source activate "${ENV_NAME}"

python3 "$@" >> py.out 2>&1
