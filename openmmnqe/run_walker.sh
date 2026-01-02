#!/usr/bin/env bash
set -euo pipefail

NW=4
mkdir -p BIAS

for i in $(seq 0 $((NW-1))); do
  mkdir -p "Walker_${i}"
  cp plumed.dat "Walker_${i}/plumed.dat"
  cp run_walker.py "Walker_${i}/run_walker.py"
done

for i in $(seq 0 $((NW-1))); do
  (
    cd "Walker_${i}"
    # Optional: prevent CPU oversubscription if using CPU platform
    export OPENMM_CPU_THREADS=1
    python3 run_walker.py --walker-id "${i}" --n-walkers "${NW}" --steps 200000 --platform CPU
  ) > "Walker_${i}/screen.out" 2>&1 &
done

wait
echo "All walkers finished."
