# Examples

Run these scripts from the repository root. The simulation inputs are shared
with the regression suite under `tests/data/`, and output is written to the
current directory.

The grouped scripts expose their available workflows through `--help`:

```bash
python examples/compare.py --help
python examples/potentials.py --help
python examples/rates.py --help
python examples/rpmd.py --help
python examples/workflows.py --help
```

`opes.py`, `steered_path.py` and `walkers.py` each run one complete workflow;
`walkers.py` launches several independent OPES walkers as parallel processes
and combines them into one free-energy surface by reweighting.

These are research-scale examples, not CI tests. Depending on the workflow,
they require a CUDA GPU, MACE model downloads, PLUMED, ORCA, or
AmberTools.
