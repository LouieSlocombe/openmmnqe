# Examples

Run these scripts from the repository root. The simulation inputs are shared
with the regression suite under `tests/data/`, and output is written to the
current directory.

The grouped scripts expose their available workflows through `--help`:

```bash
python examples/compare.py --help
python examples/potentials.py --help
python examples/rpmd.py --help
python examples/workflows.py --help
```

`opes.py` and `steered_path.py` each run one complete workflow. For
multiple-walker metadynamics, prepare `plumed.dat` in the current directory
and run `examples/run_walker.sh`; `run_walker.py --help` documents the
single-walker options.

These are research-scale examples, not CI tests. Depending on the workflow,
they require a CUDA GPU, MACE model downloads, PLUMED, ORCA, WHAM, or
AmberTools.
