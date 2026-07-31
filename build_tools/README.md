# Installation guide

There are three ways to install `openmmnqe`, depending on what you need:

| Route | Use when | Script |
|---|---|---|
| Conda environment | Normal use. Everything comes prebuilt from conda-forge. | `environment.yml` |
| Sol cluster | Running on Sol. Compiles PLUMED for the `opes` module. | `custom_install_sol.sh` |
| Source build | You need an unreleased OpenMM, openmm-torch or NNPOps. | `custom_install.sh` |

## Prerequisites

- A compatible operating system: Linux, macOS, or Windows via WSL.
- Python 3.12 or higher.
- Conda or Mamba.
- A CUDA-capable GPU for the `pytorch=*=cuda*` builds pinned in the environment files.
- ORCA, if you intend to use the QM helpers in `openmmnqe.qm` (see [ORCA](#orca) below).

## Conda environment

From this directory:

```bash
conda env create -f environment.yml
conda activate openmmnqe
pip install -e ..
```

The last step installs `openmmnqe` itself in editable mode; the environment files only
cover its dependencies.

## Sol cluster

`custom_install_sol.sh` builds the `openmmnqe` environment on Sol. Most dependencies come
from conda-forge, but PLUMED is compiled from source because the conda-forge build does
not include the `opes` module. Sources are cloned into `$SCRATCH/openmmnqe_sources`, and
both the environment and the sources are recreated from scratch on each run.

Submit it as a batch job from this directory:

```bash
sbatch sub_sol_install.sh
```

Or run it directly from an interactive session:

```bash
interactive -t 60 -p htc -c 12 --mem=128G -G a100:1
```

Once installed, `sub_sol_run.sh` runs a single simulation script inside that environment:

```bash
sbatch sub_sol_run.sh 1_minimise.py
```

It defaults to `g_enol_t.py` if no script is given. For multi-stage runs, the per-system
`run.sh` files under `examples/` chain the stages together in one job.

## Source build

`custom_install.sh` compiles OpenMM, openmm-torch, NNPOps, OpenMM-ML and PLUMED from
source into a separate `openmmnqe_custom` environment, leaving any existing `openmmnqe`
environment untouched. Pin the versions at the top of the script, then:

```bash
bash custom_install.sh
```

Sources are cloned into `../../openmmnqe_sources`, a sibling of the repository. Both the
environment and the sources are recreated from scratch on each run, so a full build takes
a while.

Both installers share `build_plumed.sh`, which is where the PLUMED and OpenMM-PLUMED
versions are pinned.

## ORCA

The QM helpers in `openmmnqe.qm` shell out to ORCA, which is licensed separately and must
be installed by hand:

1. Download it from the [ORCA website](https://www.faccts.de/orca/).
2. Extract it: `tar -xf orca-x.y.z.tar.gz`
3. Point `ORCA_PATH` at the `orca` binary, adding this to your `~/.bashrc`:

   ```bash
   export ORCA_PATH="/path/to/orca_6_1_1/orca"
   ```

`orca_calc_preset()`, `orca_optimise_atoms()` and `orca_calculate_goat()` read `ORCA_PATH`
when no explicit path is passed, as do the ORCA tests.

## Next steps

Worked examples live in `examples/`, which contains complete staged workflows (minimise →
heat → NPT → production → RPMD → adQTB) for a couple of systems.
