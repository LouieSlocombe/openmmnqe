# Installation guide

There are three ways to install `openmmnqe`, depending on what you need:

| Route | Use when | Script |
|---|---|---|
| Conda environment | Normal use. Everything from conda-forge except PLUMED. | `conda_install.sh` |
| Sol cluster | Running on Sol. Same split, plus the module loads and SLURM wrapper. | `custom_install_sol.sh` |
| Source build | You need an unreleased OpenMM, openmm-torch or NNPOps. | `custom_install.sh` |

Every route compiles PLUMED, the `openmm-plumed` plugin and the PLUMED Python
bindings (py-plumed), because there is no prebuilt combination that works:
conda-forge's `openmm-plumed` requires `openmm <8.5`, while `openmm-ml >=1.6` —
the first release with the `'ase'` potential that `openmmnqe.openmm` uses —
requires `openmm >=8.5`. Building from source also gets you PLUMED's `opes`
module, which the conda-forge build omits, and py-plumed — the `plumed` module
that `reactiontools`' `plumed_calculator` imports on first use — matched to the
same PLUMED version.

## Prerequisites

- A compatible operating system: Linux, macOS, or Windows via WSL.
- Python 3.12 or higher.
- Conda or Mamba.
- A CUDA-capable GPU for the `pytorch=*=cuda*` builds pinned in the environment files.
  Check the CUDA version in the top right of `nvidia-smi` and make sure the `cuda-version`
  pin in `environment.yml` does not exceed it, or OpenMM will fail to create a `Context`
  with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.

## Conda environment

From this directory:

```bash
bash conda_install.sh
```

The `openmmnqe` environment is recreated **from scratch on every run** — any existing
environment with that name is removed first. Set `ENV_NAME` to install into a
differently named environment instead:

```bash
ENV_NAME=openmmnqe2 bash conda_install.sh
```

The script creates the environment from `environment.yml`, compiles PLUMED, the
`openmm-plumed` plugin and py-plumed into it (sources are cloned into the gitignored
`build_tools/sources/`, wiped on each run), installs `openmmnqe` in editable mode, and
finishes with import checks. It is equivalent to running, from this directory:

```bash
conda env create -f environment.yml
conda activate openmmnqe
src_dir="$(mktemp -d)"
source build_plumed.sh && build_plumed "${src_dir}" && build_py_plumed "${src_dir}"
pip install -e ..
```

(`build_plumed.sh` is a function library rather than a script; `build_py_plumed` reuses
the plumed2 checkout that `build_plumed` leaves behind, so both take the same working
directory.)

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

It defaults to `gc.py` if no script is given. For multi-stage runs, chain the stages
together in one job with a `run.sh` that calls each script in turn.

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

All three installers share `build_plumed.sh`, which is where the PLUMED and
OpenMM-PLUMED versions are pinned.

## reactiontools

Reaction paths (NEB, transition states, IRC), ORCA and free-energy-surface plotting live
in [reactiontools](https://github.com/LouieSlocombe/reactiontools), which the environment
files install from git. If you need ORCA, follow the install steps in that repository's
`build_tools/README.md` — it is licensed separately and has to be put on the machine by
hand.

## Next steps

Worked examples live in `examples/`: `gc.py` (guanine-cytosine) and `ma.py`
(malonaldehyde) each run a single-file proton-transfer workflow end to end.
