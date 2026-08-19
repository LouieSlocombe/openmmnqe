# Installation guide

There are three ways to install `openmmnqe`, depending on what you need:

| Route | Use when | Script |
|---|---|---|
| Conda environment | Normal use. Everything from conda-forge except PLUMED and the editable checkouts. | `conda_install.sh` |
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
- Git, to clone the PLUMED sources and the editable dependencies. The compiler,
  `cmake` and `make` come from the environment; git does not.
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
`build_tools/sources/`, wiped on each run), installs `openmmnqe` and its four git
dependencies in editable mode so changes to the source are picked up without
reinstalling, and finishes with import checks. It is equivalent to running, from this
directory:

```bash
conda env create -f environment.yml
conda activate openmmnqe
src_dir="$(mktemp -d)"
source build_plumed.sh && build_plumed "${src_dir}" && build_py_plumed "${src_dir}"
pip install -e ..
source editable_repos.sh && install_editable_repos ../..
```

(`build_plumed.sh` and `editable_repos.sh` are function libraries rather than scripts.
`build_py_plumed` reuses the plumed2 checkout that `build_plumed` leaves behind, so both
take the same working directory, and the PLUMED version is pinned there in one place.)

### Editable dependencies

`forcefill`, `reactiontools`, `geodesic_interpolate` and `sella` are all repositories
that get edited alongside this package, so every installer clones them **next to the
repository** and installs them editable rather than pulling them from GitHub on each
install:

```
skunkworks/
├── openmmnqe/
├── forcefill/
├── reactiontools/
├── geodesic_interpolate/
└── sella/
```

Set `SRC_DIR` to keep them elsewhere. A checkout that is already there is used exactly
as it is — the installer never pulls, resets or removes one, so uncommitted work is safe
across a rebuild. Only a missing one is cloned.

The editable installs run **after** `pip install -e ..`, not before: `pyproject.toml`
declares `forcefill` and `reactiontools` as `name @ git+...` dependencies, and pip
re-clones those even when the package is already installed, so an editable install done
first would be replaced by the copy pip pulls. The same applies within the list itself,
which is why `reactiontools` is installed before `geodesic_interpolate` and `sella` —
the two it declares the same way. Every installer finishes by checking each one imports
from its checkout rather than from `site-packages`.

`environment_ci.yml` is the exception: a GitHub runner has only this repository checked
out, so CI keeps the `git+` pip entries and takes all four straight from GitHub. Those
`git+` installs track each repository's default branch, so CI failing here is the
integration signal that a sibling moved; forcefill bumps its `version` on API-visible
changes, so comparing `pip show forcefill` against a checkout's `pyproject.toml` tells
a stale install from a new one.

## Sol cluster

`custom_install_sol.sh` builds the `openmmnqe` environment on Sol. Most dependencies come
from conda-forge, but PLUMED is compiled from source because the conda-forge build does
not include the `opes` module. PLUMED sources are cloned into
`$SCRATCH/openmmnqe_sources`, and both the environment and those sources are recreated
from scratch on each run.

`openmmnqe` itself and the four editable dependencies are cloned into
`$HOME/openmmnqe_src` instead — outside the build area, since that is wiped — and
installed editable, so `git pull` in a checkout is enough to update it. Set `SRC_DIR` to
put them somewhere else.

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
a while. The editable checkouts are shared with the other two routes and are not wiped.

All three installers share `build_plumed.sh`, which is where the PLUMED and
OpenMM-PLUMED versions are pinned, and `editable_repos.sh`, which is where the git
dependencies are listed.

## One environment for openmmnqe + openmmqmmm

Running the stages on a QM/MM potential (`PreparedSystem`, see the main README) needs
[openmmqmmm](https://github.com/LouieSlocombe/openmmqmmm) importable from this
environment. This environment is the superset, so add openmmqmmm to it editable, with
`--no-deps` because the environment files here are the authority on dependencies:

```bash
conda activate openmmnqe
pip install -e /path/to/skunkworks/openmmqmmm --no-deps
python -c "import openmm, openmmqmmm, openmmnqe; assert hasattr(openmm, 'PythonForce')"
```

`editable_repos.sh` deliberately does not list openmmqmmm: the packages share no imports
and are composed by user scripts, so neither installer should provision the other. The
cross-package tests live in openmmqmmm's `tests/test_nqe_interop.py` and run whenever
both packages share an environment.

## reactiontools

Reaction paths (NEB, transition states, IRC), ORCA and free-energy-surface plotting live
in [reactiontools](https://github.com/LouieSlocombe/reactiontools), one of the editable
checkouts above. If you need ORCA, follow the install steps in that repository's
`build_tools/README.md` — it is licensed separately and has to be put on the machine by
hand.

That repository ships its own `build_tools/` with the same layout, for a CPU-only
environment without OpenMM. The two share `geodesic_interpolate` and `sella`, so a single
set of checkouts serves both.

## Next steps

Worked examples live in `examples/`: `gc.py` (guanine-cytosine) and `ma.py`
(malonaldehyde) each run a single-file proton-transfer workflow end to end.
