# OpenMM NQE

This repo is for running Molecular Dynamics (MD) simulations using OpenMM, specifically there are convenience functions
for running and analysing nuclear quantum effects such as Ring-Polymer Dynamics and adaptive Quantum Thermal Bath
approaches.

## Scope

`openmmnqe` covers the simulation itself: system preparation, the OpenMM stages (minimise, heat, NPT, production,
RPMD, adQTB), the PLUMED input those stages are driven with, and the reporters that read them back.

What sits either side of that lives in [reactiontools](https://github.com/LouieSlocombe/reactiontools), which is a
dependency — building a reaction path with NEB, refining a transition state, running ORCA, and plotting the
free-energy surface that comes out. Import those from there:

```python
import openmmnqe as nqe
import reactiontools as rt

product = rt.swap_bonding_configuration(reactant, 0, 8, 1)
neb_path = rt.quick_guess_path(reactant, product)
nqe.convert_xyz_to_plumed_ref("neb_path.xyz", "index_atoms.pdb", "neb_path.pdb")
...
rt.plot_plumed_fes("fes.dat", filename="fes")
```

## Installation

Some dependencies (openmm-ml, openmm-plumed) are not installable from PyPI, and openmm-plumed has to be compiled, so
the package is installed into a conda environment. See [build_tools/README.md](build_tools/README.md) for the full
instructions and the environment files.

## Citations

References for the methods and software used are collected in [CITATIONS.bib](CITATIONS.bib).
