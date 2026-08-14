# OpenMM NQE

This repo is for running Molecular Dynamics (MD) simulations using OpenMM, specifically there are convenience functions
for running and analysing nuclear quantum effects such as Ring-Polymer Dynamics and adaptive Quantum Thermal Bath
approaches.

## Scope

`openmmnqe` covers the simulation itself: the structure edits that get a system ready, the OpenMM stages (minimise,
heat, NPT, production, RPMD, adQTB), and the reporters that read them back.

What sits either side of that lives in two dependencies.

**Ligand parameters** come from [forcefill](https://github.com/LouieSlocombe/forcefill). It asks the base force field
which residues it cannot match, parameterises those with GAFF2 and AM1-BCC, and writes an ffxml you load underneath
the standard files. The force field every stage takes is built from that:

```python
import forcefill as ff
import openmm.app as app

names = ("amber14-all.xml", "amber14/tip3pfb.xml")
result = ff.build_forcefield_xml(input_pdb, "ligands.xml", base_forcefield=names)

pdb_data = app.PDBFile(input_pdb)
modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
forcefield = app.ForceField(*names, result.forcefield_xml)
```

The templates describe the residues exactly as `input_pdb` spells them, so nothing may edit the topology between
those two blocks. A raw crystal structure has to be repaired *before* it is parameterised — forcefill is subtractive
only, so `nqe.fix_pdb` runs first, then `ff.clean_pdb`, then `build_forcefield_xml`.

**Everything reaction-side** lives in [reactiontools](https://github.com/LouieSlocombe/reactiontools) — building a
reaction path with NEB, refining a transition state, running ORCA, the PLUMED collective variables these stages are
biased along, turning a steered trajectory back into a reference path, and plotting the free-energy surface that
comes out:

```python
import openmmnqe as nqe
import reactiontools as rt

product = rt.swap_bonding_configuration(reactant, 0, 8, 1)
neb_path = rt.quick_guess_path(reactant, product)
rt.convert_xyz_to_plumed_ref("neb_path.xyz", "index_atoms.pdb", "neb_path.pdb")

plumed_input, fes_command = rt.plumed_input_neb_path(temperature)
nqe.run_openmm_prod(modeller, forcefield, plumed_script_path="plumed.dat")

rt.run_sum_hills()
rt.plot_plumed_fes("fes.dat", filename="fes")
```

The reactiontools builders take the `openmm.app.Modeller` and `openmm.unit.Quantity` this package works in, so they
can be called with whatever is already to hand.

## Installation

Some dependencies (openmm-ml, openmm-plumed) are not installable from PyPI, and openmm-plumed has to be compiled, so
the package is installed into a conda environment. AmberTools is conda-only too — forcefill's GAFF backend runs the
`antechamber` and `parmchk2` executables, which have to be on `PATH`. See
[build_tools/README.md](build_tools/README.md) for the full instructions and the environment files.

## Citations

References for the methods and software used are collected in [CITATIONS.bib](CITATIONS.bib).
