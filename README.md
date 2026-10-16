# OpenMM NQE

[![Documentation Status](https://readthedocs.org/projects/openmmnqe/badge/?version=latest)](https://openmmnqe.readthedocs.io/en/latest/?badge=latest)

This repo is for running Molecular Dynamics (MD) simulations using OpenMM, specifically there are convenience functions
for running and analysing nuclear quantum effects such as Ring-Polymer Dynamics and adaptive Quantum Thermal Bath
approaches.

## Scope

`openmmnqe` covers the simulation itself: the structure edits that get a system ready, the OpenMM stages (minimise,
heat, NPT, production, RPMD, adQTB), and the reporters that read them back. The RPMD production stages also run
thermostat-off (microcanonical) ring-polymer dynamics, and `openmmnqe.rates` builds on that to turn dividing-surface
snapshots into transmission coefficients and Bennett–Chandler rate constants — the dynamical half of a kinetic
isotope effect that a free-energy surface alone cannot give (see the
[rate-dynamics guide](https://openmmnqe.readthedocs.io/en/latest/guides/rate-dynamics.html)).

What sits either side of that lives in two dependencies.

**Ligand parameters** come from [forcefill](https://github.com/LouieSlocombe/forcefill). It asks the base force field
which residues it cannot match, parameterises those with GAFF2 and AM1-BCC, and writes an ffxml you load underneath
the standard files. The force field every stage takes is built from that:

```python
import forcefill as ff
import openmm.app as app

names = ("amber14-all.xml", "amber14/tip3pfb.xml")
result = ff.build_forcefield_xml(input_pdb, "ligands.xml", base_forcefield=names)
if result.skipped:
    raise RuntimeError(f"forcefill skipped residues: {result.skipped}")
extra = [] if result.forcefield_xml is None else [result.forcefield_xml]

pdb_data = app.PDBFile(input_pdb)
modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
forcefield = app.ForceField(*names, *extra)
```

A skipped residue would otherwise surface later, as a template error at `createSystem`, and `forcefield_xml` is
`None` when the base files already matched every residue, which `ForceField()` will not take.

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

**Pre-built Systems** run through the same stages via `PreparedSystem`. Every stage builds its System by calling
`forcefield.createSystem(topology, **kwargs)`, and `PreparedSystem` is a force-field stand-in that returns a System
built elsewhere — the motivating case being a QM/MM System exported by
[openmmqmmm](https://github.com/LouieSlocombe/openmmqmmm) with its `openmm.PythonForce` already attached (neither
package imports the other; the seam is plain OpenMM objects):

```python
export = openmmqmmm.export_rpmd_potential(theory=qm_mm, num_beads=32)
prepared = nqe.PreparedSystem(export.system)

nqe.run_openmm_rpmd_equilibration(export.modeller, prepared, n_beads=32)
nqe.run_openmm_rpmd_prod(export.modeller, prepared, checkpoint_file="rpmd_ready.chk",
                         n_beads=32, barostat_freq=None)
```

Bridge from the classical preparation stages into RPMD-on-a-prepared-System through the stage-final PDB, not the
binary `.chk`: an ordinary Context checkpoint only loads into an identical System, while the RPMD bead archive
validates masses, topology, temperature, and periodicity but not forces, so it survives the System gaining a force.
Stage options that mutate the System (`deuterate`, a non-None `barostat_freq`, `plumed_script_path`) mutate the held
instance, so build a fresh `PreparedSystem` per mutating stage. A barostat on a System carrying a `PythonForce` warns
rather than refusing: OpenMM builds its molecule list from the bonded pairs each force reports and a `PythonForce`
reports none, so those atoms are scaled one at a time instead of as molecules. That is still a valid volume move, but
watch the acceptance rate and check the potential responds to the box vectors it is handed; pass `barostat_freq=None`
to run at fixed volume instead.

## Examples

The `examples/` directory contains complete simulation workflows. Run the
scripts from the repository root so they can find the structures in
`tests/data/`. The grouped examples list their available workflows with
`--help`; for example:

```bash
python examples/potentials.py --help
python examples/potentials.py openmm_ml
python examples/rpmd.py openmm_rpmd --platform CUDA
python examples/workflows.py malonaldehyde_pt
```

These examples can require a CUDA GPU, downloaded MACE models, PLUMED, ORCA,
or AmberTools depending on the selected workflow. The focused regression
suite remains under `tests/` and runs with `pytest`.

## Documentation

The guides at [openmmnqe.readthedocs.io](https://openmmnqe.readthedocs.io/en/latest/)
cover each part of a run in depth, with their code pulled from the scripts in
`examples/`:

- [Running a workflow](https://openmmnqe.readthedocs.io/en/latest/guides/workflow.html) —
  the stage sequence, the arguments every stage shares, and seeding.
- [Nuclear quantum effects](https://openmmnqe.readthedocs.io/en/latest/guides/nqe-methods.html) —
  RPMD, adQTB and equilibrium isotope effects.
- [Reporters and analysis](https://openmmnqe.readthedocs.io/en/latest/guides/reporters.html) —
  bead spread, trajectory formats, velocity spectra and ring-polymer thermodynamics.
- [Verifying an adQTB run](https://openmmnqe.readthedocs.io/en/latest/guides/adqtb-verification.html)
- [Rate dynamics](https://openmmnqe.readthedocs.io/en/latest/guides/rate-dynamics.html) —
  transmission coefficients and Bennett–Chandler rate constants.
- [Enhanced sampling](https://openmmnqe.readthedocs.io/en/latest/guides/enhanced-sampling.html) —
  OPES, steered paths and independent walkers.
- [ML/MM potentials and QM/MM](https://openmmnqe.readthedocs.io/en/latest/guides/potentials.html)
- [API reference](https://openmmnqe.readthedocs.io/en/latest/api/index.html)

## Installation

OpenMM 8.6.1 or higher, OpenMM-ML 1.8 or higher, and Python 3.12 or higher are required.

Some dependencies (openmm-ml, openmm-plumed) are not installable from PyPI, and openmm-plumed has to be compiled, so
the package is installed into a conda environment. AmberTools is conda-only too — forcefill's GAFF backend runs the
`antechamber` and `parmchk2` executables, which have to be on `PATH`. See
[build_tools/README.md](build_tools/README.md) for the full instructions and the environment files.

## Citations

References for the methods and software used are collected in [CITATIONS.bib](CITATIONS.bib).
