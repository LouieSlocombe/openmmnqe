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
instance, so build a fresh `PreparedSystem` per mutating stage, and pass `barostat_freq=None` when the System carries
a QM/MM force.

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

## Bead expansion along a reaction coordinate

RPMD spread logs can include centroid atom-pair distances sampled on the same
steps as a transferring atom's ring-polymer expansion. Use `"mean"` for the
mean bead-centroid radius; the default `"rms"` preserves the package's earlier
radius-of-gyration output.

```python
nqe.run_openmm_rpmd_prod(
    modeller,
    forcefield,
    atoms_to_watch=[17],
    expansion_metric="mean",
    distance_pairs_to_watch=[(4, 17), (9, 17)],
)

# Direct expansion-versus-distance view.
nqe.plot_rpmd_atom_expansion(
    "rpmd_prod_spread.log",
    distance_columns="Distance_Atom4-Atom17(nm)",
    length_unit="angstrom",
    filename="expansion-vs-donor-distance.png",
)
```

To make a Figure-7-style diagnostic from a PLUMED `PATHMSD` run, print
`path.sss` at the same stride as `n_report`, align it to the reporter rows with
reactiontools, and normalise PLUMED's one-based path-image coordinate:

```python
import reactiontools as rt

spread_log = "rpmd_prod_spread.log"
with open(spread_log) as handle:
    n_samples = sum(1 for _ in handle) - 1

path_image = rt.cv_from_colvar(
    "COLVAR",
    n_frames=n_samples,
    cv_name="path.sss",
)
normalised_path_progress = (path_image - 1.0) / (n_images - 1)

figure, axes = nqe.plot_rpmd_atom_expansion(
    spread_log,
    path_progress=normalised_path_progress,
    progress_bins=n_images,
    length_unit="angstrom",
    filename="expansion-along-path.png",
)
```

Repeated path/window values are averaged automatically, and `progress_bins`
forms conditional means for a continuous progress coordinate. For separate
umbrella windows, repeat each window's normalised centre for its reporter rows
before concatenating the logs. Matplotlib is available through the `plot`
optional dependency.

## Installation

Some dependencies (openmm-ml, openmm-plumed) are not installable from PyPI, and openmm-plumed has to be compiled, so
the package is installed into a conda environment. AmberTools is conda-only too — forcefill's GAFF backend runs the
`antechamber` and `parmchk2` executables, which have to be on `PATH`. See
[build_tools/README.md](build_tools/README.md) for the full instructions and the environment files.

## Citations

References for the methods and software used are collected in [CITATIONS.bib](CITATIONS.bib).
