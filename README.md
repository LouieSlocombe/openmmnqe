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

Repeated path values are averaged automatically, and `progress_bins`
forms conditional means for a continuous progress coordinate. Matplotlib is
available through the `plot` optional dependency.

## Trajectory formats

Every stage writes a PDB trajectory by default, which is self-describing but
large and slow. Pass `trajectory='dcd'` or `'xtc'` for anything long — a
solvated production run especially — and the stage writes
`<prefix>_topology.pdb` alongside, so the binary trajectory stays readable.
`trajectory='h5'` needs `pip install openmmnqe[traj]` and is the only format
that can carry velocities in the trajectory itself; `trajectory='none'` writes
none at all. `velocity_record_interval` writes a `<prefix>_velocities.npz`
archive on any stage, classical or ring-polymer, and
`nqe.vibrational_spectrum` turns it into a vibrational density of states.

The ring-polymer centroid and bead trajectories wrap molecules into the box
against the *Topology*'s bonds rather than by asking OpenMM to do it, and
honour `TrajectoryOptions.enforce_periodic_box` like the classical stages do.
The distinction matters on a mixed ML/MM system: `createMixedSystem` deletes
every bonded term inside the ML region, so a molecule modelled entirely by the
ML potential is invisible to the molecule list OpenMM wraps by, and its atoms
would be scattered across the box one at a time. A non-periodic System is now
left alone rather than folded into OpenMM's default 2 nm box.

## Ring-polymer thermodynamics

Every RPMD stage writes `<prefix>_thermo.log` alongside its spread, centroid and
bead outputs. An RPMD `Context` holds one copy of the system rather than the ring
polymer, so its energy is not a bead average and its kinetic temperature is not
the ring polymer's; the log is where the thermodynamics actually lives.

The physical observables are the centroid-virial kinetic estimator `KE_cv`, the
mean bead potential energy `PE_mean`, and their sum `E_quantum`. The rest are
diagnostics: `E_ring` is the ring-polymer Hamiltonian, worth watching for drift
rather than for physics, and `T_ring` and `T_centroid` should both settle at the
integrator's setpoint.

```python
nqe.run_openmm_rpmd_prod(modeller, forcefield, n_report=1000)

averages = nqe.rpmd_thermodynamic_averages("rpmd_prod_thermo.log", discard=0.1)
energy, error = averages["E_quantum(kJ/mol)"]
print(f"quantum internal energy {energy:.2f} +/- {error:.2f} kJ/mol")

nqe.plot_rpmd_thermodynamics(
    "rpmd_prod_thermo.log",
    energy_columns=["KE_cv(kJ/mol)", "PE_mean(kJ/mol)", "E_quantum(kJ/mol)"],
    filename="rpmd-thermodynamics.png",
)
```

Errors come from block averaging, because consecutive samples down one
trajectory are correlated and `std / sqrt(n)` would flatter them. Call
`nqe.rpmd_thermodynamics(simulation)` to take the same set of readings once,
outside any reporter.

Each report reads every bead with its forces, costing about one RPMD step, so
the default `n_report` of 1000 makes it a fraction of a percent. That holds on a
mixed ML/MM system too: the centroid and bead trajectory reporters ask for
positions alone, which OpenMM answers without running the ML model at all.

Two caveats are worth knowing: `KE_cv` is biased for a system with constraints,
because OpenMM's forces omit constraint forces -- the reporter warns, and the
fix is to run the beads flexible -- and under ring-polymer contraction the
estimators describe the full potential rather than the contracted one that
drives the dynamics.

`E_ring` and `E_spring` are reconstructed from the bead pass rather than read
from `RPMDIntegrator.getTotalEnergy()`. That method agrees with the
reconstruction to within one part in a million. Before OpenMM 8.6.1, that
method could deadlock on CUDA or OpenCL when a `PythonForce` callback needed
the GIL held by the reporting thread. OpenMM 8.6.1 disables that worker-thread
path for `PythonForce`; the reconstruction still reuses the bead states read
for the other observables. Reconstructing costs one assumption --
that OpenMM links neighbouring copies with springs of frequency
`P k_B T / hbar` -- which the test suite pins against OpenMM itself.

`nqe.RPMDKineticDecompositionReporter` splits that same `KE_cv` over individual
atoms, which is the diagnostic that says how quantum one particular proton is: a
classical atom sits at `3kT/2`, 3.74 kJ/mol at 300 K, and a proton in a stiff
bond several times above it. Because that per-atom energy is the exact
derivative of the free energy with respect to log mass, integrating it over mass
gives an equilibrium isotope effect from a couple of short runs at fictitious
intermediate masses, rather than from differencing separate H and D
trajectories -- `nqe.rpmd_mass_integration_nodes` plans the quadrature,
`nqe.rpmd_isotope_free_energy` combines it, and `nqe.rpmd_fractionation_factor`
compares two sites.

A third caveat, fixed rather than documented: `getForces` reports a virtual
site's own force as well as the shares it redistributes onto that site's
parents, so only particles with mass enter the virial. Summing every row would
count a TIP4P or Drude site twice.

Heat capacity and pressure are deliberately absent. The centroid-virial heat
capacity needs second derivatives OpenMM will not supply, and the
`k_B beta^2 Var(E)` fluctuation formula that looks like a substitute is wrong
for path integrals; a centroid-virial pressure needs the true virial, which
forces alone do not give under periodic boundary conditions.

## Installation

OpenMM 8.6.1 or higher, OpenMM-ML 1.8 or higher, and Python 3.12 or higher are required.

Some dependencies (openmm-ml, openmm-plumed) are not installable from PyPI, and openmm-plumed has to be compiled, so
the package is installed into a conda environment. AmberTools is conda-only too — forcefill's GAFF backend runs the
`antechamber` and `parmchk2` executables, which have to be on `PATH`. See
[build_tools/README.md](build_tools/README.md) for the full instructions and the environment files.

## Citations

References for the methods and software used are collected in [CITATIONS.bib](CITATIONS.bib).
