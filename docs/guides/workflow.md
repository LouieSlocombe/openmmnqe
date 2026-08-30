# Running a workflow

Each `run_openmm_*` function is one stage, and they are meant to run in order,
each starting from the structure the previous stage wrote:

1. {func}`~openmmnqe.openmm.run_openmm_relaxation` or
   {func}`~openmmnqe.openmm.run_openmm_relaxation_simple` — take the strain out
   of the starting structure,
2. {func}`~openmmnqe.openmm.run_openmm_heating` — warm it to temperature under
   backbone restraints,
3. {func}`~openmmnqe.openmm.run_openmm_npt` — relax the box density,
4. {func}`~openmmnqe.openmm.run_openmm_prod` — classical production, optionally
   biased,
5. the nuclear-quantum stages, covered in [](nqe-methods.md).

Every stage takes the same shape: build the system, optionally deuterate it,
attach a PLUMED bias and the reporters, run, then save its final structure.
Classical stages initialise new velocities when started from that structure.

## Building the force field first

Nothing in this package parameterises a ligand. That is
[forcefill](https://github.com/LouieSlocombe/forcefill)'s job: it asks the base
force field which residues it cannot match, parameterises those with GAFF and
AM1-BCC, and writes an ffxml that loads underneath the standard files.

```{literalinclude} ../../examples/workflows.py
:pyobject: ligand_forcefield
:language: python
```

Two details in there are easy to get wrong. A skipped residue would otherwise
surface much later as a template error at `createSystem`, so it is checked
immediately; and `forcefield_xml` is `None` when the base files already matched
every residue, which `ForceField()` will not accept.

Repair the structure *before* this, with {func}`~openmmnqe.io.fix_pdb` —
forcefill decides what needs parameters by asking what the base force field
cannot match, and a protein missing its hydrogens matches nothing. But nothing
may edit the topology *between* the repair and `createSystem`: the templates
describe the residues exactly as the input PDB spells them.

## The equilibration sequence

```{literalinclude} ../../examples/workflows.py
:pyobject: run_eq_workflow
:language: python
```

`output_prefix` names every file a stage writes, so giving each stage its own
prefix is what keeps the trajectories, logs and restart files apart.

## Shared arguments

These behave the same across every stage:

`potential` (with `ml_idx`)
: Runs an ML/MM mixed system and forces the CUDA platform. See
  [](potentials.md).

`plumed_script_path`
: Attaches a PLUMED bias. See [](enhanced-sampling.md).

`output_prefix`
: Names every file the stage writes.

`trajectory`
: Picks the trajectory format — `'pdb'` by default, `'dcd'` or `'xtc'` for
  anything long, `'none'` for nothing at all. Pass a
  {class}`~openmmnqe.openmm.TrajectoryOptions` instead of a bare name to set the
  interval or write only a subset of the atoms. See
  [](reporters.md#trajectory-formats).

`velocity_record_interval` (with `velocity_atom_indices`)
: Writes `<prefix>_velocities.npz`, which
  {func}`~openmmnqe.reporters.vibrational_spectrum` turns into a vibrational
  density of states. See [](reporters.md#velocities-and-spectra).

`seed`
: Fixes every random stream the stage draws. See [](#reproducibility).

Full signatures are in [](../api/openmm.rst).

## Reproducibility

Left alone, OpenMM takes the starting velocities, the thermostat noise and the
barostat's volume moves from system entropy, so two runs of the same script
never agree exactly. `seed` fixes all of them, and one seed is meant to be
handed to the whole workflow:

```python
nqe.run_openmm_heating(modeller, forcefield, output_prefix="heat", seed=2024)
nqe.run_openmm_npt(modeller, forcefield, output_prefix="npt", seed=2024)
nqe.run_openmm_prod(modeller, forcefield, output_prefix="prod", seed=2024)
```

It is a *master* seed rather than the number passed straight to OpenMM. Each
stage splits it into one independent stream per consumer, so the thermostat is
never driven by the numbers that chose the velocities -- which would correlate
the noise with the state it acts on -- and a stream keeps its meaning whichever
stages draw it. The two nuclear-quantum-effect routes split it the same way:
{func}`~openmmnqe.openmm.run_openmm_rpmd_equilibration` places the ring polymer
from one stream and drives the PILE thermostat from another, so the beads and
their noise are independent.

Reproducibility holds for the same platform, hardware, and OpenMM build.
Neither a change of platform nor the same platform on a different GPU is
expected to agree bit for bit, so a comparison that has to be exact -- the H
and D runs of a kinetic isotope effect, say -- belongs on one machine.
