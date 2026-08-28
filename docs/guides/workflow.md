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

Full signatures are in [](../api/openmm.rst).
