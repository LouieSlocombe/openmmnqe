# ML/MM potentials and QM/MM

Passing `potential=` to any stage runs a machine-learned potential. Passing
`ml_idx=` alongside it makes the run a **mixed** ML/MM system: the atoms in
`ml_idx` get the ML potential, everything else stays on the classical force
field, and the two are coupled. Either argument forces the CUDA platform.

```{literalinclude} ../../examples/potentials.py
:pyobject: run_openmm_ml
:language: python
```

## Mixed ML/MM

The usual arrangement for a reaction in solution: the reacting solute is treated
with the ML potential, the water stays classical.

```{literalinclude} ../../examples/potentials.py
:pyobject: run_openmm_ml_mixed_system
:language: python
```

Choosing `ml_idx` is the substantive decision. It must cover every atom whose
bonding changes over the trajectory — a proton that transfers, and both the donor
and acceptor it moves between. An ML region that clips one of those describes a
bond breaking against a force field that says it cannot.

## Bringing your own system

{class}`~openmmnqe.openmm.PreparedSystem` is a stand-in force field: it hands a
pre-built `openmm.System` to a stage instead of having the stage build one. Any
stage that accepts a `ForceField` accepts it, which is how a system assembled
elsewhere — by a QM/MM setup, or by hand — gets access to the same
relaxation/heating/NPT/RPMD sequence as everything else.

## ASE and quantum chemistry

For structures that need a real electronic-structure method rather than an ML
surrogate, the examples also drive ASE calculators:

```{literalinclude} ../../examples/potentials.py
:pyobject: run_ase_orca
:language: python
```

This is a relaxation path, not a dynamics one — ORCA at every step of an RPMD run
is not a realistic proposition. The usual pattern is to refine geometries and
barrier heights quantum-mechanically, then run the dynamics on an ML potential
fitted to that level of theory.
