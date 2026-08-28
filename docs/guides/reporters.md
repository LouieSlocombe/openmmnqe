# Reporters and analysis

OpenMM's own reporters see only the `Context`, which for an `RPMDIntegrator`
holds a single copy of the system rather than the ring polymer. Anything that
needs the beads themselves — their spread, their individual trajectories, their
centroid, or their energies — has to ask the integrator. That is what these four
reporters do, and the `run_openmm_rpmd_*` drivers attach them for you.

All four follow OpenMM's reporter protocol: `describeNextReport` says when the
next report is due and what state it needs, and `report` writes it.

| Reporter | Writes |
|---|---|
| {class}`~openmmnqe.reporters.RPMDQuantumSpreadReporter` | Quantum spread of selected atoms |
| {class}`~openmmnqe.reporters.RPMDBeadReporter` | Every bead's trajectory, one PDB each |
| {class}`~openmmnqe.reporters.RPMDCentroidReporter` | The ring-polymer centroid, one PDB |
| {class}`~openmmnqe.reporters.RPMDThermodynamicReporter` | Ring-polymer thermodynamic estimators |

## Quantum spread

The spread of a ring polymer is the direct, visual measure of how quantum a
nucleus is behaving. {func}`~openmmnqe.reporters.track_rpmd_atom_expansion`
attaches the reporter for one target atom without constructing it directly:

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_rpmd_quantum_spread_reporter
:language: python
```

{func}`~openmmnqe.reporters.plot_rpmd_atom_expansion` plots the result against
either a centroid atom-pair distance or a supplied reference-path progress
coordinate — so the expansion can be read against the reaction coordinate it
matters for.

## Bead and centroid trajectories

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_rpmd_bead_reporter
:language: python
```

The centroid reporter is the one to use for anything that expects a normal
trajectory, since the centroid is the closest classical analogue of "the
position of the atom".

## Thermodynamics

{class}`~openmmnqe.reporters.RPMDThermodynamicReporter` covers the quantities a
`Context` cannot give: the centroid-virial kinetic estimator, the mean bead
potential energy, and the total quantum energy, alongside ring-polymer
diagnostics that say whether the trajectory is worth analysing at all.

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_rpmd_thermodynamic_reporter
:language: python
```

{func}`~openmmnqe.reporters.rpmd_thermodynamics` computes the same set once, off
any simulation. {func}`~openmmnqe.reporters.rpmd_thermodynamic_averages` reads
the log back with block-averaged standard errors, and
{func}`~openmmnqe.reporters.plot_rpmd_thermodynamics` plots it.

### Two quantities deliberately absent

A **heat capacity** would need the exact centroid-virial estimator's
second-derivative term, which OpenMM will not supply. The fluctuation formula
`k_B beta**2 Var(E)` that looks like a substitute is simply wrong for path
integrals — the estimator carries its own explicit `beta` dependence.

A **centroid-virial pressure** would need the true virial, which forces alone do
not give under periodic boundary conditions.

Neither is worth a plausible-looking wrong number, so neither is computed.

```{note}
The plotting helpers need `matplotlib`, which is an optional dependency:
install it with `pip install openmmnqe[plot]`.
```
