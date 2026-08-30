# Reporters and analysis

OpenMM's own reporters see only the `Context`, which for an `RPMDIntegrator`
holds a single copy of the system rather than the ring polymer. Anything that
needs the beads themselves — their spread, their individual trajectories, their
centroid, or their energies — has to ask the integrator. That is what these six
reporters do. The `run_openmm_rpmd_*` drivers attach the first four for you;
the per-atom kinetic decomposition and the velocity recorder are opt-in,
because each reads the beads a second time per report.

All six follow OpenMM's reporter protocol: `describeNextReport` says when the
next report is due and what state it needs, and `report` writes it.

| Reporter | Writes |
|---|---|
| {class}`~openmmnqe.reporters.RPMDQuantumSpreadReporter` | Quantum spread of selected atoms |
| {class}`~openmmnqe.reporters.RPMDBeadReporter` | Every bead's trajectory, one PDB each |
| {class}`~openmmnqe.reporters.RPMDCentroidReporter` | The ring-polymer centroid, one PDB |
| {class}`~openmmnqe.reporters.RPMDThermodynamicReporter` | Ring-polymer thermodynamic estimators |
| {class}`~openmmnqe.reporters.RPMDKineticDecompositionReporter` | Per-atom quantum kinetic energy |
| {class}`~openmmnqe.reporters.RPMDVelocityReporter` | Centroid velocities, for spectra |

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

### Per-atom kinetic decomposition

The system `KE_cv` answers "how quantum is this system". Keeping the atom axis
of the same virial instead of summing it away answers the question a
ring-polymer run is usually kept for — *how quantum is this particular proton*:

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_rpmd_kinetic_decomposition
:language: python
```

A classical atom sits at `3kT/2`, which is 3.74 kJ/mol at 300 K; a proton in a
stiff bond sits several times above it.
{func}`~openmmnqe.reporters.plot_rpmd_kinetic_decomposition` draws that
reference as a dashed line, because the gap to it is the whole reading.
{func}`~openmmnqe.reporters.rpmd_kinetic_decomposition` takes one reading off
any simulation, and
{func}`~openmmnqe.reporters.rpmd_kinetic_decomposition_averages` reads the log
back with the same block-averaged errors.

The per-atom free-particle term is `3kT/2`, three degrees of freedom per atom,
rather than the system's `d` shared out — there is no non-arbitrary way to
divide up the constraint and centre-of-mass reductions that `d` carries. So the
decomposition sums back to `KE_cv` exactly when `d == 3 * n_massive`, and
otherwise differs by `(N_constraints + 3 * n_CMMotionRemover) * kT / 2`. For a
flexible system with no centre-of-mass removal — the only case where the
estimator is unbiased anyway — the two agree.

The drivers attach it on request rather than automatically:
`run_openmm_rpmd_prod(..., atoms_to_watch=[...], kinetic_decomposition=True)`.
It reads the beads a second time per report, so turning it on for everyone who
passes `atoms_to_watch` would quietly double their per-report cost. At the
default `n_report` of 1000 that second pass is irrelevant; at the interval of
ten or so that per-atom statistics want, give the thermodynamic reporter the
coarser interval of the two.

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
