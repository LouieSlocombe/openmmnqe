# Reporters and analysis

OpenMM's own reporters see only the `Context`, which for an `RPMDIntegrator`
holds a single copy of the system rather than the ring polymer. Anything that
needs the beads themselves — their spread, their individual trajectories, their
centroid, or their energies — has to ask the integrator. That is what six of
these seven reporters do. The `run_openmm_rpmd_*` drivers attach the first four
for you; the per-atom kinetic decomposition and the velocity recorder are
opt-in, because each reads the beads a second time per report. The seventh has
nothing to do with beads: it records a classical run's velocities into the same
archive, which is the baseline a ring-polymer spectrum is read against.

All seven follow OpenMM's reporter protocol: `describeNextReport` says when the
next report is due and what state it needs, and `report` writes it.

| Reporter | Writes |
|---|---|
| {class}`~openmmnqe.reporters.RPMDQuantumSpreadReporter` | Quantum spread of selected atoms |
| {class}`~openmmnqe.reporters.RPMDBeadReporter` | Every bead's trajectory, one PDB each |
| {class}`~openmmnqe.reporters.RPMDCentroidReporter` | The ring-polymer centroid, one PDB |
| {class}`~openmmnqe.reporters.RPMDThermodynamicReporter` | Ring-polymer thermodynamic estimators |
| {class}`~openmmnqe.reporters.RPMDKineticDecompositionReporter` | Per-atom quantum kinetic energy |
| {class}`~openmmnqe.reporters.RPMDVelocityReporter` | Centroid velocities, for spectra |
| {class}`~openmmnqe.reporters.VelocityArchiveReporter` | A classical run's velocities, same archive |

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

Both reporters read the raw stored coordinates and wrap molecules into the box
themselves, following the *Topology*'s bonds. OpenMM's own
`enforcePeriodicBox` cannot be used here: it wraps by a molecule list built
from the bonded pairs the forces report, and
`MLPotential.createMixedSystem` deletes every bonded term inside the ML region
while the `openmm.PythonForce` that replaces them reports none. A molecule
modelled entirely by the ML potential would therefore have each of its atoms
moved into the box separately, which tears it apart. Where OpenMM's list *is*
complete the two agree exactly, so nothing changes for a plain MM run. Pass
`enforce_periodic_box=False` in the trajectory options to keep the raw
coordinates instead; a non-periodic System is never wrapped either way.

## Trajectory formats

Every stage writes a PDB by default, and for a long solvated run that is the
wrong choice: PDB is text, so it is roughly an order of magnitude larger than
DCD and two larger than XTC, and it is slower to write and to read back. The
bead reporter is the sharpest case — one whole trajectory per bead.

`trajectory` takes a bare format name, or a
{class}`~openmmnqe.openmm.TrajectoryOptions` when the interval or an atom subset
matters too:

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_rpmd_binary_trajectory
:language: python
```

| Format | Notes |
|---|---|
| `pdb` | The default. Self-describing, and what {func}`reactiontools.path_from_steered_md` reads unaided. Large and slow. |
| `dcd` | Compact binary, from OpenMM. Needs no extra dependency. |
| `xtc` | Smaller still, from OpenMM. Lossy: coordinates are quantized to 1e-3 nm. |
| `h5` | mdtraj's HDF5. The only format carrying velocities and topology in one file; needs `pip install openmmnqe[traj]`, and cannot be written from bead states. |
| `none` | No trajectory, for a run analysed entirely through its logs. |

Because DCD and XTC carry no topology of their own, a stage writing one also
writes `<prefix>_topology.pdb` up front — before the run has had a chance to
crash without it. That is the file to pass as `top=` to mdtraj, or to
{func}`reactiontools.path_from_steered_md`. An atom subset is applied to it
too, so the two always match, and chains and residues survive the selection so
a `resname` query still works downstream.

## Velocities and spectra

Positions alone cannot give a vibrational density of states.
{class}`~openmmnqe.reporters.VelocityArchiveReporter` records a classical run's
velocities into a `.npz` archive, and
{class}`~openmmnqe.reporters.RPMDVelocityReporter` records a ring polymer's
centroid velocities into the same one — so
{func}`~openmmnqe.reporters.velocity_autocorrelation` and
{func}`~openmmnqe.reporters.vibrational_spectrum` read either without knowing
which wrote it. That is what makes the classical spectrum a like-for-like
baseline. Ask for one with `velocity_record_interval` on any stage:

```{literalinclude} ../../examples/rates.py
:pyobject: run_classical_spectrum
:language: python
```

Frames are held in memory until the run ends, so pass `velocity_atom_indices`:
every frame otherwise costs `3 * n_atoms` doubles. A spectrum also wants frames
close enough together to resolve the fastest mode, which is a much finer
cadence than the logs want — hence a separate interval rather than `n_report`.

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

`E_ring` and `E_spring` are reconstructed from the bead pass — the bead
kinetic and potential energies plus springs of frequency `P k_B T / hbar` —
rather than read from `RPMDIntegrator.getTotalEnergy()`. The two agree to
within one part in a million, but that method cannot be called at all on a
mixed ML/MM System on CUDA or OpenCL: the ML potential is an
`openmm.PythonForce`, those platforms evaluate forces on a worker thread, and
the method holds the GIL while it waits for one. `step()` and `getState()` do
release it, so such a run used to advance normally and then stop dead at its
first report.

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
coarser interval of the two — which means attaching the reporters by hand,
because the drivers hand every reporter the same `n_report`.

These are the only two reporters that cost anything. The centroid and bead
trajectory reporters ask for positions alone, which OpenMM answers without
evaluating any force — including, on a mixed ML/MM system, without running the
ML model. So the several bead passes a report makes are not several force
evaluations.

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
