# Nuclear quantum effects: RPMD and adQTB

There are two routes to nuclear quantum effects here, and they trade accuracy
against cost in opposite directions.

**Ring-polymer molecular dynamics** is the reference method. It converges to
the exact quantum statistics given enough beads, but costs one force evaluation
per bead — a 32-bead run is 32 times the work of the classical one.

**The adaptive quantum thermal bath** costs no more than a classical run, but
is an approximation.

Both are paired stages: an equilibration that writes a restart, then a
production run that consumes it. RPMD restart files contain every bead, rather
than the single-Context checkpoint an ordinary OpenMM run would write.

## RPMD

{func}`~openmmnqe.openmm.run_openmm_rpmd_equilibration` then
{func}`~openmmnqe.openmm.run_openmm_rpmd_prod`:

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_openmm_rpmd
:language: python
```

{func}`~openmmnqe.openmm.run_openmm_rpmd_contracted` is the alternative
production stage. Ring-polymer contraction evaluates the expensive part of the
force on fewer beads than the cheap part, which is what makes a large bead
count affordable when only some interactions need the full ring.

### Continuing a completed production run

Both production drivers save every bead's positions and velocities to
`<output_prefix>.chk` when they finish. Pass that file as `checkpoint_file`
to continue production; there is no need to repeat equilibration:

```python
import openmm.app as app
import openmm.unit as unit
import openmmnqe as nqe

# Recreate the same topology and force field used for the previous segment.
# Its final centroid PDB supplies the topology; the .chk restores all beads.
pdb = app.PDBFile("rpmd_prod_final.pdb")
modeller = app.Modeller(pdb.topology, pdb.positions)
nqe.run_openmm_rpmd_prod(
    modeller,
    forcefield,
    checkpoint_file="rpmd_prod.chk",
    output_prefix="rpmd_prod_2",
    n_beads=32,
    temperature=300 * unit.kelvin,
    barostat_freq=None,  # Match the previous segment's ensemble.
    steps=500_000,
)
```

Here `steps` means **additional** steps. The saved time and step count carry
forward, along with the periodic box. The same pattern works with
{func}`~openmmnqe.openmm.run_openmm_rpmd_contracted`; retain the previous
segment's contraction settings. A runnable equilibration → production →
continuation example is available as
`python examples/rpmd.py openmm_rpmd_prod --platform Reference`.

Use a **new output prefix** for each segment: trajectories, logs, velocity
archives, and numbered snapshots are written afresh. Keep the same ordered
atoms, bead count, masses/deuteration, and temperature; the loader checks
these. Also recreate the same forces, timestep, thermostat/barostat settings,
and bias configuration, which the archive does not validate. When using
`PreparedSystem`, build a fresh System if stage options modify it.

These archives preserve the bead state, but do not contain thermostat or
barostat random-generator state or external bias history. A stochastic
continuation therefore does not reproduce an uninterrupted trajectory bit
for bit, even with the same `seed`. A final centroid PDB alone is insufficient
to restore the ring polymer.

### Working with beads directly

{func}`~openmmnqe.tools.init_beads` seeds bead positions and velocities at the
simulation temperature, {func}`~openmmnqe.tools.step_rpmd` advances the
simulation while keeping the step count synchronised, and
{func}`~openmmnqe.tools.centroid_positions` averages the beads back into a
single structure.

To decide how many beads a system needs, compare the thermal de Broglie
wavelength against the length scale of the motion in question:

```python
import openmmnqe as nqe

wavelength = nqe.get_thermal_de_broglie_wavelength(mass, temperature)
```

## adQTB

{func}`~openmmnqe.openmm.run_openmm_adqtb_eq` then
{func}`~openmmnqe.openmm.run_openmm_adqtb_prod`:

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_openmm_adqtb
:language: python
```

The bath needs to know which particles are which, because it adapts a separate
random force spectrum per type, and a particle left untyped gets a type to
itself. Both drivers therefore default to `particle_types="element"`, which
assigns one type per element through
{func}`~openmmnqe.tools.set_adqtb_particle_types_by_element`, splitting a
symbol when masses differ so that deuterium does not share hydrogen's bath.

Adaptation has to have converged before the production run means anything.
Both drivers log the adapted spectra to `<prefix>_friction.log`, and
[verifying an adQTB run](adqtb-verification.md) covers how to read that back.

## Isotope effects

{func}`~openmmnqe.tools.deuterate_system` replaces hydrogens with deuterium,
either all of them or a chosen subset. Every stage also takes a deuteration
argument directly, so an isotope substitution does not need a separate pass
over the topology.

Because a kinetic isotope effect is a *ratio*, the H and D runs have to be
identical in every other respect — same seed handling, same bead count, same
bias. Running them from one script rather than two is the cheap way to
guarantee that.

### Equilibrium isotope effects without differencing two runs

For an *equilibrium* isotope effect there is a better route than running both
isotopes and subtracting. The centroid-virial kinetic energy of an atom is the
exact derivative of the free energy with respect to that atom's log mass,

```{math}
\frac{\partial F}{\partial \ln m_i} = -\left\langle K_i \right\rangle
```

so the substitution free energy is an integral of a quantity
{class}`~openmmnqe.reporters.RPMDKineticDecompositionReporter` already logs:

```{math}
\Delta F = -\int_{\ln m_\mathrm{light}}^{\ln m_\mathrm{heavy}}
           \left\langle K \right\rangle_m \, \mathrm{d}\ln m
```

{func}`~openmmnqe.isotopes.rpmd_mass_integration_nodes` says which masses to run
at, {func}`~openmmnqe.isotopes.rpmd_isotope_free_energy` combines the averaged
kinetic energies, and {func}`~openmmnqe.isotopes.rpmd_fractionation_factor`
turns two sites into their H/D fractionation ratio:

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_rpmd_isotope_free_energy
:language: python
```

What this costs is one short run per quadrature node, at a *fictitious* mass
between the two isotopes — not one trajectory. For a 3600 cm⁻¹ O–H stretch at
300 K, where the exact H→D free energy is −18.898 kJ/mol:

| scheme | trajectories | masses (Da) | error (kJ/mol) |
|---|---|---|---|
| integrand at the H mass | 1 | 1.008 | **3.46 (18%)** |
| `nodes=1` | 1 | 1.425 | 0.09 |
| trapezoid on the endpoints | 2 | 1.008, 2.014 | 0.19 |
| **`nodes=2`** (the default) | 2 | 1.167, 1.740 | **0.00006** |

Two nodes cost the same as evaluating both endpoints and are some three
thousand times more accurate, which is why the endpoints are not offered. The
tempting one-trajectory shortcut — evaluating the integrand only at the
physical hydrogen mass — is not a free energy at all.

Mass *perturbation*, reweighting a single trajectory from one mass to another,
is deliberately absent. Mass enters the ring-polymer weight through the
free-particle normalisation and the spring term rather than as a
potential-energy difference, and the overlap between the light and heavy
distributions degrades as the bead count grows.
