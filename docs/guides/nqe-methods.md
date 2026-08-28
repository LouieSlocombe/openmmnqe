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
random force spectrum per type.
{func}`~openmmnqe.tools.set_adqtb_particle_types_by_element` assigns one type
per element, which is the usual choice.

## Isotope effects

{func}`~openmmnqe.tools.deuterate_system` replaces hydrogens with deuterium,
either all of them or a chosen subset. Every stage also takes a deuteration
argument directly, so an isotope substitution does not need a separate pass
over the topology.

Because a kinetic isotope effect is a *ratio*, the H and D runs have to be
identical in every other respect — same seed handling, same bead count, same
bias. Running them from one script rather than two is the cheap way to
guarantee that.
