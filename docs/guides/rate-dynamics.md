# Rates, recrossing, and thermostat-off RPMD

A metadynamics free-energy surface gives a transition-state-theory rate, and
TST assumes no trajectory ever recrosses the dividing surface. That assumption
is exactly what {func}`~openmmnqe.tools.deuterate_system` has always warned
about: changing masses on the same surface does not recover the dynamical
part of a kinetic isotope effect. The `openmmnqe.rates` module supplies that
part. The factorization is

```
k = kappa * k_QTST
```

where `k_QTST` comes from the free-energy surface and the thermal flux
through the dividing surface, and the transmission coefficient `kappa` is
measured by shooting swarms of short, unbiased, *microcanonical*
ring-polymer trajectories from configurations pinned at the surface and
watching which side they commit to.

Microcanonical means thermostat off: every RPMD production stage takes
`apply_thermostat=False`, which turns the PILE thermostat off and leaves
pure ring-polymer dynamics -- the ensemble every RPMD time-correlation
observable is defined in. The temperature argument is still required (it
sets the ring-polymer springs, not a thermostat target), and the barostat
must be off.

## The whole workflow

```{literalinclude} ../../examples/rates.py
:pyobject: run_malonaldehyde_rate
:language: python
```

Step by step:

1. **Surface.** Metadynamics under thermostatted RPMD
   ({func}`reactiontools.plumed_input_1pt` into
   {func}`~openmmnqe.openmm.run_openmm_rpmd_prod`), then
   {func}`reactiontools.summarise_fes` for the barrier position `s_dagger`.
2. **Harvest.** A stiff PLUMED `RESTRAINT` pins a thermostatted run at
   `s_dagger`; `snapshot_interval` makes the production stage write periodic
   full-bead restart archives. Thermalize inside the restraint first, then
   harvest from a second run, so every snapshot is an equilibrated
   dividing-surface configuration.
3. **Shoot.** {func}`~openmmnqe.rates.run_openmm_rpmd_recrossing` builds the
   *unbiased* system (no PLUMED, no barostat, no thermostat), and for every
   snapshot launches children with fresh thermal bead momenta, recording the
   collective variable. The restart archive checks masses, topology, and
   temperature but deliberately not forces, which is why a snapshot from the
   biased harvest loads cleanly into the unbiased system. The Python `cv`
   callable must be the same coordinate PLUMED biased -- here it is rebuilt
   from {func}`reactiontools.switching_value` with the `R_0` the script
   carries.
4. **Analyse.** {func}`~openmmnqe.rates.transmission_coefficient` turns the
   per-parent logs into `kappa(t)`, and {func}`~openmmnqe.rates.rpmd_rate`
   assembles the rate from its plateau, its forward-flux factor, and the
   surface.

## Reading kappa(t)

With snapshots from a finite-width restraint, `kappa` near time zero
reflects the restraint width rather than dynamics: it rises from around
zero as trajectories clear the initial scatter, then relaxes to the plateau
that multiplies the TST rate. Read the plateau, never the early rise, and
check the `s0_std` diagnostic against the barrier width -- a soft restraint
shows up there. `plot_transmission_coefficient` draws the curve, its
blocked error band, and the plateau.

Error bars are estimated by splitting parents into contiguous blocks in
harvest order, which absorbs both the correlation between children of one
parent and the correlation between parents along the harvest trajectory.
They cover the transmission statistics only; the surface's own statistical
error is not propagated (in a KIE ratio much of it cancels).

## Verifying the dynamics

Thermostat-off dynamics has no bath to absorb integration error, so its
conserved quantity is worth checking. The thermodynamic log every RPMD
stage writes already carries it -- `E_ring(kJ/mol)`, the ring-polymer
Hamiltonian -- and {func}`~openmmnqe.reporters.rpmd_energy_conservation`
turns that column into a drift-against-fluctuation verdict, in the same
spirit as [adQTB verification](adqtb-verification.md):

```{literalinclude} ../../examples/rates.py
:pyobject: run_rpmd_energy_conservation
:language: python
```

If the verdict says drifting, shrink the time step: the production default
of 1 fs is chosen for thermostatted runs, and the recrossing driver
defaults to 0.5 fs for exactly this reason.

## Kinetic isotope effects

Run the harvest and shooting stages twice, once with `deuterate=True`
(matching `deuterate_option`) on every stage, and take the ratio of the two
assembled rates. The snapshot loader's particle-mass check makes it
impossible to shoot deuterated children from protiated snapshots, so the
two campaigns cannot be crossed by accident. Both the surface and the
transmission coefficient change under deuteration -- that is the point.

## Correlation functions and spectra

The same thermostat-off dynamics supports time-correlation functions
directly. `velocity_record_interval` on the production stage attaches an
{class}`~openmmnqe.reporters.RPMDVelocityReporter`, which records
bead-averaged (centroid) velocities;
{func}`~openmmnqe.reporters.rpmd_velocity_autocorrelation` and
{func}`~openmmnqe.reporters.rpmd_vibrational_spectrum` turn the archive
into the RPMD approximation to the Kubo-transformed velocity
autocorrelation and a vibrational density of states:

```{literalinclude} ../../examples/rates.py
:pyobject: run_rpmd_spectrum
:language: python
```

Read high frequencies with care: the free ring-polymer spring frequencies
contaminate the spectrum near and above `n_beads * kB * T / hbar`, and
their resonances with physical modes can split or shift stretch bands.

## Limitations, stated

- The harvest restraint acts on each bead separately, so the harvested
  ensemble approximates the centroid-constrained one; `s0_mean`/`s0_std`
  measure the difference, and `max_s0_deviation` can prune outliers.
- The rate is first order and per molecule, in 1/ps, and it is only as
  dividing-surface-independent as the plateau is flat.
- `rpmd_rate` propagates the transmission error only.
