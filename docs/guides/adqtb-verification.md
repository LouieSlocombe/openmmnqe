# Verifying an adQTB run

An adQTB thermostat spends the whole of an equilibration adjusting one noise
spectrum per particle type, and OpenMM is explicit that a run must be
equilibrated *"long enough for the friction coefficients to converge"* before
production. Nothing in a `Context` says whether that has happened.
{class}`~openmmnqe.adqtb.QTBFrictionReporter` logs the adapted spectra at the
adaptation cadence, and the readers here turn that log into a
fluctuation-dissipation residual, a convergence verdict, and two plots.

The `run_openmm_adqtb_*` drivers attach the reporter for you, so a default run
already writes `<prefix>_friction.log`.

```{literalinclude} ../../examples/rpmd.py
:pyobject: run_adqtb_verification
:language: python
```

## What the spectra are

{func}`~openmmnqe.adqtb.adqtb_friction_spectra` returns one array per particle
type, shaped `(n_segments, n_frequencies)`. The values are the **dimensionless
ratio** `gamma_r(w)/gamma`, which starts at `1.0` in every bin: a converged
spectrum is a *fixed* curve, not a flat one, and its distance from one is the
size of the zero-point-energy-leakage correction the bath is applying.

{func}`~openmmnqe.adqtb.adqtb_frequencies` rebuilds the grid the spectrum is
sampled on, `w_j = pi*j/(numFreq*dt)` in rad/ps, running from zero up to just
below the Nyquist frequency. Both `numFreq` and the step size come out of the
log itself, so no sidecar metadata is needed.

## Why the residual is exact

OpenMM adapts by projected gradient descent, one step per segment:

```
dfdt(w)     = sum over the 3*N_k components of type k of
                  gamma_r(w)*m*Cvv(w) - Re Cvf(w)
gamma_r(w) <- max(0, gamma_r(w) - eta_k*dfdt(w))
eta_k       = dt*A_k/(3*N_k*n)
```

That bracket is the adQTB fluctuation-dissipation residual: the velocity power
spectrum weighted by the current friction, less the cross spectrum between
velocity and the random force. Neither `Cvv` nor `Cvf` is exposed to Python —
but the *increment* is, because the update is a subtraction. So differencing
consecutive rows of the log recovers the residual exactly, up to the known
positive constant `eta_k`. That is what
{func}`~openmmnqe.adqtb.adqtb_fdt_residual` returns, and why it is a
measurement rather than a proxy.

It is returned as the friction increment rather than converted into physical
units. The accumulator it comes from carries the scaling of an unnormalised
internal FFT, and a number with an invented unit on it would be worse than no
number at all; `eta_k` is written out above for anyone who needs to convert.

## Reading the verdict

{func}`~openmmnqe.adqtb.adqtb_convergence` returns one
{class}`~openmmnqe.adqtb.QTBConvergence` per type.

The important thing it encodes: adaptation is a *stochastic* gradient step, so
the size of a single correction plateaus at a noise floor instead of decaying
to zero. **The step size is not the convergence signal — the drift is.** Over
the analysed window of `W` increments the net change is compared against the
`sqrt(W)*step_rms` a pure random walk of the same step size would accumulate:

| Field | Reading |
|---|---|
| `drift_ratio` | Below `tolerance` (default 2.0) means the spectrum is diffusing about a fixed curve, not marching towards one. This is the verdict. |
| `step_rms` | The noise floor itself. Useful for scale; not a convergence signal on its own. |
| `max_deviation` | Largest `abs(gamma_r - 1)`. How hard the bath is working. |
| `clamped_fraction` | Bins pinned at exactly zero by OpenMM's `max(0, ...)`. |

## The plots

{func}`~openmmnqe.adqtb.plot_adqtb_friction_spectra` draws one panel per type,
with a colour-graded line per sampled segment. Adaptation has converged when
the late lines lie on top of one another; a spectrum still fanning out has not
finished.

{func}`~openmmnqe.adqtb.plot_adqtb_fdt_residual` puts the residual spectrum
over its history. A converged bath scatters about zero in the upper panel,
while a band of one sign marks a frequency range still leaking or gaining
energy. The lower panel is the per-segment RMS residual, drawn as the raw
trace under a running mean — that mean is what says whether the noise floor is
flat or still falling.

Both crop the frequency axis to the part of the spectrum that actually moved,
since a friction spectrum runs to the Nyquist frequency but a system only has
vibrational density over the bottom of that range. Pass `max_frequency` to
override the crop.

A non-zero `clamped_fraction` is the one result that needs care. Those bins are
saturated: the residual inferred for them is a lower bound, not a measurement,
because the clamp swallowed part of the step. A few clamped bins at the top of
the spectrum are ordinary; a large fraction means the adaptation rate is too
high for the number of particles being averaged over.

The criterion is deliberately conservative rather than rigorous. A converged
spectrum is mean-reverting, not a free random walk, so its increments are
anticorrelated and its drift ratio settles well below one. It is a practical
check that adaptation has stopped going anywhere, not a statistical test with a
calibrated false-positive rate.

## Particle types

The bath adapts one spectrum per *type*, and a particle left untyped gets a
type to itself — so an untyped run adapts one independent spectrum per atom,
which is both far noisier at a given adaptation rate and unreadable once
logged. The drivers default to `particle_types="element"`, which groups by
element and splits a symbol when masses differ, so deuterium does not share
hydrogen's bath. OpenMM enforces that split as well: a `Context` refuses to
build when one particle type spans more than one mass.

To group by hand instead, pass a mapping of particle index to type index, or
call {func}`~openmmnqe.tools.set_adqtb_particle_types_by_element` yourself
before the `Context` is created.

## What the progress log does not contain

The adQTB drivers deliberately omit temperature, kinetic energy and total
energy from `<prefix>.log`. An adQTB thermostat drives the velocities to a
quantum distribution, and the standard estimators of temperature and pressure
assume a classical one; OpenMM's own documentation warns that they *"do not
produce correct results for an adQTB simulation"*. Step, time, potential
energy, speed and box volume are still reported, as they are for RPMD.
