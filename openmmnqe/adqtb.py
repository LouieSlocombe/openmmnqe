"""Verification tooling for adaptive quantum thermal bath (adQTB) runs.

An ``QTBIntegrator`` spends the whole of an equilibration adjusting one noise
spectrum per particle type, and the OpenMM documentation is explicit that a
run must be equilibrated "long enough for the friction coefficients to
converge" before production.  Nothing in a Context reveals whether that has
happened.  :class:`QTBFrictionReporter` logs the adapted spectra at segment
cadence, and the readers here turn that log into a fluctuation-dissipation
residual, a convergence verdict, and two plots.

The residual is exact rather than a proxy.  OpenMM adapts by projected
gradient descent, one step per segment::

    dfdt(w)     = sum over the 3*N_k components of type k of
                      gamma_r(w)*m*Cvv(w) - Re Cvf(w)
    gamma_r(w) <- max(0, gamma_r(w) - eta_k*dfdt(w))
    eta_k       = dt*A_k/(3*N_k*n)

where *n* is the number of steps in a segment, *A_k* the type's adaptation
rate and *N_k* its particle count.  The bracket is the adQTB
fluctuation-dissipation residual: the velocity power spectrum weighted by the
current friction, less the cross spectrum between velocity and the random
force.  Neither ``Cvv`` nor ``Cvf`` is exposed to Python, but the *increment*
is, because the update is a subtraction.  Differencing
``getAdaptedFriction`` across a segment therefore recovers the residual
exactly, up to the known positive constant ``eta_k``.

Two quantities are deliberately absent.  There is no velocity-spectrum
reporter: recomputing ``Cvv`` independently would mean pulling velocities
every step, and matching it to OpenMM's residual would mean guessing the
normalisation of an unnormalised internal FFT.  And the residual is reported
as the per-segment friction increment rather than converted into physical
units, because the accumulator it comes from carries OpenMM's own FFT
scaling; ``eta_k`` is documented above for anyone who wants to convert, but
this module will not invent units for it.

Note that ``getAdaptedFriction`` returns the *dimensionless* ratio
``gamma_r(w)/gamma``, one entry per frequency bin, starting at ``1.0``
everywhere.  A converged spectrum is a fixed curve, not a flat one.
"""
from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Any, NamedTuple, Self

import numpy as np
import numpy.typing as npt
import openmm.unit as unit
from openmm import app, openmm
from scipy import constants

from ._logs import _read_reporter_log
from ._validation import require_integer

# Friction columns are named ``Gamma_<label>_<bin>``.  The label may itself
# contain underscores and digits, so the bin is matched greedily from the
# right and the label takes whatever is left.
# A bin counts as adapted, for the purpose of cropping a plot's frequency
# axis, once it deviates from its baseline by this fraction of the largest
# deviation anywhere in the spectrum.
_ADAPTED_FRACTION = 0.05

_GAMMA_PREFIX = "Gamma_"
_GAMMA_COLUMN = re.compile(r"^Gamma_(?P<label>.+)_(?P<bin>\d{4,})$")

# Factors converting an angular frequency in rad/ps into the plotted unit,
# as ``(factor, axis label)``.
_FREQUENCY_UNITS: dict[str, tuple[float, str]] = {
    "rad/ps": (1.0, "rad/ps"),
    "1/ps": (1.0 / (2.0 * math.pi), "1/ps"),
    "cm^-1": (1.0e12 / (2.0 * math.pi * constants.c * 100.0), "cm$^{-1}$"),
}


class QTBConvergence(NamedTuple):
    """
    Verdict on whether one type's friction spectrum has stopped adapting.

    Attributes
    ----------
    step_rms : float
        Root-mean-square per-segment increment over the analysed window.
        This is the size of a single fluctuation-dissipation correction and
        it plateaus at a noise floor; it is not itself a convergence signal.
    drift_rms : float
        Root-mean-square net change across the whole analysed window.
    drift_ratio : float
        *drift_rms* divided by the ``sqrt(window)*step_rms`` a pure random
        walk of the same step size would accumulate. Small means the
        spectrum is diffusing about a fixed curve; large means it is still
        marching towards one.
    clamped_fraction : float
        Fraction of the analysed entries sitting at exactly zero, where
        OpenMM's ``max(0, ...)`` clamp is active. The residual inferred for
        those bins is only a lower bound.
    max_deviation : float
        Largest ``abs(gamma_r - 1)`` in the final spectrum, i.e. how much
        correction the bath is applying relative to a plain Langevin
        thermostat.
    converged : bool
        True when *drift_ratio* is within the requested tolerance.
    """

    step_rms: float
    drift_rms: float
    drift_ratio: float
    clamped_fraction: float
    max_deviation: float
    converged: bool


def _integrator_type_particles(integrator: Any) -> dict[int, list[int]]:
    """
    Group the integrator's particles by adQTB type.

    Parameters
    ----------
    integrator : openmm.QTBIntegrator
        The integrator whose particle types are read. Only particles passed
        to ``setParticleType`` appear in its map.

    Returns
    -------
    dict of int to list of int
        Particle indices for each type, each list sorted ascending and the
        mapping ordered by type.

    Raises
    ------
    TypeError
        If *integrator* does not provide ``getParticleTypes``.
    ValueError
        If no particle types have been assigned.
    """
    getter = getattr(integrator, "getParticleTypes", None)
    if getter is None:
        raise TypeError(
            "integrator must provide getParticleTypes(); expected an OpenMM "
            "QTBIntegrator."
        )
    assignments = dict(getter())
    if not assignments:
        raise ValueError(
            "integrator has no adQTB particle types assigned, so every "
            "particle adapts its own spectrum; call "
            "set_adqtb_particle_types_by_element before creating the Context"
        )
    grouped: dict[int, list[int]] = {}
    for particle, type_index in sorted(assignments.items()):
        grouped.setdefault(int(type_index), []).append(int(particle))
    return {key: grouped[key] for key in sorted(grouped)}


def _segment_steps(integrator: Any) -> int:
    """
    Return the number of integration steps in one adaptation segment.

    Parameters
    ----------
    integrator : openmm.QTBIntegrator
        The integrator to read the step size and segment length from.

    Returns
    -------
    int
        Steps per segment, the interval at which the friction is adapted.

    Raises
    ------
    ValueError
        If the segment length is not a whole number of steps.
    """
    step_size = integrator.getStepSize().value_in_unit(unit.picosecond)
    segment_length = integrator.getSegmentLength().value_in_unit(
        unit.picosecond
    )
    steps = int(round(segment_length / step_size))
    if steps < 1 or abs(steps * step_size - segment_length) > 1e-9:
        raise ValueError(
            "segment length must be a whole number of steps, but "
            f"{segment_length} ps is not a multiple of {step_size} ps"
        )
    return steps


def _num_frequencies(segment_steps: int) -> int:
    """
    Return the length of an adapted friction spectrum.

    Parameters
    ----------
    segment_steps : int
        Number of integration steps in one segment.

    Returns
    -------
    int
        ``(3*segment_steps + 1)//2``, matching OpenMM's noise buffer, which
        holds three segments so the coloured-noise filter has context either
        side of the one it uses.
    """
    return (3 * segment_steps + 1) // 2


def _frequency_grid(num_freq: int, step_size: float) -> npt.NDArray[np.float64]:
    """
    Build the angular-frequency grid the friction spectrum is sampled on.

    Parameters
    ----------
    num_freq : int
        Number of frequency bins.
    step_size : float
        Integration step size in picoseconds.

    Returns
    -------
    numpy.ndarray
        Angular frequencies ``pi*j/(num_freq*step_size)`` in rad/ps, one per
        bin, running from zero up to just below the Nyquist frequency.
    """
    return np.arange(num_freq, dtype=float) * math.pi / (num_freq * step_size)


class QTBFrictionReporter:
    """
    Log a ``QTBIntegrator``'s adapted friction spectra as they adapt.

    One block of tab-separated columns is written per particle type, named
    ``Gamma_<label>_<bin>``, holding the dimensionless ratio
    ``gamma_r(w)/gamma`` for each frequency bin. Because particles of the
    same type share a spectrum exactly, one representative particle per type
    is read rather than all of them.

    *reportInterval* must be a whole number of adaptation segments. OpenMM
    adapts exactly once per segment, so a report every ``k`` segments makes
    each row-to-row difference the sum of ``k`` fluctuation-dissipation
    corrections; an interval that straddles a segment boundary would mix
    rows containing different numbers of corrections and make the residual
    meaningless. Use :func:`track_adqtb_friction` to pick the interval from
    the integrator instead of computing it by hand.

    Parameters
    ----------
    file : str
        Path to write the friction log to.
    reportInterval : int
        Interval between reports, in steps. Must be a positive multiple of
        the integrator's segment length in steps.
    integrator : openmm.QTBIntegrator
        The integrator whose spectra are logged. It is read at construction
        for the frequency-bin count and the particle types, both of which
        are fixed before the Context exists, and again on every report.
    type_names : dict of int to str or None, optional
        Column labels for each particle type, e.g. ``{0: "H", 1: "O"}`` as
        returned inverted from
        :func:`~openmmnqe.tools.set_adqtb_particle_types_by_element`. Types
        without an entry fall back to ``T<type>``. Default is None.

    Raises
    ------
    TypeError
        If *reportInterval* is not an integer, or *integrator* is not a QTB
        integrator.
    ValueError
        If *reportInterval* is not a positive multiple of the segment
        length, no particle types have been assigned, or a label is unusable.

    Examples
    --------
    ::

        reporter = QTBFrictionReporter('adqtb_friction.log', 500, integrator)
        simulation.reporters.append(reporter)
    """

    def __init__(self, file: str | os.PathLike[str], reportInterval: int,
                 integrator: Any,
                 type_names: Mapping[int, str] | None = None) -> None:
        report_interval = require_integer(
            reportInterval,
            name="reportInterval",
            minimum=1,
        )
        type_particles = _integrator_type_particles(integrator)
        segment_steps = _segment_steps(integrator)
        if report_interval % segment_steps != 0:
            raise ValueError(
                "reportInterval must be a multiple of the segment length in "
                f"steps ({segment_steps}), so that each report spans a whole "
                "number of adaptations"
            )
        num_freq = _num_frequencies(segment_steps)

        labels: list[str] = []
        for type_index in type_particles:
            label = f"T{type_index}"
            if type_names is not None:
                label = str(type_names.get(type_index, label))
            if not label or any(char in label for char in "\t\n\r"):
                raise ValueError(
                    f"type_names entry for type {type_index} must be a "
                    "non-empty string without whitespace separators"
                )
            labels.append(label)
        if len(set(labels)) != len(labels):
            raise ValueError("type_names must give each type a distinct label")

        self._reportInterval = report_interval
        self._segment_steps = segment_steps
        self._num_freq = num_freq
        self._labels = labels
        self._particles = [
            particles[0] for particles in type_particles.values()
        ]

        columns = [
            f"{_GAMMA_PREFIX}{label}_{index:04d}"
            for label in labels
            for index in range(num_freq)
        ]
        header = "Step\tTime(ps)\t" + "\t".join(columns)
        self._out = open(file, "w")
        self._out.write(header + "\n")

    def describeNextReport(self, simulation: app.Simulation,
                           ) -> tuple[int, bool, bool, bool, bool]:
        """
        Report when the next report is due and what state it needs.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.

        Returns
        -------
        tuple
            ``(steps, positions, velocities, forces, energies)``. No state is
            requested: the spectra come from the integrator instead.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """
        Write one row of adapted friction spectra.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation being reported on. Its integrator supplies the
            spectra.
        state : openmm.State
            Used only for the simulation clock, so that a run resumed from a
            checkpoint carries on with the right time.

        Raises
        ------
        ValueError
            If the integrator returns a spectrum of unexpected length, which
            would mean OpenMM's frequency grid no longer matches the header
            written at construction.
        """
        integrator = simulation.integrator
        line = f"{simulation.currentStep}"
        line += f"\t{state.getTime().value_in_unit(unit.picosecond):.6f}"
        for label, particle in zip(self._labels, self._particles, strict=True):
            friction = np.asarray(
                integrator.getAdaptedFriction(particle),
                dtype=float,
            )
            if friction.shape != (self._num_freq,):
                raise ValueError(
                    f"integrator returned {friction.size} friction "
                    f"coefficients for type {label}, but the log was opened "
                    f"for {self._num_freq}"
                )
            line += "".join(f"\t{value:.6f}" for value in friction)
        self._out.write(line + "\n")
        self._out.flush()

    def close(self) -> None:
        """Close the output file, safely allowing repeated calls."""
        out = getattr(self, "_out", None)
        if out is not None and not out.closed:
            out.close()

    def __enter__(self) -> Self:
        """Return this reporter for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the output file when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort fallback for callers that did not close the reporter."""
        try:
            self.close()
        except Exception:
            pass


def track_adqtb_friction(simulation: app.Simulation,
                         file: str | os.PathLike[str],
                         *,
                         segments_per_report: int = 1,
                         type_names: Mapping[int, str] | None = None,
                         ) -> QTBFrictionReporter:
    """
    Attach a friction reporter whose interval matches the adaptation cadence.

    The reporting interval is the one thing about this reporter that is easy
    to get wrong, and it is fully determined by the integrator, so this
    helper reads it from there rather than asking the caller for it.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation to attach the reporter to. Its integrator must be a
        ``QTBIntegrator`` with particle types assigned.
    file : str or os.PathLike
        Path to write the friction log to.
    segments_per_report : int, optional
        Number of adaptation segments between reports. One row per segment
        gives the finest residual; raise it to thin a long run. Default
        is 1.
    type_names : dict of int to str or None, optional
        Column labels for each particle type. Default is None, giving
        ``T<type>``.

    Returns
    -------
    QTBFrictionReporter
        The reporter, already appended to ``simulation.reporters``.

    Examples
    --------
    ::

        track_adqtb_friction(simulation, 'adqtb_friction.log',
                             type_names={0: 'H', 1: 'O'})
    """
    segments = require_integer(
        segments_per_report,
        name="segments_per_report",
        minimum=1,
    )
    integrator = simulation.integrator
    interval = _segment_steps(integrator) * segments
    reporter = QTBFrictionReporter(file, interval, integrator, type_names)
    simulation.reporters.append(reporter)
    return reporter


def adqtb_friction(simulation: app.Simulation,
                   *,
                   type_names: Mapping[int, str] | None = None,
                   ) -> dict[str, npt.NDArray[np.float64]]:
    """
    Read the current adapted friction spectra off a live simulation.

    Useful for checking a run restored from a checkpoint, where the log that
    produced it may not be at hand.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation whose ``QTBIntegrator`` is read.
    type_names : dict of int to str or None, optional
        Labels for each particle type. Default is None, giving ``T<type>``.

    Returns
    -------
    dict of str to numpy.ndarray
        One spectrum per particle type, each of length ``(3*n + 1)//2`` for
        *n* steps per segment, holding the dimensionless ratio
        ``gamma_r(w)/gamma``.

    Examples
    --------
    ::

        spectra = adqtb_friction(simulation, type_names={0: 'H', 1: 'O'})
        print(max(abs(spectra['H'] - 1.0)))
    """
    integrator = simulation.integrator
    type_particles = _integrator_type_particles(integrator)
    spectra: dict[str, npt.NDArray[np.float64]] = {}
    for type_index, particles in type_particles.items():
        label = f"T{type_index}"
        if type_names is not None:
            label = str(type_names.get(type_index, label))
        spectra[label] = np.asarray(
            integrator.getAdaptedFriction(particles[0]),
            dtype=float,
        )
    return spectra


def _read_friction_log(file: str | os.PathLike[str],
                       ) -> tuple[list[str], np.ndarray]:
    """
    Read a tab-separated friction reporter log.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`QTBFrictionReporter`.

    Returns
    -------
    header : list of str
        Column names, the first of which is ``"Step"``.
    values : numpy.ndarray
        Row values, shaped ``(n_rows, len(header))``.

    Raises
    ------
    ValueError
        If the header is malformed or duplicated, the file holds no data
        rows, or a row does not match the header.
    """
    return _read_reporter_log(file, "friction log")


def _friction_blocks(header: list[str],
                     ) -> dict[str, list[int]]:
    """
    Locate each particle type's block of friction columns in a header.

    Parameters
    ----------
    header : list of str
        Column names read from a friction log.

    Returns
    -------
    dict of str to list of int
        Column indices for each type label, ordered by frequency bin.

    Raises
    ------
    ValueError
        If the log carries no friction columns, or a type's bins are not the
        contiguous range the reporter writes.
    """
    blocks: dict[str, dict[int, int]] = {}
    for index, name in enumerate(header):
        match = _GAMMA_COLUMN.match(name)
        if match is None:
            continue
        blocks.setdefault(match["label"], {})[int(match["bin"])] = index
    if not blocks:
        raise ValueError("friction log contains no friction columns")

    resolved: dict[str, list[int]] = {}
    for label, bins in blocks.items():
        expected = list(range(len(bins)))
        if sorted(bins) != expected:
            raise ValueError(
                f"friction log has gaps in the frequency bins for type "
                f"{label}"
            )
        resolved[label] = [bins[index] for index in expected]
    # Every type is sampled on the same frequency grid, so unequal widths
    # mean an edited log. Saying so here beats a shape mismatch surfacing
    # later from inside a plot.
    widths = {len(columns) for columns in resolved.values()}
    if len(widths) > 1:
        raise ValueError(
            "friction log gives its types different numbers of frequency "
            f"bins: {sorted(widths)}"
        )
    return resolved


def _selected_labels(available: Iterable[str],
                     types: str | Iterable[str] | None) -> list[str]:
    """
    Resolve a requested type selection against a log's type labels.

    Parameters
    ----------
    available : iterable of str
        Type labels present in the log.
    types : str or iterable of str or None
        Labels wanted. A bare string selects one; None selects all.

    Returns
    -------
    list of str
        The selected labels, in the order they appear in the log.

    Raises
    ------
    ValueError
        If a requested label is absent.
    """
    present = list(available)
    if types is None:
        return present
    requested = [types] if isinstance(types, str) else list(types)
    missing = [label for label in requested if label not in present]
    if missing:
        raise ValueError(
            f"unknown friction log type(s): {', '.join(map(str, missing))}"
        )
    return [label for label in present if label in requested]


def adqtb_friction_spectra(file: str | os.PathLike[str],
                           *,
                           types: str | Iterable[str] | None = None,
                           ) -> dict[str, npt.NDArray[np.float64]]:
    """
    Read the adapted friction spectra back out of a friction log.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`QTBFrictionReporter`.
    types : str or iterable of str or None, optional
        Type labels to return. If None, every type in the log is returned.
        Default is None.

    Returns
    -------
    dict of str to numpy.ndarray
        One array per particle type, shaped ``(n_segments, n_frequencies)``,
        holding the dimensionless ratio ``gamma_r(w)/gamma``.

    Raises
    ------
    ValueError
        If the log is malformed or a requested type is absent.

    Examples
    --------
    ::

        spectra = adqtb_friction_spectra('adqtb_friction.log')
        final = spectra['H'][-1]
    """
    header, values = _read_friction_log(file)
    blocks = _friction_blocks(header)
    labels = _selected_labels(blocks, types)
    return {
        label: np.ascontiguousarray(values[:, blocks[label]])
        for label in labels
    }


def adqtb_frequencies(file: str | os.PathLike[str],
                      ) -> npt.NDArray[np.float64]:
    """
    Rebuild the angular-frequency grid a friction log was sampled on.

    The grid is ``w_j = pi*j/(numFreq*dt)``, and both ``numFreq`` and the
    step size are recoverable from the log itself: the first from the number
    of friction columns per type, the second from the time and step columns.
    No sidecar metadata is needed.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`QTBFrictionReporter`.

    Returns
    -------
    numpy.ndarray
        Angular frequencies in rad/ps, one per frequency bin, running from
        zero up to just below the Nyquist frequency ``pi/dt``.

    Raises
    ------
    ValueError
        If the log is malformed or lacks the ``Time(ps)`` column.

    Examples
    --------
    ::

        omega = adqtb_frequencies('adqtb_friction.log')
        wavenumber = omega * 5.3088  # rad/ps to cm^-1
    """
    header, values = _read_friction_log(file)
    blocks = _friction_blocks(header)
    if "Time(ps)" not in header:
        raise ValueError("friction log must carry a Time(ps) column")
    steps = values[:, header.index("Step")]
    times = values[:, header.index("Time(ps)")]
    if steps[-1] <= 0 or not np.isfinite(times[-1]):
        raise ValueError("friction log must carry positive steps and times")
    step_size = float(times[-1] / steps[-1])
    if step_size <= 0:
        raise ValueError("friction log must carry a positive step size")
    num_freq = len(next(iter(blocks.values())))
    return _frequency_grid(num_freq, step_size)


def adqtb_fdt_residual(file: str | os.PathLike[str],
                       *,
                       types: str | Iterable[str] | None = None,
                       ) -> dict[str, npt.NDArray[np.float64]]:
    """
    Recover the fluctuation-dissipation residual from a friction log.

    OpenMM adapts by subtracting ``eta_k`` times the residual from the
    friction once per segment, so the per-segment increment returned here is
    exactly ``-eta_k*dfdt(w)`` with
    ``eta_k = dt*A_k/(3*N_k*n)`` for a type of *N_k* particles, adaptation
    rate *A_k*, and *n* steps per segment. The residual is reported in this
    form, as a friction increment, rather than converted: the accumulator it
    comes from carries the scaling of an unnormalised internal FFT, and this
    module will not invent physical units for it. What matters for
    verification is that it stops having a preferred sign, which
    :func:`adqtb_convergence` measures.

    Bins clamped at zero by OpenMM's ``max(0, ...)`` are a lower bound on
    the residual rather than a measurement of it;
    :attr:`QTBConvergence.clamped_fraction` counts them.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`QTBFrictionReporter`.
    types : str or iterable of str or None, optional
        Type labels to return. If None, every type in the log is returned.
        Default is None.

    Returns
    -------
    dict of str to numpy.ndarray
        One array per particle type, shaped
        ``(n_segments - 1, n_frequencies)``.

    Raises
    ------
    ValueError
        If the log is malformed, a requested type is absent, or the log holds
        fewer than two rows.

    Examples
    --------
    ::

        residual = adqtb_fdt_residual('adqtb_friction.log')
        print(abs(residual['H'][-1]).max())
    """
    spectra = adqtb_friction_spectra(file, types=types)
    for label, values in spectra.items():
        if values.shape[0] < 2:
            raise ValueError(
                f"friction log holds only {values.shape[0]} row(s) for type "
                f"{label}; a residual needs at least two"
            )
    return {
        label: np.diff(values, axis=0) for label, values in spectra.items()
    }


def adqtb_convergence(file: str | os.PathLike[str],
                      *,
                      types: str | Iterable[str] | None = None,
                      discard: float = 0.5,
                      tolerance: float = 2.0,
                      ) -> dict[str, QTBConvergence]:
    """
    Judge whether each type's friction spectrum has stopped adapting.

    Adaptation is a *stochastic* gradient step, so the size of a single
    correction plateaus at a noise floor instead of decaying to zero. The
    step size is therefore not the convergence signal; the drift is. Over
    the analysed window of *W* increments this compares the net change
    against the ``sqrt(W)*step_rms`` a pure random walk of the same step
    size would accumulate, and calls the spectrum converged when the ratio
    is within *tolerance*.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`QTBFrictionReporter`.
    discard : float, optional
        Leading fraction of the log to skip as still-equilibrating. Default
        is 0.5.
    types : str or iterable of str or None, optional
        Type labels to judge. If None, every type in the log is judged.
        Default is None.
    tolerance : float, optional
        Largest drift ratio still counted as converged. Default is 2.0.

    Returns
    -------
    dict of str to QTBConvergence
        One verdict per particle type.

    Raises
    ------
    ValueError
        If the log is malformed, a requested type is absent, *discard* is
        outside ``[0, 1)``, *tolerance* is not positive and finite, or the
        log holds fewer than three rows.

    Notes
    -----
    The criterion is deliberately conservative rather than rigorous. A
    converged spectrum is mean-reverting, not a free random walk, so its
    increments are anticorrelated and its drift ratio settles below one; a
    tolerance above one therefore leaves room for a short window without
    calling a genuinely marching spectrum converged. It is a practical
    check that adaptation has stopped going anywhere, not a statistical
    test with a calibrated false-positive rate.

    Examples
    --------
    ::

        verdicts = adqtb_convergence('adqtb_friction.log')
        assert all(v.converged for v in verdicts.values())
    """
    if not math.isfinite(discard) or not 0.0 <= discard < 1.0:
        raise ValueError("discard must be in the interval [0, 1)")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    spectra = adqtb_friction_spectra(file, types=types)
    verdicts: dict[str, QTBConvergence] = {}
    for label, values in spectra.items():
        n_segments = values.shape[0]
        if n_segments < 3:
            raise ValueError(
                f"friction log holds only {n_segments} row(s) for type "
                f"{label}; a convergence verdict needs at least three"
            )
        start = min(int(round(discard * n_segments)), n_segments - 2)
        window = values[start:]
        increments = np.diff(window, axis=0)
        n_increments = increments.shape[0]

        step_rms = float(np.sqrt(np.mean(np.square(increments))))
        drift = window[-1] - window[0]
        drift_rms = float(np.sqrt(np.mean(np.square(drift))))
        expected = math.sqrt(n_increments) * step_rms
        drift_ratio = drift_rms / expected if expected > 0.0 else 0.0
        verdicts[label] = QTBConvergence(
            step_rms=step_rms,
            drift_rms=drift_rms,
            drift_ratio=drift_ratio,
            clamped_fraction=float(np.mean(window == 0.0)),
            max_deviation=float(np.max(np.abs(values[-1] - 1.0))),
            converged=bool(drift_ratio <= tolerance),
        )
    return verdicts


def _frequency_axis(unit_name: str) -> tuple[float, str]:
    """
    Resolve a frequency unit name into a conversion factor and axis label.

    Parameters
    ----------
    unit_name : str
        One of ``"cm^-1"``, ``"1/ps"`` or ``"rad/ps"``.

    Returns
    -------
    factor : float
        Multiplier taking an angular frequency in rad/ps into *unit_name*.
    label : str
        Axis label for the unit, ready for matplotlib.

    Raises
    ------
    ValueError
        If *unit_name* is not one of the supported units.
    """
    try:
        return _FREQUENCY_UNITS[unit_name]
    except KeyError:
        choices = ", ".join(sorted(_FREQUENCY_UNITS))
        raise ValueError(
            f"frequency_unit must be one of: {choices}"
        ) from None


def _require_finite(values: np.ndarray, description: str) -> None:
    """
    Reject non-finite values before they reach matplotlib.

    Parameters
    ----------
    values : numpy.ndarray
        Values about to be plotted.
    description : str
        What the values are, used verbatim in the error message.

    Raises
    ------
    ValueError
        If any value is not finite.
    """
    if not np.all(np.isfinite(values)):
        raise ValueError(f"selected {description} values must be finite")


def _running_mean(values: np.ndarray, window: int | None = None,
                  ) -> npt.NDArray[np.float64]:
    """
    Smooth a per-segment series enough to read a trend off it.

    Parameters
    ----------
    values : numpy.ndarray
        One value per segment.
    window : int or None, optional
        Averaging window in segments. Default is None, giving a twentieth of
        the series, at least three points wide.

    Returns
    -------
    numpy.ndarray
        The smoothed series, the same length as *values*, averaged over the
        window that fits at each end.
    """
    if window is None:
        window = max(3, values.size // 20)
    window = min(window, values.size)
    padded = np.cumsum(np.insert(np.asarray(values, dtype=float), 0, 0.0))
    smoothed = np.empty(values.size, dtype=float)
    for index in range(values.size):
        low = max(0, index - window // 2)
        high = min(values.size, low + window)
        smoothed[index] = (padded[high] - padded[low]) / (high - low)
    return smoothed


def _adapted_frequency_limit(blocks: Iterable[np.ndarray], baseline: float,
                             frequencies: np.ndarray) -> float:
    """
    Find the frequency above which nothing is being adapted.

    A friction spectrum runs to the Nyquist frequency, but a system only has
    vibrational density over the bottom of that range, so the adapted part of
    the plot is usually a tenth of the axis and everything above it is a flat
    line at the baseline.  This picks a limit that keeps the part worth
    looking at, which is what the hand-rolled version of this plot always
    ended up doing with an explicit ``xlim``.

    Parameters
    ----------
    blocks : iterable of numpy.ndarray
        Arrays to scan, each shaped ``(n_rows, n_frequencies)``.
    baseline : float
        The value an unadapted bin holds: ``1.0`` for a friction spectrum,
        ``0.0`` for a residual.
    frequencies : numpy.ndarray
        The frequency grid, already in the plotted unit.

    Returns
    -------
    float
        Upper frequency limit, padded past the highest adapted bin, or the
        whole range when nothing stands out.
    """
    highest = 0
    for block in blocks:
        deviation = np.abs(np.asarray(block) - baseline).max(axis=0)
        peak = deviation.max()
        if peak <= 0.0:
            continue
        adapted = np.nonzero(deviation > _ADAPTED_FRACTION * peak)[0]
        if adapted.size:
            highest = max(highest, int(adapted[-1]))
    if highest == 0:
        return float(frequencies[-1])
    padded = min(int(highest * 1.2) + 1, frequencies.size - 1)
    return float(frequencies[padded])


def _segment_selection(n_segments: int,
                       segments: int | Iterable[int] | None) -> list[int]:
    """
    Choose which logged segments to draw.

    Parameters
    ----------
    n_segments : int
        Number of rows in the log.
    segments : int or iterable of int or None
        A count of evenly spaced segments to draw, explicit row indices, or
        None for the default of six.

    Returns
    -------
    list of int
        Row indices, ascending and without duplicates.

    Raises
    ------
    TypeError
        If a requested index is not an integer.
    ValueError
        If a count is not positive, or an index lies outside the log.
    """
    if segments is None or isinstance(segments, int):
        count = 6 if segments is None else require_integer(
            segments, name="segments", minimum=1,
        )
        count = min(count, n_segments)
        chosen = np.unique(
            np.linspace(0, n_segments - 1, count).round().astype(int)
        )
        return [int(index) for index in chosen]

    indices = [
        require_integer(index, name="segments entry") for index in segments
    ]
    for index in indices:
        if not -n_segments <= index < n_segments:
            raise ValueError(
                f"segment index {index} lies outside the {n_segments} rows "
                "of the friction log"
            )
    return sorted({index % n_segments for index in indices})


def plot_adqtb_friction_spectra(file: str | os.PathLike[str],
                                *,
                                types: str | Iterable[str] | None = None,
                                segments: int | Iterable[int] | None = None,
                                frequency_unit: str = "cm^-1",
                                max_frequency: float | None = None,
                                filename: str | os.PathLike[str] | None = None,
                                show: bool = False,
                                ) -> tuple[Any, tuple[Any, ...]]:
    """
    Plot how a run's adapted friction spectra evolved.

    One panel per particle type, with a colour-graded line per sampled
    segment running from the start of the log to its end. Adaptation has
    converged when the late lines lie on top of one another; a spectrum
    still fanning out has not finished. The flat line at
    ``gamma_r/gamma = 1`` is where the bath started, so distance from it is
    the size of the zero-point-energy-leakage correction being applied.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`QTBFrictionReporter`.
    types : str or iterable of str or None, optional
        Type labels to plot. If None, every type in the log is plotted.
        Default is None.
    segments : int or iterable of int or None, optional
        Number of evenly spaced segments to draw, or explicit row indices.
        Default is None, giving six.
    frequency_unit : {"cm^-1", "1/ps", "rad/ps"}, optional
        Unit for the frequency axis. Default is ``"cm^-1"``.
    max_frequency : float or None, optional
        Upper limit of the frequency axis, in *frequency_unit*. Default is
        None, which crops to the adapted part of the spectrum rather than
        showing the flat run up to the Nyquist frequency; pass the full
        range explicitly to see all of it.
    filename : str or os.PathLike or None, optional
        Path to save the figure to. If None, nothing is written. Default is
        None.
    show : bool, optional
        Call ``pyplot.show`` after drawing. Default is False.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The figure drawn.
    axes : tuple
        One axis per particle type, in log order.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If the log is malformed, a requested type or segment is absent, the
        frequency unit is unknown, or a selected value is not finite.

    Examples
    --------
    ::

        figure, axes = plot_adqtb_friction_spectra(
            'adqtb_friction.log', filename='friction.png',
        )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_adqtb_friction_spectra requires matplotlib; install the "
            "'plot' optional dependency"
        ) from exc

    factor, unit_label = _frequency_axis(frequency_unit)
    spectra = adqtb_friction_spectra(file, types=types)
    frequencies = adqtb_frequencies(file) * factor
    header, values = _read_friction_log(file)
    times = values[:, header.index("Time(ps)")]

    labels = list(spectra)
    n_segments = next(iter(spectra.values())).shape[0]
    chosen = _segment_selection(n_segments, segments)

    figure, raw_axes = plt.subplots(
        len(labels), 1,
        sharex=True,
        figsize=(6.4, 2.6 * len(labels) + 1.0),
        squeeze=False,
    )
    axes = tuple(raw_axes[:, 0])
    shades = plt.cm.viridis(np.linspace(0.0, 0.9, len(chosen)))

    for axis, label in zip(axes, labels, strict=True):
        block = spectra[label]
        _require_finite(block[chosen], "friction-log")
        axis.axhline(1.0, color="0.6", linewidth=0.8, linestyle=":")
        for shade, index in zip(shades, chosen, strict=True):
            axis.plot(
                frequencies,
                block[index],
                color=shade,
                linewidth=1.0,
                label=f"{times[index]:.1f} ps",
            )
        axis.set_ylabel(rf"$\gamma_r/\gamma$  ({label})")
        axis.legend(frameon=False, fontsize="small", ncol=2)
    axes[-1].set_xlabel(f"Frequency ({unit_label})")
    if max_frequency is None:
        max_frequency = _adapted_frequency_limit(
            (block[chosen] for block in spectra.values()), 1.0, frequencies,
        )
    axes[0].set_xlim(0.0, max_frequency)

    if filename is not None:
        figure.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_adqtb_fdt_residual(file: str | os.PathLike[str],
                            *,
                            types: str | Iterable[str] | None = None,
                            discard: float = 0.5,
                            frequency_unit: str = "cm^-1",
                            max_frequency: float | None = None,
                            filename: str | os.PathLike[str] | None = None,
                            show: bool = False,
                            ) -> tuple[Any, tuple[Any, ...]]:
    """
    Plot the fluctuation-dissipation residual and how it is settling.

    The upper panel is the mean residual over the analysed window as a
    function of frequency: a converged bath scatters about zero, while a
    band of one sign marks a frequency range still leaking or gaining
    energy. The lower panel is the per-segment root-mean-square residual
    against time, whose plateau is the noise floor that
    :func:`adqtb_convergence` measures drift against.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`QTBFrictionReporter`.
    types : str or iterable of str or None, optional
        Type labels to plot. If None, every type in the log is plotted.
        Default is None.
    discard : float, optional
        Leading fraction of the log to skip when averaging the residual
        spectrum. Default is 0.5.
    frequency_unit : {"cm^-1", "1/ps", "rad/ps"}, optional
        Unit for the frequency axis. Default is ``"cm^-1"``.
    max_frequency : float or None, optional
        Upper limit of the frequency axis, in *frequency_unit*. Default is
        None, which crops to the adapted part of the spectrum rather than
        showing the flat run up to the Nyquist frequency.
    filename : str or os.PathLike or None, optional
        Path to save the figure to. If None, nothing is written. Default is
        None.
    show : bool, optional
        Call ``pyplot.show`` after drawing. Default is False.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The figure drawn.
    axes : tuple
        The residual-spectrum axis and the residual-against-time axis.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If the log is malformed, a requested type is absent, *discard* is
        outside ``[0, 1)``, the frequency unit is unknown, or a selected
        value is not finite.

    Examples
    --------
    ::

        figure, axes = plot_adqtb_fdt_residual(
            'adqtb_friction.log', filename='residual.png',
        )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_adqtb_fdt_residual requires matplotlib; install the "
            "'plot' optional dependency"
        ) from exc

    if not math.isfinite(discard) or not 0.0 <= discard < 1.0:
        raise ValueError("discard must be in the interval [0, 1)")

    factor, unit_label = _frequency_axis(frequency_unit)
    residuals = adqtb_fdt_residual(file, types=types)
    frequencies = adqtb_frequencies(file) * factor
    header, values = _read_friction_log(file)
    times = values[1:, header.index("Time(ps)")]

    figure, raw_axes = plt.subplots(
        2, 1,
        figsize=(6.4, 6.0),
        gridspec_kw={"height_ratios": (1.3, 1), "hspace": 0.28},
    )
    spectrum_axis, history_axis = raw_axes

    for label, block in residuals.items():
        _require_finite(block, "friction-log residual")
        start = min(int(round(discard * block.shape[0])), block.shape[0] - 1)
        line, = spectrum_axis.plot(
            frequencies,
            block[start:].mean(axis=0),
            linewidth=1.0,
            label=label,
        )
        # The raw trace is a flat noise floor once converged, and dense
        # enough at one point per segment to hide whether it is still
        # falling, so a running mean goes on top of it to answer that.
        history = np.sqrt(np.mean(np.square(block), axis=1))
        history_axis.plot(times, history, linewidth=0.5, alpha=0.35,
                          color=line.get_color())
        history_axis.plot(times, _running_mean(history), linewidth=1.4,
                          color=line.get_color(), label=label)

    if max_frequency is None:
        max_frequency = _adapted_frequency_limit(
            residuals.values(), 0.0, frequencies,
        )
    spectrum_axis.set_xlim(0.0, max_frequency)
    spectrum_axis.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
    spectrum_axis.set_xlabel(f"Frequency ({unit_label})")
    spectrum_axis.set_ylabel(r"mean $\Delta\gamma_r/\gamma$ per segment")
    spectrum_axis.legend(frameon=False)

    history_axis.set_yscale("log")
    history_axis.set_xlabel("Time (ps)")
    history_axis.set_ylabel(r"RMS $\Delta\gamma_r/\gamma$")
    history_axis.legend(frameon=False)

    if filename is not None:
        figure.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return figure, (spectrum_axis, history_axis)
