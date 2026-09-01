"""Ring-polymer rate dynamics: recrossing trajectories and rate constants.

A free-energy surface alone gives a transition-state-theory rate, and TST
assumes no trajectory ever recrosses the dividing surface.  The
Bennett-Chandler factorization repairs that with dynamics::

    k = kappa * k_QTST

where ``k_QTST`` comes from the free-energy surface and the thermal flux
through the dividing surface, and the transmission coefficient ``kappa`` is
measured by launching swarms of short, unbiased, microcanonical trajectories
from configurations pinned at the surface and watching which side they
commit to.  On a ring polymer this is ring-polymer TST with an RPMD
recrossing correction (Craig and Manolopoulos, J. Chem. Phys. 122, 084106
(2005); 123, 034102 (2005)), the machinery behind RPMD rate studies of
proton transfer (Collepardo-Guevara et al., J. Chem. Phys. 128, 144502
(2008)).

The workflow this module sits in:

1. Metadynamics on the collective variable gives the free-energy surface and
   its barrier position ``s_dagger`` (:func:`reactiontools.summarise_fes`).
2. A PLUMED ``RESTRAINT`` pins a thermostatted RPMD run at ``s_dagger``, and
   :func:`openmmnqe.openmm.run_openmm_rpmd_prod` with ``snapshot_interval``
   harvests full-bead snapshots of it.
3. :func:`run_openmm_rpmd_recrossing` spawns unbiased thermostat-off
   trajectories from every snapshot, with fresh thermal bead momenta per
   child, and records the collective variable.
4. :func:`transmission_coefficient` turns the records into ``kappa(t)`` and
   :func:`rpmd_rate` assembles the rate constant from it and the surface.

Kinetic isotope effects follow by running steps 2-4 twice, once with
``deuterate=True``, and taking the ratio of the rates.

Two limitations are built in rather than hidden.  The harvest restraint acts
on each bead separately, so the harvested ensemble approximates the
centroid-constrained one; the ``s0_mean``/``s0_std`` diagnostics measure the
scatter, and ``max_s0_deviation`` can prune outliers.  And the reported
uncertainties cover the transmission-coefficient statistics only -- the
free-energy surface's own error is not propagated (in an isotope-effect
ratio much of it cancels).
"""
from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from numbers import Real
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import openmm.unit as unit
from openmm import app, openmm
from openmmml import MLPotential
from scipy import constants

from ._logs import _read_reporter_log
from ._validation import require_integer, require_positive_finite_scalar_in_unit
from .openmm import (
    PreparedSystem,
    _build_system,
    _load_rpmd_restart,
    _maybe_deuterate,
    _validate_rpmd_n_beads,
)
from .reporters import _rpmd_ring_energies
from .tools import (
    WorkflowDeuterationOption,
    _centroid_of_beads,
    _particle_masses_dalton,
    _sample_maxwell_boltzmann_velocities,
    step_rpmd,
)

# Namespaces the recrossing driver's random streams away from the stage seed
# streams in openmmnqe.openmm._derive_seeds: a user reusing one master seed
# for harvest and shooting must not correlate the shooting momenta with any
# stage stream. The value is arbitrary but fixed ("RPMD" in ASCII).
_RECROSSING_SEED_TAG = 0x52504D44

# Column order of a recrossing log. Flux0 and CV are in the collective
# variable's own units (per picosecond for Flux0), which this module does
# not know, so those two headers carry no unit suffix.
_RECROSSING_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Step",
    "Time(ps)",
    "Parent",
    "Child",
    "Flux0",
    "CV",
)
_RECROSSING_ENERGY_COLUMN = "E_ring(kJ/mol)"

_GAS_CONSTANT_KJ_PER_MOL_K = constants.R / 1000.0
_KCAL_PER_KJ = 4.184


class TransmissionResult(NamedTuple):
    """
    A transmission coefficient with its provenance and diagnostics.

    Produced by :func:`transmission_coefficient` from recrossing logs.

    Attributes
    ----------
    times_ps : numpy.ndarray
        Child-local times of the records, in picoseconds, starting at zero.
    kappa_t : numpy.ndarray
        Transmission coefficient at each time. It may transiently leave
        ``[0, 1]``; it is reported as computed, never clamped.
    kappa_stderr : numpy.ndarray
        Standard error of ``kappa_t`` from contiguous parent blocks.
    plateau : float
        Mean of ``kappa_t`` over the trailing plateau window, the value to
        multiply a TST rate by.
    plateau_stderr : float
        Standard error of the plateau from the scatter of the per-block
        plateau means.
    forward_flux : float
        Mean forward initial flux ``<max(s_dot(0), 0)>`` over the analysed
        children, in collective-variable units per picosecond. This is the
        thermal-flux factor :func:`rpmd_rate` needs.
    s0_mean : float
        Mean initial collective-variable value over the analysed children.
    s0_std : float
        Spread of the initial value, the harvest-restraint stiffness
        diagnostic: ``kappa_t`` near time zero reflects this width, not
        dynamics, so the plateau is read from the tail.
    n_parents : int
        Parent snapshots analysed, after any ``max_s0_deviation`` pruning.
    n_parents_excluded : int
        Parent snapshots pruned by ``max_s0_deviation``.
    n_children : int
        Total child trajectories analysed.
    """

    times_ps: npt.NDArray[np.float64]
    kappa_t: npt.NDArray[np.float64]
    kappa_stderr: npt.NDArray[np.float64]
    plateau: float
    plateau_stderr: float
    forward_flux: float
    s0_mean: float
    s0_std: float
    n_parents: int
    n_parents_excluded: int
    n_children: int


class RPMDRate(NamedTuple):
    """
    A Bennett-Chandler rate constant and its factors.

    Produced by :func:`rpmd_rate`. The rate is first order and per molecule,
    in reciprocal picoseconds; unit conversions beyond that are left to the
    caller.

    Attributes
    ----------
    rate : float
        ``transmission * qtst_rate``, in 1/ps.
    rate_stderr : float
        ``qtst_rate * transmission_stderr``, in 1/ps. Only the transmission
        statistics are propagated; the free-energy surface's own uncertainty
        is not.
    qtst_rate : float
        The quantum-TST rate ``forward_flux * p(s_dagger) / Z_reactant``,
        in 1/ps.
    transmission : float
        The plateau transmission coefficient used.
    transmission_stderr : float
        Its standard error.
    forward_flux : float
        The thermal forward flux used, in collective-variable units per
        picosecond.
    dividing_surface_probability : float
        ``exp(-beta F(s_dagger))`` divided by the reactant-window integral of
        ``exp(-beta F(s))``, in reciprocal collective-variable units.
    """

    rate: float
    rate_stderr: float
    qtst_rate: float
    transmission: float
    transmission_stderr: float
    forward_flux: float
    dividing_surface_probability: float


def _require_finite_number(value: object, *, name: str) -> float:
    """
    Return *value* as a finite float, rejecting booleans.

    Parameters
    ----------
    value : object
        Value to check.
    name : str
        Argument name for the error message.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If *value* is a bool, not a real number, or not finite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _validate_recrossing_seed(seed: int | None) -> int | None:
    """
    Validate a master seed the way the simulation stages do.

    Parameters
    ----------
    seed : int or None
        Master seed, or None for entropy-based seeding.

    Returns
    -------
    int or None
        The seed as a plain int, or None.

    Raises
    ------
    ValueError
        If *seed* is a bool, not an integer, or negative.
    """
    if seed is None:
        return None
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")
    return int(seed)


def _spawn_child_sequences(seed: int | None, n_parents: int, n_children: int,
                           ) -> list[list[np.random.SeedSequence]]:
    """
    Derive one independent random stream per (parent, child) pair.

    The root sequence carries :data:`_RECROSSING_SEED_TAG` alongside the
    master seed, so the streams are disjoint from every stage stream
    ``openmmnqe.openmm._derive_seeds`` hands out for the same master seed.
    Each stream depends only on ``(seed, parent, child)``, never on
    execution order, so a resumed run redraws exactly the momenta a fresh
    run would.

    Parameters
    ----------
    seed : int or None
        Master seed, or None for entropy-based seeding.
    n_parents : int
        Number of parent snapshots.
    n_children : int
        Number of children per parent.

    Returns
    -------
    list of list of numpy.random.SeedSequence
        ``result[parent][child]`` seeds that child's velocity draw.
    """
    if seed is None:
        root = np.random.SeedSequence()
    else:
        root = np.random.SeedSequence([int(seed), _RECROSSING_SEED_TAG])
    return [parent.spawn(n_children) for parent in root.spawn(n_parents)]


def _recrossing_columns(record_energy: bool) -> tuple[str, ...]:
    """
    Return the header columns of a recrossing log.

    Parameters
    ----------
    record_energy : bool
        Whether the ring-polymer Hamiltonian column is included.

    Returns
    -------
    tuple of str
        The column names, in file order.
    """
    if record_energy:
        return _RECROSSING_REQUIRED_COLUMNS + (_RECROSSING_ENERGY_COLUMN,)
    return _RECROSSING_REQUIRED_COLUMNS


def _recrossing_log_complete(path: str, columns: Sequence[str],
                             expected_rows: int, n_steps: int) -> bool:
    """
    Decide whether an existing recrossing log holds a finished parent.

    Parameters
    ----------
    path : str
        Log file to inspect.
    columns : sequence of str
        Header the log must carry, in order.
    expected_rows : int
        Rows a complete parent writes.
    n_steps : int
        Final Step value of a complete child.

    Returns
    -------
    bool
        True only when the file parses, matches the header, and holds every
        row through the final step; anything else means redo the parent.
    """
    if not os.path.exists(path):
        return False
    try:
        header, values = _read_reporter_log(path, "recrossing log")
    except (ValueError, OSError):
        return False
    if header != list(columns):
        return False
    if len(values) != expected_rows:
        return False
    return int(values[:, 0].max()) == n_steps


def _evaluate_cv(cv: Callable[[npt.NDArray[np.float64]], float],
                 positions_nm: npt.NDArray[np.float64],
                 parent: int, child: int, step: int) -> float:
    """
    Evaluate the collective variable, insisting on a finite scalar.

    Parameters
    ----------
    cv : callable
        Collective-variable function of ``(n_atoms, 3)`` positions in
        nanometres.
    positions_nm : numpy.ndarray
        Positions to evaluate at.
    parent : int
        Parent index, named in error messages.
    child : int
        Child index, named in error messages.
    step : int
        Step count, named in error messages.

    Returns
    -------
    float
        The collective-variable value.

    Raises
    ------
    ValueError
        If the callable returns something that is not a finite scalar.
    """
    raw = cv(positions_nm)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cv returned a non-scalar ({raw!r}) at parent {parent}, "
            f"child {child}, step {step}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"cv returned a non-finite value at parent {parent}, "
            f"child {child}, step {step}"
        )
    return value


def _bead_state_arrays(simulation: app.Simulation, n_beads: int,
                       periodic: bool, with_velocities: bool,
                       ) -> tuple[npt.NDArray[np.float64],
                                  npt.NDArray[np.float64] | None,
                                  npt.NDArray[np.float64] | None]:
    """
    Pull every bead's positions, and optionally velocities, as bare arrays.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``.
    n_beads : int
        Number of ring-polymer beads.
    periodic : bool
        Whether the System is periodic; wrapped coordinates and the box are
        returned when it is.
    with_velocities : bool
        Whether bead velocities are pulled too.

    Returns
    -------
    positions : numpy.ndarray
        Bead positions in nanometres, shaped ``(n_beads, n_atoms, 3)``.
    velocities : numpy.ndarray or None
        Bead velocities in nm/ps, same shape, or None when not requested.
    box : numpy.ndarray or None
        Periodic box vectors in nanometres, shaped ``(3, 3)``, or None for
        a nonperiodic System.
    """
    integrator = simulation.integrator
    positions: npt.NDArray[np.float64] | None = None
    velocities: npt.NDArray[np.float64] | None = None
    box: npt.NDArray[np.float64] | None = None
    for bead in range(n_beads):
        state = integrator.getState(
            bead,
            getPositions=True,
            getVelocities=with_velocities,
            enforcePeriodicBox=periodic,
        )
        bead_positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer
        )
        if positions is None:
            positions = np.empty((n_beads,) + bead_positions.shape)
            if with_velocities:
                velocities = np.empty_like(positions)
            if periodic:
                box = np.asarray(
                    state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                        unit.nanometer
                    ),
                    dtype=float,
                )
        positions[bead] = bead_positions
        if with_velocities:
            assert velocities is not None
            velocities[bead] = state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )
    assert positions is not None
    return positions, velocities, box


def _cv_of_beads(cv: Callable[[npt.NDArray[np.float64]], float],
                 cv_mode: str,
                 positions_nm: npt.NDArray[np.float64],
                 box_nm: npt.NDArray[np.float64] | None,
                 parent: int, child: int, step: int) -> float:
    """
    Evaluate the collective variable of a ring-polymer configuration.

    Parameters
    ----------
    cv : callable
        Collective-variable function of ``(n_atoms, 3)`` positions in
        nanometres.
    cv_mode : str
        ``"centroid"`` evaluates at the bead centroid; ``"bead-mean"``
        averages the value over the beads.
    positions_nm : numpy.ndarray
        Bead positions, shaped ``(n_beads, n_atoms, 3)``.
    box_nm : numpy.ndarray or None
        Periodic box vectors for centroid unwrapping, or None.
    parent : int
        Parent index, named in error messages.
    child : int
        Child index, named in error messages.
    step : int
        Step count, named in error messages.

    Returns
    -------
    float
        The collective-variable value.
    """
    if cv_mode == "centroid":
        centroid = _centroid_of_beads(positions_nm, box_nm)
        return _evaluate_cv(cv, centroid, parent, child, step)
    values = [
        _evaluate_cv(cv, positions_nm[bead], parent, child, step)
        for bead in range(positions_nm.shape[0])
    ]
    return float(np.mean(values))


def _initial_cv_and_flux(cv: Callable[[npt.NDArray[np.float64]], float],
                         cv_mode: str,
                         positions_nm: npt.NDArray[np.float64],
                         velocities_nm_ps: npt.NDArray[np.float64],
                         box_nm: npt.NDArray[np.float64] | None,
                         epsilon_ps: float,
                         parent: int, child: int) -> tuple[float, float]:
    """
    Evaluate the initial collective variable and its time derivative.

    The derivative is a central finite difference of the collective variable
    displaced along the velocities, ``[s(x + eps*v) - s(x - eps*v)] /
    (2*eps)``, which needs no analytic gradient and works for any callable.

    Parameters
    ----------
    cv : callable
        Collective-variable function of ``(n_atoms, 3)`` positions in
        nanometres.
    cv_mode : str
        ``"centroid"`` differentiates along the centroid velocity;
        ``"bead-mean"`` averages per-bead derivatives.
    positions_nm : numpy.ndarray
        Bead positions, shaped ``(n_beads, n_atoms, 3)``.
    velocities_nm_ps : numpy.ndarray
        Bead velocities in nm/ps, same shape.
    box_nm : numpy.ndarray or None
        Periodic box vectors for centroid unwrapping, or None.
    epsilon_ps : float
        Finite-difference time offset in picoseconds.
    parent : int
        Parent index, named in error messages.
    child : int
        Child index, named in error messages.

    Returns
    -------
    s0 : float
        Initial collective-variable value.
    flux0 : float
        Initial time derivative, in collective-variable units per
        picosecond.
    """
    if cv_mode == "centroid":
        centroid = _centroid_of_beads(positions_nm, box_nm)
        velocity = velocities_nm_ps.mean(axis=0)
        s0 = _evaluate_cv(cv, centroid, parent, child, 0)
        forward = _evaluate_cv(
            cv, centroid + epsilon_ps * velocity, parent, child, 0
        )
        backward = _evaluate_cv(
            cv, centroid - epsilon_ps * velocity, parent, child, 0
        )
        return s0, (forward - backward) / (2.0 * epsilon_ps)

    values = []
    derivatives = []
    for bead in range(positions_nm.shape[0]):
        values.append(
            _evaluate_cv(cv, positions_nm[bead], parent, child, 0)
        )
        forward = _evaluate_cv(
            cv,
            positions_nm[bead] + epsilon_ps * velocities_nm_ps[bead],
            parent, child, 0,
        )
        backward = _evaluate_cv(
            cv,
            positions_nm[bead] - epsilon_ps * velocities_nm_ps[bead],
            parent, child, 0,
        )
        derivatives.append((forward - backward) / (2.0 * epsilon_ps))
    return float(np.mean(values)), float(np.mean(derivatives))


def run_openmm_rpmd_recrossing(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        snapshot_files: Sequence[str | os.PathLike[str]],
        cv: Callable[[npt.NDArray[np.float64]], float],
        output_prefix: str = 'rpmd_recrossing',
        n_beads: int = 32,
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        time_step: unit.Quantity = 0.5 * unit.femtoseconds,
        n_children: int = 10,
        n_steps: int = 200,
        record_interval: int = 5,
        cv_mode: Literal["centroid", "bead-mean"] = "centroid",
        flux_epsilon: unit.Quantity | float = 1.0e-4 * unit.picosecond,
        record_energy: bool = False,
        resume: bool = False,
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        seed: int | None = None,
) -> None:
    """
    Shoot unbiased microcanonical RPMD trajectories from saved snapshots.

    For every full-bead snapshot -- typically harvested by
    :func:`openmmnqe.openmm.run_openmm_rpmd_prod` with ``snapshot_interval``
    from a run restrained at the dividing surface -- this launches *n_children*
    short thermostat-off ring-polymer trajectories, each starting from the
    snapshot's bead positions with fresh thermal bead momenta, and records
    the collective variable along each.  One tab-separated log per parent,
    ``<output_prefix>_recrossing_<parent>.log``, holds every child's time
    series; feed the logs to :func:`transmission_coefficient`.

    The trajectories run on the plain physical system: no PLUMED bias, no
    barostat, no thermostat.  A snapshot saved from a biased harvest loads
    cleanly because the restart archive checks masses, topology and
    temperature but deliberately not forces.  The dividing-surface value is
    likewise not an argument here -- shooting only records the collective
    variable, so one campaign can be re-analysed at different surfaces.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions. Positions are
        only used to build the System; every trajectory starts from a
        snapshot.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    snapshot_files : sequence of str or os.PathLike
        Full-bead RPMD restart archives, one per parent. Their order defines
        the parent numbering, and analysis blocks parents in that order for
        its error bars, so pass them in harvest order (sort the glob).
    cv : callable
        The collective variable: a function of an ``(n_atoms, 3)`` positions
        array in nanometres returning a number, in whatever units the
        free-energy surface uses. It must match the coordinate the surface
        and the harvest restraint were built on. For a periodic System the
        positions are wrapped molecule by molecule, so a collective variable
        spanning molecules must handle periodicity itself.
    output_prefix : str, optional
        Prefix for the per-parent logs. Default is ``'rpmd_recrossing'``.
    n_beads : int, optional
        Number of ring-polymer beads; must match the snapshots. Default
        is 32.
    temperature : openmm.unit.Quantity, optional
        Temperature the ring-polymer springs and the resampled momenta are
        built at; it must equal the harvest temperature, and the restart
        loader enforces that. Default is 300.0 K.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Microcanonical dynamics has no thermostat to
        absorb integration error, so this defaults smaller than the
        production stages'. Default is 0.5 fs.
    n_children : int, optional
        Trajectories launched per snapshot, each with independent thermal
        momenta. Default is 10.
    n_steps : int, optional
        Steps per child; with *record_interval* it must give whole records.
        The span ``n_steps * time_step`` must comfortably cover the plateau
        time of the transmission coefficient. Default is 200.
    record_interval : int, optional
        Steps between recorded collective-variable values. Default is 5.
    cv_mode : {"centroid", "bead-mean"}, optional
        ``"centroid"`` evaluates the collective variable at the bead
        centroid, the conventional RPMD reaction coordinate;
        ``"bead-mean"`` averages the value over beads, matching how a
        per-bead PLUMED bias acts. The two agree for a near-linear
        collective variable. Default is ``"centroid"``.
    flux_epsilon : openmm.unit.Quantity or float, optional
        Time offset of the central finite difference that measures the
        initial flux. A bare number is read as picoseconds. Default is
        1.0e-4 ps.
    record_energy : bool, optional
        If True, also record the ring-polymer Hamiltonian each record as an
        ``E_ring(kJ/mol)`` column -- a per-child conservation check costing
        one full energy evaluation per record. Default is False.
    resume : bool, optional
        If True, skip parents whose log is already complete and redo any
        partial one; with a fixed *seed* the result is identical to an
        uninterrupted run. Default is False.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before shooting. The snapshots must
        come from an equally deuterated harvest -- the restart loader's mass
        check enforces it. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is
        None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is
        None.
    seed : int or None, optional
        Master random seed. Each child's momenta come from an independent
        stream derived from ``(seed, parent, child)``, disjoint from the
        simulation stages' streams for the same master seed. If None, every
        draw is non-deterministic. Default is None.

    Raises
    ------
    FileNotFoundError
        If a snapshot file does not exist.
    TypeError
        If *cv* is not callable.
    ValueError
        If *snapshot_files* is empty, a count is not a positive integer,
        *n_steps* is not a multiple of *record_interval*, *cv_mode* is
        unknown, *seed* is negative or not an integer, a snapshot disagrees
        with the System (beads, masses, topology, temperature,
        periodicity), or the collective variable returns a non-finite or
        non-scalar value.

    Notes
    -----
    One OpenMM Context serves the whole campaign: each child resets the
    loaded snapshot state, its clock, and its momenta rather than paying for
    a fresh Context.  When *forcefield* is a
    :class:`~openmmnqe.openmm.PreparedSystem`, nothing here mutates the held
    System, so one instance can serve harvest and shooting -- but only if
    the harvest stage's own mutating options are respected (see
    :class:`~openmmnqe.openmm.PreparedSystem`).
    """
    snapshot_paths = [os.fspath(path) for path in snapshot_files]
    if not snapshot_paths:
        raise ValueError("snapshot_files must not be empty")
    for path in snapshot_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"snapshot file {path!r} does not exist; recrossing needs "
                "the harvest run's full-bead archives"
            )

    n_beads = _validate_rpmd_n_beads(n_beads)
    n_children = require_integer(n_children, name="n_children", minimum=1)
    n_steps = require_integer(n_steps, name="n_steps", minimum=1)
    record_interval = require_integer(
        record_interval,
        name="record_interval",
        minimum=1,
    )
    if n_steps % record_interval != 0:
        raise ValueError(
            f"n_steps={n_steps} must be a multiple of "
            f"record_interval={record_interval}, or the final records would "
            "silently truncate each trajectory"
        )
    require_positive_finite_scalar_in_unit(
        temperature,
        unit.kelvin,
        name="temperature",
    )
    time_step_ps = require_positive_finite_scalar_in_unit(
        time_step,
        unit.picosecond,
        name="time_step",
    )
    epsilon_ps = require_positive_finite_scalar_in_unit(
        flux_epsilon,
        unit.picosecond,
        name="flux_epsilon",
    )
    if not callable(cv):
        raise TypeError("cv must be a callable of an (n_atoms, 3) array")
    if cv_mode not in ("centroid", "bead-mean"):
        raise ValueError(
            f"unknown cv_mode {cv_mode!r}; use 'centroid' or 'bead-mean'"
        )
    seed = _validate_recrossing_seed(seed)

    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)
    _maybe_deuterate(modeller, system, deuterate, deuterate_option)
    integrator = openmm.RPMDIntegrator(
        n_beads,
        temperature,
        1.0 / unit.picosecond,
        time_step,
    )
    integrator.setApplyThermostat(False)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    periodic = system.usesPeriodicBoundaryConditions()
    particle_masses = _particle_masses_dalton(system)

    child_sequences = _spawn_child_sequences(
        seed,
        len(snapshot_paths),
        n_children,
    )
    n_records = n_steps // record_interval
    expected_rows = n_children * (n_records + 1)
    columns = _recrossing_columns(record_energy)

    for parent_index, snapshot in enumerate(snapshot_paths):
        log_path = f'{output_prefix}_recrossing_{parent_index:05d}.log'
        if resume and _recrossing_log_complete(
            log_path, columns, expected_rows, n_steps
        ):
            print(
                f"Recrossing parent {parent_index + 1}/{len(snapshot_paths)}: "
                f"{log_path} is complete, skipping",
                flush=True,
            )
            continue

        print(
            f"Recrossing parent {parent_index + 1}/{len(snapshot_paths)}: "
            f"{n_children} children x {n_steps} steps from {snapshot}",
            flush=True,
        )
        with open(log_path, "w") as log:
            log.write("\t".join(columns) + "\n")
            for child_index in range(n_children):
                rng = np.random.default_rng(
                    child_sequences[parent_index][child_index]
                )
                _load_rpmd_restart(simulation, snapshot, n_beads)
                # The restart carries the harvest run's clock; each child
                # starts its own at zero.
                simulation.context.setTime(0.0 * unit.picosecond)
                simulation.currentStep = 0
                velocities = _sample_maxwell_boltzmann_velocities(
                    system,
                    temperature,
                    n_beads,
                    rng,
                )
                for bead in range(n_beads):
                    integrator.setVelocities(bead, velocities[bead])

                positions_nm, velocities_nm_ps, box_nm = _bead_state_arrays(
                    simulation, n_beads, periodic, with_velocities=True
                )
                assert velocities_nm_ps is not None
                s_value, flux0 = _initial_cv_and_flux(
                    cv, cv_mode, positions_nm, velocities_nm_ps, box_nm,
                    epsilon_ps, parent_index, child_index,
                )

                for record in range(n_records + 1):
                    step = record * record_interval
                    if record > 0:
                        step_rpmd(simulation, record_interval)
                        positions_nm, _, box_nm = _bead_state_arrays(
                            simulation, n_beads, periodic,
                            with_velocities=False,
                        )
                        s_value = _cv_of_beads(
                            cv, cv_mode, positions_nm, box_nm,
                            parent_index, child_index, step,
                        )
                    fields = [
                        str(step),
                        f"{step * time_step_ps:.17g}",
                        str(parent_index),
                        str(child_index),
                        f"{flux0:.17g}",
                        f"{s_value:.17g}",
                    ]
                    if record_energy:
                        # Not RPMDIntegrator.getTotalEnergy(): that method
                        # deadlocks on a mixed ML/MM System on CUDA or
                        # OpenCL, which is exactly what this driver runs.
                        energy, _ = _rpmd_ring_energies(
                            integrator, particle_masses
                        )
                        fields.append(f"{energy:.17g}")
                    log.write("\t".join(fields) + "\n")
                log.flush()


class _ParentRecord(NamedTuple):
    """
    One parent snapshot's children, as analysis-ready arrays.

    Attributes
    ----------
    flux0 : numpy.ndarray
        Initial flux of each child, shaped ``(n_children,)``.
    cv_values : numpy.ndarray
        Collective-variable time series, shaped ``(n_children, n_times)``.
    """

    flux0: npt.NDArray[np.float64]
    cv_values: npt.NDArray[np.float64]


def _read_recrossing_logs(
    files: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
) -> tuple[npt.NDArray[np.float64], list[_ParentRecord]]:
    """
    Read recrossing logs back into per-parent arrays on one time grid.

    Parameters
    ----------
    files : str or os.PathLike or iterable of str
        Recrossing logs written by :func:`run_openmm_rpmd_recrossing`.
        Their order defines the parent order the error analysis blocks by.

    Returns
    -------
    times_ps : numpy.ndarray
        The shared record times in picoseconds, starting at zero.
    parents : list of _ParentRecord
        One record per parent, in first-appearance order across the given
        files.

    Raises
    ------
    ValueError
        If no file is given, a log is malformed or lacks a required column,
        values are non-finite, children disagree on the record grid, or a
        child's initial flux is not constant across its rows.
    """
    if isinstance(files, (str, os.PathLike)):
        file_list = [os.fspath(files)]
    else:
        file_list = [os.fspath(file) for file in files]
    if not file_list:
        raise ValueError("no recrossing logs given")

    chunks = []
    for path in file_list:
        header, values = _read_reporter_log(path, "recrossing log")
        missing = [
            name for name in _RECROSSING_REQUIRED_COLUMNS
            if name not in header
        ]
        if missing:
            raise ValueError(
                f"recrossing log {path!s} lacks column(s): "
                f"{', '.join(missing)}"
            )
        selected = [
            header.index(name) for name in _RECROSSING_REQUIRED_COLUMNS
        ]
        chunks.append(values[:, selected])
    rows = np.vstack(chunks)
    if not np.isfinite(rows).all():
        raise ValueError("recrossing logs contain non-finite values")

    steps = rows[:, 0]
    times_column = rows[:, 1]
    parents_column = rows[:, 2].astype(int)
    children_column = rows[:, 3].astype(int)

    grid = np.unique(steps)
    if len(grid) < 2:
        raise ValueError(
            "recrossing logs hold a single record per child; nothing to "
            "analyse"
        )

    # The time must be a function of the step alone, or two campaigns with
    # different time steps were mixed into one analysis.
    grid_times = np.empty_like(grid)
    for index, step in enumerate(grid):
        grid_times[index] = times_column[steps == step][0]
    expected_times = grid_times[np.searchsorted(grid, steps)]
    if not np.allclose(times_column, expected_times, rtol=1.0e-9, atol=0.0):
        raise ValueError(
            "recrossing logs disagree on record times; do not mix "
            "campaigns with different time steps"
        )

    parent_order: list[int] = []
    for parent in parents_column:
        if int(parent) not in parent_order:
            parent_order.append(int(parent))

    records: list[_ParentRecord] = []
    for parent in parent_order:
        parent_mask = parents_column == parent
        child_ids = sorted(set(children_column[parent_mask].tolist()))
        flux0_list = []
        series_list = []
        for child in child_ids:
            child_rows = rows[parent_mask & (children_column == child)]
            child_rows = child_rows[np.argsort(child_rows[:, 0])]
            if (
                len(child_rows) != len(grid)
                or not np.array_equal(child_rows[:, 0], grid)
            ):
                raise ValueError(
                    "recrossing children do not share one record grid; "
                    "analyse runs with one record_interval and n_steps at "
                    "a time"
                )
            flux0_values = child_rows[:, 4]
            if not np.all(flux0_values == flux0_values[0]):
                raise ValueError(
                    f"Flux0 varies within parent {parent}, child {child}; "
                    "the log is corrupt"
                )
            flux0_list.append(float(flux0_values[0]))
            series_list.append(child_rows[:, 5])
        records.append(_ParentRecord(
            flux0=np.asarray(flux0_list),
            cv_values=np.vstack(series_list),
        ))
    return grid_times, records


def transmission_coefficient(
    files: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    s_dagger: float,
    blocks: int = 5,
    plateau_fraction: float = 0.25,
    max_s0_deviation: float | None = None,
) -> TransmissionResult:
    """
    Compute the transmission coefficient kappa(t) from recrossing logs.

    The estimator is the flux-side correlation ratio

    .. math::

        \\kappa(t) = \\frac{\\langle \\dot{s}(0)\\,
        \\theta[s(t) - s^\\ddagger] \\rangle}
        {\\langle \\dot{s}(0)\\, \\theta[\\dot{s}(0)] \\rangle}

    over every child trajectory: each child is weighted by its signed
    initial flux, counted when it sits on the product side, and normalized
    by the mean forward flux.  A child launched backwards that ends on the
    product side contributes negative weight, exactly the recrossing this
    factor exists to measure.

    With snapshots harvested by a finite-width restraint, kappa near time
    zero reflects the restraint width rather than dynamics -- it rises from
    around zero as trajectories clear the initial scatter, then relaxes to
    the plateau that multiplies the TST rate.  Read the plateau, never the
    early rise, and check ``s0_std`` against the barrier width.

    Parameters
    ----------
    files : str or os.PathLike or iterable of str
        Recrossing logs written by :func:`run_openmm_rpmd_recrossing`, in
        harvest order; the error analysis blocks parents contiguously in
        this order, which is what absorbs their correlation along the
        harvest trajectory.
    s_dagger : float
        Dividing-surface value of the collective variable, in the same
        units the logs' ``CV`` column carries.
    blocks : int, optional
        Contiguous parent blocks the standard errors are estimated from.
        Default is 5.
    plateau_fraction : float, optional
        Trailing fraction of the time span averaged into the plateau, in
        ``(0, 1]``. Default is 0.25.
    max_s0_deviation : float or None, optional
        If set, parents whose initial collective-variable value sits
        further than this from *s_dagger* are excluded -- a pragmatic prune
        of harvest outliers, at the price of truncating the sampled
        ensemble. Default is None, which keeps every parent.

    Returns
    -------
    TransmissionResult
        The kappa(t) curve, its plateau, the forward flux, and diagnostics.

    Raises
    ------
    ValueError
        If the logs are malformed or inconsistent, *s_dagger* or a control
        is out of domain, fewer parents survive than *blocks*, a block or
        the whole set carries no forward flux, or the plateau window holds
        fewer than two records.

    Examples
    --------
    Analyse a finished campaign at the barrier position of the surface::

        from openmmnqe import transmission_coefficient

        result = transmission_coefficient(
            sorted(glob.glob("rpmd_recrossing_recrossing_*.log")),
            s_dagger=0.12,
        )
        print(result.plateau, "+/-", result.plateau_stderr)
    """
    s_dagger = _require_finite_number(s_dagger, name="s_dagger")
    blocks = require_integer(blocks, name="blocks", minimum=2)
    plateau_fraction = _require_finite_number(
        plateau_fraction,
        name="plateau_fraction",
    )
    if not 0.0 < plateau_fraction <= 1.0:
        raise ValueError("plateau_fraction must be in (0, 1]")
    if max_s0_deviation is not None:
        max_s0_deviation = _require_finite_number(
            max_s0_deviation,
            name="max_s0_deviation",
        )
        if max_s0_deviation <= 0.0:
            raise ValueError("max_s0_deviation must be positive")

    times, parents = _read_recrossing_logs(files)

    kept = parents
    n_excluded = 0
    if max_s0_deviation is not None:
        kept = [
            parent for parent in parents
            if abs(float(parent.cv_values[:, 0].mean()) - s_dagger)
            <= max_s0_deviation
        ]
        n_excluded = len(parents) - len(kept)
        if not kept:
            raise ValueError(
                f"max_s0_deviation={max_s0_deviation} excludes every "
                "parent; the harvest restraint centre and s_dagger "
                "disagree"
            )
    if len(kept) < blocks:
        raise ValueError(
            f"{len(kept)} parents survive but {blocks} blocks were "
            "requested; harvest more snapshots or lower blocks"
        )

    # Per parent: signed-flux side sums and the forward-flux normalization.
    numerators = np.stack([
        (parent.flux0[:, np.newaxis]
         * (parent.cv_values > s_dagger)).sum(axis=0)
        for parent in kept
    ])
    denominators = np.asarray([
        np.maximum(parent.flux0, 0.0).sum() for parent in kept
    ])
    total_denominator = float(denominators.sum())
    if total_denominator <= 0.0:
        raise ValueError(
            "no forward-moving children; check the cv callable and "
            "flux_epsilon"
        )
    kappa_t = numerators.sum(axis=0) / total_denominator

    block_slices = np.array_split(np.arange(len(kept)), blocks)
    block_kappas = []
    for block in block_slices:
        block_denominator = float(denominators[block].sum())
        if block_denominator <= 0.0:
            raise ValueError(
                "a parent block carries no forward flux; use fewer blocks "
                "or more children per parent"
            )
        block_kappas.append(
            numerators[block].sum(axis=0) / block_denominator
        )
    block_matrix = np.stack(block_kappas)
    kappa_stderr = block_matrix.std(axis=0, ddof=1) / np.sqrt(blocks)

    tail = times >= times[0] + (1.0 - plateau_fraction) * (times[-1] - times[0])
    if tail.sum() < 2:
        raise ValueError(
            "the plateau window holds fewer than two records; raise "
            "plateau_fraction or record more often"
        )
    plateau = float(kappa_t[tail].mean())
    plateau_stderr = float(
        block_matrix[:, tail].mean(axis=1).std(ddof=1) / np.sqrt(blocks)
    )

    initial_values = np.concatenate([
        parent.cv_values[:, 0] for parent in kept
    ])
    n_children = sum(len(parent.flux0) for parent in kept)
    return TransmissionResult(
        times_ps=times - times[0],
        kappa_t=kappa_t,
        kappa_stderr=kappa_stderr,
        plateau=plateau,
        plateau_stderr=plateau_stderr,
        forward_flux=total_denominator / n_children,
        s0_mean=float(initial_values.mean()),
        s0_std=float(initial_values.std()),
        n_parents=len(kept),
        n_parents_excluded=n_excluded,
        n_children=n_children,
    )


def plot_transmission_coefficient(
    files: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    s_dagger: float,
    blocks: int = 5,
    plateau_fraction: float = 0.25,
    max_s0_deviation: float | None = None,
    filename: str | os.PathLike[str] | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """
    Plot kappa(t) with its error band and plateau.

    Parameters
    ----------
    files : str or os.PathLike or iterable of str
        Recrossing logs written by :func:`run_openmm_rpmd_recrossing`.
    s_dagger : float
        Dividing-surface value of the collective variable.
    blocks : int, optional
        Contiguous parent blocks for the error band. Default is 5.
    plateau_fraction : float, optional
        Trailing fraction averaged into the plateau. Default is 0.25.
    max_s0_deviation : float or None, optional
        Parent-pruning cutoff, as in :func:`transmission_coefficient`.
        Default is None.
    filename : str or os.PathLike or None, optional
        If set, save the figure here. Default is None.
    show : bool, optional
        If True, display the figure. Default is False.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The created figure.
    axis : matplotlib.axes.Axes
        Its single axis.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        As :func:`transmission_coefficient` raises.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_transmission_coefficient requires matplotlib; install "
            "the 'plot' optional dependency"
        ) from exc

    result = transmission_coefficient(
        files,
        s_dagger=s_dagger,
        blocks=blocks,
        plateau_fraction=plateau_fraction,
        max_s0_deviation=max_s0_deviation,
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.axhline(1.0, color="0.8", linewidth=1.0)
    axis.fill_between(
        result.times_ps,
        result.kappa_t - result.kappa_stderr,
        result.kappa_t + result.kappa_stderr,
        alpha=0.3,
        linewidth=0.0,
    )
    axis.plot(result.times_ps, result.kappa_t)
    axis.axhline(
        result.plateau,
        linestyle="--",
        color="C1",
        label=(
            f"plateau {result.plateau:.3f} "
            f"$\\pm$ {result.plateau_stderr:.3f}"
        ),
    )
    axis.set_xlabel("Time (ps)")
    axis.set_ylabel(r"$\kappa(t)$")
    axis.legend(frameon=False)

    if filename is not None:
        figure.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axis


def rpmd_rate(
    transmission: TransmissionResult | float,
    cv_grid: npt.ArrayLike,
    free_energy: npt.ArrayLike,
    *,
    temperature: unit.Quantity | float,
    s_dagger: float,
    reactant_window: tuple[float, float],
    forward_flux: float | None = None,
    transmission_stderr: float = 0.0,
    energy_unit: Literal[
        "kilojoule_per_mole", "kilocalorie_per_mole"
    ] = "kilojoule_per_mole",
) -> RPMDRate:
    """
    Assemble a Bennett-Chandler rate from kappa and a free-energy surface.

    The factorization is

    .. math::

        k = \\kappa \\cdot k_\\mathrm{QTST}, \\qquad
        k_\\mathrm{QTST} = \\langle \\dot{s}\\,\\theta(\\dot{s}) \\rangle
        \\; \\frac{e^{-\\beta F(s^\\ddagger)}}
        {\\int_\\mathrm{reactant} e^{-\\beta F(s)}\\, \\mathrm{d}s}

    Measuring the forward flux in the shooting ensemble is what makes this
    work for any collective variable: the Maxwell-Boltzmann average over the
    real atoms carries the coordinate's effective-mass geometry, so no
    analytic gradient or reduced mass is ever needed.  The units compose as
    (CV/ps) times (1/CV), so the rate is in reciprocal picoseconds, first
    order and per molecule.

    The surface must be the free energy of the same collective variable, in
    the same units, that the recrossing run recorded -- for this package's
    PLUMED convention that is nanometres and kJ/mol.  Its overall
    normalization cancels; only the used ranges must be finite, so a
    metadynamics surface that never explored far corners is fine.

    Parameters
    ----------
    transmission : TransmissionResult or float
        Either the result of :func:`transmission_coefficient`, from which
        the plateau, its error, and the forward flux are taken, or a bare
        transmission coefficient, in which case *forward_flux* is required.
    cv_grid : sequence of float
        Strictly increasing 1-D grid of collective-variable values.
    free_energy : sequence of float
        Free energy on *cv_grid*, in *energy_unit*.
    temperature : openmm.unit.Quantity or float
        Temperature of the runs the surface and kappa came from. A bare
        number is read as kelvin.
    s_dagger : float
        Dividing-surface value, strictly inside the grid. Use the same value
        the transmission coefficient was analysed at.
    reactant_window : tuple of float
        ``(low, high)`` collective-variable range Boltzmann-integrated as
        the reactant state. It must not contain *s_dagger*.
    forward_flux : float or None, optional
        Mean forward initial flux in collective-variable units per
        picosecond; required exactly when *transmission* is a bare number.
        Default is None.
    transmission_stderr : float, optional
        Standard error of a bare *transmission* number; ignored when a
        :class:`TransmissionResult` carries its own. Default is 0.0.
    energy_unit : {"kilojoule_per_mole", "kilocalorie_per_mole"}, optional
        Unit of *free_energy*. Default is ``"kilojoule_per_mole"``.

    Returns
    -------
    RPMDRate
        The rate and every factor it was assembled from.

    Raises
    ------
    ValueError
        If the grid is not strictly increasing 1-D matching *free_energy*,
        *s_dagger* lies outside it or inside the reactant window, the
        surface is non-finite where it is used, the flux specification is
        missing or doubled, or a scalar is out of domain.

    Warns
    -----
    UserWarning
        If the surface rises above ``F(s_dagger)`` between the reactant
        window and the dividing surface -- the signature of a mis-placed
        *s_dagger*.
    """
    if isinstance(transmission, TransmissionResult):
        if forward_flux is not None:
            raise ValueError(
                "forward_flux is taken from the TransmissionResult; "
                "passing both is ambiguous"
            )
        kappa = _require_finite_number(
            transmission.plateau,
            name="transmission.plateau",
        )
        kappa_stderr = _require_finite_number(
            transmission.plateau_stderr,
            name="transmission.plateau_stderr",
        )
        flux = _require_finite_number(
            transmission.forward_flux,
            name="transmission.forward_flux",
        )
    else:
        kappa = _require_finite_number(transmission, name="transmission")
        if forward_flux is None:
            raise ValueError(
                "a bare transmission number needs forward_flux, the mean "
                "forward initial flux of the shooting ensemble"
            )
        flux = _require_finite_number(forward_flux, name="forward_flux")
        kappa_stderr = _require_finite_number(
            transmission_stderr,
            name="transmission_stderr",
        )
    if flux <= 0.0:
        raise ValueError("forward_flux must be positive")
    if kappa_stderr < 0.0:
        raise ValueError("transmission_stderr must be non-negative")

    temperature_k = require_positive_finite_scalar_in_unit(
        temperature,
        unit.kelvin,
        name="temperature",
    )
    s_dagger = _require_finite_number(s_dagger, name="s_dagger")

    grid = np.asarray(cv_grid, dtype=float)
    surface = np.asarray(free_energy, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("cv_grid must be a 1-D grid of at least two values")
    if surface.shape != grid.shape:
        raise ValueError(
            f"free_energy has shape {surface.shape}; expected {grid.shape}"
        )
    if not np.isfinite(grid).all():
        raise ValueError("cv_grid contains non-finite values")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("cv_grid must be strictly increasing")
    if energy_unit == "kilocalorie_per_mole":
        surface = surface * _KCAL_PER_KJ
    elif energy_unit != "kilojoule_per_mole":
        raise ValueError(
            f"unknown energy_unit {energy_unit!r}; use "
            "'kilojoule_per_mole' or 'kilocalorie_per_mole'"
        )

    if not grid[0] < s_dagger < grid[-1]:
        raise ValueError(
            f"s_dagger={s_dagger} lies outside the grid "
            f"[{grid[0]}, {grid[-1]}]"
        )
    try:
        window_low, window_high = reactant_window
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reactant_window must be a (low, high) pair"
        ) from exc
    window_low = _require_finite_number(window_low, name="reactant_window[0]")
    window_high = _require_finite_number(
        window_high,
        name="reactant_window[1]",
    )
    if window_low >= window_high:
        raise ValueError("reactant_window must satisfy low < high")
    if window_low <= s_dagger <= window_high:
        raise ValueError(
            "reactant_window must not contain s_dagger; the reactant basin "
            "and the dividing surface are different places"
        )

    bracket = int(np.searchsorted(grid, s_dagger))
    if not (
        np.isfinite(surface[bracket - 1]) and np.isfinite(surface[bracket])
    ):
        raise ValueError(
            "the free energy is non-finite next to s_dagger; the surface "
            "was never sampled there"
        )
    barrier_energy = float(np.interp(s_dagger, grid, surface))

    window_mask = (grid >= window_low) & (grid <= window_high)
    if window_mask.sum() < 2:
        raise ValueError(
            "reactant_window covers fewer than two grid points"
        )
    window_energies = surface[window_mask]
    if not np.isfinite(window_energies).all():
        raise ValueError(
            "the free energy is non-finite inside reactant_window"
        )

    # Between the window edge nearest the surface and s_dagger, nothing
    # should stand taller than F(s_dagger) -- otherwise the true barrier is
    # elsewhere and the assembled rate is not a rate over it.
    if s_dagger > window_high:
        path_mask = (grid > window_high) & (grid < s_dagger)
    else:
        path_mask = (grid > s_dagger) & (grid < window_low)
    path_energies = surface[path_mask]
    path_energies = path_energies[np.isfinite(path_energies)]
    if path_energies.size and float(path_energies.max()) > barrier_energy:
        warnings.warn(
            "the free energy rises above F(s_dagger) between the reactant "
            "window and the dividing surface; s_dagger is probably not the "
            "barrier top",
            UserWarning,
            stacklevel=2,
        )

    beta = 1.0 / (_GAS_CONSTANT_KJ_PER_MOL_K * temperature_k)
    # A common shift cancels in the ratio and keeps the exponentials tame.
    shift = min(float(window_energies.min()), barrier_energy)
    surface_probability = math.exp(-beta * (barrier_energy - shift))
    reactant_partition = float(np.trapezoid(
        np.exp(-beta * (window_energies - shift)),
        grid[window_mask],
    ))
    dividing_surface_probability = surface_probability / reactant_partition

    qtst_rate = flux * dividing_surface_probability
    return RPMDRate(
        rate=kappa * qtst_rate,
        rate_stderr=qtst_rate * kappa_stderr,
        qtst_rate=qtst_rate,
        transmission=kappa,
        transmission_stderr=kappa_stderr,
        forward_flux=flux,
        dividing_surface_probability=dividing_surface_probability,
    )
