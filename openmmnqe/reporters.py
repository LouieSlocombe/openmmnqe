"""Reporters for ring-polymer molecular dynamics runs.

OpenMM's own reporters see only the context, which for an ``RPMDIntegrator``
holds a single copy of the system rather than the ring polymer.  Anything that
needs the beads themselves -- their spread, their individual trajectories,
their centroid, their energies, or their centroid velocities -- has to ask
the integrator, which is what six of these seven reporters do.  They are
attached by the ``run_openmm_rpmd_*`` drivers in :mod:`openmmnqe.openmm`.

All seven follow OpenMM's reporter protocol: ``describeNextReport`` says
when the next report is due and what state it needs, and ``report`` writes
it.  Each trajectory reporter takes a ``format``, so a long solvated run can
write DCD or XTC instead of the PDB that is the default.
Use :func:`track_rpmd_atom_expansion` to attach the quantum-spread reporter
for one target atom without constructing it directly, and
:func:`plot_rpmd_atom_expansion` to plot the result against a centroid
atom-pair distance or a supplied reference-path progress coordinate.

:class:`RPMDThermodynamicReporter` covers the quantities a Context cannot
give: the centroid-virial kinetic estimator, the mean bead potential energy,
and the total quantum energy, alongside ring-polymer diagnostics that say
whether the trajectory is worth analysing at all.  Call
:func:`rpmd_thermodynamics` to compute the same set once, off any simulation,
and :func:`rpmd_thermodynamic_averages` or :func:`plot_rpmd_thermodynamics`
to read the log it writes back.  For a thermostat-off run,
:func:`rpmd_energy_conservation` turns the same log's ring-polymer
Hamiltonian column into a conservation verdict, and
:class:`RPMDVelocityReporter` with :func:`velocity_autocorrelation` and
:func:`vibrational_spectrum` turn recorded centroid velocities into
Kubo-style correlation functions and vibrational spectra.
:class:`VelocityArchiveReporter` writes the identical archive from a
classical run, which is the baseline those spectra are read against; it is
the one reporter here that has nothing to do with beads.

:class:`RPMDKineticDecompositionReporter` splits that same centroid-virial
kinetic energy over individual atoms, which is the diagnostic that says how
quantum one particular proton is, and -- through
:func:`~openmmnqe.isotopes.rpmd_isotope_free_energy` -- the integrand of the
mass thermodynamic integration that gives equilibrium isotope effects.
:func:`rpmd_kinetic_decomposition` computes it once off any simulation, and
:func:`rpmd_kinetic_decomposition_averages` and
:func:`plot_rpmd_kinetic_decomposition` read the log back.

Two obvious quantities are deliberately absent.  A heat capacity would need
the exact centroid-virial estimator's second-derivative term, which OpenMM
will not supply, and the fluctuation formula ``k_B beta**2 Var(E)`` that looks
like a substitute is simply wrong for path integrals -- the estimator carries
its own explicit ``beta`` dependence.  A centroid-virial pressure would need
the true virial, which forces alone do not give under periodic boundary
conditions.  Neither is worth a plausible-looking wrong number.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from types import TracebackType
from typing import Any, Literal, NamedTuple, Self

import numpy as np
import numpy.typing as npt
import openmm.unit as unit
from openmm import app, openmm

from ._logs import (
    _block_averaged_columns,
    _column_label,
    _read_reporter_log,
    _select_log_columns,
)
from ._validation import require_integer, require_positive_finite_scalar_in_unit
from .tools import (
    _particle_masses_dalton,
    _ring_spring_energy,
    _topology_molecule_tree,
    _wrap_molecules,
    centroid_positions,
)

_SPREAD_METRICS = {"rms", "mean"}

_BOLTZMANN_KJ_PER_MOL_K = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin
)

# Column order of the thermodynamic log, as ``(result key, header)`` pairs.
# The reporter writes them in this order and the reader and plot helpers
# resolve columns through it, so the three cannot drift apart.
_THERMO_COLUMNS: tuple[tuple[str, str], ...] = (
    ("time", "Time(ps)"),
    ("kinetic_centroid_virial", "KE_cv(kJ/mol)"),
    ("potential_mean", "PE_mean(kJ/mol)"),
    ("energy_quantum", "E_quantum(kJ/mol)"),
    ("potential_sd", "PE_sd(kJ/mol)"),
    ("energy_ring", "E_ring(kJ/mol)"),
    ("energy_spring", "E_spring(kJ/mol)"),
    ("temperature_ring", "T_ring(K)"),
    ("temperature_centroid", "T_centroid(K)"),
)

_THERMO_UNITS: dict[str, Any] = {
    "time": unit.picosecond,
    "kinetic_centroid_virial": unit.kilojoule_per_mole,
    "potential_mean": unit.kilojoule_per_mole,
    "energy_quantum": unit.kilojoule_per_mole,
    "potential_sd": unit.kilojoule_per_mole,
    "energy_ring": unit.kilojoule_per_mole,
    "energy_spring": unit.kilojoule_per_mole,
    "temperature_ring": unit.kelvin,
    "temperature_centroid": unit.kelvin,
}


def _bead_coordinates(integrator: openmm.RPMDIntegrator,
                      atom_indices: Sequence[int] | None = None) -> np.ndarray:
    """
    Collect the positions of every bead of the ring polymer.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        The integrator holding the ring polymer.
    atom_indices : list of int or None, optional
        Atoms to keep. If None, every atom is returned, which for a solvated
        system is a lot of memory. Default is None.

    Returns
    -------
    numpy.ndarray
        Bead coordinates in nanometres, with shape
        ``(n_beads, n_selected_atoms, 3)``.
    """
    all_bead_positions = []
    for bead in range(integrator.getNumCopies()):
        state = integrator.getState(copy=bead, getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometers
        )
        if atom_indices is not None:
            positions = positions[atom_indices]
        all_bead_positions.append(positions)
    return np.asarray(all_bead_positions)


def _spread_from_coordinates(coordinates: np.ndarray,
                             metric: Literal["rms", "mean"]) -> unit.Quantity:
    """
    Reduce bead coordinates to a per-atom radius about their centroid.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Bead coordinates in nanometres, shaped
        ``(n_beads, n_atoms, 3)`` as returned by :func:`_bead_coordinates`.
    metric : {"rms", "mean"}
        ``"rms"`` gives the root-mean-square radius, ``"mean"`` the mean
        radius.

    Returns
    -------
    openmm.unit.Quantity
        Radius per atom, in nanometres, with shape ``(n_atoms,)``.
    """
    centroid = np.mean(coordinates, axis=0)
    radii = np.linalg.norm(coordinates - centroid, axis=2)
    if metric == "rms":
        values = np.sqrt(np.mean(radii ** 2, axis=0))
    else:
        values = np.mean(radii, axis=0)
    return values * unit.nanometers


def _calculate_quantum_spread(integrator: openmm.RPMDIntegrator,
                              atom_indices: Sequence[int] | None = None,
                              ) -> unit.Quantity:
    """
    Compute the RMS distance of the beads from their ring-polymer centroid.

    This radius of gyration is a measure of how delocalised an atom is: a
    classical particle collapses to zero, while a light atom at low
    temperature spreads over an appreciable fraction of a bond length.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        The integrator running the simulation.
    atom_indices : list of int or None, optional
        Atoms to compute the spread for. If None, every atom is included,
        which for a solvated system is a lot of memory. Default is None.

    Returns
    -------
    openmm.unit.Quantity
        Quantum radius of gyration per selected atom, in nanometres, with
        shape ``(n_selected_atoms,)``.
    """
    coordinates = _bead_coordinates(integrator, atom_indices)
    return _spread_from_coordinates(coordinates, metric="rms")


def _calculate_bead_expansion(integrator: openmm.RPMDIntegrator,
                              atom_indices: Sequence[int] | None = None,
                              ) -> unit.Quantity:
    r"""
    Compute the mean bead-centroid distance for selected atoms.

    The proton ring-polymer degree of expansion is

    .. math::

        \Delta |r| = \frac{1}{P}\sum_{i=1}^{P}
        \left|\mathbf{r}_i-\bar{\mathbf{r}}\right|.

    Unlike :func:`_calculate_quantum_spread`, this is a mean radius rather
    than a root-mean-square radius.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        The integrator running the simulation.
    atom_indices : list of int or None, optional
        Atoms to compute the expansion for. If None, every atom is included.
        Default is None.

    Returns
    -------
    openmm.unit.Quantity
        Degree of expansion per selected atom, in nanometres, with shape
        ``(n_selected_atoms,)``.
    """
    coordinates = _bead_coordinates(integrator, atom_indices)
    return _spread_from_coordinates(coordinates, metric="mean")


def _simulation_bead_coordinates(simulation: app.Simulation,
                                 atom_indices: Sequence[int],
                                 ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read one bead state per copy, undoing any periodic wrapping.

    Beads of the same ring polymer can land on opposite sides of a periodic
    box, which would inflate every spread computed from them. Each bead is
    therefore imaged onto the first one before being returned.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The simulation whose integrator holds the ring polymer.
    atom_indices : list of int
        Atoms to read, in the order the returned array uses.

    Returns
    -------
    coordinates : numpy.ndarray
        Bead coordinates in nanometres, shaped
        ``(n_beads, len(atom_indices), 3)``.
    box : numpy.ndarray or None
        Periodic box vectors in nanometres, or None if the system is not
        periodic.
    """
    integrator = simulation.integrator
    system = getattr(simulation, "system", None)
    periodic = (
        system is not None and system.usesPeriodicBoundaryConditions()
    )
    coordinates = []
    reference = None
    box = None

    for bead in range(integrator.getNumCopies()):
        state = integrator.getState(
            copy=bead,
            getPositions=True,
            enforcePeriodicBox=periodic,
        )
        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer
        )[atom_indices]
        if bead == 0:
            reference = positions
            if periodic:
                box = state.getPeriodicBoxVectors(
                    asNumpy=True
                ).value_in_unit(unit.nanometer)
        elif periodic:
            assert reference is not None and box is not None
            displacement = positions - reference
            for axis in (2, 1, 0):
                displacement -= box[axis] * np.round(
                    displacement[:, axis:axis + 1] / box[axis][axis]
                )
            positions = reference + displacement
        coordinates.append(positions)

    return np.asarray(coordinates), box


def _validate_metric(metric: str) -> None:
    """
    Check that a spread metric is one this module implements.

    Parameters
    ----------
    metric : str
        Metric name to check.

    Raises
    ------
    ValueError
        If *metric* is not ``"rms"`` or ``"mean"``.
    """
    if metric not in _SPREAD_METRICS:
        choices = ", ".join(sorted(_SPREAD_METRICS))
        raise ValueError(f"metric must be one of: {choices}")


def _validate_atom_indices(atom_indices: Iterable[int]) -> list[int]:
    """
    Validate and normalise selected zero-based atom indices.

    Parameters
    ----------
    atom_indices : iterable of int
        Atoms to monitor. Booleans are rejected even though they are
        integers in Python.

    Returns
    -------
    list of int
        The same indices as plain ints.

    Raises
    ------
    ValueError
        If the selection is empty or holds a negative index.
    TypeError
        If any index is not an integer.
    """
    atom_indices = list(atom_indices)
    if not atom_indices:
        raise ValueError("atom_indices must not be empty")
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in atom_indices
    ):
        raise TypeError("atom_indices must be integers")
    if any(index < 0 for index in atom_indices):
        raise ValueError("atom_indices must be non-negative")
    return [int(index) for index in atom_indices]


def _validate_distance_pairs(distance_pairs: Iterable[tuple[int, int]] | None,
                             ) -> list[tuple[int, int]]:
    """
    Validate and normalise zero-based atom-index pairs.

    Parameters
    ----------
    distance_pairs : iterable of pair of int or None
        Atom-index pairs whose centroid distance is wanted. None means no
        pairs.

    Returns
    -------
    list of tuple of int
        One ``(first, second)`` pair per entry, as plain ints. Empty if
        *distance_pairs* is None.

    Raises
    ------
    ValueError
        If an entry does not hold exactly two indices, or holds a negative
        one.
    TypeError
        If any index is not an integer.
    """
    if distance_pairs is None:
        return []

    normalised = []
    for pair in distance_pairs:
        try:
            first, second = pair
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each distance pair must contain exactly two atom indices"
            ) from exc
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in (first, second)
        ):
            raise TypeError("distance-pair atom indices must be integers")
        if first < 0 or second < 0:
            raise ValueError("distance-pair atom indices must be non-negative")
        normalised.append((int(first), int(second)))
    return normalised


def _validate_observable_indices(atom_indices: Iterable[int],
                                 distance_pairs: Iterable[tuple[int, int]] | None,
                                 n_atoms: int | None = None,
                                 ) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Validate expansion atoms and distance pairs together.

    Parameters
    ----------
    atom_indices : iterable of int
        Atoms whose expansion is to be reported.
    distance_pairs : iterable of pair of int or None
        Atom-index pairs whose centroid distance is to be reported.
    n_atoms : int or None, optional
        Number of atoms in the topology the indices refer to. If given,
        every index is bounds-checked against it. Default is None.

    Returns
    -------
    atom_indices : list of int
        Normalised expansion atom indices.
    distance_pairs : list of tuple of int
        Normalised distance pairs.

    Raises
    ------
    ValueError
        If either selection is invalid, or an index is at or beyond
        *n_atoms*.
    TypeError
        If any index is not an integer.
    """
    atom_indices = _validate_atom_indices(atom_indices)
    distance_pairs = _validate_distance_pairs(distance_pairs)
    selected = [
        *atom_indices,
        *(index for pair in distance_pairs for index in pair),
    ]
    if n_atoms is not None:
        invalid = [index for index in selected if index >= n_atoms]
        if invalid:
            raise ValueError(
                f"atom index {invalid[0]} is outside topology with "
                f"{n_atoms} atoms"
            )
    return atom_indices, distance_pairs


def _calculate_report_observables(simulation: app.Simulation,
                                  atom_indices: Sequence[int],
                                  metric: Literal["rms", "mean"],
                                  distance_pairs: Sequence[tuple[int, int]],
                                  ) -> tuple[unit.Quantity, unit.Quantity]:
    """
    Calculate expansion and centroid distances in one bead-state pass.

    Reading the bead states is the expensive part of a report, so the atoms
    needed by both observables are gathered once and shared.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The simulation to read from.
    atom_indices : list of int
        Atoms whose expansion is reported.
    metric : {"rms", "mean"}
        Spread metric applied to the beads.
    distance_pairs : list of tuple of int
        Atom pairs whose centroid distance is reported. May be empty.

    Returns
    -------
    spreads : openmm.unit.Quantity
        Expansion per entry of *atom_indices*, in nanometres.
    distances : openmm.unit.Quantity
        Minimum-image centroid distance per entry of *distance_pairs*, in
        nanometres. Empty if there are no pairs.

    Raises
    ------
    ValueError
        If an index lies outside the simulation topology.
    """
    selected_atoms = list(dict.fromkeys([
        *atom_indices,
        *(index for pair in distance_pairs for index in pair),
    ]))
    topology = getattr(simulation, "topology", None)
    if topology is not None:
        n_atoms = topology.getNumAtoms()
        invalid = [index for index in selected_atoms if index >= n_atoms]
        if invalid:
            raise ValueError(
                f"atom index {invalid[0]} is outside topology with "
                f"{n_atoms} atoms"
            )

    coordinates, box = _simulation_bead_coordinates(
        simulation,
        selected_atoms,
    )
    local_index = {
        atom_index: index for index, atom_index in enumerate(selected_atoms)
    }
    expansion_indices = [local_index[index] for index in atom_indices]
    spreads = _spread_from_coordinates(
        coordinates[:, expansion_indices, :],
        metric=metric,
    )

    if not distance_pairs:
        return spreads, np.asarray([]) * unit.nanometer

    centroid = np.mean(coordinates, axis=0)
    deltas = np.asarray([
        centroid[local_index[second]] - centroid[local_index[first]]
        for first, second in distance_pairs
    ])
    if box is not None:
        # OpenMM stores box vectors in reduced form. Remove whole c, b, then a
        # vectors to obtain the minimum image of each centroid displacement.
        for axis in (2, 1, 0):
            deltas -= box[axis] * np.round(
                deltas[:, axis:axis + 1] / box[axis][axis]
            )
    distances = np.linalg.norm(deltas, axis=1) * unit.nanometer
    return spreads, distances


class RPMDQuantumSpreadReporter:
    """
    Log the quantum spread of selected atoms during an RPMD simulation.

    One tab-separated expansion column is written per monitored atom, with
    optional centroid atom-pair distances in the same row. The default metric
    is the radius of gyration computed by :func:`_calculate_quantum_spread`.

    Parameters
    ----------
    file : str
        Path to write the spread log to.
    reportInterval : int
        Interval between reports, in steps.
    atom_indices : list of int
        Atoms to monitor, e.g. the transferring proton.
    names : list of str or None, optional
        Column names, one per atom, e.g. ``["Proton_H1", "Donor_N"]``. If
        None, the atom indices are used. Default is None.
    metric : {"rms", "mean"}, optional
        ``"rms"`` records the existing quantum radius of gyration.
        ``"mean"`` records the mean bead-centroid distance used as the degree
        of expansion. Default is ``"rms"``.
    distance_pairs : iterable of pair of int or None, optional
        Zero-based atom-index pairs whose centroid distances are written in
        the same row as the expansion values. Default is None.
    distance_names : list of str or None, optional
        Column names for *distance_pairs*. By default ``AtomI-AtomJ`` is used.
    """

    def __init__(self, file: str | os.PathLike[str], reportInterval: int,
                 atom_indices: Iterable[int],
                 names: Sequence[str] | None = None,
                 metric: Literal["rms", "mean"] = "rms",
                 distance_pairs: Iterable[tuple[int, int]] | None = None,
                 distance_names: Sequence[str] | None = None) -> None:
        report_interval = require_integer(
            reportInterval,
            name="reportInterval",
            minimum=1,
        )
        atom_indices, distance_pairs = _validate_observable_indices(
            atom_indices,
            distance_pairs,
        )
        if names is not None and len(names) != len(atom_indices):
            raise ValueError("names must contain one entry per atom index")
        _validate_metric(metric)
        if distance_names is not None and len(distance_names) != len(distance_pairs):
            raise ValueError(
                "distance_names must contain one entry per distance pair"
            )
        self._reportInterval = report_interval
        self._atom_indices = atom_indices
        self._metric = metric
        self._distance_pairs = distance_pairs

        prefix = "Rg" if metric == "rms" else "Expansion"
        if names:
            columns = [f"{prefix}_{name}(nm)" for name in names]
        else:
            columns = [
                f"{prefix}_Atom{index}(nm)" for index in atom_indices
            ]

        if distance_names is None:
            distance_names = [
                f"Atom{first}-Atom{second}"
                for first, second in distance_pairs
            ]
        columns.extend(
            f"Distance_{name}(nm)" for name in distance_names
        )
        if len(set(columns)) != len(columns):
            raise ValueError("reporter column names must be unique")
        header = "Step\t" + "\t".join(columns)
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
            requested: the bead positions come from the integrator instead.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """
        Write the quantum spread and optional centroid distances.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            Unused; the bead positions come from the RPMD integrator.
        """
        spreads, distances = _calculate_report_observables(
            simulation,
            self._atom_indices,
            self._metric,
            self._distance_pairs,
        )

        step = simulation.currentStep
        spread_values = spreads.value_in_unit(unit.nanometers)
        distance_values = distances.value_in_unit(unit.nanometers)

        line = f"{step}"
        for val in [*spread_values, *distance_values]:
            line += f"\t{val:.6f}"
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


def track_rpmd_atom_expansion(simulation: app.Simulation, atom_index: int,
                              file: str | os.PathLike[str],
                              report_interval: int, name: str | None = None,
                              metric: Literal["rms", "mean"] = "rms",
                              distance_pairs: Iterable[tuple[int, int]] | None = None,
                              distance_names: Sequence[str] | None = None,
                              ) -> RPMDQuantumSpreadReporter:
    r"""
    Track one atom's ring-polymer spread or degree of expansion.

    The returned reporter is appended to ``simulation.reporters``. At every
    reporting interval it writes the current simulation step and

    .. math::

        R_g = \sqrt{\frac{1}{P}\sum_{i=1}^{P}
              \left|\mathbf{r}_i-\bar{\mathbf{r}}\right|^2}

    where ``P`` is the number of beads, ``r_i`` is the target atom's position
    in bead ``i``, and ``r_bar`` is its ring-polymer centroid. This is the
    default ``metric="rms"``. Set ``metric="mean"`` to record
    ``mean(|r_i-r_bar|)``, the mean bead-centroid degree of expansion. Optional
    centroid atom-pair distances are sampled into the same row, which keeps
    the reaction coordinate aligned with the expansion.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``openmm.RPMDIntegrator``. The reporter is
        attached to this object but the simulation is not advanced.
    atom_index : int
        Zero-based topology index of the atom whose bead expansion to track.
    file : str or os.PathLike
        Output path for the tab-separated time series.
    report_interval : int
        Number of integration steps between samples.
    name : str or None, optional
        Optional label for the radius-of-gyration column. By default the atom
        index is used.
    metric : {"rms", "mean"}, optional
        Quantum radius-of-gyration or mean-radius expansion. Default is
        ``"rms"`` for compatibility with existing spread logs.
    distance_pairs : iterable of pair of int or None, optional
        Zero-based atom pairs whose centroid distances should be recorded.
    distance_names : list of str or None, optional
        Optional labels for the distance columns.

    Returns
    -------
    RPMDQuantumSpreadReporter
        The reporter appended to ``simulation.reporters``.

    Raises
    ------
    TypeError
        If ``atom_index`` is not an integer.
    ValueError
        If ``atom_index`` is negative or ``report_interval`` is not positive.

    Examples
    --------
    Track atom 17 every 100 steps before starting the simulation::

        from openmmnqe import step_rpmd, track_rpmd_atom_expansion

        track_rpmd_atom_expansion(
            simulation,
            atom_index=17,
            file="proton_expansion.tsv",
            report_interval=100,
            name="proton",
            metric="mean",
            distance_pairs=[(4, 17), (9, 17)],
            distance_names=["donor-H", "acceptor-H"],
        )
        step_rpmd(simulation, 10_000)
    """
    if isinstance(atom_index, bool) or not isinstance(atom_index, Integral):
        raise TypeError("atom_index must be an integer")
    if atom_index < 0:
        raise ValueError("atom_index must be non-negative")
    _validate_metric(metric)
    topology = getattr(simulation, "topology", None)
    n_atoms = None if topology is None else topology.getNumAtoms()
    atom_indices, distance_pairs = _validate_observable_indices(
        [atom_index],
        distance_pairs,
        n_atoms=n_atoms,
    )

    reporter = RPMDQuantumSpreadReporter(
        file=file,
        reportInterval=report_interval,
        atom_indices=atom_indices,
        names=None if name is None else [name],
        metric=metric,
        distance_pairs=distance_pairs,
        distance_names=distance_names,
    )
    simulation.reporters.append(reporter)
    return reporter


def _read_expansion_log(file: str | os.PathLike[str],
                        ) -> tuple[list[str], np.ndarray]:
    """
    Read a tab-separated expansion reporter log.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`RPMDQuantumSpreadReporter`.

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
    return _read_reporter_log(file, "expansion log")


def _average_by_progress(progress: np.ndarray, values: np.ndarray,
                         progress_bins: int | None = None,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort path samples and average rows sharing a progress group.

    Parameters
    ----------
    progress : numpy.ndarray
        Progress coordinate, one value per row of *values*.
    values : numpy.ndarray
        Log values, shaped ``(n_rows, n_columns)``.
    progress_bins : int or None, optional
        Number of equal-width bins to average within. With None, only rows
        at exactly equal progress values are averaged. Default is None.

    Returns
    -------
    grouped_progress : numpy.ndarray
        Progress value of each populated group, in ascending order.
    grouped_values : numpy.ndarray
        Mean of the rows in each populated group.

    Raises
    ------
    ValueError
        If *progress_bins* is not a positive integer or None.
    """
    if progress_bins is not None:
        if (
            isinstance(progress_bins, bool)
            or not isinstance(progress_bins, Integral)
            or progress_bins <= 0
        ):
            raise ValueError("progress_bins must be a positive integer or None")
        progress_bins = int(progress_bins)

    if progress_bins is None:
        grouped_progress, group_index = np.unique(
            progress,
            return_inverse=True,
        )
    elif np.ptp(progress) == 0.0:
        grouped_progress = np.asarray([progress[0]])
        group_index = np.zeros(len(progress), dtype=int)
    else:
        edges = np.linspace(progress.min(), progress.max(), progress_bins + 1)
        group_index = np.searchsorted(edges, progress, side="right") - 1
        group_index = np.clip(group_index, 0, progress_bins - 1)
        grouped_progress = np.zeros(progress_bins)
        np.add.at(grouped_progress, group_index, progress)

    group_count = np.bincount(
        group_index,
        minlength=len(grouped_progress),
    )
    populated = group_count > 0
    if progress_bins is not None and np.ptp(progress) != 0.0:
        grouped_progress[populated] /= group_count[populated]

    grouped_values = np.zeros((len(grouped_progress), values.shape[1]))
    np.add.at(grouped_values, group_index, values)
    grouped_values[populated] /= group_count[populated, np.newaxis]
    return grouped_progress[populated], grouped_values[populated]


def plot_rpmd_atom_expansion(file: str | os.PathLike[str], *,
                             expansion_columns: str | Iterable[str] | None = None,
                             distance_columns: str | Iterable[str] | None = None,
                             path_progress: npt.ArrayLike | None = None,
                             progress_bins: int | None = None,
                             length_unit: Literal["nanometer", "angstrom"] = "nanometer",
                             filename: str | os.PathLike[str] | None = None,
                             show: bool = False) -> tuple[Any, tuple[Any, ...]]:
    """
    Plot an RPMD atom expansion against distance or path progress.

    With no *path_progress*, one selected centroid-distance column is used on
    the x axis and a direct expansion-versus-distance scatter is produced. With
    *path_progress*, expansion and all selected distances are drawn in stacked
    panels sharing that coordinate. Repeated progress values are averaged and
    sorted; *progress_bins* can aggregate a continuous path coordinate.

    Parameters
    ----------
    file : str or os.PathLike
        Tab-separated output from :class:`RPMDQuantumSpreadReporter`.
    expansion_columns : str or iterable of str or None, optional
        Expansion/Rg columns to draw. By default all are used.
    distance_columns : str or iterable of str or None, optional
        Centroid-distance columns to draw. Direct distance mode requires
        exactly one; path-progress mode accepts any number.
    path_progress : array-like or None, optional
        Progress value for every log row. Supplying this selects the stacked
        Figure-7-style layout. Normalised values in ``[0, 1]`` are customary.
    progress_bins : int or None, optional
        Number of equal-width bins used to average continuous path-progress
        samples. With None, rows at identical progress values are averaged.
        Ignored in direct distance mode. Default is None.
    length_unit : {"nanometer", "angstrom"}, optional
        Unit used for both plotted expansion and distance values. Reporter
        logs are stored in nanometres. Default is ``"nanometer"``.
    filename : str or os.PathLike or None, optional
        If given, save the figure at this path.
    show : bool, optional
        Display the figure with Matplotlib. Default is False.

    Returns
    -------
    tuple
        ``(figure, axes)`` where *axes* is a tuple containing one direct-plot
        axis or the expansion and distance axes for path-progress mode.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_rpmd_atom_expansion requires matplotlib; install the "
            "'plot' optional dependency"
        ) from exc

    length_units = {
        "nanometer": (1.0, "nm"),
        "angstrom": (10.0, r"$\AA$"),
    }
    if length_unit not in length_units:
        choices = ", ".join(length_units)
        raise ValueError(f"length_unit must be one of: {choices}")
    scale, unit_label = length_units[length_unit]

    header, values = _read_expansion_log(file)
    column_index = {name: index for index, name in enumerate(header)}
    expansion_columns = _select_log_columns(
        header,
        expansion_columns,
        ("Expansion_", "Rg_"),
        "expansion",
    )
    distance_columns = _select_log_columns(
        header,
        distance_columns,
        ("Distance_",),
        "distance",
    )

    selected_indices = [
        column_index[name]
        for name in [*expansion_columns, *distance_columns]
    ]
    if not np.isfinite(values[:, selected_indices]).all():
        raise ValueError("selected expansion-log values must be finite")

    is_mean_expansion = all(
        name.startswith("Expansion_") for name in expansion_columns
    )
    y_label = (
        f"Bead expansion ({unit_label})"
        if is_mean_expansion
        else f"Quantum radius of gyration ({unit_label})"
    )

    axes: tuple[Any, ...]
    if path_progress is None:
        if len(distance_columns) != 1:
            raise ValueError(
                "direct distance mode requires exactly one distance column"
            )
        figure, axis = plt.subplots(figsize=(6.0, 4.2))
        x_values = values[:, column_index[distance_columns[0]]] * scale
        for column in expansion_columns:
            axis.scatter(
                x_values,
                values[:, column_index[column]] * scale,
                label=_column_label(column),
                s=20,
                alpha=0.75,
            )
        axis.set_xlabel(
            f"{_column_label(distance_columns[0])} distance ({unit_label})"
        )
        axis.set_ylabel(y_label)
        axis.legend(frameon=False)
        axes = (axis,)
    else:
        progress = np.asarray(path_progress, dtype=float)
        if progress.ndim != 1 or len(progress) != len(values):
            raise ValueError(
                "path_progress must contain one value per expansion-log row"
            )
        if not np.isfinite(progress).all():
            raise ValueError("path_progress values must be finite")
        progress_limits = (progress.min(), progress.max())
        progress, plotted_values = _average_by_progress(
            progress,
            values,
            progress_bins=progress_bins,
        )

        if distance_columns:
            figure, (expansion_axis, distance_axis) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(6.4, 6.0),
                gridspec_kw={"height_ratios": (1, 1.15), "hspace": 0.06},
            )
            axes = (expansion_axis, distance_axis)
        else:
            figure, expansion_axis = plt.subplots(figsize=(6.0, 4.2))
            distance_axis = None
            axes = (expansion_axis,)

        for column in expansion_columns:
            expansion_axis.plot(
                progress,
                plotted_values[:, column_index[column]] * scale,
                label=_column_label(column),
            )
        expansion_axis.set_ylabel(y_label)
        expansion_axis.legend(frameon=False)
        if progress_limits[0] < progress_limits[1]:
            expansion_axis.set_xlim(*progress_limits)

        if distance_axis is not None:
            for column in distance_columns:
                distance_axis.plot(
                    progress,
                    plotted_values[:, column_index[column]] * scale,
                    label=_column_label(column),
                )
            distance_axis.set_ylabel(f"Distance ({unit_label})")
            distance_axis.set_xlabel("Path progress (unitless)")
            distance_axis.legend(frameon=False)
        else:
            expansion_axis.set_xlabel("Path progress (unitless)")

    if filename is not None:
        figure.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


TrajectoryFormat = Literal["pdb", "dcd", "xtc", "h5", "none"]

_TRAJECTORY_SUFFIX: dict[str, str] = {
    "pdb": ".pdb",
    "dcd": ".dcd",
    "xtc": ".xtc",
    "h5": ".h5",
}

#: Every trajectory format a stage can be asked for, ``'none'`` included.
_TRAJECTORY_FORMATS = ("pdb", "dcd", "xtc", "h5", "none")

#: Formats a reporter reading the RPMD integrator can write.  ``h5`` is
#: absent because mdtraj's HDF5 reporter reads a Context, which for an
#: ``RPMDIntegrator`` holds one copy rather than the ring polymer.
_INTEGRATOR_TRAJECTORY_FORMATS = ("pdb", "dcd", "xtc")


def _require_integrator_format(format: TrajectoryFormat) -> str:
    """
    Check a format a reporter reading the RPMD integrator can write.

    Parameters
    ----------
    format : str
        Requested trajectory format.

    Returns
    -------
    str
        The file suffix that format uses, ``'.pdb'`` and friends.

    Raises
    ------
    ValueError
        If *format* is not one this reporter can write. ``'h5'`` is named
        separately, since it is a real format that simply cannot be produced
        from bead states.
    """
    if format == "h5":
        raise ValueError(
            "trajectory format 'h5' cannot be written from bead states: "
            "mdtraj's HDF5 reporter reads a Context, which for an "
            "RPMDIntegrator holds one copy rather than the ring polymer; "
            "use 'dcd' or 'xtc'"
        )
    if format not in _INTEGRATOR_TRAJECTORY_FORMATS:
        raise ValueError(
            f"unknown trajectory format {format!r}; use one of "
            f"{', '.join(map(repr, _INTEGRATOR_TRAJECTORY_FORMATS))}"
        )
    return _TRAJECTORY_SUFFIX[format]


def _subset_topology(topology: app.Topology,
                     atom_indices: Sequence[int],
                     ) -> app.Topology:
    """
    Build the topology of an atom subset, keeping chains and residues.

    Deleting through a :class:`~openmm.app.Modeller` preserves the chain and
    residue each surviving atom belonged to, so a ``resname`` selection still
    works in whatever reads the trajectory back.  OpenMM's own ``atomSubset``
    support instead flattens the selection into a single residue.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the full system.
    atom_indices : sequence of int
        0-based indices of the atoms to keep.

    Returns
    -------
    openmm.app.Topology
        Topology holding only the selected atoms, with the periodic box of
        the original.

    Raises
    ------
    ValueError
        If an index lies outside *topology*.
    """
    n_atoms = topology.getNumAtoms()
    out_of_range = [index for index in atom_indices if index >= n_atoms]
    if out_of_range:
        raise ValueError(
            f"atom_indices {out_of_range} lie outside the topology with "
            f"{n_atoms} atoms"
        )
    placeholder = [openmm.Vec3(0.0, 0.0, 0.0)] * n_atoms * unit.nanometer
    modeller = app.Modeller(topology, placeholder)
    keep = frozenset(atom_indices)
    modeller.delete([
        atom for atom in modeller.topology.atoms() if atom.index not in keep
    ])
    subset = modeller.topology
    subset.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    return subset


class _TrajectoryWriter:
    """
    Write frames to a PDB, DCD or XTC trajectory behind one interface.

    The three formats disagree about almost everything: ``PDBFile`` writes
    header, models and footer onto a text handle; ``DCDFile`` wraps a binary
    handle and needs the time per frame up front; ``XTCFile`` takes a file
    name and owns the file itself.  This hides that, so a reporter pulling
    positions from an RPMD integrator can offer a format switch without
    caring which of the three it got.

    The file is opened eagerly, so a bad path fails where it was asked for
    rather than at the first report, but the ``DCDFile``/``XTCFile`` object
    is built on the first :meth:`write` -- that is the earliest the time step
    is known, and it is what OpenMM's own ``DCDReporter`` does.

    A binary trajectory also survives a crash better than a PDB one: both
    ``DCDFile`` and ``XTCFile`` finalize each frame as they write it, whereas
    a PDB's closing ``END`` is only written on close.

    Parameters
    ----------
    file_name : str or os.PathLike
        Path to write to. Nothing appends the format's suffix; pass the name
        you want.
    topology : openmm.app.Topology
        Topology of the full system, before any subset is applied.
    format : {"pdb", "dcd", "xtc"}, optional
        Trajectory format. Default is ``"pdb"``.
    reportInterval : int, optional
        Interval between frames, in steps. DCD and XTC record it in their
        headers so the frame times come out right. Default is 1.
    atom_indices : sequence of int or None, optional
        Atoms to write. Default is None, which writes every atom. For a
        solvated system, selecting the solute is the difference between a
        trajectory worth keeping and one worth deleting.

    Raises
    ------
    TypeError
        If *reportInterval* is not an integer.
    ValueError
        If *format* is not one of the three, *reportInterval* is not
        positive, or an atom index lies outside *topology*.
    """

    def __init__(self, file_name: str | os.PathLike[str],
                 topology: app.Topology, *,
                 format: TrajectoryFormat = "pdb",
                 reportInterval: int = 1,
                 atom_indices: Sequence[int] | None = None) -> None:
        _require_integrator_format(format)
        self._format = format
        self._reportInterval = require_integer(
            reportInterval,
            name="reportInterval",
            minimum=1,
        )
        self._closed = False
        self._file_name = os.fspath(file_name)
        self._frame_index = 0
        self._handle: Any = None
        self._writer: Any = None

        self._kept: npt.NDArray[np.intp] | None = None
        if atom_indices is None:
            self._topology = topology
        else:
            self._topology = _subset_topology(topology, atom_indices)
            self._kept = np.asarray(atom_indices, dtype=np.intp)

        if format == "pdb":
            self._handle = open(self._file_name, "w")
            app.PDBFile.writeHeader(self._topology, self._handle)
        elif format == "dcd":
            self._handle = open(self._file_name, "wb")
        else:
            # XTCFile refuses to overwrite a non-empty file unless appending,
            # so truncate first -- what OpenMM's own XTCReporter does.
            open(self._file_name, "wb").close()

    @property
    def is_binary(self) -> bool:
        """
        bool: Whether this is a binary format.

        A binary trajectory needs periodic box vectors on every
        :meth:`write` and the time step on the first one; a PDB needs
        neither, since its box lives in the header it has already written.
        """
        return self._format in ("dcd", "xtc")

    @property
    def topology(self) -> app.Topology:
        """openmm.app.Topology: The topology actually written, subset included."""
        return self._topology

    def select(self, positions: Any) -> Any:
        """
        Cut *positions* down to the atoms this writer keeps.

        Parameters
        ----------
        positions : openmm.unit.Quantity
            Positions for every atom in the full system.

        Returns
        -------
        openmm.unit.Quantity
            The selected positions, or *positions* unchanged when no subset
            was asked for.
        """
        if self._kept is None:
            return positions
        values = np.asarray(positions.value_in_unit(unit.nanometer))
        return values[self._kept] * unit.nanometer

    def write(self, positions: Any, *,
              step_size: Any,
              periodic_box_vectors: Any = None) -> None:
        """
        Append one frame.

        Parameters
        ----------
        positions : openmm.unit.Quantity
            Positions of the atoms this writer keeps, already selected with
            :meth:`select`.
        step_size : openmm.unit.Quantity or None
            Integration time step, used once to set the frame spacing in a
            DCD or XTC header. Unused, and so safely None, for PDB.
        periodic_box_vectors : openmm.unit.Quantity or None, optional
            Box vectors for this frame. Ignored for PDB, whose box lives in
            the header. Default is None.
        """
        if self._format == "pdb":
            app.PDBFile.writeModel(
                self._topology,
                positions,
                self._handle,
                self._frame_index + 1,
            )
            # Flushing every frame would cost more than it buys.
            if self._frame_index % 10 == 0:
                self._handle.flush()
        else:
            if self._writer is None:
                self._writer = self._open_binary_writer(step_size)
            self._writer.writeModel(
                positions,
                periodicBoxVectors=periodic_box_vectors,
            )
        self._frame_index += 1

    def _open_binary_writer(self, step_size: Any) -> Any:
        """
        Build the DCD or XTC writer once the time step is known.

        Parameters
        ----------
        step_size : openmm.unit.Quantity
            Integration time step.

        Returns
        -------
        openmm.app.DCDFile or openmm.app.XTCFile
            The writer, with a header describing the frame spacing.
        """
        interval = self._reportInterval
        if self._format == "dcd":
            return app.DCDFile(self._handle, self._topology, step_size,
                               interval, interval, False)
        return app.XTCFile(self._file_name, self._topology, step_size,
                           interval, interval, False)

    def close(self) -> None:
        """Finalize and close the trajectory, once."""
        if getattr(self, "_closed", True):
            return
        self._closed = True

        handle = getattr(self, "_handle", None)
        if self._format == "pdb":
            if handle is not None and not handle.closed:
                try:
                    app.PDBFile.writeFooter(self._topology, handle)
                finally:
                    handle.close()
            return
        # DCDFile flushes every frame and XTCFile owns its own file, so
        # there is nothing to finalize beyond releasing the handle.
        if handle is not None and not handle.closed:
            handle.close()

    def __enter__(self) -> Self:
        """Return this writer for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize the trajectory when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort fallback for callers that did not close the writer."""
        try:
            self.close()
        except Exception:
            pass


class RPMDBeadReporter:
    """
    Write the trajectory of every individual bead to its own file.

    Each bead of the ring polymer gets a separate file, so the beads can be
    inspected one by one rather than only through their centroid.  That also
    makes this the most expensive output the package writes -- one whole
    trajectory per bead -- so a solvated run wants a binary *format*, an
    *atom_indices* selection, or both.

    Parameters
    ----------
    file_base_name : str
        Prefix for the output files: ``'output'`` gives
        ``'output_bead_0.pdb'``, ``'output_bead_1.pdb'`` and so on, with the
        suffix following *format*.
    reportInterval : int
        Interval between reports, in steps.
    num_beads : int
        Number of beads in the RPMD integrator.
    topology : openmm.app.Topology
        Topology of the full system, written into the trajectory headers.
    format : {"pdb", "dcd", "xtc"}, optional
        Trajectory format. DCD and XTC are far smaller and faster than PDB
        but carry no topology, so pair them with a structure file. Default
        is ``"pdb"``. HDF5 is not available here: it is written by reading a
        Context, which for an ``RPMDIntegrator`` holds one copy rather than
        the ring polymer.
    atom_indices : sequence of int or None, optional
        Atoms to write. Default is None, which writes every atom.
    enforce_periodic_box : bool or None, optional
        Whether to wrap molecules into the periodic box. Default is None,
        which wraps whenever the System is periodic. A nonperiodic System is
        never wrapped, whatever this is set to, because it has no box to
        wrap into.

    Raises
    ------
    TypeError
        If *reportInterval* or *num_beads* is not an integer.
    ValueError
        If *reportInterval* or *num_beads* is not positive, *format* is not
        recognized, or an atom index lies outside *topology*.

    Notes
    -----
    The wrapping is done here, against *topology*, rather than by asking
    OpenMM for it. OpenMM builds its molecule list from the bonded pairs the
    forces report, and ``MLPotential.createMixedSystem`` deletes every
    bonded term inside the ML region while an ``openmm.PythonForce`` reports
    none, so a molecule modelled entirely by the ML potential would have
    each of its atoms moved into the box separately.
    """

    def __init__(self, file_base_name: str, reportInterval: int,
                 num_beads: int, topology: app.Topology,
                 format: TrajectoryFormat = "pdb",
                 atom_indices: Sequence[int] | None = None,
                 enforce_periodic_box: bool | None = None) -> None:
        self._reportInterval = require_integer(
            reportInterval,
            name="reportInterval",
            minimum=1,
        )
        self._num_beads = require_integer(
            num_beads,
            name="num_beads",
            minimum=1,
        )
        self._closed = False
        self._enforce_periodic_box = enforce_periodic_box
        self._molecules = _topology_molecule_tree(topology)

        suffix = _require_integrator_format(format)
        self._is_binary = format != "pdb"
        self._writers: list[_TrajectoryWriter] = []
        try:
            for i in range(self._num_beads):
                self._writers.append(_TrajectoryWriter(
                    f"{file_base_name}_bead_{i}{suffix}",
                    topology,
                    format=format,
                    reportInterval=self._reportInterval,
                    atom_indices=atom_indices,
                ))
        except BaseException:
            # A later bead failing must not strand the files already open.
            for writer in self._writers:
                writer.close()
            raise

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
            requested: the bead positions come from the integrator instead.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """
        Write the current position of every bead to its own file.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            Unused; the bead positions come from the RPMD integrator.
        """
        integrator = simulation.integrator
        step_size = integrator.getStepSize() if self._is_binary else None

        system = getattr(simulation, "system", None)
        periodic = (
            system is not None and system.usesPeriodicBoundaryConditions()
        )
        wrap = periodic and self._enforce_periodic_box is not False

        for i, writer in enumerate(self._writers):
            # getState(bead_index, ...) is specific to RPMDIntegrator. The
            # raw stored coordinates are asked for and wrapped below against
            # the Topology: OpenMM's own molecule list is incomplete on a
            # mixed ML/MM System.
            bead_state = integrator.getState(
                i, getPositions=True, enforcePeriodicBox=False
            )
            box = bead_state.getPeriodicBoxVectors() if self._is_binary else None
            positions = bead_state.getPositions(asNumpy=True)
            if wrap:
                positions = _wrap_molecules(
                    positions.value_in_unit(unit.nanometer),
                    bead_state.getPeriodicBoxVectors(
                        asNumpy=True
                    ).value_in_unit(unit.nanometer),
                    self._molecules,
                ) * unit.nanometer
            writer.write(
                writer.select(positions),
                step_size=step_size,
                periodic_box_vectors=box,
            )

    def close(self) -> None:
        """Finalize and close every bead trajectory, once."""
        if getattr(self, "_closed", True):
            return
        self._closed = True

        first_error: Exception | None = None
        for writer in getattr(self, "_writers", []):
            try:
                writer.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> Self:
        """Return this reporter for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize the bead trajectories when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort fallback for callers that did not close the reporter."""
        try:
            self.close()
        except Exception:
            pass


class RPMDCentroidReporter:
    """
    Write the centroid of the ring polymer to a single PDB file.

    The centroid trajectory is the classical-looking one: it is what the
    ring polymer's centre of mass does, and the trajectory to analyse when
    the beads themselves are not of interest.

    Parameters
    ----------
    file_name : str
        Path to write the centroid trajectory to, suffix included.
    reportInterval : int
        Interval between reports, in steps.
    num_beads : int
        Number of beads in the RPMD integrator.
    topology : openmm.app.Topology
        Topology of the full system, written into the trajectory header.
    format : {"pdb", "dcd", "xtc"}, optional
        Trajectory format. DCD and XTC are far smaller and faster than PDB
        but carry no topology, so pair them with a structure file. Default
        is ``"pdb"``. HDF5 is not available here: it is written by reading a
        Context, which for an ``RPMDIntegrator`` holds one copy rather than
        the ring polymer.
    atom_indices : sequence of int or None, optional
        Atoms to write. Default is None, which writes every atom.
    enforce_periodic_box : bool or None, optional
        Whether to wrap molecules into the periodic box. Default is None,
        which wraps whenever the System is periodic. The wrapping follows
        the Topology's bonds rather than OpenMM's molecule list, which a
        mixed ML/MM System leaves incomplete.

    Raises
    ------
    TypeError
        If *reportInterval* or *num_beads* is not an integer.
    ValueError
        If *reportInterval* or *num_beads* is not positive, *format* is not
        recognized, or an atom index lies outside *topology*.
    """

    def __init__(self, file_name: str, reportInterval: int,
                 num_beads: int, topology: app.Topology,
                 format: TrajectoryFormat = "pdb",
                 atom_indices: Sequence[int] | None = None,
                 enforce_periodic_box: bool | None = None) -> None:
        self._reportInterval = require_integer(
            reportInterval,
            name="reportInterval",
            minimum=1,
        )
        self._num_beads = require_integer(
            num_beads,
            name="num_beads",
            minimum=1,
        )
        _require_integrator_format(format)
        self._is_binary = format != "pdb"
        # The full atom count, which is what centroid_positions needs -- the
        # writer's topology may hold only a subset.
        self._num_atoms = topology.getNumAtoms()
        self._enforce_periodic_box = enforce_periodic_box
        self._closed = False
        self._writer = _TrajectoryWriter(
            file_name,
            topology,
            format=format,
            reportInterval=self._reportInterval,
            atom_indices=atom_indices,
        )

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
            requested: the bead positions come from the integrator instead.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """
        Write the bead centroid for the current step.

        Each bead state is wrapped into the box independently, so beads of a
        ring polymer (or copies of a whole molecule) that straddle a periodic
        boundary can land on opposite sides of the box.  Averaging those
        wrapped coordinates directly would put the centroid in the middle of
        the box, so each bead is first unwrapped relative to bead 0 via the
        minimum image of its displacement -- valid because bead spreads are
        far smaller than half a box length.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            Unused; the bead positions come from the RPMD integrator.
        """
        centroid_pos = centroid_positions(
            simulation,
            self._num_atoms,
            self._num_beads,
            self._enforce_periodic_box,
        )

        box = None
        step_size = None
        if self._is_binary:
            # The centroid has no state of its own to carry a box, but the
            # box is a Context property shared across the copies, and asking
            # for it alone requests no positions or forces.
            box = simulation.context.getState().getPeriodicBoxVectors()
            step_size = simulation.integrator.getStepSize()
        self._writer.write(
            self._writer.select(centroid_pos),
            step_size=step_size,
            periodic_box_vectors=box,
        )

    def close(self) -> None:
        """Finalize and close the centroid trajectory, once."""
        if getattr(self, "_closed", True):
            return
        self._closed = True

        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.close()

    def __enter__(self) -> Self:
        """Return this reporter for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize the centroid trajectory when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort fallback for callers that did not close the reporter."""
        try:
            self.close()
        except Exception:
            pass


class _BeadThermodynamicStates(NamedTuple):
    """
    One pass of bead states, as bare arrays in OpenMM's MD units.

    Attributes
    ----------
    positions : numpy.ndarray
        Bead positions in nanometres, shaped ``(n_beads, n_atoms, 3)``.
    velocities : numpy.ndarray
        Bead velocities in nanometres per picosecond, same shape.
    forces : numpy.ndarray
        Forces on each bead in kJ/mol/nm, same shape.
    potential : numpy.ndarray
        Potential energy of each bead in kJ/mol, shaped ``(n_beads,)``.
    kinetic : numpy.ndarray
        Kinetic energy of each bead in kJ/mol, shaped ``(n_beads,)``.
    time : float
        Simulation time in picoseconds.
    """

    positions: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64]
    forces: npt.NDArray[np.float64]
    potential: npt.NDArray[np.float64]
    kinetic: npt.NDArray[np.float64]
    time: float


def _rpmd_ring_energies(integrator: openmm.RPMDIntegrator,
                        masses_amu: np.ndarray,
                        *,
                        states: _BeadThermodynamicStates | None = None,
                        ) -> tuple[float, float]:
    """
    Ring-polymer Hamiltonian and its spring term, without ``getTotalEnergy``.

    Before OpenMM 8.6.1, ``RPMDIntegrator.getTotalEnergy()`` could deadlock
    on CUDA or OpenCL when a ``PythonForce`` worker needed the GIL held by
    the caller. OpenMM 8.6.1 disables that worker-thread path. These energies
    still reuse the states collected in the thermodynamic bead pass: the bead
    kinetic and potential energies OpenMM reports per copy, plus the springs
    linking neighbouring copies.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        The integrator holding the ring polymer.
    masses_amu : numpy.ndarray
        Particle masses in daltons, shaped ``(n_particles,)``.
    states : _BeadThermodynamicStates or None, optional
        Bead states already read from *integrator*. If None they are read
        here, at one force evaluation per bead. Default is None.

    Returns
    -------
    energy_ring : float
        The ring-polymer Hamiltonian in kJ/mol.
    energy_spring : float
        The spring term alone in kJ/mol.

    Notes
    -----
    Reading its own bead pass costs a little more than the method it replaces
    -- measured at 1.0 ms against 0.7 ms for a 900-atom, four-bead System on
    CUDA, the difference being that each copy is read back separately rather
    than summed on the device. Asking for fewer fields per copy barely helps:
    the cost is in the round trips, not the arrays. A caller that already
    holds the beads should pass them in and pay nothing.
    """
    if states is None:
        states = _bead_thermodynamic_states(integrator)
    energy_spring = _ring_spring_energy(
        states.positions,
        masses_amu,
        integrator.getTemperature().value_in_unit(unit.kelvin),
    )
    energy_ring = float(
        states.potential.sum() + states.kinetic.sum()
    ) + energy_spring
    return energy_ring, energy_spring


def _thermodynamic_degrees_of_freedom(system: openmm.System) -> int:
    """
    Count the momentum degrees of freedom of one ring-polymer copy.

    This follows the same rule as ``openmm.app.StateDataReporter``: three per
    particle that has mass, less one per distance constraint, less three more
    when the system removes centre-of-mass motion.

    Parameters
    ----------
    system : openmm.System
        System the ring polymer is built from.

    Returns
    -------
    int
        Degrees of freedom of a single copy.

    Raises
    ------
    ValueError
        If the count is not positive, which means the system is entirely
        massless or over-constrained.
    """
    zero_mass = 0 * unit.dalton
    dof = 3 * sum(
        1
        for index in range(system.getNumParticles())
        if system.getParticleMass(index) > zero_mass
    )
    dof -= system.getNumConstraints()
    if any(
        isinstance(system.getForce(index), openmm.CMMotionRemover)
        for index in range(system.getNumForces())
    ):
        dof -= 3
    if dof <= 0:
        raise ValueError(
            "system has no positive degrees of freedom to report on"
        )
    return dof


def _bead_thermodynamic_states(integrator: openmm.RPMDIntegrator,
                               ) -> _BeadThermodynamicStates:
    """
    Read positions, velocities, forces and energies of every bead in one pass.

    Each bead costs a force evaluation, so all four quantities are asked for
    in a single ``getState`` call per copy rather than one call each.

    ``enforcePeriodicBox`` is deliberately left False.  Wrapping each copy
    into the box independently can place beads of one ring polymer on
    opposite sides of it, which would wreck the bead-centroid displacements
    the virial estimator is built from.  The raw stored coordinates are
    contiguous within a ring polymer, so they need no unwrapping -- unlike
    :func:`_simulation_bead_coordinates`, which wants wrapped molecules and
    therefore has to undo the wrapping itself.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        The integrator holding the ring polymer.

    Returns
    -------
    _BeadThermodynamicStates
        Bead arrays in nm, nm/ps, kJ/mol/nm and kJ/mol, plus the time in ps.
    """
    positions = []
    velocities = []
    forces = []
    potential = []
    kinetic = []
    time = 0.0

    for bead in range(integrator.getNumCopies()):
        state = integrator.getState(
            copy=bead,
            getPositions=True,
            getVelocities=True,
            getForces=True,
            getEnergy=True,
            enforcePeriodicBox=False,
        )
        positions.append(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        velocities.append(
            state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )
        )
        forces.append(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        )
        potential.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        kinetic.append(
            state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        if bead == 0:
            time = float(state.getTime().value_in_unit(unit.picosecond))

    return _BeadThermodynamicStates(
        positions=np.asarray(positions, dtype=np.float64),
        velocities=np.asarray(velocities, dtype=np.float64),
        forces=np.asarray(forces, dtype=np.float64),
        potential=np.asarray(potential, dtype=np.float64),
        kinetic=np.asarray(kinetic, dtype=np.float64),
        time=time,
    )


def rpmd_thermodynamics(simulation: app.Simulation, *,
                        temperature: unit.Quantity | float | None = None,
                        degrees_of_freedom: int | None = None,
                        ) -> dict[str, unit.Quantity]:
    r"""
    Compute the ring polymer's thermodynamic estimators for the current state.

    The physically meaningful quantities are ``kinetic_centroid_virial``,
    ``potential_mean`` and their sum ``energy_quantum``.  The centroid-virial
    kinetic estimator is

    .. math::

        K_{cv} = \frac{d}{2\beta} - \frac{1}{2P}\sum_{i=1}^{P}
                 \left(\mathbf{r}_i-\bar{\mathbf{r}}\right)\cdot\mathbf{F}_i

    with ``d`` the degrees of freedom of one copy, ``P`` the bead count and
    ``F_i`` the force on bead ``i``.  It is far quieter than the primitive
    estimator, whose variance grows with ``P``.  The potential estimator is
    the mean bead potential energy :math:`\frac{1}{P}\sum_i V(\mathbf{r}_i)`.

    The remaining entries are diagnostics rather than observables.
    ``energy_spring`` is the harmonic spring term alone, summed over the
    springs of angular frequency ``P k_B T / hbar`` that link neighbouring
    copies, and ``energy_ring`` is the ring-polymer Hamiltonian: the bead
    kinetic and potential energies plus those springs.  It is worth watching
    for drift, not for physics.  ``temperature_ring`` and
    ``temperature_centroid`` should both sit at the integrator's setpoint once
    the thermostat has taken hold, the first covering all ``P`` copies and the
    second the centroid mode alone.

    ``energy_ring`` is reconstructed rather than read from
    ``RPMDIntegrator.getTotalEnergy()``, which agrees with it to within one
    part in a million. Before OpenMM 8.6.1, that method could deadlock on
    CUDA or OpenCL when a ``PythonForce`` worker needed the GIL held by the
    caller. The reconstruction still reuses the states read for the other
    observables. It assumes that OpenMM links neighbouring copies with springs of
    angular frequency ``P k_B T / hbar`` -- which
    ``test_reported_ring_energy_matches_openmm_get_total_energy`` pins
    against OpenMM itself.

    Every bead read is a force evaluation, so one call costs roughly what one
    RPMD step does.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``openmm.RPMDIntegrator``.
    temperature : openmm.unit.Quantity or float or None, optional
        Temperature the ring polymer is sampled at, in kelvin if given as a
        bare number. With None, the integrator's own setpoint is used.
        Default is None.
    degrees_of_freedom : int or None, optional
        Degrees of freedom of one copy. With None, they are counted from the
        system by :func:`_thermodynamic_degrees_of_freedom`. Default is None.

    Returns
    -------
    dict of str to openmm.unit.Quantity
        One entry per key of ``_THERMO_COLUMNS``: ``time``,
        ``kinetic_centroid_virial``, ``potential_mean``, ``energy_quantum``,
        ``potential_sd``, ``energy_ring``, ``energy_spring``,
        ``temperature_ring`` and ``temperature_centroid``.

    Raises
    ------
    ValueError
        If *temperature* is not finite and positive, *degrees_of_freedom* is
        not positive, or the system has no degrees of freedom to report on.
    TypeError
        If *degrees_of_freedom* is not an integer.

    Notes
    -----
    Forces from ``getForces`` exclude constraint forces, so the
    centroid-virial estimator is biased for a system with rigid bonds or
    rigid water. Run the beads flexible.

    Only particles with mass contribute to the virial. OpenMM reports the
    force on a virtual site as well as the shares that force redistributes
    onto the site's parents, and for an average site the two are identically
    equal, so counting every row would count the site twice.

    Under ring-polymer contraction the forces read back are the full,
    uncontracted ones evaluated at each bead. That is the wanted behaviour --
    the exact estimator applied to the approximate distribution the
    contracted dynamics samples -- but it does mean these numbers describe
    the full potential, not the contracted one.

    Examples
    --------
    Take a single reading part way through a run::

        from openmmnqe import rpmd_thermodynamics

        values = rpmd_thermodynamics(simulation)
        print(values["energy_quantum"])
    """
    integrator = simulation.integrator
    if temperature is None:
        temperature = integrator.getTemperature()
    temperature_k = require_positive_finite_scalar_in_unit(
        temperature,
        unit.kelvin,
        name="temperature",
    )
    if degrees_of_freedom is None:
        dof = _thermodynamic_degrees_of_freedom(simulation.system)
    else:
        dof = require_integer(
            degrees_of_freedom,
            name="degrees_of_freedom",
            minimum=1,
        )
    masses = _particle_masses_dalton(simulation.system)

    values = _rpmd_thermodynamic_values(integrator, temperature_k, dof, masses)
    return {key: value * _THERMO_UNITS[key] for key, value in values.items()}


def _rpmd_thermodynamic_values(integrator: openmm.RPMDIntegrator,
                               temperature_k: float,
                               dof: int,
                               masses: np.ndarray,
                               *,
                               states: _BeadThermodynamicStates | None = None,
                               ) -> dict[str, float]:
    """
    Evaluate the estimators from already-resolved constants.

    Split out from :func:`rpmd_thermodynamics` so a reporter can look the
    temperature, degree-of-freedom count and particle masses up once rather
    than at every report; on a solvated system the mass read alone is tens of
    milliseconds.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        The integrator holding the ring polymer.
    temperature_k : float
        Sampling temperature in kelvin.
    dof : int
        Degrees of freedom of one copy.
    masses : numpy.ndarray
        Particle masses in daltons, shaped ``(n_particles,)``.
    states : _BeadThermodynamicStates or None, optional
        Bead states already read from *integrator*. If None they are read
        here. Passing them lets a caller that needs the beads for something
        else share the one pass rather than paying for a second set of force
        evaluations. Default is None.

    Returns
    -------
    dict of str to float
        One entry per key of ``_THERMO_COLUMNS``, in the units of
        ``_THERMO_UNITS``.
    """
    n_beads = integrator.getNumCopies()
    if states is None:
        states = _bead_thermodynamic_states(integrator)
    kt = _BOLTZMANN_KJ_PER_MOL_K * temperature_k

    # Only massive particles carry the virial. OpenMM reports a virtual
    # site's own force as well as the shares it redistributes onto its
    # parents, so summing every row counts the site twice; for an average
    # site the two are identically equal. A frozen particle contributes
    # nothing either way, because its beads never separate.
    massive = masses > 0.0
    centroid = states.positions.mean(axis=0)
    virial = float(np.sum(
        ((states.positions - centroid) * states.forces)[:, massive, :]
    ))
    kinetic_centroid_virial = 0.5 * dof * kt - 0.5 * virial / n_beads

    potential_mean = float(states.potential.mean())
    potential_sd = float(states.potential.std())

    energy_ring, energy_spring = _rpmd_ring_energies(
        integrator, masses, states=states
    )

    # Ring-polymer momenta are sampled at P times the physical temperature,
    # which is where the extra factor of P in the normalisation comes from.
    temperature_ring = 2.0 * float(states.kinetic.sum()) / (
        dof * n_beads ** 2 * _BOLTZMANN_KJ_PER_MOL_K
    )

    centroid_velocity = states.velocities.mean(axis=0)
    kinetic_centroid = 0.5 * float(
        np.sum(masses[:, np.newaxis] * centroid_velocity ** 2)
    )
    temperature_centroid = (
        2.0 * kinetic_centroid / (dof * _BOLTZMANN_KJ_PER_MOL_K)
    )

    return {
        "time": states.time,
        "kinetic_centroid_virial": kinetic_centroid_virial,
        "potential_mean": potential_mean,
        "energy_quantum": kinetic_centroid_virial + potential_mean,
        "potential_sd": potential_sd,
        "energy_ring": energy_ring,
        "energy_spring": energy_spring,
        "temperature_ring": temperature_ring,
        "temperature_centroid": temperature_centroid,
    }


class RPMDThermodynamicReporter:
    """
    Log ring-polymer thermodynamic estimators during an RPMD simulation.

    Writes one tab-separated row per report, holding the centroid-virial
    kinetic estimator, the mean bead potential energy and their sum, followed
    by the ring-polymer diagnostics described in
    :func:`rpmd_thermodynamics`.  An RPMD ``Context`` state cannot supply any
    of these: it mirrors a single copy, so its energy is not a bead average
    and its kinetic temperature is not the ring polymer's.

    Each report reads every bead once, which costs about as much as one RPMD
    step. At the drivers' default report interval of 1000 steps that is a
    fraction of a percent.  The degree-of-freedom count and the particle
    masses are resolved on the first report and cached, since nothing in a run
    can change them.

    Parameters
    ----------
    file : str or os.PathLike
        Path to write the thermodynamic log to.
    reportInterval : int
        Interval between reports, in steps.
    temperature : openmm.unit.Quantity or float or None, optional
        Temperature the ring polymer is sampled at, in kelvin if given as a
        bare number. With None, the integrator's setpoint is read at every
        report, so a temperature ramp is followed. Default is None.
    degrees_of_freedom : int or None, optional
        Degrees of freedom of one copy. With None, they are counted from the
        system. Default is None.

    Warns
    -----
    UserWarning
        If the system carries constraints. ``getForces`` omits constraint
        forces, so the centroid-virial estimator is biased for a constrained
        system.
    """

    def __init__(self, file: str | os.PathLike[str], reportInterval: int,
                 temperature: unit.Quantity | float | None = None,
                 degrees_of_freedom: int | None = None) -> None:
        self._reportInterval = require_integer(
            reportInterval,
            name="reportInterval",
            minimum=1,
        )
        if temperature is not None:
            require_positive_finite_scalar_in_unit(
                temperature,
                unit.kelvin,
                name="temperature",
            )
        if degrees_of_freedom is not None:
            degrees_of_freedom = require_integer(
                degrees_of_freedom,
                name="degrees_of_freedom",
                minimum=1,
            )
        self._temperature = temperature
        self._degrees_of_freedom = degrees_of_freedom
        self._dof: int | None = None
        self._masses: np.ndarray | None = None

        header = "Step\t" + "\t".join(
            column for _, column in _THERMO_COLUMNS
        )
        self._out = open(file, "w")
        self._out.write(header + "\n")

    def _prepare(self, simulation: app.Simulation) -> None:
        """
        Resolve and cache the constants of the run, warning once on the way.

        The degree-of-freedom count and the particle masses are read from the
        System, which neither the beads nor the integrator can change, so they
        are looked up on the first report and kept.
        """
        if self._masses is not None:
            return

        system = simulation.system
        if self._degrees_of_freedom is None:
            self._dof = _thermodynamic_degrees_of_freedom(system)
        else:
            self._dof = self._degrees_of_freedom
        self._masses = _particle_masses_dalton(system)

        if system.getNumConstraints() > 0:
            warnings.warn(
                "centroid-virial kinetic energy is biased by constraints: "
                "OpenMM's forces omit constraint forces, so run the ring "
                "polymer flexible if the reported energies are to be trusted",
                UserWarning,
                stacklevel=2,
            )

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
            requested: everything comes from the RPMD integrator instead.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """
        Write one row of ring-polymer thermodynamic estimators.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            Unused; the bead states come from the RPMD integrator.
        """
        self._prepare(simulation)
        assert self._dof is not None and self._masses is not None

        temperature = self._temperature
        if temperature is None:
            temperature = simulation.integrator.getTemperature()
        temperature_k = require_positive_finite_scalar_in_unit(
            temperature,
            unit.kelvin,
            name="temperature",
        )
        values = _rpmd_thermodynamic_values(
            simulation.integrator,
            temperature_k,
            self._dof,
            self._masses,
        )

        line = f"{simulation.currentStep}"
        for key, _ in _THERMO_COLUMNS:
            line += f"\t{values[key]:.6f}"
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


def _read_thermodynamic_log(file: str | os.PathLike[str],
                            ) -> tuple[list[str], np.ndarray]:
    """
    Read a tab-separated thermodynamic reporter log.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`RPMDThermodynamicReporter`.

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
    return _read_reporter_log(file, "thermodynamic log")


def rpmd_thermodynamic_averages(file: str | os.PathLike[str], *,
                                discard: float = 0.0,
                                blocks: int = 5,
                                ) -> dict[str, tuple[float, float]]:
    """
    Average a thermodynamic log, with block-averaged standard errors.

    Consecutive samples from one trajectory are correlated, so the naive
    ``std / sqrt(n)`` understates the uncertainty, often by a large factor.
    The retained rows are instead split into *blocks* contiguous chunks and
    the error taken from the scatter of the block means, which is only fooled
    if the correlation time approaches the block length.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`RPMDThermodynamicReporter`.
    discard : float, optional
        Leading fraction of the rows to drop as equilibration, in ``[0, 1)``.
        Default is 0.0.
    blocks : int, optional
        Number of blocks the retained rows are split into. Default is 5.

    Returns
    -------
    dict of str to tuple of float
        ``{column: (mean, standard_error)}`` for every column but ``"Step"``,
        in the units the column name carries.

    Raises
    ------
    ValueError
        If *discard* is outside ``[0, 1)``, *blocks* is below 2, or too few
        rows survive to fill the blocks.
    TypeError
        If *blocks* is not an integer.

    Examples
    --------
    Drop the first tenth of a production log and average the rest::

        from openmmnqe import rpmd_thermodynamic_averages

        averages = rpmd_thermodynamic_averages("rpmd_prod_thermo.log", discard=0.1)
        mean, error = averages["E_quantum(kJ/mol)"]
    """
    return _block_averaged_columns(
        file,
        "thermodynamic log",
        discard=discard,
        blocks=blocks,
    )


def _energy_unit_scale(energy_unit: str) -> tuple[float, str]:
    """
    Resolve an energy-unit name to its scale factor and axis label.

    Parameters
    ----------
    energy_unit : str
        ``"kilojoule_per_mole"`` or ``"kilocalorie_per_mole"``.

    Returns
    -------
    scale : float
        Factor converting a logged kJ/mol value into *energy_unit*.
    label : str
        Compact unit name for an axis label, e.g. ``"kJ/mol"``.

    Raises
    ------
    ValueError
        If *energy_unit* is not one this module plots.
    """
    energy_units = {
        "kilojoule_per_mole": (1.0, "kJ/mol"),
        "kilocalorie_per_mole": (
            (1.0 * unit.kilojoule_per_mole).value_in_unit(
                unit.kilocalorie_per_mole
            ),
            "kcal/mol",
        ),
    }
    if energy_unit not in energy_units:
        choices = ", ".join(energy_units)
        raise ValueError(f"energy_unit must be one of: {choices}")
    return energy_units[energy_unit]


def plot_rpmd_thermodynamics(file: str | os.PathLike[str], *,
                             energy_columns: str | Iterable[str] | None = None,
                             temperature_columns: str | Iterable[str] | None = None,
                             x_axis: Literal["time", "step"] = "time",
                             energy_unit: Literal["kilojoule_per_mole", "kilocalorie_per_mole"] = "kilojoule_per_mole",
                             filename: str | os.PathLike[str] | None = None,
                             show: bool = False) -> tuple[Any, tuple[Any, ...]]:
    """
    Plot an RPMD thermodynamic log as energy and temperature traces.

    Energies and temperatures are drawn in stacked panels sharing the time or
    step axis, which is the view that answers the two questions a log like
    this is kept for: has the thermostat settled, and is the ring-polymer
    Hamiltonian drifting.

    Parameters
    ----------
    file : str or os.PathLike
        Tab-separated output from :class:`RPMDThermodynamicReporter`.
    energy_columns : str or iterable of str or None, optional
        Energy columns to draw. By default all of them are used, which
        includes the ring-polymer Hamiltonian and spring energy; those are
        orders of magnitude larger than the estimators, so a selection such
        as ``["KE_cv(kJ/mol)", "PE_mean(kJ/mol)", "E_quantum(kJ/mol)"]`` is
        usually the readable choice.
    temperature_columns : str or iterable of str or None, optional
        Temperature columns to draw. By default all of them are used.
    x_axis : {"time", "step"}, optional
        Whether to plot against simulation time or step number. Default is
        ``"time"``.
    energy_unit : {"kilojoule_per_mole", "kilocalorie_per_mole"}, optional
        Unit used for the plotted energies. Logs are stored in kJ/mol.
        Default is ``"kilojoule_per_mole"``.
    filename : str or os.PathLike or None, optional
        If given, save the figure at this path.
    show : bool, optional
        Display the figure with Matplotlib. Default is False.

    Returns
    -------
    tuple
        ``(figure, axes)`` where *axes* holds the energy axis and, when any
        temperature column is drawn, the temperature axis.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    ValueError
        If *x_axis* or *energy_unit* is unknown, a requested column is
        absent, no energy column is selected, or a plotted value is not
        finite.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_rpmd_thermodynamics requires matplotlib; install the "
            "'plot' optional dependency"
        ) from exc

    scale, unit_label = _energy_unit_scale(energy_unit)
    if x_axis not in {"time", "step"}:
        raise ValueError('x_axis must be one of: step, time')

    header, values = _read_thermodynamic_log(file)
    column_index = {name: index for index, name in enumerate(header)}
    energy_columns = _select_log_columns(
        header,
        energy_columns,
        ("KE_", "PE_", "E_"),
        "energy",
    )
    if not energy_columns:
        raise ValueError("thermodynamic log contains no energy columns")
    temperature_columns = _select_log_columns(
        header,
        temperature_columns,
        ("T_",),
        "temperature",
    )

    if x_axis == "time" and "Time(ps)" in column_index:
        x_values = values[:, column_index["Time(ps)"]]
        x_label = "Time (ps)"
    else:
        x_values = values[:, column_index["Step"]]
        x_label = "Step"

    selected = [
        column_index[name] for name in [*energy_columns, *temperature_columns]
    ]
    if not np.isfinite(values[:, selected]).all():
        raise ValueError("selected thermodynamic-log values must be finite")

    axes: tuple[Any, ...]
    if temperature_columns:
        figure, (energy_axis, temperature_axis) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6.4, 6.0),
            gridspec_kw={"height_ratios": (1.3, 1), "hspace": 0.06},
        )
        axes = (energy_axis, temperature_axis)
    else:
        figure, energy_axis = plt.subplots(figsize=(6.4, 4.2))
        temperature_axis = None
        axes = (energy_axis,)

    for column in energy_columns:
        energy_axis.plot(
            x_values,
            values[:, column_index[column]] * scale,
            label=_column_label(column),
        )
    energy_axis.set_ylabel(f"Energy ({unit_label})")
    energy_axis.legend(frameon=False)

    if temperature_axis is not None:
        for column in temperature_columns:
            temperature_axis.plot(
                x_values,
                values[:, column_index[column]],
                label=_column_label(column),
            )
        temperature_axis.set_ylabel("Temperature (K)")
        temperature_axis.set_xlabel(x_label)
        temperature_axis.legend(frameon=False)
    else:
        energy_axis.set_xlabel(x_label)

    if filename is not None:
        figure.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def _per_atom_centroid_virial_kinetic(states: _BeadThermodynamicStates,
                                      temperature_k: float,
                                      masses: npt.NDArray[np.float64],
                                      ) -> npt.NDArray[np.float64]:
    """
    Decompose the centroid-virial kinetic energy over particles.

    The system estimator in :func:`_rpmd_thermodynamic_values` collapses the
    virial to a scalar. Keeping the particle axis instead costs nothing --
    the same bead states, one fewer axis summed -- and gives each particle's
    own quantum kinetic energy.

    Parameters
    ----------
    states : _BeadThermodynamicStates
        Bead positions and forces, as read by
        :func:`_bead_thermodynamic_states`.
    temperature_k : float
        Sampling temperature in kelvin.
    masses : numpy.ndarray
        Particle masses in daltons, shaped ``(n_particles,)``.

    Returns
    -------
    numpy.ndarray
        Per-particle kinetic energy in kJ/mol, shaped ``(n_particles,)``,
        with NaN wherever the mass is zero.

    Notes
    -----
    The free-particle term is ``3 kT / 2`` per particle, not the system
    ``d kT / 2`` divided up: three degrees of freedom belong to each
    particle, and there is no non-arbitrary way to share out the constraint
    and centre-of-mass reductions that ``d`` carries.  Massless rows are NaN
    rather than zero so that summing them is loud rather than quietly wrong.
    """
    n_beads = states.positions.shape[0]
    kt = _BOLTZMANN_KJ_PER_MOL_K * temperature_k

    centroid = states.positions.mean(axis=0)
    virial = np.sum((states.positions - centroid) * states.forces, axis=(0, 2))
    kinetic = 1.5 * kt - 0.5 * virial / n_beads
    kinetic[masses <= 0.0] = np.nan
    return kinetic


def _resolve_decomposition_atoms(atom_indices: Iterable[int] | None,
                                 masses: npt.NDArray[np.float64],
                                 ) -> list[int]:
    """
    Normalise an atom selection for the kinetic decomposition.

    Parameters
    ----------
    atom_indices : iterable of int or None
        Atoms wanted. None selects every particle that has mass.
    masses : numpy.ndarray
        Particle masses in daltons, shaped ``(n_particles,)``.

    Returns
    -------
    list of int
        The selected indices, as plain ints.

    Raises
    ------
    ValueError
        If a requested index is outside the System, a requested particle is
        massless, or the System has no massive particles at all.
    TypeError
        If any index is not an integer.
    """
    n_particles = len(masses)
    if atom_indices is None:
        selected = [
            index for index in range(n_particles) if masses[index] > 0.0
        ]
        if not selected:
            raise ValueError(
                "System has no particles with mass to decompose"
            )
        return selected

    selected = _validate_atom_indices(atom_indices)
    outside = [index for index in selected if index >= n_particles]
    if outside:
        raise ValueError(
            f"atom index {outside[0]} is outside System with "
            f"{n_particles} particles"
        )
    massless = [index for index in selected if masses[index] <= 0.0]
    if massless:
        raise ValueError(
            f"atom {massless[0]} has zero mass; a virtual site has no "
            "kinetic energy to decompose"
        )
    return selected


def rpmd_kinetic_decomposition(simulation: app.Simulation,
                               atom_indices: Iterable[int] | None = None,
                               *,
                               temperature: unit.Quantity | float | None = None,
                               ) -> dict[int, unit.Quantity]:
    r"""
    Split the centroid-virial kinetic energy over individual atoms.

    :func:`rpmd_thermodynamics` reports one number for the whole system. The
    same bead pass answers the question a ring-polymer run is usually kept
    for -- *how quantum is this particular proton* -- one atom at a time:

    .. math::

        K_i = \frac{3}{2\beta} - \frac{1}{2P}\sum_{j=1}^{P}
              \left(\mathbf{r}_i^{(j)}-\bar{\mathbf{r}}_i\right)
              \cdot\mathbf{F}_i^{(j)}

    with ``P`` the bead count, ``r_i^(j)`` atom ``i`` in bead ``j`` and
    ``r_bar_i`` its ring-polymer centroid.  A classical atom sits at
    ``3 kT / 2``; a quantum one sits above it, and a light atom in a stiff
    bond sits far above it.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        A simulation driven by an ``openmm.RPMDIntegrator``.
    atom_indices : iterable of int or None, optional
        Atoms to report. Default is None, which reports every particle that
        has mass.
    temperature : openmm.unit.Quantity or float or None, optional
        Sampling temperature. A bare number is read as kelvin. Default is
        None, which reads the integrator's setpoint.

    Returns
    -------
    dict of int to openmm.unit.Quantity
        ``{atom index: kinetic energy}``, in kJ/mol.

    Raises
    ------
    ValueError
        If *temperature* is not finite and positive, an index is outside the
        System, or a selected particle is massless.
    TypeError
        If any index is not an integer.

    See Also
    --------
    rpmd_thermodynamics : The system totals this decomposes.
    RPMDKineticDecompositionReporter : Log this along a trajectory.

    Notes
    -----
    The per-atom free-particle term is ``3 kT / 2``, so summing over every
    massive particle does not in general reproduce the ``KE_cv`` column,
    which uses the system degree-of-freedom count:

    .. math::

        \sum_i K_i = K_{cv}
                   + \left(N_{c} + 3 n_{\mathrm{CMM}}\right)\frac{kT}{2}

    with ``N_c`` the number of constraints and ``n_CMM`` one if the System
    carries a ``CMMotionRemover``.  The two agree exactly for a flexible
    System with no centre-of-mass removal, which is the only case where the
    estimator is unbiased anyway: ``getForces`` omits constraint forces.

    Examples
    --------
    Compare a proton against the heavy atom it is bonded to::

        from openmmnqe import rpmd_kinetic_decomposition

        kinetic = rpmd_kinetic_decomposition(simulation, [proton, donor])
        print(kinetic[proton], kinetic[donor])
    """
    integrator = simulation.integrator
    if temperature is None:
        temperature = integrator.getTemperature()
    temperature_k = require_positive_finite_scalar_in_unit(
        temperature,
        unit.kelvin,
        name="temperature",
    )
    masses = _particle_masses_dalton(simulation.system)
    selected = _resolve_decomposition_atoms(atom_indices, masses)

    states = _bead_thermodynamic_states(integrator)
    kinetic = _per_atom_centroid_virial_kinetic(states, temperature_k, masses)
    return {
        index: float(kinetic[index]) * unit.kilojoule_per_mole
        for index in selected
    }


class RPMDKineticDecompositionReporter:
    """
    Log per-atom centroid-virial kinetic energies during an RPMD simulation.

    One ``Kcv_<name>(kJ/mol)`` column is written per selected atom, after the
    step and simulation time.  The quantity is the one described by
    :func:`rpmd_kinetic_decomposition`: a classical atom sits at ``3 kT / 2``
    and a quantum one above it, so the column read against that reference is
    a direct measure of how much nuclear quantum character an atom carries.

    Parameters
    ----------
    file : str or os.PathLike
        Path to write the log to.
    reportInterval : int
        Interval between reports, in steps.
    atom_indices : iterable of int
        Atoms to report, e.g. the transferring proton and its donor. A
        selection is required: one column per atom of a solvated System is
        not a usable file.
    names : list of str or None, optional
        Column names, one per atom, e.g. ``["H1", "Donor_N"]``. If None, the
        atom indices are used. Default is None.
    temperature : openmm.unit.Quantity or float or None, optional
        Sampling temperature. A bare number is read as kelvin. Default is
        None, which re-reads the integrator's setpoint at every report and so
        follows a temperature ramp.

    Raises
    ------
    ValueError
        If *reportInterval* is below 1, the selection is empty or negative,
        *names* does not match the selection, the column names collide, or
        *temperature* is not finite and positive.
    TypeError
        If *reportInterval* or any index is not an integer.

    See Also
    --------
    RPMDThermodynamicReporter : The system totals this decomposes.

    Notes
    -----
    Every report reads each bead once, with forces, which costs roughly one
    RPMD step.  Attached alongside :class:`RPMDThermodynamicReporter` a due
    step therefore costs two bead passes rather than one.  At the drivers'
    default reporting interval of 1000 steps that is a fraction of a percent
    either way; at the interval of ten or so that per-atom statistics want,
    it is the difference between a tenth and a fifth of the run.  Give the
    thermodynamic reporter the coarser interval of the two -- drift and
    thermostat health need far fewer samples than a per-atom estimator does.

    Like every centroid-virial estimator here, this one is biased by
    constraints, because ``getForces`` omits constraint forces.  A selected
    particle must have mass; a virtual site has no kinetic energy to
    decompose and is rejected at the first report, once the System is known.
    """

    def __init__(self, file: str | os.PathLike[str], reportInterval: int,
                 atom_indices: Iterable[int],
                 names: Sequence[str] | None = None,
                 temperature: unit.Quantity | float | None = None) -> None:
        report_interval = require_integer(
            reportInterval,
            name="reportInterval",
            minimum=1,
        )
        atom_indices = _validate_atom_indices(atom_indices)
        if names is not None and len(names) != len(atom_indices):
            raise ValueError("names must contain one entry per atom index")
        if temperature is not None:
            require_positive_finite_scalar_in_unit(
                temperature, unit.kelvin, name="temperature",
            )
        self._reportInterval = report_interval
        self._atom_indices = atom_indices
        self._temperature = temperature
        self._masses: np.ndarray | None = None

        if names:
            columns = [f"Kcv_{name}(kJ/mol)" for name in names]
        else:
            columns = [
                f"Kcv_Atom{index}(kJ/mol)" for index in atom_indices
            ]
        if len(set(columns)) != len(columns):
            raise ValueError("reporter column names must be unique")
        header = "Step\tTime(ps)\t" + "\t".join(columns)
        self._out = open(file, "w")
        self._out.write(header + "\n")

    def _prepare(self, simulation: app.Simulation) -> None:
        """
        Read and check the particle masses once, at the first report.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.

        Raises
        ------
        ValueError
            If a selected index is outside the System or names a massless
            particle.

        Warns
        -----
        UserWarning
            If the System is constrained, because the estimator is then
            biased.
        """
        if self._masses is not None:
            return

        system = simulation.system
        masses = _particle_masses_dalton(system)
        _resolve_decomposition_atoms(self._atom_indices, masses)

        if system.getNumConstraints() > 0:
            warnings.warn(
                "centroid-virial kinetic energy is biased by constraints: "
                "OpenMM's forces omit constraint forces, so run the ring "
                "polymer flexible if the reported energies are to be trusted",
                UserWarning,
                stacklevel=2,
            )
        self._masses = masses

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
            requested: the beads come from the integrator instead.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """
        Write one row of per-atom kinetic energies.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            Unused; the beads come from the RPMD integrator.
        """
        self._prepare(simulation)
        assert self._masses is not None

        temperature = self._temperature
        if temperature is None:
            temperature = simulation.integrator.getTemperature()
        temperature_k = require_positive_finite_scalar_in_unit(
            temperature,
            unit.kelvin,
            name="temperature",
        )
        states = _bead_thermodynamic_states(simulation.integrator)
        kinetic = _per_atom_centroid_virial_kinetic(
            states, temperature_k, self._masses,
        )

        line = f"{simulation.currentStep}\t{states.time:.6f}"
        for index in self._atom_indices:
            line += f"\t{kinetic[index]:.6f}"
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


def _read_kinetic_decomposition_log(file: str | os.PathLike[str],
                                    ) -> tuple[list[str], np.ndarray]:
    """
    Read a tab-separated per-atom kinetic-energy log.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`RPMDKineticDecompositionReporter`.

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
    return _read_reporter_log(file, "kinetic decomposition log")


def rpmd_kinetic_decomposition_averages(file: str | os.PathLike[str], *,
                                        discard: float = 0.0,
                                        blocks: int = 5,
                                        ) -> dict[str, tuple[float, float]]:
    """
    Average a per-atom kinetic-energy log, with block standard errors.

    The errors are block-averaged for the reason set out in
    :func:`rpmd_thermodynamic_averages`: consecutive samples are correlated,
    so ``std / sqrt(n)`` understates the uncertainty.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`RPMDKineticDecompositionReporter`.
    discard : float, optional
        Leading fraction of the rows to drop as equilibration, in ``[0, 1)``.
        Default is 0.0.
    blocks : int, optional
        Number of blocks the retained rows are split into. Default is 5.

    Returns
    -------
    dict of str to tuple of float
        ``{column: (mean, standard_error)}`` for every column but ``"Step"``,
        in the units the column name carries.

    Raises
    ------
    ValueError
        If *discard* is outside ``[0, 1)``, *blocks* is below 2, or too few
        rows survive to fill the blocks.
    TypeError
        If *blocks* is not an integer.

    Examples
    --------
    Feed a mass-integration node straight from its log::

        from openmmnqe import rpmd_kinetic_decomposition_averages

        averages = rpmd_kinetic_decomposition_averages(
            "node_0_kinetic.log", discard=0.2,
        )
        mean, error = averages["Kcv_H1(kJ/mol)"]
    """
    return _block_averaged_columns(
        file,
        "kinetic decomposition log",
        discard=discard,
        blocks=blocks,
    )


def plot_rpmd_kinetic_decomposition(file: str | os.PathLike[str], *,
                                    columns: str | Iterable[str] | None = None,
                                    x_axis: Literal["time", "step"] = "time",
                                    energy_unit: Literal["kilojoule_per_mole", "kilocalorie_per_mole"] = "kilojoule_per_mole",
                                    temperature: unit.Quantity | float | None = None,
                                    filename: str | os.PathLike[str] | None = None,
                                    show: bool = False) -> tuple[Any, tuple[Any, ...]]:
    """
    Plot per-atom kinetic energies against a classical reference.

    One trace per atom, drawn against time or step.  Given *temperature* a
    dashed line is added at ``3 kT / 2``, the classical equipartition value:
    the gap between an atom's trace and that line is the quantity the log
    exists to show.

    Parameters
    ----------
    file : str or os.PathLike
        Tab-separated output from
        :class:`RPMDKineticDecompositionReporter`.
    columns : str or iterable of str or None, optional
        Columns to draw. By default every ``Kcv_`` column is used.
    x_axis : {"time", "step"}, optional
        Whether to plot against simulation time or step number. Default is
        ``"time"``.
    energy_unit : {"kilojoule_per_mole", "kilocalorie_per_mole"}, optional
        Unit used for the plotted energies. Logs are stored in kJ/mol.
        Default is ``"kilojoule_per_mole"``.
    temperature : openmm.unit.Quantity or float or None, optional
        If given, draw the classical ``3 kT / 2`` reference at this
        temperature. A bare number is read as kelvin. Default is None.
    filename : str or os.PathLike or None, optional
        If given, save the figure at this path.
    show : bool, optional
        Display the figure with Matplotlib. Default is False.

    Returns
    -------
    tuple
        ``(figure, axes)`` where *axes* holds the single kinetic-energy axis.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    ValueError
        If *x_axis* or *energy_unit* is unknown, a requested column is
        absent, no column is selected, *temperature* is not finite and
        positive, or a plotted value is not finite.

    Examples
    --------
    Draw a proton against its classical reference::

        from openmmnqe import plot_rpmd_kinetic_decomposition

        plot_rpmd_kinetic_decomposition(
            "rpmd_prod_kinetic.log",
            temperature=300.0,
            filename="kinetic.png",
        )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_rpmd_kinetic_decomposition requires matplotlib; install "
            "the 'plot' optional dependency"
        ) from exc

    scale, unit_label = _energy_unit_scale(energy_unit)
    if x_axis not in {"time", "step"}:
        raise ValueError('x_axis must be one of: step, time')
    if temperature is not None:
        temperature_k = require_positive_finite_scalar_in_unit(
            temperature,
            unit.kelvin,
            name="temperature",
        )

    header, values = _read_kinetic_decomposition_log(file)
    column_index = {name: index for index, name in enumerate(header)}
    columns = _select_log_columns(header, columns, ("Kcv_",), "kinetic")
    if not columns:
        raise ValueError("kinetic decomposition log contains no Kcv_ columns")

    if x_axis == "time" and "Time(ps)" in column_index:
        x_values = values[:, column_index["Time(ps)"]]
        x_label = "Time (ps)"
    else:
        x_values = values[:, column_index["Step"]]
        x_label = "Step"

    selected = [column_index[name] for name in columns]
    if not np.isfinite(values[:, selected]).all():
        raise ValueError(
            "selected kinetic-decomposition values must be finite"
        )

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for column in columns:
        axis.plot(
            x_values,
            values[:, column_index[column]] * scale,
            label=_column_label(column),
        )
    if temperature is not None:
        classical = 1.5 * _BOLTZMANN_KJ_PER_MOL_K * temperature_k * scale
        axis.axhline(
            classical,
            linestyle="--",
            color="0.4",
            linewidth=1.0,
            label=f"classical, 3kT/2 at {temperature_k:g} K",
        )
    axis.set_ylabel(f"Kinetic energy ({unit_label})")
    axis.set_xlabel(x_label)
    axis.legend(frameon=False)

    if filename is not None:
        figure.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return figure, (axis,)


class RPMDEnergyConservation(NamedTuple):
    """
    Verdict on whether a run conserved the ring-polymer Hamiltonian.

    Produced by :func:`rpmd_energy_conservation` from the ``E_ring(kJ/mol)``
    column of a thermodynamic log.  A microcanonical (thermostat-off) RPMD
    run should show a drift small against its own short-time fluctuation; a
    thermostatted run's ``E_ring`` wanders by design and carries no verdict
    worth reading.

    Attributes
    ----------
    drift_rate : float
        Least-squares slope of ``E_ring`` against time, in kJ/mol/ps.
    drift_per_step : float
        Least-squares slope of ``E_ring`` against the step count, in
        kJ/mol per step.
    total_drift : float
        ``drift_rate`` times the analysed time span, in kJ/mol.
    fluctuation : float
        Root-mean-square residual of ``E_ring`` about the fitted line, in
        kJ/mol.  For a symplectic integrator this is the bounded shadow-
        Hamiltonian oscillation, which shrinks with the time step.
    drift_per_ps_over_kbt : float
        ``drift_rate`` divided by ``k_B T``, so the drift reads in thermal
        energies per picosecond.
    drift_ratio : float
        ``abs(total_drift) / fluctuation``.  Infinite when the residuals
        vanish but the drift does not, as for a perfectly linear ramp.
    conserved : bool
        Whether ``drift_ratio`` is at or below the tolerance.
    """

    drift_rate: float
    drift_per_step: float
    total_drift: float
    fluctuation: float
    drift_per_ps_over_kbt: float
    drift_ratio: float
    conserved: bool


def rpmd_energy_conservation(file: str | os.PathLike[str], *,
                             temperature: unit.Quantity | float,
                             discard: float = 0.0,
                             tolerance: float = 2.0,
                             ) -> RPMDEnergyConservation:
    """
    Check a thermodynamic log for ring-polymer energy conservation.

    ``E_ring(kJ/mol)`` is the full ring-polymer Hamiltonian -- bead kinetic
    and potential energies plus the springs linking neighbouring copies.  With
    the thermostat off it is the conserved quantity of the dynamics, so its
    net drift is the integration-quality signal for a microcanonical run.  The
    verdict compares the drift across the retained rows with the size of the
    fluctuation about it: a good thermostat-off run drifts by less than it
    oscillates.

    This is a practical check that the dynamics is not losing energy, not a
    statistical test with a calibrated false-positive rate.  It also cannot
    tell a thermostat-off run from a thermostatted one whose wander happens
    to be driftless -- read ``T_ring(K)`` for that.  The hard-wired
    ``CMMotionRemover`` in the stage-built System introduces tiny
    non-Hamiltonian corrections; for the strictest checks build the System
    yourself with ``removeCMMotion=False`` and hand it in through
    :class:`openmmnqe.openmm.PreparedSystem`.

    Parameters
    ----------
    file : str or os.PathLike
        Log written by :class:`RPMDThermodynamicReporter`.
    temperature : openmm.unit.Quantity or float
        Simulation temperature, used only to express the drift in thermal
        energies. A bare number is read as kelvin.
    discard : float, optional
        Leading fraction of the rows to drop as equilibration, in ``[0, 1)``.
        Default is 0.0.
    tolerance : float, optional
        Largest ``drift_ratio`` still reported as conserved. Default is 2.0.

    Returns
    -------
    RPMDEnergyConservation
        The verdict and the numbers behind it.

    Raises
    ------
    ValueError
        If *discard* is outside ``[0, 1)``, *tolerance* or *temperature* is
        not positive and finite, fewer than three rows survive the discard,
        or the retained rows do not advance in time and step.

    Examples
    --------
    Check a thermostat-off production run::

        from openmmnqe import rpmd_energy_conservation

        verdict = rpmd_energy_conservation("rpmd_nve_thermo.log",
                                           temperature=300.0)
        assert verdict.conserved, verdict
    """
    temperature_k = require_positive_finite_scalar_in_unit(
        temperature,
        unit.kelvin,
        name="temperature",
    )
    if isinstance(discard, bool) or not isinstance(discard, Real):
        raise ValueError("discard must be a number in [0, 1)")
    discard = float(discard)
    if not np.isfinite(discard) or not 0.0 <= discard < 1.0:
        raise ValueError("discard must be a number in [0, 1)")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise ValueError("tolerance must be a positive, finite number")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive, finite number")

    header, values = _read_thermodynamic_log(file)
    required = ("Step", "Time(ps)", "E_ring(kJ/mol)")
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(
            f"thermodynamic log lacks column(s): {', '.join(missing)}"
        )
    retained = values[int(discard * len(values)):]
    if len(retained) < 3:
        raise ValueError(
            f"thermodynamic log has {len(retained)} rows after discarding; "
            "an energy-conservation check needs at least 3"
        )

    steps = retained[:, header.index("Step")]
    times = retained[:, header.index("Time(ps)")]
    energies = retained[:, header.index("E_ring(kJ/mol)")]
    if not np.isfinite(energies).all():
        raise ValueError("E_ring column contains non-finite values")
    time_span = float(times[-1] - times[0])
    step_span = float(steps[-1] - steps[0])
    if time_span <= 0.0 or step_span <= 0.0:
        raise ValueError(
            "thermodynamic log rows must advance in time and step"
        )

    drift_rate, intercept = np.polyfit(times, energies, 1)
    drift_per_step = float(np.polyfit(steps, energies, 1)[0])
    residuals = energies - (drift_rate * times + intercept)
    fluctuation = float(np.sqrt(np.mean(np.square(residuals))))
    total_drift = float(drift_rate * time_span)
    if fluctuation > 0.0:
        drift_ratio = abs(total_drift) / fluctuation
    else:
        drift_ratio = 0.0 if total_drift == 0.0 else float("inf")

    kbt = _BOLTZMANN_KJ_PER_MOL_K * temperature_k
    return RPMDEnergyConservation(
        drift_rate=float(drift_rate),
        drift_per_step=drift_per_step,
        total_drift=total_drift,
        fluctuation=fluctuation,
        drift_per_ps_over_kbt=float(drift_rate) / kbt,
        drift_ratio=float(drift_ratio),
        conserved=bool(drift_ratio <= tolerance),
    )


class _VelocityArchiveBase:
    """
    Shared machinery for the reporters that record a velocity archive.

    Subclasses differ only in where a frame's velocities come from -- a
    Context for a classical run, the bead states for a ring polymer -- so
    everything else lives here: index validation, the mass lookup, frame
    accumulation and the single ``.npz`` written on close.  Sharing it is
    what makes the two archives the same file format by construction rather
    than by inspection, so :func:`velocity_autocorrelation` and
    :func:`vibrational_spectrum` read either without knowing which wrote it.

    Frames accumulate in memory until close, so a selection in
    *atom_indices* is the difference between an archive and an
    out-of-memory error: every frame holds ``3 * n_atoms`` doubles.

    Parameters
    ----------
    file : str or os.PathLike
        Path the ``.npz`` archive is written to on close. Nothing is written
        if no frame was ever recorded.
    reportInterval : int
        Interval between recorded frames, in steps. Correlation functions
        resolve nothing faster than twice this interval times the time step.
    atom_indices : sequence of int or None, optional
        Atoms whose velocities are kept. Default is None, which keeps every
        atom.

    Raises
    ------
    TypeError
        If *reportInterval* or an atom index is not an integer.
    ValueError
        If *reportInterval* is not positive, *atom_indices* is empty,
        contains a duplicate, or a negative index.
    """

    def __init__(self, file: str | os.PathLike[str], reportInterval: int,
                 atom_indices: Sequence[int] | None = None) -> None:
        self._reportInterval = require_integer(
            reportInterval,
            name="reportInterval",
            minimum=1,
        )
        self._atom_indices: list[int] | None = None
        if atom_indices is not None:
            indices = [
                require_integer(
                    index,
                    name=f"atom_indices[{position}]",
                    minimum=0,
                )
                for position, index in enumerate(atom_indices)
            ]
            if not indices:
                raise ValueError("atom_indices must not be empty")
            if len(set(indices)) != len(indices):
                raise ValueError("atom_indices contains duplicate indices")
            self._atom_indices = indices
        self._file = os.fspath(file)
        self._times_ps: list[float] = []
        self._frames: list[npt.NDArray[np.float64]] = []
        self._masses: npt.NDArray[np.float64] | None = None
        self._kept: npt.NDArray[np.intp] | None = None
        self._closed = False

    def _check_integrator(self, simulation: app.Simulation) -> None:
        """
        Reject a simulation this reporter would silently mis-sample.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        """

    def _sample(self, simulation: app.Simulation, state: openmm.State,
                ) -> tuple[float, npt.NDArray[np.float64]]:
        """
        Take one frame of velocities for the kept atoms.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            The state OpenMM built for this report.

        Returns
        -------
        time_ps : float
            Simulation time of the frame, in picoseconds.
        velocities : numpy.ndarray
            Velocities of the kept atoms, in nm/ps.
        """
        raise NotImplementedError

    def _prepare(self, simulation: app.Simulation) -> None:
        """
        Resolve the atom selection and masses on the first report.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.

        Raises
        ------
        TypeError
            If the Simulation's integrator is the wrong kind for this
            reporter.
        ValueError
            If an atom index lies outside the System.
        """
        self._check_integrator(simulation)
        masses = _particle_masses_dalton(simulation.system)
        if self._atom_indices is None:
            kept = np.arange(len(masses), dtype=np.intp)
        else:
            out_of_range = [
                index for index in self._atom_indices
                if index >= len(masses)
            ]
            if out_of_range:
                raise ValueError(
                    f"atom_indices {out_of_range} lie outside the System "
                    f"with {len(masses)} particles"
                )
            kept = np.asarray(self._atom_indices, dtype=np.intp)
        self._kept = kept
        self._masses = masses[kept]

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """
        Record one frame of velocities.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            The state OpenMM built for this report.
        """
        if self._kept is None:
            self._prepare(simulation)
        assert self._kept is not None

        time_ps, velocities = self._sample(simulation, state)
        self._times_ps.append(time_ps)
        self._frames.append(velocities)

    def close(self) -> None:
        """Write the accumulated frames as one ``.npz`` archive, once."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if not self._frames:
            return
        assert self._kept is not None and self._masses is not None
        np.savez(
            self._file,
            times_ps=np.asarray(self._times_ps, dtype=np.float64),
            velocities_nm_per_ps=np.stack(self._frames),
            atom_indices=np.asarray(self._kept, dtype=np.int64),
            masses_dalton=np.asarray(self._masses, dtype=np.float64),
        )

    def __enter__(self) -> Self:
        """Return this reporter for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize the velocity archive when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort fallback for callers that did not close the reporter."""
        try:
            self.close()
        except Exception:
            pass


class VelocityArchiveReporter(_VelocityArchiveBase):
    """
    Record velocities from a classical run for correlation functions.

    This is the classical counterpart of :class:`RPMDVelocityReporter`, and
    it writes the identical archive, so the same
    :func:`velocity_autocorrelation` and :func:`vibrational_spectrum` read
    it.  That is the point of it: a classical vibrational density of states
    is the cheap baseline a ring-polymer spectrum is only interesting
    against.

    Record from a run whose thermostat is gentle or absent.  A strong
    Langevin friction damps the very dynamics a correlation function
    measures, which broadens every peak in the spectrum.

    Frames accumulate in memory until close, so pass *atom_indices*: a
    solvated system keeps ``3 * n_atoms`` doubles per frame otherwise, and a
    spectrum wants frames close enough together to resolve the fastest mode.

    Parameters
    ----------
    file : str or os.PathLike
        Path the ``.npz`` archive is written to on close. Nothing is written
        if no frame was ever recorded.
    reportInterval : int
        Interval between recorded frames, in steps. Correlation functions
        resolve nothing faster than twice this interval times the time step.
    atom_indices : sequence of int or None, optional
        Atoms whose velocities are kept. Default is None, which keeps every
        atom.

    Raises
    ------
    TypeError
        If *reportInterval* or an atom index is not an integer.
    ValueError
        If *reportInterval* is not positive, *atom_indices* is empty,
        contains a duplicate, or a negative index.
    """

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
            ``(steps, positions, velocities, forces, energies)``, asking for
            velocities alone.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, True, False, False)

    def _check_integrator(self, simulation: app.Simulation) -> None:
        """
        Refuse an RPMD integrator, whose Context is one copy.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.

        Raises
        ------
        TypeError
            If the integrator is an ``RPMDIntegrator``. Its Context holds a
            single copy of the system, so these velocities would be one
            bead's rather than the centroid's -- a spectrum that looks
            perfectly reasonable and is wrong.
        """
        if hasattr(simulation.integrator, "getNumCopies"):
            raise TypeError(
                "VelocityArchiveReporter cannot record an RPMD run: the "
                "Context holds one copy, not the centroid; use "
                "RPMDVelocityReporter"
            )

    def _sample(self, simulation: app.Simulation, state: openmm.State,
                ) -> tuple[float, npt.NDArray[np.float64]]:
        """
        Take the kept atoms' velocities from the reported state.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            State carrying this report's velocities.

        Returns
        -------
        time_ps : float
            Simulation time of the frame, in picoseconds.
        velocities : numpy.ndarray
            Velocities of the kept atoms, in nm/ps.
        """
        assert self._kept is not None
        velocities = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )[self._kept]
        time_ps = state.getTime().value_in_unit(unit.picosecond)
        return float(time_ps), np.array(velocities, dtype=np.float64)


class RPMDVelocityReporter(_VelocityArchiveBase):
    """
    Record centroid (bead-averaged) velocities for correlation functions.

    RPMD approximates a Kubo-transformed correlation function of operators
    linear in position or momentum by the corresponding centroid correlation
    function, so the centroid velocities are the raw material for velocity
    autocorrelation functions and vibrational spectra.  Frames accumulate in
    memory and are written as one ``.npz`` archive when the reporter is
    closed, which the ``run_openmm_rpmd_*`` drivers do on exit; read it back
    with :func:`velocity_autocorrelation` or :func:`vibrational_spectrum`.
    :class:`VelocityArchiveReporter` writes the identical archive from a
    classical run, which is the baseline to compare against.

    Record from a thermostat-off run (``apply_thermostat=False``): the PILE
    thermostat's friction and noise contaminate the very dynamics a
    correlation function is meant to measure.

    A spectrum from these velocities carries known ring-polymer artifacts:
    the free ring-polymer spring frequencies contaminate it near and above
    ``n_beads k_B T / hbar``, and resonances between them and physical modes
    can split or shift high-frequency peaks (Witt et al., J. Chem. Phys.
    130, 194510 (2009)).  Read high-frequency features with that in mind.

    Parameters
    ----------
    file : str or os.PathLike
        Path the ``.npz`` archive is written to on close. Nothing is written
        if no frame was ever recorded.
    reportInterval : int
        Interval between recorded frames, in steps. Correlation functions
        resolve nothing faster than twice this interval times the time step.
    atom_indices : sequence of int or None, optional
        Atoms whose centroid velocities are kept. Default is None, which
        keeps every atom -- for a solvated system that is a lot of memory,
        since every frame holds ``3 * n_atoms`` doubles until close.

    Raises
    ------
    TypeError
        If *reportInterval* or an atom index is not an integer.
    ValueError
        If *reportInterval* is not positive, *atom_indices* is empty,
        contains a duplicate, or a negative index.
    """

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
            requested: the bead velocities come from the integrator instead.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def _check_integrator(self, simulation: app.Simulation) -> None:
        """
        Require an RPMD integrator, the only source of bead velocities.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.

        Raises
        ------
        TypeError
            If the Simulation does not use an RPMD-style integrator.
        """
        if not hasattr(simulation.integrator, "getNumCopies"):
            raise TypeError(
                "RPMDVelocityReporter requires an RPMDIntegrator"
            )

    def _sample(self, simulation: app.Simulation, state: openmm.State,
                ) -> tuple[float, npt.NDArray[np.float64]]:
        """
        Average the kept atoms' velocities over the beads.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            Unused; the bead velocities come from the RPMD integrator.

        Returns
        -------
        time_ps : float
            Simulation time of the frame, in picoseconds.
        velocities : numpy.ndarray
            Centroid velocities of the kept atoms, in nm/ps.
        """
        assert self._kept is not None
        integrator = simulation.integrator
        n_beads = integrator.getNumCopies()
        time_ps: float | None = None
        mean_velocities: npt.NDArray[np.float64] | None = None
        for bead in range(n_beads):
            bead_state = integrator.getState(bead, getVelocities=True)
            velocities = bead_state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )[self._kept]
            if mean_velocities is None:
                time_ps = bead_state.getTime().value_in_unit(unit.picosecond)
                mean_velocities = np.array(velocities, dtype=np.float64)
            else:
                mean_velocities += velocities
        assert mean_velocities is not None and time_ps is not None
        return float(time_ps), mean_velocities / n_beads

def _read_velocity_archive(file: str | os.PathLike[str],
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a velocity archive back as validated bare arrays.

    Parameters
    ----------
    file : str or os.PathLike
        Archive written by :class:`VelocityArchiveReporter` or
        :class:`RPMDVelocityReporter`.

    Returns
    -------
    times_ps : numpy.ndarray
        Frame times in picoseconds, shaped ``(n_frames,)``, uniformly
        spaced and increasing.
    velocities : numpy.ndarray
        Centroid velocities in nm/ps, shaped ``(n_frames, n_atoms, 3)``.
    masses : numpy.ndarray
        Masses of the kept atoms in daltons, shaped ``(n_atoms,)``.

    Raises
    ------
    ValueError
        If the archive lacks a field, holds fewer than two frames, has
        mismatched shapes or non-finite values, or its frames are not
        uniformly spaced in time.
    """
    with np.load(os.fspath(file)) as archive:
        missing = [
            name
            for name in ("times_ps", "velocities_nm_per_ps", "masses_dalton")
            if name not in archive
        ]
        if missing:
            raise ValueError(
                f"velocity archive lacks field(s): {', '.join(missing)}"
            )
        times = np.asarray(archive["times_ps"], dtype=np.float64)
        velocities = np.asarray(
            archive["velocities_nm_per_ps"], dtype=np.float64
        )
        masses = np.asarray(archive["masses_dalton"], dtype=np.float64)

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("velocity archive must hold at least two frames")
    if velocities.shape != (len(times), len(masses), 3):
        raise ValueError(
            f"velocity archive shapes disagree: {len(times)} times, "
            f"velocities {velocities.shape}, {len(masses)} masses"
        )
    if not (np.isfinite(times).all() and np.isfinite(velocities).all()
            and np.isfinite(masses).all()):
        raise ValueError("velocity archive contains non-finite values")
    intervals = np.diff(times)
    if np.any(intervals <= 0.0) or not np.allclose(
        intervals, intervals[0], rtol=1.0e-6, atol=0.0
    ):
        raise ValueError(
            "velocity archive frames must be uniformly spaced in time"
        )
    return times, velocities, masses


def velocity_autocorrelation(file: str | os.PathLike[str], *,
                             max_time: unit.Quantity | float | None = None,
                             mass_weighted: bool = True,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the centroid velocity autocorrelation function from an archive.

    For operators linear in momentum, the centroid correlation function of a
    thermostat-off RPMD run is the ring-polymer approximation to the
    Kubo-transformed quantum correlation function, so this is the
    ``C_vv(t)`` that vibrational spectra and diffusion coefficients start
    from.  Each atom's Cartesian components are correlated by FFT with the
    unbiased ``1/(n-k)`` lag normalization and summed.

    Parameters
    ----------
    file : str or os.PathLike
        Archive written by :class:`VelocityArchiveReporter` or
        :class:`RPMDVelocityReporter`.
    max_time : openmm.unit.Quantity or float or None, optional
        Longest lag to keep. A bare number is read as picoseconds. Long lags
        average few frame pairs and are mostly noise, so a fraction of the
        run length is usual. Default is None, which keeps every lag.
    mass_weighted : bool, optional
        If True, weight each atom's term by its mass, giving ``C_vv`` in
        dalton nm^2/ps^2 -- the weighting under a vibrational density of
        states. If False, atoms are averaged unweighted, in nm^2/ps^2.
        Default is True.

    Returns
    -------
    times_ps : numpy.ndarray
        Lag times in picoseconds, starting at zero.
    vacf : numpy.ndarray
        The autocorrelation at each lag, in the units *mass_weighted*
        selects.

    Raises
    ------
    ValueError
        If the archive is malformed (see
        :class:`VelocityArchiveReporter`), or *max_time* is not positive and
        finite.
    """
    times, velocities, masses = _read_velocity_archive(file)
    dt = float(times[1] - times[0])
    n_frames = len(times)

    n_lags = n_frames
    if max_time is not None:
        max_time_ps = require_positive_finite_scalar_in_unit(
            max_time,
            unit.picosecond,
            name="max_time",
        )
        n_lags = min(n_frames, int(np.floor(max_time_ps / dt)) + 1)

    # Wiener-Khinchin: correlate each atom's Cartesian component by FFT,
    # zero-padded to double length so the circular correlation is linear.
    n_fft = 2 * n_frames
    spectra = np.fft.rfft(velocities, n=n_fft, axis=0)
    correlations = np.fft.irfft(
        spectra * np.conj(spectra), n=n_fft, axis=0
    )[:n_lags].real
    correlations /= (n_frames - np.arange(n_lags))[:, np.newaxis, np.newaxis]

    per_atom = correlations.sum(axis=2)
    if mass_weighted:
        vacf = per_atom @ masses
    else:
        vacf = per_atom.mean(axis=1)
    return np.arange(n_lags) * dt, vacf


def vibrational_spectrum(file: str | os.PathLike[str], *,
                         window: Literal["hann", "none"] = "hann",
                         max_time: unit.Quantity | float | None = None,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a vibrational density of states from a velocity archive.

    The spectrum is the cosine transform of the mass-weighted centroid
    velocity autocorrelation function, so a peak sits at each vibrational
    frequency the centroid dynamics carries.  No normalization is invented:
    the intensity is in dalton nm^2/ps, proportional to the vibrational
    density of states, and only relative heights are meaningful.

    Parameters
    ----------
    file : str or os.PathLike
        Archive written by :class:`VelocityArchiveReporter` or
        :class:`RPMDVelocityReporter`.
    window : {"hann", "none"}, optional
        Taper applied to the autocorrelation before transforming. The Hann
        window suppresses the ringing a truncated correlation function
        otherwise scatters around every peak. Default is ``"hann"``.
    max_time : openmm.unit.Quantity or float or None, optional
        Longest correlation lag transformed, which sets the frequency
        resolution to roughly ``1 / max_time``. A bare number is read as
        picoseconds. Default is None, which uses every lag.

    Returns
    -------
    frequencies_invcm : numpy.ndarray
        Frequency axis in reciprocal centimetres.
    intensities : numpy.ndarray
        Spectral intensity at each frequency, in dalton nm^2/ps.

    Raises
    ------
    ValueError
        If the archive is malformed, *max_time* is not positive and finite,
        or *window* is not a recognized name.

    Notes
    -----
    A ring-polymer archive carries high-frequency artifacts a classical one
    does not; see :class:`RPMDVelocityReporter` for what they are and where
    they sit.
    """
    if window not in ("hann", "none"):
        raise ValueError(f"unknown window {window!r}; use 'hann' or 'none'")
    times, vacf = velocity_autocorrelation(
        file,
        max_time=max_time,
        mass_weighted=True,
    )
    dt = float(times[1] - times[0])
    n_lags = len(vacf)

    tapered = vacf
    if window == "hann":
        # Descending half of an odd Hann window: exactly 1 at lag zero,
        # exactly 0 at the last lag.
        tapered = vacf * np.hanning(2 * n_lags - 1)[n_lags - 1:]

    # Cosine transform of the one-sided C(t), zero-padded fourfold so the
    # returned grid samples each peak smoothly.
    n_fft = 4 * n_lags
    intensities = 2.0 * dt * np.fft.rfft(tapered, n=n_fft).real
    frequencies_per_ps = np.fft.rfftfreq(n_fft, d=dt)

    # 1/ps to cm^-1: divide by c = 0.0299792458 cm/ps.
    speed_of_light_cm_per_ps = 1.0e2 * unit.SPEED_OF_LIGHT_C.value_in_unit(
        unit.meter / unit.picosecond
    )
    return frequencies_per_ps / speed_of_light_cm_per_ps, intensities
