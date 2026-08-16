"""Reporters for ring-polymer molecular dynamics runs.

OpenMM's own reporters see only the context, which for an ``RPMDIntegrator``
holds a single copy of the system rather than the ring polymer.  Anything that
needs the beads themselves -- their spread, their individual trajectories, or
their centroid -- has to ask the integrator, which is what these three
reporters do.  They are attached by the ``run_openmm_rpmd_*`` drivers in
:mod:`openmmnqe.openmm`.

All three follow OpenMM's reporter protocol: ``describeNextReport`` says when
the next report is due and what state it needs, and ``report`` writes it.
Use :func:`track_rpmd_atom_expansion` to attach the quantum-spread reporter
for one target atom without constructing it directly, and
:func:`plot_rpmd_atom_expansion` to plot the result against a centroid
atom-pair distance or a supplied reference-path progress coordinate.
"""
from numbers import Integral

import numpy as np
import openmm.unit as unit

from openmm import app

from .tools import centroid_positions


_SPREAD_METRICS = {"rms", "mean"}


def _bead_coordinates(integrator, atom_indices=None):
    """Return bead coordinates in nanometres, optionally selecting atoms."""
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


def _spread_from_coordinates(coordinates, metric):
    """Calculate an RMS or mean bead-centroid radius from coordinates."""
    centroid = np.mean(coordinates, axis=0)
    radii = np.linalg.norm(coordinates - centroid, axis=2)
    if metric == "rms":
        values = np.sqrt(np.mean(radii ** 2, axis=0))
    else:
        values = np.mean(radii, axis=0)
    return values * unit.nanometers


def _calculate_quantum_spread(integrator, atom_indices=None):
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


def _calculate_bead_expansion(integrator, atom_indices=None):
    r"""Compute the mean bead-centroid distance for selected atoms.

    The proton ring-polymer degree of expansion is

    .. math::

        \Delta |r| = \frac{1}{P}\sum_{i=1}^{P}
        \left|\mathbf{r}_i-\bar{\mathbf{r}}\right|.

    Unlike :func:`_calculate_quantum_spread`, this is a mean radius rather
    than a root-mean-square radius.
    """
    coordinates = _bead_coordinates(integrator, atom_indices)
    return _spread_from_coordinates(coordinates, metric="mean")


def _simulation_bead_coordinates(simulation, atom_indices):
    """Return selected bead coordinates and their periodic box."""
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
            displacement = positions - reference
            for axis in (2, 1, 0):
                displacement -= box[axis] * np.round(
                    displacement[:, axis:axis + 1] / box[axis][axis]
                )
            positions = reference + displacement
        coordinates.append(positions)

    return np.asarray(coordinates), box


def _validate_metric(metric):
    if metric not in _SPREAD_METRICS:
        choices = ", ".join(sorted(_SPREAD_METRICS))
        raise ValueError(f"metric must be one of: {choices}")


def _validate_atom_indices(atom_indices):
    """Validate and normalise selected zero-based atom indices."""
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


def _validate_distance_pairs(distance_pairs):
    """Validate and normalise zero-based atom-index pairs."""
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


def _validate_observable_indices(atom_indices, distance_pairs, n_atoms=None):
    """Validate expansion atoms and distance pairs against a topology size."""
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


def _calculate_report_observables(simulation, atom_indices, metric,
                                  distance_pairs):
    """Calculate expansion and centroid distances in one bead-state pass."""
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


class RPMDQuantumSpreadReporter(object):
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

    def __init__(self, file, reportInterval, atom_indices, names=None,
                 metric="rms", distance_pairs=None, distance_names=None):
        if reportInterval <= 0:
            raise ValueError("reportInterval must be positive")
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
        self._reportInterval = reportInterval
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
        self._out = open(file, 'w')
        self._out.write(header + "\n")

    def describeNextReport(self, simulation):
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

    def report(self, simulation, state):
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

    def __del__(self):
        """Close the output file."""
        out = getattr(self, "_out", None)
        if out is not None:
            out.close()


def track_rpmd_atom_expansion(simulation, atom_index, file, report_interval,
                              name=None, metric="rms", distance_pairs=None,
                              distance_names=None):
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


def _read_expansion_log(file):
    """Read a tab-separated expansion reporter log."""
    with open(file) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        has_data = any(line.strip() for line in handle)
    if len(header) < 2 or header[0] != "Step":
        raise ValueError("expansion log must start with a Step column")
    if len(set(header)) != len(header):
        raise ValueError("expansion log contains duplicate column names")
    if not has_data:
        raise ValueError("expansion log contains no data rows")

    try:
        values = np.loadtxt(file, delimiter="\t", skiprows=1, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"could not parse expansion log {file!s}") from exc
    if values.shape[1] != len(header):
        raise ValueError("expansion log rows do not match its header")
    return header, values


def _select_log_columns(header, requested, prefixes, description):
    available = [
        name for name in header
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if requested is None:
        selected = available
    elif isinstance(requested, str):
        selected = [requested]
    else:
        selected = list(requested)

    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(
            f"unknown {description} column(s): {', '.join(missing)}"
        )
    if not selected and description == "expansion":
        raise ValueError("expansion log contains no expansion columns")
    return selected


def _column_label(column):
    """Turn a reporter column name into a compact legend label."""
    label = column
    for prefix in ("Expansion_", "Rg_", "Distance_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    if label.endswith("(nm)"):
        label = label[:-4]
    return label


def _average_by_progress(progress, values, progress_bins=None):
    """Sort path samples and average rows sharing a progress group."""
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


def plot_rpmd_atom_expansion(file, *, expansion_columns=None,
                             distance_columns=None, path_progress=None,
                             progress_bins=None, length_unit="nanometer",
                             filename=None, show=False):
    """Plot an RPMD atom expansion against distance or path progress.

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


class RPMDBeadReporter(object):
    """
    Write the trajectory of every individual bead to its own PDB file.

    Each bead of the ring polymer gets a separate file, so the beads can be
    inspected one by one rather than only through their centroid.

    Parameters
    ----------
    file_base_name : str
        Prefix for the output files: ``'output'`` gives
        ``'output_bead_0.pdb'``, ``'output_bead_1.pdb'`` and so on.
    reportInterval : int
        Interval between reports, in steps.
    num_beads : int
        Number of beads in the RPMD integrator.
    topology : openmm.app.Topology
        Topology written into the PDB headers and models.
    """

    def __init__(self, file_base_name, reportInterval, num_beads, topology):
        if reportInterval <= 0:
            raise ValueError("reportInterval must be positive")
        if num_beads <= 0:
            raise ValueError("num_beads must be positive")
        self._reportInterval = reportInterval
        self._num_beads = num_beads
        self._topology = topology
        self._next_frame_index = 0

        self._files = []
        for i in range(num_beads):
            filename = f"{file_base_name}_bead_{i}.pdb"
            f = open(filename, 'w')
            app.PDBFile.writeHeader(topology, f)
            self._files.append(f)

    def describeNextReport(self, simulation):
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

    def report(self, simulation, state):
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

        for i in range(self._num_beads):
            # getState(bead_index, ...) is specific to RPMDIntegrator.
            bead_state = integrator.getState(i, getPositions=True, enforcePeriodicBox=True)
            positions = bead_state.getPositions()

            app.PDBFile.writeModel(self._topology, positions, self._files[i], self._next_frame_index)

            # Flushing every frame would cost more than it buys with one file
            # per bead.
            if self._next_frame_index % 10 == 0:
                self._files[i].flush()

        self._next_frame_index += 1

    def __del__(self):
        """Write the PDB footers and close every bead file."""
        for f in getattr(self, "_files", []):
            try:
                # The footer has to go in before the close, or the file is not
                # valid PDB.
                app.PDBFile.writeFooter(self._topology, f)
                f.close()
            except Exception:
                pass


class RPMDCentroidReporter(object):
    """
    Write the centroid of the ring polymer to a single PDB file.

    The centroid trajectory is the classical-looking one: it is what the
    ring polymer's centre of mass does, and the trajectory to analyse when
    the beads themselves are not of interest.

    Parameters
    ----------
    file_name : str
        Path to write the centroid trajectory to.
    reportInterval : int
        Interval between reports, in steps.
    num_beads : int
        Number of beads in the RPMD integrator.
    topology : openmm.app.Topology
        Topology written into the PDB header and models.
    """

    def __init__(self, file_name, reportInterval, num_beads, topology):
        if reportInterval <= 0:
            raise ValueError("reportInterval must be positive")
        if num_beads <= 0:
            raise ValueError("num_beads must be positive")
        self._reportInterval = reportInterval
        self._num_beads = num_beads
        self._topology = topology
        self._next_frame_index = 0
        self._out = open(file_name, 'w')
        app.PDBFile.writeHeader(topology, self._out)

    def describeNextReport(self, simulation):
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

    def report(self, simulation, state):
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
            self._topology.getNumAtoms(),
            self._num_beads,
        )

        app.PDBFile.writeModel(self._topology, centroid_pos, self._out, self._next_frame_index)
        self._next_frame_index += 1

        if self._next_frame_index % 10 == 0:
            self._out.flush()

    def __del__(self):
        """Write the PDB footer and close the output file."""
        try:
            app.PDBFile.writeFooter(self._topology, self._out)
            self._out.close()
        except Exception:
            pass
