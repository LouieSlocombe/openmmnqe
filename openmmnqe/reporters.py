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
for one target atom without constructing it directly.
"""
from numbers import Integral

import numpy as np
import openmm.unit as unit

from openmm import app

from .tools import centroid_positions


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
    num_beads = integrator.getNumCopies()
    all_bead_positions = []
    for i in range(num_beads):
        state = integrator.getState(copy=i, getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        if atom_indices is not None:
            pos = pos[atom_indices]
        all_bead_positions.append(pos)

    coords = np.array(all_bead_positions)
    centroid = np.mean(coords, axis=0)
    diff = coords - centroid
    sq_dist = np.sum(diff ** 2, axis=2)  # over x, y, z -> (n_beads, n_atoms)
    mean_sq_dist = np.mean(sq_dist, axis=0)
    quantum_rg = np.sqrt(mean_sq_dist)
    return quantum_rg * unit.nanometers


class RPMDQuantumSpreadReporter(object):
    """
    Log the quantum spread of selected atoms during an RPMD simulation.

    One tab-separated column per monitored atom, holding the radius of
    gyration of its ring polymer as computed by
    :func:`_calculate_quantum_spread`.

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
    """

    def __init__(self, file, reportInterval, atom_indices, names=None):
        if reportInterval <= 0:
            raise ValueError("reportInterval must be positive")
        if names is not None and len(names) != len(atom_indices):
            raise ValueError("names must contain one entry per atom index")
        self._reportInterval = reportInterval
        self._atom_indices = atom_indices
        self._out = open(file, 'w')

        if names:
            header = "Step\t" + "\t".join([f"Rg_{n}(nm)" for n in names])
        else:
            header = "Step\t" + "\t".join([f"Rg_Atom{i}(nm)" for i in atom_indices])
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
        Write the quantum spread of the monitored atoms.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation this reporter is attached to.
        state : openmm.State
            Unused; the bead positions come from the RPMD integrator.
        """
        integrator = simulation.integrator
        spreads = _calculate_quantum_spread(integrator, self._atom_indices)

        step = simulation.currentStep
        spread_values = spreads.value_in_unit(unit.nanometers)

        line = f"{step}"
        for val in spread_values:
            line += f"\t{val:.6f}"
        self._out.write(line + "\n")
        self._out.flush()

    def __del__(self):
        """Close the output file."""
        out = getattr(self, "_out", None)
        if out is not None:
            out.close()


def track_rpmd_atom_expansion(simulation, atom_index, file, report_interval,
                              name=None):
    r"""
    Track one atom's ring-polymer radius of gyration during an RPMD run.

    The returned reporter is appended to ``simulation.reporters``. At every
    reporting interval it writes the current simulation step and

    .. math::

        R_g = \sqrt{\frac{1}{P}\sum_{i=1}^{P}
              \left|\mathbf{r}_i-\bar{\mathbf{r}}\right|^2}

    where ``P`` is the number of beads, ``r_i`` is the target atom's position
    in bead ``i``, and ``r_bar`` is its ring-polymer centroid. The output is a
    tab-separated time series with the radius in nanometres.

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
        )
        step_rpmd(simulation, 10_000)
    """
    if isinstance(atom_index, bool) or not isinstance(atom_index, Integral):
        raise TypeError("atom_index must be an integer")
    if atom_index < 0:
        raise ValueError("atom_index must be non-negative")

    reporter = RPMDQuantumSpreadReporter(
        file=file,
        reportInterval=report_interval,
        atom_indices=[int(atom_index)],
        names=None if name is None else [name],
    )
    simulation.reporters.append(reporter)
    return reporter


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
