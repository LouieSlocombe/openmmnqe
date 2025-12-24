import numpy as np
import openmm.unit as unit
from openmm import app

def _calculate_quantum_spread(integrator, atom_indices=None):
    """
    Calculates the root-mean-square distance of beads from the ring polymer centroid.
    This is a measure of quantum delocalization (quantum spread).

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        The integrator running the simulation.
    atom_indices : list of int, optional
        List of atom indices to calculate the spread for.
        If None, calculates for ALL atoms (can be memory intensive).

    Returns
    -------
    spreads : openmm.unit.Quantity (numpy array)
        An array of shape (n_selected_atoms,) containing the quantum Rg
        for each selected atom in nanometers.
    """
    num_beads = integrator.getNumCopies()
    all_bead_positions = []
    for i in range(num_beads):
        state = integrator.getState(copy=i, getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        if atom_indices is not None:
            pos = pos[atom_indices]
        all_bead_positions.append(pos)

    # Convert to numpy array for vector math
    coords = np.array(all_bead_positions)
    # Calculate Centroid (average position across beads)
    centroid = np.mean(coords, axis=0)
    # Calculate Squared Distance of each bead from the Centroid
    diff = coords - centroid
    sq_dist = np.sum(diff ** 2, axis=2)  # Sum x,y,z components -> (n_beads, n_atoms)
    # Average over beads (Mean Squared Displacement from Centroid)
    mean_sq_dist = np.mean(sq_dist, axis=0)
    quantum_rg = np.sqrt(mean_sq_dist)
    return quantum_rg * unit.nanometers



class RPMDQuantumSpreadReporter(object):
    """
    A Reporter class to log the quantum spread (delocalization) of specific atoms
    during an RPMD simulation.
    """

    def __init__(self, file, reportInterval, atom_indices, names=None):
        """
        Parameters
        ----------
        file : str
            Filename to write to.
        reportInterval : int
            The interval (in steps) at which to write frames.
        atom_indices : list of int
            The indices of the atoms to monitor (e.g., the transferring proton).
        names : list of str, optional
            Names for the columns (e.g., ["Proton_H1", "Donor_N"]).
            If None, uses indices.
        """
        self._reportInterval = reportInterval
        self._atom_indices = atom_indices
        self._out = open(file, 'w')

        # Header
        if names:
            header = "Step\t" + "\t".join([f"Rg_{n}(nm)" for n in names])
        else:
            header = "Step\t" + "\t".join([f"Rg_Atom{i}(nm)" for i in atom_indices])
        self._out.write(header + "\n")

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        # We need access to the integrator to get bead positions, not just the simulation state
        integrator = simulation.integrator

        # Calculate spreads using the helper function defined above
        # Note: This requires the helper function to be available or methodized
        spreads = _calculate_quantum_spread(integrator, self._atom_indices)

        # Write to file
        step = simulation.currentStep
        spread_values = spreads.value_in_unit(unit.nanometers)

        line = f"{step}"
        for val in spread_values:
            line += f"\t{val:.6f}"
        self._out.write(line + "\n")
        self._out.flush()

    def __del__(self):
        self._out.close()


class RPMDBeadReporter(object):
    """
    A custom reporter for OpenMM that saves the trajectory of EVERY individual
    bead in an RPMD simulation to separate PDB files.
    """

    def __init__(self, file_base_name, reportInterval, num_beads, topology):
        """
        args:
            file_base_name (str): Prefix for files (e.g., 'output' -> 'output_bead_0.pdb')
            reportInterval (int): How often to write frames (steps)
            num_beads (int): Number of beads in the RPMD integrator
            topology (Topology): The system topology
        """
        self._reportInterval = reportInterval
        self._num_beads = num_beads
        self._topology = topology
        self._next_frame_index = 0

        # Create a list of open file handles, one for each bead
        self._files = []
        for i in range(num_beads):
            filename = f"{file_base_name}_bead_{i}.pdb"
            f = open(filename, 'w')
            # Write the PDB Header for each file
            app.PDBFile.writeHeader(topology, f)
            self._files.append(f)

    def describeNextReport(self, simulation):
        """
        Tells the Simulation when the next report is due.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        """
        Called by the Simulation to generate the report.
        """
        # We must access the integrator specifically to get bead positions
        integrator = simulation.integrator

        # Loop through every bead
        for i in range(self._num_beads):
            # getState(bead_index, ...) is specific to RPMDIntegrator
            # Note: enforcePeriodicBox must match your system settings
            bead_state = integrator.getState(i, getPositions=True, enforcePeriodicBox=True)
            positions = bead_state.getPositions()

            # Write the frame (Model) to the specific bead's file
            app.PDBFile.writeModel(self._topology, positions, self._files[i], self._next_frame_index)

            # Flush periodically to ensure data is written to disk
            if self._next_frame_index % 10 == 0:
                self._files[i].flush()

        self._next_frame_index += 1

    def __del__(self):
        """
        Cleanup: Close all file handles when the reporter is destroyed.
        """
        for f in self._files:
            try:
                # Write footer before closing to ensure valid PDB syntax
                app.PDBFile.writeFooter(self._topology, f)
                f.close()
            except:
                pass


class RPMDCentroidReporter(object):
    """
    A custom reporter that calculates the centroid (average position) of all beads
    and writes it to a single PDB file.
    """

    def __init__(self, file_name, reportInterval, num_beads, topology):
        self._reportInterval = reportInterval
        self._num_beads = num_beads
        self._topology = topology
        self._next_frame_index = 0
        self._out = open(file_name, 'w')
        app.PDBFile.writeHeader(topology, self._out)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        integrator = simulation.integrator

        # Get first bead to initialize sum
        # We use asNumpy=True for vector efficiency, though OpenMM Quantities also support math.
        sum_pos = integrator.getState(0, getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)

        # Sum positions of remaining beads
        for i in range(1, self._num_beads):
            pos = integrator.getState(i, getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
            sum_pos += pos

        # Calculate Average
        centroid_pos = sum_pos / self._num_beads

        # Write to file
        app.PDBFile.writeModel(self._topology, centroid_pos, self._out, self._next_frame_index)
        self._next_frame_index += 1

        if self._next_frame_index % 10 == 0:
            self._out.flush()

    def __del__(self):
        try:
            app.PDBFile.writeFooter(self._topology, self._out)
            self._out.close()
        except:
            pass
