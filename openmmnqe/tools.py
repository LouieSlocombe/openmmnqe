import numpy as np
import openmm.unit as unit
from openmm import openmm, app
from scipy import constants


def zero_velocities(n_atoms):
    """
    Generates a list of zero velocity vectors for a given number of atoms.

    Parameters
    ----------
    n_atoms : int
        The number of atoms for which zero velocity vectors are to be created.

    Returns
    -------
    list of openmm.Vec3
        A list of zero velocity vectors, each scaled by the unit of nanometer/picosecond.
    """
    return [openmm.Vec3(0, 0, 0) for _ in range(n_atoms)] * (unit.nanometer / unit.picosecond)


def write_multimodel_pdb(topology, positions, fh, model_index):
    """
    Writes a single model to a multi-model PDB file.

    This function appends a model to an existing PDB file, allowing the creation
    of a multi-model PDB file. Each model is identified by a unique index.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology of the system to be written.
    positions : openmm.unit.Quantity
        The atomic positions to be written, with units of length.
    fh : file-like object
        An open file handle where the PDB model will be written.
    model_index : int
        The index of the model to be written, used to distinguish models in the PDB file.

    Returns
    -------
    None
    """
    app.PDBFile.writeModel(topology, positions, fh, modelIndex=model_index)


def centroid_positions(simulation, n_atoms, n_beads):
    """
    Computes the centroid positions of atoms across multiple beads in a simulation.

    This function calculates the average positions of atoms over a specified number
    of beads in a ring-polymer molecular dynamics (RPMD) simulation.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The OpenMM simulation object containing the integrator and system state.
    n_atoms : int
        The number of atoms in the system.
    n_beads : int
        The number of beads in the RPMD simulation.

    Returns
    -------
    list of openmm.Vec3
        A list of centroid positions for each atom, with units of nanometers.
    """
    acc = np.zeros((n_atoms, 3), dtype=float)  # Initialize accumulator for positions.
    for b in range(n_beads):
        state = simulation.integrator.getState(b, getPositions=True)  # Get state for bead `b`.
        r = state.getPositions(asNumpy=True)  # Extract positions as a NumPy array.
        acc += r.value_in_unit(unit.nanometer)  # Accumulate positions in nanometers.
    acc /= n_beads  # Compute the average positions across all beads.
    return [openmm.Vec3(*acc[i]) for i in range(n_atoms)] * unit.nanometer  # Return centroid positions.


def get_thermal_de_broglie_wavelength(mass, temperature):
    """
    Calculates the thermal de Broglie wavelength for a given mass and temperature.

    The thermal de Broglie wavelength is a quantum mechanical property that
    characterizes the wave-like behavior of particles at a given temperature.

    Parameters
    ----------
    mass : openmm.unit.Quantity or float
        The mass of the particle. If a `Quantity`, it should have units of daltons.
        If a float, it is assumed to be in atomic mass units (amu).
    temperature : openmm.unit.Quantity or float
        The temperature of the system. If a `Quantity`, it should have units of kelvin.
        If a float, it is assumed to be in kelvin.

    Returns
    -------
    openmm.unit.Quantity
        The thermal de Broglie wavelength with units of meters.
    """
    if unit.is_quantity(mass):
        mass_amu = mass.value_in_unit(unit.dalton)
    else:
        mass_amu = mass

    if unit.is_quantity(temperature):
        temp_k = temperature.value_in_unit(unit.kelvin)
    else:
        temp_k = temperature

    mass_kg = mass_amu * constants.atomic_mass

    h = constants.h
    k_b = constants.k
    lambda_meters = h / np.sqrt(2 * np.pi * mass_kg * k_b * temp_k)
    return lambda_meters * unit.meter


def init_beads_scaled(simulation, positions, n_beads, temperature, scale_factor=0.1):
    """
    Initializes bead positions for a ring-polymer molecular dynamics (RPMD) simulation.

    This function perturbs the initial positions of atoms in the system to create
    multiple beads, scaled by the thermal de Broglie wavelength of each atom.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The OpenMM simulation object containing the system and integrator.
    positions : openmm.unit.Quantity or np.ndarray
        The initial atomic positions. If not a Quantity, it is assumed to be in nanometers.
    n_beads : int
        The number of beads to initialize for the RPMD simulation.
    temperature : openmm.unit.Quantity
        The temperature of the system, used to calculate the thermal de Broglie wavelength.
    scale_factor : float, optional
        A scaling factor applied to the thermal wavelength perturbation. Default is 0.1.

    Returns
    -------
    None
    """
    system = simulation.system
    n_atoms = system.getNumParticles()

    # Get the masses of all particles in daltons.
    masses_val = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_atoms)])
    masses_quantity = masses_val * unit.dalton

    # Calculate the thermal de Broglie wavelength for each particle.
    lambdas = get_thermal_de_broglie_wavelength(masses_quantity, temperature)
    lambdas_nm = lambdas.value_in_unit(unit.nanometer)

    # Ensure positions are in the correct unit (nanometers).
    if not unit.is_quantity(positions):
        positions = positions * unit.nanometer
    pos0 = positions.value_in_unit(unit.nanometer)

    # Initialize a random number generator with a fixed seed.
    rng = np.random.default_rng(0)

    # Log information about the thermal wavelengths.
    print(f"Initializing {n_beads} beads scaled by thermal wavelengths...")
    print(f"Max Lambda (lightest atom): {np.max(lambdas_nm):.4f} nm")
    print(f"Min Lambda (heaviest atom): {np.min(lambdas_nm):.4f} nm")

    # Perturb the positions for each bead.
    for b in range(n_beads):
        noise = rng.normal(size=(n_atoms, 3)) * lambdas_nm[:, np.newaxis] * scale_factor
        bead_pos = pos0 + noise
        simulation.integrator.setPositions(b, bead_pos * unit.nanometer)

    # Set the velocities of the system to match the target temperature.
    simulation.context.setVelocitiesToTemperature(temperature)


def init_beads(modeller, simulation, n_beads, perturb=0.002):
    """
    Initializes bead positions and velocities for a ring-polymer molecular dynamics (RPMD) simulation.

    This function perturbs the initial positions of atoms to create multiple beads
    and sets their velocities to zero.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM modeller object containing the system topology and positions.
    simulation : openmm.app.Simulation
        The OpenMM simulation object containing the integrator and system state.
    n_beads : int
        The number of beads to initialize for the RPMD simulation.
    perturb : float, optional
        The magnitude of the random perturbation applied to the initial positions.
        Default is 0.002.

    Returns
    -------
    None
    """
    rng = np.random.default_rng(0)
    pos0 = modeller.positions
    n_atoms = len(pos0)
    for b in range(n_beads):
        jiggle = perturb * rng.normal(size=(n_atoms, 3))
        bead_pos = [openmm.Vec3(p.x + dx, p.y + dy, p.z + dz)
                    for p, (dx, dy, dz) in zip(pos0, jiggle)]
        simulation.integrator.setPositions(b, bead_pos * unit.nanometer)
        simulation.integrator.setVelocities(b, zero_velocities(n_atoms))
