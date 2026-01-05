import re
from typing import Dict, Sequence, Any, List, Union, Literal, Optional

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


def count_dna_and_estimate_charge(topology):
    """
    Counts the number of DNA residues in a topology and estimates the total charge.

    This function identifies DNA residues in the given topology based on their residue names
    and calculates the total charge of the DNA. Each DNA residue is assumed to contribute
    a charge of -1 e.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology object containing the system's residues.

    Returns
    -------
    int
        The estimated total charge of the DNA in the topology.
        Negative value indicates the total charge contributed by the DNA residues.
    """
    dna_residue_names = {
        "DA", "DC", "DG", "DT",  # internal
        "DA5", "DC5", "DG5", "DT5",  # 5'-terminal
        "DA3", "DC3", "DG3", "DT3",  # 3'-terminal
    }

    num_dna_residues = 0

    for residue in topology.residues():
        if residue.name.strip() in dna_residue_names:
            num_dna_residues += 1

    # Estimate: -1 e per nucleotide
    estimated_charge = -num_dna_residues

    return estimated_charge


def deuterate_system(modeller, system, option='all', target_resname=None):
    """
    Replaces hydrogen atoms with deuterium in a molecular system.

    This function modifies the masses of hydrogen atoms in the system to the mass of deuterium
    based on the specified option. It supports deuteration of all hydrogens, or specific subsets
    such as water, protein, DNA, RNA, nucleic acids, or a specific ligand.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM modeller object containing the system topology and positions.
    system : openmm.System
        The OpenMM system object to be modified.
    option : str, optional
        Specifies the subset of the system to deuterate. Options include:
        'all', 'water', 'protein', 'dna', 'rna', 'nucleic', or 'ligand'. Default is 'all'.
    target_resname : str, optional
        The residue name of the ligand to deuterate. Required if `option` is 'ligand'.

    Raises
    ------
    ValueError
        If `option` is 'ligand' and `target_resname` is not provided.
        If `option` is not one of the supported values.

    Notes
    -----
    - If `option` is 'all', all hydrogen atoms in the system are deuterated.
    - If `option` is 'ligand' and no residues match `target_resname`, a warning is printed.
    - If no residues match the specified `option`, a warning is printed.

    Returns
    -------
    None
    """
    deuterium_mass = app.element.deuterium.mass

    # Define residue sets for different options
    protein_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
        'TYR', 'VAL', 'HID', 'HIE', 'HIP', 'CYX', 'LYN', 'ASH', 'GLH',
        'ACE', 'NME', 'NAC'
    }

    dna_residues = {
        'DA', 'DC', 'DG', 'DT',
        'DA5', 'DC5', 'DG5', 'DT5',  # 5' terminals
        'DA3', 'DC3', 'DG3', 'DT3'  # 3' terminals
    }

    rna_residues = {
        'A', 'C', 'G', 'U',
        'RA', 'RC', 'RG', 'RU',  # Common in Amber force fields
        'A5', 'C5', 'G5', 'U5',  # 5' terminals
        'A3', 'C3', 'G3', 'U3'  # 3' terminals
    }

    water_residues = {'HOH', 'H2O', 'TIP3', 'WAT', 'SOL'}

    nucleic_residues = dna_residues.union(rna_residues)

    # Determine target residues based on the option
    target_residues = set()
    if option == 'all':
        pass
    elif option == 'water':
        target_residues = water_residues
    elif option == 'protein':
        target_residues = protein_residues
    elif option == 'dna':
        target_residues = dna_residues
    elif option == 'rna':
        target_residues = rna_residues
    elif option == 'nucleic':
        target_residues = nucleic_residues
    elif option == 'ligand':
        if target_resname is None:
            raise ValueError("If option is 'ligand', you must provide a 'target_resname'.")
        target_residues = {target_resname}
    else:
        raise ValueError("Option must be 'all', 'water', 'protein', 'dna', 'rna', 'nucleic', or 'ligand'")

    # Deuterate all hydrogens if option is 'all'
    if option == 'all':
        for atom in modeller.topology.atoms():
            if atom.element and atom.element.symbol == 'H':
                system.setParticleMass(atom.index, deuterium_mass)
    else:
        # Deuterate hydrogens in the specified residue set
        found_target = False
        for residue in modeller.topology.residues():
            if residue.name in target_residues:
                found_target = True
                for atom in residue.atoms():
                    if atom.element and atom.element.symbol == 'H':
                        system.setParticleMass(atom.index, deuterium_mass)

        # Print warnings if no matching residues are found
        if not found_target and option == 'ligand':
            print(f"Warning: No ligand named '{target_resname}' was found.")
        elif not found_target and option != 'all':
            print(f"Warning: No residues matching option '{option}' were found.")


def get_atoms_in_residue(pdb_file_path, residue_index, chain_id=None):
    """
    Retrieves the atom indices of a specific residue in a PDB file.

    This function reads a PDB file, identifies the specified residue by its index
    and optionally its chain ID, and returns the indices of all atoms in that residue.

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file to be read.
    residue_index : int
        The index of the residue whose atom indices are to be retrieved.
    chain_id : str, optional
        The ID of the chain containing the residue. If None, the residue is
        searched in the global topology. Default is None.

    Returns
    -------
    list of int or None
        A list of atom indices in the specified residue. Returns None if the
        residue or chain is not found, or if the residue index is out of bounds.

    Notes
    -----
    - If `chain_id` is provided, the residue is searched within the specified chain.
    - If `chain_id` is None, the residue is searched in the global topology.
    - Prints error messages if the chain or residue index is invalid.
    """
    pdb = app.PDBFile(pdb_file_path)
    topology = pdb.topology
    if chain_id is not None:
        found_chain = None
        for chain in topology.chains():
            if chain.id == chain_id:
                found_chain = chain
                break

        if found_chain is None:
            available_chains = [c.id for c in topology.chains()]
            print(f"Error: Chain '{chain_id}' not found. Available chains: {available_chains}")
            return None

        residues = list(found_chain.residues())
        if residue_index < 0 or residue_index >= len(residues):
            print(f"Error: Residue index {residue_index} is out of bounds for Chain {chain_id}.")
            print(f"Chain {chain_id} contains {len(residues)} residues.")
            return None

        target_residue = residues[residue_index]
        print(f"Looking in Chain {chain_id}, Residue Index {residue_index}...")

    else:
        residues = list(topology.residues())

        if residue_index < 0 or residue_index >= len(residues):
            print(f"Error: Residue index {residue_index} is out of bounds.")
            print(f"The file contains {len(residues)} residues (indices 0 to {len(residues) - 1}).")
            return None

        target_residue = residues[residue_index]
        print(f"Looking in global topology, Residue Index {residue_index}...")

    atom_indices = [atom.index for atom in target_residue.atoms()]

    print(
        f"Successfully retrieved residue: {target_residue.name} (Chain: {target_residue.chain.id}, Index: {target_residue.index}, PDB ID: {target_residue.id})")
    return atom_indices


def set_adqtb_particle_types_by_element(
        integrator: Any,
        *,
        topology: Optional[Any] = None,
        system: Optional[Any] = None,
        particle_elements: Optional[Sequence[Any]] = None,
        start_type: int = 0,
        unknown_symbol: str = "X",
) -> Dict[str, int]:
    """
    Assign OpenMM adQTB (QTBIntegrator) particle types so that all particles with the same
    chemical element share the same integer type.

    IMPORTANT: call this BEFORE creating a Context/Simulation; particle types are fixed at
    Context creation time for QTBIntegrator.

    Parameters
    ----------
    integrator
        An OpenMM QTBIntegrator (adQTB). Must provide setParticleType(index, type).
    topology
        openmm.app.Topology used to infer element per particle (atom.element), if
        particle_elements is not provided.
    system
        openmm.System (optional) used to sanity-check that the number of particles matches.
    particle_elements
        Optional explicit per-particle element spec (length == system.getNumParticles()).
        Each entry can be an openmm.app.element.Element, a symbol string like "C", or None.
        Use this if your System has extra particles not present in the Topology (e.g., Drudes).
    start_type
        First type id to use (default 0).
    unknown_symbol
        Symbol to use when an element is missing/None (all such particles share a type).

    Returns
    -------
    element_to_type
        Mapping from element symbol (e.g., "H", "C") to assigned integer type id.
    """

    if not hasattr(integrator, "setParticleType"):
        raise TypeError(
            "integrator must support setParticleType(index, type); expected an OpenMM QTBIntegrator."
        )

    def _sym_and_Z(el: Any) -> tuple[str, int]:
        # OpenMM Element has .symbol and .atomic_number; allow strings too.
        if el is None:
            return unknown_symbol, 10 ** 9
        sym = getattr(el, "symbol", None)
        if sym is None:
            sym = str(el)
        Z = getattr(el, "atomic_number", 10 ** 9)
        try:
            Z = int(Z)
        except Exception:
            Z = 10 ** 9
        return sym, Z

    # Infer per-particle elements
    if particle_elements is None:
        if topology is None:
            raise ValueError("Provide topology or particle_elements.")
        atoms = list(topology.atoms())
        particle_elements = [a.element for a in atoms]

    n = len(particle_elements)

    # Optional sanity check against the System particle count
    if system is not None:
        n_sys = system.getNumParticles()
        if n_sys != n:
            raise ValueError(
                f"Element list has length {n}, but System has {n_sys} particles. "
                "If your System includes extra particles (Drudes/virtual sites/etc.), "
                "pass particle_elements with one entry per System particle."
            )

    # Build a stable symbol -> type_id mapping (sorted by atomic number, then symbol)
    symbol_to_Z: dict[str, int] = {}
    for el in particle_elements:
        sym, Z = _sym_and_Z(el)
        symbol_to_Z.setdefault(sym, Z)

    ordered_symbols = sorted(symbol_to_Z.items(), key=lambda kv: (kv[1], kv[0]))
    element_to_type = {sym: start_type + i for i, (sym, _) in enumerate(ordered_symbols)}

    # Assign types to particles
    for idx, el in enumerate(particle_elements):
        sym, _ = _sym_and_Z(el)
        integrator.setParticleType(idx, int(element_to_type[sym]))

    return element_to_type


_VMD_PICK_RE = re.compile(
    r"^\s*([A-Za-z]+)\s*(-?\d+)\s*([A-Za-z]?)\s*:\s*([A-Za-z0-9'_*]+)\s*$"
)


def atom_indices_from_vmd_picks(
        modeller,
        picks: List[str],
        *,
        match_mode: Literal["unique", "first", "all"] = "unique",
        chain_id: Optional[str] = None,
) -> List[Union[int, List[int]]]:
    """
    Maps VMD pick strings to OpenMM Topology atom indices.

    This function parses VMD-style atom selection strings (e.g., "HIE258:CD2")
    and returns the corresponding atom indices from an OpenMM topology.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller object whose topology will be searched.
    picks : list of str
        VMD pick strings specifying residue and atom names.
        Format: "RESNAME<RESID>[INSERTION_CODE]:ATOMNAME"
        Examples: ["HIE258:CD2", "ALA12:CA", "HIE258A:CD2"]
    match_mode : {"unique", "first", "all"}, optional
        Controls behavior when multiple atoms match a pick string:
        - "unique": Requires exactly one match; raises ValueError otherwise.
        - "first": Returns the first matching atom index if multiple matches exist.
        - "all": Returns a list of all matching indices for each pick.
        Default is "unique".
    chain_id : str or None, optional
        If provided, restricts the search to residues in the specified chain.
        Default is None (searches all chains).

    Returns
    -------
    list of int or list of list of int
        Atom indices corresponding to each pick string. If match_mode is "all",
        returns a list of lists.
    """
    topo = modeller.topology

    # Build lookup: (chain_id_or_None, resname, resid_str, atomname) -> [atom.index,...]
    lookup = {}
    for atom in topo.atoms():
        res = atom.residue
        key = (res.chain.id, res.name, str(res.id), atom.name)
        lookup.setdefault(key, []).append(atom.index)

    out: List[Union[int, List[int]]] = []

    for s in picks:
        m = _VMD_PICK_RE.match(s)
        if not m:
            raise ValueError(
                f"Pick '{s}' is malformed; expected like 'HIE258:CD2' (optionally 'HIE258A:CD2')."
            )

        resname, resid_num, ins_code, atomname = m.groups()
        resid_str = f"{resid_num}{ins_code or ''}"

        # collect matches across chains unless chain_id specified
        matches: List[int] = []
        if chain_id is not None:
            matches = lookup.get((chain_id, resname, resid_str, atomname), [])
        else:
            for (ch, rn, rid, an), idxs in lookup.items():
                if rn == resname and rid == resid_str and an == atomname:
                    matches.extend(idxs)

        if not matches:
            msg = f"No atom matches pick '{s}' -> (resname='{resname}', resid='{resid_str}', atomname='{atomname}')"
            if chain_id is not None:
                msg += f" in chain '{chain_id}'"
            msg += "."
            raise ValueError(msg)

        if match_mode == "all":
            out.append(matches)
        elif match_mode == "first":
            out.append(matches[0])
        elif match_mode == "unique":
            if len(matches) != 1:
                raise ValueError(
                    f"Pick '{s}' matched {len(matches)} atoms; expected exactly 1. "
                    f"This usually means multiple chains/segments share the same residue id/name. "
                    f"Use chain_id='A' (etc.), or match_mode='first'/'all'."
                )
            out.append(matches[0])
        else:
            raise ValueError(f"Unknown match_mode '{match_mode}'.")

    return out


def atom_indices_to_plumed(atom_indices: List[int]) -> List[int]:
    """
    Converts OpenMM atom indices to PLUMED atom indices.

    PLUMED uses 1-based indexing, while OpenMM uses 0-based indexing.
    This function adds 1 to each atom index to convert between the two conventions.

    Parameters
    ----------
    atom_indices : list of int
        A list of 0-based atom indices from OpenMM.

    Returns
    -------
    list of int
        A list of 1-based atom indices compatible with PLUMED.
    """
    return [idx + 1 for idx in atom_indices]
