"""Assorted helpers for setting up and interrogating an OpenMM system.

What collects here is the small stuff the simulation stages in
:mod:`openmmnqe.openmm` and the collective-variable builders in
:mod:`reactiontools.tools_cv` need but that belongs to neither: seeding
ring-polymer beads, swapping hydrogen for deuterium, measuring a distance or
angle off a Modeller, naming atoms the way VMD does, and picking a compute
platform.

Anything here that takes a temperature or a mass accepts either a
:class:`openmm.unit.Quantity` or a bare number in the obvious unit -- kelvin
and daltons respectively.
"""
import math
import re
from typing import Dict, Sequence, Any, List, Union, Literal, Optional

import numpy as np
import openmm.unit as unit
from scipy import constants

from openmm import openmm, app


def zero_velocities(n_atoms):
    """
    Build a list of zero velocity vectors, one per atom.

    Parameters
    ----------
    n_atoms : int
        Number of atoms to produce vectors for.

    Returns
    -------
    openmm.unit.Quantity
        Zero velocities, in nanometres per picosecond.
    """
    return [openmm.Vec3(0, 0, 0) for _ in range(n_atoms)] * (unit.nanometer / unit.picosecond)


def write_multimodel_pdb(topology, positions, fh, model_index):
    """
    Append one model to an open multi-model PDB file.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology of the system being written.
    positions : openmm.unit.Quantity
        Atomic positions, with units of length.
    fh : file-like object
        An open, writable text handle.
    model_index : int
        Index distinguishing this model from the others in the file.
    """
    app.PDBFile.writeModel(topology, positions, fh, modelIndex=model_index)


def centroid_positions(simulation, n_atoms, n_beads):
    """
    Average the bead positions of an RPMD simulation into a centroid structure.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``.
    n_atoms : int
        Number of atoms in the system.
    n_beads : int
        Number of beads in the ring polymer.

    Returns
    -------
    openmm.unit.Quantity
        Centroid position of each atom, in nanometres.

    Notes
    -----
    For periodic systems, each bead is wrapped and then unwrapped relative to
    bead 0 with the minimum-image convention before averaging. This prevents
    beads on opposite sides of a box face from producing a centroid near the
    middle of the box.
    """
    integrator = simulation.integrator
    periodic = simulation.system.usesPeriodicBoundaryConditions()

    ref_state = integrator.getState(
        0, getPositions=True, enforcePeriodicBox=periodic
    )
    ref = ref_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    sum_pos = ref.copy()

    if periodic:
        box = ref_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
            unit.nanometer
        )

    for bead in range(1, n_beads):
        pos = integrator.getState(
            bead, getPositions=True, enforcePeriodicBox=periodic
        ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if periodic:
            disp = pos - ref
            # OpenMM box vectors are in reduced form. Remove whole c, b, then
            # a vectors to obtain the minimum image of each displacement.
            for axis in (2, 1, 0):
                disp -= box[axis] * np.round(
                    disp[:, axis:axis + 1] / box[axis][axis]
                )
            pos = ref + disp
        sum_pos += pos

    centroid = sum_pos / n_beads
    return [openmm.Vec3(*centroid[i]) for i in range(n_atoms)] * unit.nanometer


def get_thermal_de_broglie_wavelength(mass, temperature):
    """
    Compute the thermal de Broglie wavelength of a particle.

    This length sets the scale over which a particle behaves as a wave rather
    than a point, and so how far apart the beads of its ring polymer should
    start; see :func:`init_beads_scaled`.

    Parameters
    ----------
    mass : openmm.unit.Quantity or float
        Mass of the particle. A bare number is taken to be in daltons.
    temperature : openmm.unit.Quantity or float
        Temperature of the system. A bare number is taken to be in kelvin.

    Returns
    -------
    openmm.unit.Quantity
        The thermal de Broglie wavelength, in metres.
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
    Seed RPMD bead positions, spread by each atom's thermal wavelength.

    Every bead starts at the given positions plus Gaussian noise scaled by
    the atom's thermal de Broglie wavelength, so light atoms open out further
    than heavy ones and the ring polymers start closer to their equilibrium
    extent than the uniform jiggle of :func:`init_beads` manages.  Velocities
    are then drawn from the Maxwell-Boltzmann distribution.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``, modified in place.
    positions : openmm.unit.Quantity or numpy.ndarray
        Initial atomic positions. A bare array is taken to be in nanometres.
    n_beads : int
        Number of beads in the ring polymer.
    temperature : openmm.unit.Quantity
        Temperature the wavelengths and velocities are drawn at.
    scale_factor : float, optional
        Fraction of the thermal wavelength to perturb by. Default is 0.1.
    """
    system = simulation.system
    n_atoms = system.getNumParticles()

    masses_val = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_atoms)])
    masses_quantity = masses_val * unit.dalton

    lambdas = get_thermal_de_broglie_wavelength(masses_quantity, temperature)
    lambdas_nm = lambdas.value_in_unit(unit.nanometer)

    if not unit.is_quantity(positions):
        positions = positions * unit.nanometer
    pos0 = positions.value_in_unit(unit.nanometer)

    rng = np.random.default_rng(0)

    print(f"Initializing {n_beads} beads scaled by thermal wavelengths...")
    print(f"Max Lambda (lightest atom): {np.max(lambdas_nm):.4f} nm")
    print(f"Min Lambda (heaviest atom): {np.min(lambdas_nm):.4f} nm")

    for b in range(n_beads):
        noise = rng.normal(size=(n_atoms, 3)) * lambdas_nm[:, np.newaxis] * scale_factor
        bead_pos = pos0 + noise
        simulation.integrator.setPositions(b, bead_pos * unit.nanometer)

    simulation.context.setVelocitiesToTemperature(temperature)


def init_beads(modeller, simulation, n_beads, perturb=0.002):
    """
    Seed RPMD bead positions with a uniform jiggle and zero velocities.

    Beads that all start at the same point stay collapsed on top of one
    another, so each is displaced by a small random amount.  The displacement
    is the same size for every atom; :func:`init_beads_scaled` sizes it per
    atom instead, at the cost of needing a temperature.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller the initial positions are taken from.
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``, modified in place.
    n_beads : int
        Number of beads in the ring polymer.
    perturb : float, optional
        Standard deviation of the displacement, in nanometres. Default is
        0.002.
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
    Estimate the charge of the DNA in a topology from its residue count.

    Each nucleotide carries one deprotonated phosphate, so the charge is
    simply minus the number of DNA residues.  This is the number of
    counter-ions the system needs to come out neutral.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology to scan for DNA residues.

    Returns
    -------
    int
        Estimated total DNA charge, in units of the elementary charge. Zero
        or negative.
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

    estimated_charge = -num_dna_residues

    return estimated_charge


def deuterate_system(modeller, system, option='all', target_resname=None):
    """
    Replace the hydrogens of a system, or part of it, with deuterium.

    Only the particle masses change: the topology keeps calling them
    hydrogens, and the force field keeps treating them as such.  That is all
    a kinetic isotope effect needs, since the potential energy surface is
    isotope-independent and the mass is what the dynamics sees.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller whose topology names the residues to deuterate.
    system : openmm.System
        System whose particle masses are modified in place.
    option : {'all', 'water', 'protein', 'dna', 'rna', 'nucleic', 'ligand'}, optional
        Which part of the system to deuterate. Default is ``'all'``.
    target_resname : str or None, optional
        Residue name of the ligand to deuterate. Required when *option* is
        ``'ligand'``, ignored otherwise. Default is None.

    Raises
    ------
    ValueError
        If *option* is ``'ligand'`` and *target_resname* is not given, or if
        *option* is not one of the values listed above.

    Notes
    -----
    An *option* that matches no residue at all is only warned about, not
    raised: it is a plausible thing to ask of a system that happens not to
    have that component.
    """
    deuterium_mass = app.element.deuterium.mass

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

    if option == 'all':
        for atom in modeller.topology.atoms():
            if atom.element and atom.element.symbol == 'H':
                system.setParticleMass(atom.index, deuterium_mass)
    else:
        found_target = False
        for residue in modeller.topology.residues():
            if residue.name in target_residues:
                found_target = True
                for atom in residue.atoms():
                    if atom.element and atom.element.symbol == 'H':
                        system.setParticleMass(atom.index, deuterium_mass)

        if not found_target and option == 'ligand':
            print(f"Warning: No ligand named '{target_resname}' was found.")
        elif not found_target and option != 'all':
            print(f"Warning: No residues matching option '{option}' were found.")


def get_atoms_in_residue(pdb_file_path, residue_index, chain_id=None):
    """
    Look up the atom indices of one residue in a PDB file.

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file to read.
    residue_index : int
        Position of the residue, counted from 0 within its chain when
        *chain_id* is given and within the whole topology otherwise. Note
        that this is not the residue ID written in the PDB.
    chain_id : str or None, optional
        Chain to look in. If None, the whole topology is searched.
        Default is None.

    Returns
    -------
    list of int or None
        0-based indices of the residue's atoms, or None if the chain does
        not exist or the index is out of range. The reason is printed.
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
    Assign adQTB particle types so each chemical element shares one type.

    A ``QTBIntegrator`` adapts one noise spectrum per particle type, so
    grouping by element gives every carbon the same bath as every other
    carbon.  Types are numbered by atomic number, lightest first, which puts
    hydrogen -- where the quantum correction matters most -- at the front.

    Parameters
    ----------
    integrator : openmm.QTBIntegrator
        The adQTB integrator, modified in place. Must provide
        ``setParticleType(index, type)``.
    topology : openmm.app.Topology or None, optional
        Topology the per-particle elements are read from. Required unless
        *particle_elements* is given. Default is None.
    system : openmm.System or None, optional
        System to check the particle count against. Default is None.
    particle_elements : sequence or None, optional
        Explicit element per particle, one entry per particle in the system.
        Each may be an ``openmm.app.element.Element``, a symbol such as
        ``"C"``, or None. Use this when the system holds particles the
        topology does not, such as Drude particles. Default is None.
    start_type : int, optional
        Type id to number from. Default is 0.
    unknown_symbol : str, optional
        Symbol standing in for a missing element; all such particles share
        one type, sorted last. Default is ``"X"``.

    Returns
    -------
    dict
        Mapping from element symbol to the integer type id assigned to it.

    Raises
    ------
    TypeError
        If *integrator* is not a QTB integrator.
    ValueError
        If neither *topology* nor *particle_elements* is given, or if the
        element count disagrees with the system's particle count.

    Notes
    -----
    Call this *before* the ``Context`` or ``Simulation`` is created. A
    ``QTBIntegrator`` fixes its particle types at context creation, so
    assignments made afterwards are ignored.
    """

    if not hasattr(integrator, "setParticleType"):
        raise TypeError(
            "integrator must support setParticleType(index, type); expected an OpenMM QTBIntegrator."
        )

    def _sym_and_Z(el: Any) -> tuple[str, int]:
        """
        Extract the symbol and atomic number of an element-like object.

        Parameters
        ----------
        el : openmm.app.element.Element or str or None
            An element, its symbol, or None.

        Returns
        -------
        sym : str
            The element symbol, or *unknown_symbol* if *el* is None.
        Z : int
            The atomic number, or 10**9 when unknown, which sorts such
            particles to the end.
        """
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

    if particle_elements is None:
        if topology is None:
            raise ValueError("Provide topology or particle_elements.")
        atoms = list(topology.atoms())
        particle_elements = [a.element for a in atoms]

    n = len(particle_elements)

    if system is not None:
        n_sys = system.getNumParticles()
        if n_sys != n:
            raise ValueError(
                f"Element list has length {n}, but System has {n_sys} particles. "
                "If your System includes extra particles (Drudes/virtual sites/etc.), "
                "pass particle_elements with one entry per System particle."
            )

    symbol_to_Z: dict[str, int] = {}
    for el in particle_elements:
        sym, Z = _sym_and_Z(el)
        symbol_to_Z.setdefault(sym, Z)

    ordered_symbols = sorted(symbol_to_Z.items(), key=lambda kv: (kv[1], kv[0]))
    element_to_type = {sym: start_type + i for i, (sym, _) in enumerate(ordered_symbols)}

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
    Map VMD pick strings onto OpenMM topology atom indices.

    Picking atoms in VMD and pasting what it prints is the quickest way to
    name the handful of atoms a collective variable acts on, and this turns
    those strings into the indices the rest of the package wants.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller whose topology is searched.
    picks : list of str
        Pick strings of the form ``"RESNAME<RESID>[INSERTION_CODE]:ATOMNAME"``,
        e.g. ``["HIE258:CD2", "ALA12:CA", "HIE258A:CD2"]``.
    match_mode : {'unique', 'first', 'all'}, optional
        What to do when a pick matches more than one atom: ``'unique'``
        raises, ``'first'`` takes the lowest index, ``'all'`` returns them
        all as a list. Default is ``'unique'``.
    chain_id : str or None, optional
        Chain to restrict the search to. If None, every chain is searched.
        Default is None.

    Returns
    -------
    list of int or list of list of int
        One entry per pick, a list of indices when *match_mode* is
        ``'all'`` and a single index otherwise.

    Raises
    ------
    ValueError
        If a pick is malformed, matches nothing, or -- under
        ``'unique'`` -- matches more than one atom.
    """
    topo = modeller.topology

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


def distance_between_atoms(modeller, atom_index_1: int, atom_index_2: int):
    """
    Measure the distance between two atoms of a Modeller.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller with its positions set.
    atom_index_1, atom_index_2 : int
        0-based indices into ``modeller.positions``, in the same order as
        ``modeller.topology.atoms()``.

    Returns
    -------
    openmm.unit.Quantity
        The distance, in nanometres.

    Raises
    ------
    ValueError
        If *modeller* has no positions.
    """

    positions = getattr(modeller, "positions", None)
    if positions is None:
        raise ValueError("modeller.positions is None. Make sure positions are set on the Modeller.")

    dr = positions[atom_index_1] - positions[atom_index_2]
    # unit.norm, not unit.sqrt over the components: `dr` is a Quantity wrapping
    # a Vec3, and `dr.x` hands back the bare component with the unit stripped,
    # so summing the squares gives a plain float and the result loses its units.
    return unit.norm(dr)


def angle_between_atoms(modeller, i, j, k, degrees: bool = False):
    """
    Measure the angle i-j-k, with its vertex at atom j.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller with its positions set.
    i, j, k : int
        0-based atom indices. The angle is subtended at *j*.
    degrees : bool, optional
        Whether to return degrees rather than radians. Default is False.

    Returns
    -------
    float
        The angle, in radians unless *degrees* is True.

    Raises
    ------
    ValueError
        If *i* or *k* coincides with *j*, leaving the angle undefined.
    """
    pos = modeller.positions

    ri = pos[i].value_in_unit(unit.nanometer)
    rj = pos[j].value_in_unit(unit.nanometer)
    rk = pos[k].value_in_unit(unit.nanometer)

    v1 = (ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2])
    v2 = (rk[0] - rj[0], rk[1] - rj[1], rk[2] - rj[2])

    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("Cannot compute angle: one of the vectors has zero length.")

    cos_theta = dot / (n1 * n2)

    # Rounding can push a straight angle just past +/-1, where acos is undefined
    cos_theta = max(-1.0, min(1.0, cos_theta))

    theta = math.acos(cos_theta)
    return math.degrees(theta) if degrees else theta


def check_platform(platform=None):
    """
    Pick a compute platform, preferring CUDA where it is available.

    Parameters
    ----------
    platform : str or None, optional
        Platform to force, e.g. ``'CUDA'``, ``'OpenCL'`` or ``'CPU'``. If
        None, CUDA is used when OpenMM offers it and CPU otherwise.
        Default is None.

    Returns
    -------
    str
        The chosen platform name, returned unchanged when one was given.

    Notes
    -----
    OpenMM is asked directly rather than, say, ``torch.cuda.is_available()``: a
    CUDA-capable GPU is no use here unless the OpenMM build in this environment
    also carries the CUDA platform, and only OpenMM knows that.
    """
    if platform is None:
        available = {openmm.Platform.getPlatform(i).getName()
                     for i in range(openmm.Platform.getNumPlatforms())}
        platform = 'CUDA' if 'CUDA' in available else 'CPU'
    return platform
