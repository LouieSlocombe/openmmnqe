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
from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Literal, TextIO

import numpy as np
import numpy.typing as npt
import openmm.unit as unit
from scipy import constants

from openmm import openmm, app


def zero_velocities(n_atoms: int) -> unit.Quantity:
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


def _sample_maxwell_boltzmann_velocities(system: openmm.System,
                                         temperature: unit.Quantity | float,
                                         n_copies: int,
                                         rng: np.random.Generator,
                                         ) -> unit.Quantity:
    """
    Draw independent bead velocities from a Maxwell-Boltzmann distribution.

    OpenMM's RPMD Hamiltonian thermalizes each bead at ``n_copies * k_B * T``,
    so a bead velocity is drawn with ``sigma = sqrt(n_copies*k_B*T/m)`` rather
    than the classical ``sqrt(k_B*T/m)``.

    Parameters
    ----------
    system : openmm.System
        System whose particle masses set the widths.
    temperature : openmm.unit.Quantity or float
        Target temperature. A plain number is read as kelvin.
    n_copies : int
        Number of ring-polymer beads.
    rng : numpy.random.Generator
        Random source the velocities are drawn from.

    Returns
    -------
    openmm.unit.Quantity
        Velocities in nm/ps, shaped ``(n_copies, n_particles, 3)``. Massless
        particles, typically virtual sites, are left at zero.

    Raises
    ------
    ValueError
        If *n_copies* is not positive, the temperature is not finite and
        positive, or a particle mass is not finite and non-negative.
    """
    if n_copies <= 0:
        raise ValueError("n_copies must be positive")
    if unit.is_quantity(temperature):
        temperature_k = temperature.value_in_unit(unit.kelvin)
    else:
        temperature_k = float(temperature)
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature must be finite and positive")

    masses_amu = np.asarray([
        system.getParticleMass(index).value_in_unit(unit.dalton)
        for index in range(system.getNumParticles())
    ])
    if not np.isfinite(masses_amu).all() or np.any(masses_amu < 0):
        raise ValueError("particle masses must be finite and non-negative")

    # OpenMM's RPMD Hamiltonian thermalizes each bead with n_copies*k_B*T
    # (``nkT`` in its PILE implementation).  Therefore a bead velocity has
    # sigma=sqrt(n_copies*k_B*T/m), not the classical sqrt(k_B*T/m).  The
    # result below is in m/s; one m/s is 0.001 nm/ps.  Massless particles
    # (typically virtual sites) do not carry momenta and remain at zero.
    sigma_nm_per_ps = np.zeros_like(masses_amu, dtype=float)
    massive = masses_amu > 0
    sigma_nm_per_ps[massive] = (
        np.sqrt(
            n_copies * constants.k * temperature_k
            / (masses_amu[massive] * constants.atomic_mass)
        )
        * 1.0e-3
    )
    velocities = rng.normal(size=(n_copies, len(masses_amu), 3))
    velocities *= sigma_nm_per_ps[np.newaxis, :, np.newaxis]
    return velocities * (unit.nanometer / unit.picosecond)


def _sample_free_ring_polymer_displacements(
    masses_amu: npt.ArrayLike,
    temperature: unit.Quantity | float,
    n_copies: int,
    rng: np.random.Generator,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Draw free-ring-polymer displacements with a fixed zero centroid.

    OpenMM evolves mode ``k`` at angular frequency
    ``2*P*k_B*T/hbar*sin(pi*k/P)`` and thermalizes its RPMD Hamiltonian at
    ``P*T``.  In an orthonormal normal-mode basis, each non-centroid real
    coordinate therefore has variance ``P*k_B*T/(m*omega_k**2)``.  The
    centroid mode is left at zero, so the displacements can be added to an
    existing classical configuration without moving it.

    Parameters
    ----------
    masses_amu : array-like of float
        Particle masses in daltons, one per atom.
    temperature : openmm.unit.Quantity or float
        Target temperature. A plain number is read as kelvin.
    n_copies : int
        Number of ring-polymer beads.
    rng : numpy.random.Generator
        Random source the displacements are drawn from.
    scale_factor : float, optional
        Multiplies the sampled widths, so values below 1 start the beads
        more tightly collapsed than equilibrium. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Displacements in nanometres, shaped ``(n_copies, n_atoms, 3)``, whose
        mean over the bead axis is zero. Massless particles stay at zero.

    Raises
    ------
    ValueError
        If the temperature is not finite and positive.
    """
    if unit.is_quantity(temperature):
        temperature_k = temperature.value_in_unit(unit.kelvin)
    else:
        temperature_k = float(temperature)
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature must be finite and positive")

    masses_amu = np.asarray(masses_amu, dtype=float)
    n_atoms = len(masses_amu)
    coefficients = np.zeros(
        (n_copies // 2 + 1, n_atoms, 3),
        dtype=np.complex128,
    )
    massive = masses_amu > 0
    mass_kg = masses_amu[massive] * constants.atomic_mass
    omega_p = n_copies * constants.k * temperature_k / constants.hbar

    for mode in range(1, n_copies // 2 + 1):
        omega_k = 2.0 * omega_p * np.sin(np.pi * mode / n_copies)
        sigma_nm = np.zeros(n_atoms)
        sigma_nm[massive] = (
            np.sqrt(
                n_copies * constants.k * temperature_k
                / (mass_kg * omega_k**2)
            )
            * 1.0e9
        )

        is_nyquist = n_copies % 2 == 0 and mode == n_copies // 2
        if is_nyquist:
            coefficients[mode].real = (
                rng.normal(size=(n_atoms, 3)) * sigma_nm[:, np.newaxis]
            )
        else:
            component_sigma = sigma_nm / np.sqrt(2.0)
            coefficients[mode] = (
                rng.normal(size=(n_atoms, 3))
                + 1j * rng.normal(size=(n_atoms, 3))
            ) * component_sigma[:, np.newaxis]

    # NumPy's inverse transform has a 1/P normalization. Multiplication by
    # sqrt(P) converts the orthonormal Fourier coefficients above back to
    # bead coordinates. The zero coefficient fixes the supplied centroid.
    displacements = (
        np.fft.irfft(coefficients, n=n_copies, axis=0)
        * np.sqrt(n_copies)
        * scale_factor
    )
    # Remove the last few bits of roundoff from the inverse FFT so that the
    # supplied coordinates remain the centroid to machine precision.
    displacements -= displacements.mean(axis=0, keepdims=True)
    return displacements


def write_multimodel_pdb(topology: app.Topology, positions: unit.Quantity,
                         fh: TextIO, model_index: int) -> None:
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


def centroid_positions(simulation: app.Simulation, n_atoms: int,
                       n_beads: int) -> unit.Quantity:
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


def get_thermal_de_broglie_wavelength(mass: unit.Quantity | float,
                                      temperature: unit.Quantity | float,
                                      ) -> unit.Quantity:
    """
    Compute the thermal de Broglie wavelength of a particle.

    This length sets the scale over which a particle behaves as a wave rather
    than a point and characterizes the spatial extent of its ring polymer.

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


def init_beads(modeller: app.Modeller, simulation: app.Simulation,
               n_beads: int, scale_factor: float = 1.0,
               temperature: unit.Quantity | float | None = None,
               seed: int | None = None) -> None:
    """
    Seed RPMD bead positions and velocities at the simulation temperature.

    Every bead starts at the Modeller positions plus an equilibrium free-ring-
    polymer displacement. Non-centroid normal mode ``k`` has variance
    ``n_beads*k_B*T/(m*omega_k**2)``, where OpenMM uses
    ``omega_k = 2*n_beads*k_B*T/hbar*sin(pi*k/n_beads)``. The centroid mode is
    held at zero, preserving the Modeller positions as the ring-polymer
    centroid. Each copy receives an independent Maxwell-Boltzmann velocity
    sample based on the particle masses and RPMD temperature. Following
    OpenMM's RPMD Hamiltonian convention, each copy's velocity variance is
    ``n_beads*k_B*T/m``. Massless particles remain at the supplied position
    and receive zero velocity.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller the initial positions are taken from.
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``, modified in place.
    n_beads : int
        Number of beads in the ring polymer.
    scale_factor : float, optional
        Multiplier applied to the equilibrium free-ring-polymer mode
        amplitudes. The default, 1.0, gives their exact free-particle thermal
        distribution. Values other than 1.0 deliberately contract or expand
        the ring polymer.
    temperature : openmm.unit.Quantity or float or None, optional
        Temperature for position and velocity sampling. A bare number is
        interpreted as kelvin. If specified, it must match the RPMD
        integrator temperature. If None, use the integrator temperature.
        Default is None.
    seed : int or None, optional
        NumPy random seed used for both position and velocity sampling. Pass
        an integer for reproducible initialization. Default is None for
        entropy-based seeding.
    """
    if (
        isinstance(n_beads, (bool, np.bool_))
        or not isinstance(n_beads, (int, np.integer))
        or n_beads <= 0
    ):
        raise ValueError("n_beads must be a positive integer")
    n_beads = int(n_beads)
    if (
        seed is not None
        and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        )
    ):
        raise ValueError("seed must be a non-negative integer or None")
    if seed is not None:
        seed = int(seed)
    if not np.isfinite(scale_factor) or scale_factor < 0:
        raise ValueError("scale_factor must be finite and non-negative")

    system = simulation.system
    integrator = simulation.integrator
    if hasattr(integrator, "getNumCopies"):
        integrator_beads = integrator.getNumCopies()
        if integrator_beads != n_beads:
            raise ValueError(
                f"n_beads={n_beads} does not match RPMDIntegrator "
                f"copies={integrator_beads}"
            )
    n_atoms = system.getNumParticles()

    positions = modeller.positions
    if not unit.is_quantity(positions):
        positions = positions * unit.nanometer
    pos0 = np.asarray(positions.value_in_unit(unit.nanometer))
    if pos0.shape != (n_atoms, 3):
        raise ValueError(
            f"positions must have shape ({n_atoms}, 3), got {pos0.shape}"
        )
    if not np.isfinite(pos0).all():
        raise ValueError("positions must be finite")

    integrator_temperature = integrator.getTemperature()
    if unit.is_quantity(integrator_temperature):
        integrator_temperature_k = integrator_temperature.value_in_unit(
            unit.kelvin
        )
    else:
        integrator_temperature_k = float(integrator_temperature)
    if (
        not np.isfinite(integrator_temperature_k)
        or integrator_temperature_k <= 0
    ):
        raise ValueError(
            "RPMDIntegrator temperature must be finite and positive"
        )
    if temperature is None:
        temperature = integrator_temperature
    if unit.is_quantity(temperature):
        temperature_k = temperature.value_in_unit(unit.kelvin)
    else:
        temperature_k = float(temperature)
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isclose(
        temperature_k,
        integrator_temperature_k,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError(
            "temperature must match the RPMDIntegrator temperature "
            f"({integrator_temperature_k} K)"
        )

    masses_amu = np.asarray([
        system.getParticleMass(index).value_in_unit(unit.dalton)
        for index in range(n_atoms)
    ])
    if not np.isfinite(masses_amu).all() or np.any(masses_amu < 0):
        raise ValueError("particle masses must be finite and non-negative")
    rng = np.random.default_rng(seed)
    displacements = _sample_free_ring_polymer_displacements(
        masses_amu,
        temperature,
        n_beads,
        rng,
        scale_factor,
    )
    velocities = _sample_maxwell_boltzmann_velocities(
        system,
        temperature,
        n_beads,
        rng,
    )
    for b in range(n_beads):
        integrator.setPositions(
            b,
            (pos0 + displacements[b]) * unit.nanometer,
        )
        integrator.setVelocities(b, velocities[b])


def step_rpmd(simulation: app.Simulation, steps: int) -> None:
    """
    Advance an RPMD Simulation while keeping its step count synchronized.

    OpenMM 8.5 advances an :class:`openmm.RPMDIntegrator`'s time but does not
    advance its Context step count.  :meth:`openmm.app.Simulation.step` uses
    that count to stop and schedule reporters, so it otherwise loops forever.
    This compatibility helper temporarily wraps the integrator's ``step()``
    method, repairs the count only when OpenMM did not update it, and delegates
    reporter scheduling to ``Simulation.step()``.  On OpenMM versions that do
    update the count, the wrapper leaves it untouched.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``.
    steps : int
        Number of integration steps to run.

    Raises
    ------
    TypeError
        If *steps* is not an integer or the Simulation does not use an
        RPMD-style integrator.
    ValueError
        If *steps* is negative.
    RuntimeError
        If OpenMM changes the Context step count by an unexpected amount.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, Integral):
        raise TypeError("steps must be a non-negative integer")
    if steps < 0:
        raise ValueError("steps must be a non-negative integer")
    steps = int(steps)

    integrator = simulation.integrator
    if not hasattr(integrator, "getNumCopies"):
        raise TypeError("simulation must use an RPMDIntegrator")

    native_step = integrator.step

    def synchronized_step(count: int) -> None:
        """
        Advance the integrator, repairing the step count if OpenMM did not.

        Parameters
        ----------
        count : int
            Number of steps to advance.

        Raises
        ------
        RuntimeError
            If the Context step count moved by anything other than *count*.
        """
        before = simulation.currentStep
        native_step(count)
        after = simulation.currentStep
        expected = before + count
        if after == before:
            simulation.currentStep = expected
        elif after != expected:
            raise RuntimeError(
                "RPMDIntegrator changed the Context step count from "
                f"{before} to {after} while advancing {count} steps"
            )

    integrator.step = synchronized_step
    try:
        simulation.step(steps)
    finally:
        integrator.step = native_step


def count_dna_and_estimate_charge(topology: app.Topology) -> int:
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


def deuterate_system(
        modeller: app.Modeller,
        system: openmm.System,
        option: Literal['all', 'water', 'protein', 'dna', 'rna', 'nucleic',
                        'ligand'] = 'all',
        target_resname: str | None = None) -> None:
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


def get_atoms_in_residue(pdb_file_path: str | os.PathLike[str],
                         residue_index: int,
                         chain_id: str | None = None) -> list[int] | None:
    """
    Look up the atom indices of one residue in a PDB file.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
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
    pdb = app.PDBFile(os.fspath(pdb_file_path))
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
        topology: app.Topology | None = None,
        system: openmm.System | None = None,
        particle_elements: Sequence[Any] | None = None,
        start_type: int = 0,
        unknown_symbol: str = "X",
) -> dict[str, int]:
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

    def _sym_and_Z(el: app.Element | str | None) -> tuple[str, int]:
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
        if isinstance(el, str):
            try:
                element = app.Element.getBySymbol(el)
            except KeyError:
                return el, 10 ** 9
            return element.symbol, int(element.atomic_number)
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
        modeller: app.Modeller,
        picks: list[str],
        *,
        match_mode: Literal["unique", "first", "all"] = "unique",
        chain_id: str | None = None,
) -> list[int | list[int]]:
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
        resid = f"{res.id}{res.insertionCode or ''}"
        key = (res.chain.id, res.name, resid, atom.name)
        lookup.setdefault(key, []).append(atom.index)

    out: list[int | list[int]] = []

    for s in picks:
        m = _VMD_PICK_RE.match(s)
        if not m:
            raise ValueError(
                f"Pick '{s}' is malformed; expected like 'HIE258:CD2' (optionally 'HIE258A:CD2')."
            )

        resname, resid_num, ins_code, atomname = m.groups()
        resid_str = f"{resid_num}{ins_code or ''}"

        matches: list[int] = []
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


def distance_between_atoms(modeller: app.Modeller, atom_index_1: int,
                           atom_index_2: int) -> unit.Quantity:
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


def angle_between_atoms(modeller: app.Modeller, i: int, j: int, k: int,
                        degrees: bool = False) -> float:
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


def check_platform(platform: str | None = None) -> str:
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
