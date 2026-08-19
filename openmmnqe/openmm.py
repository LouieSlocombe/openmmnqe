"""The OpenMM simulation stages, from minimisation through to production.

Each ``run_openmm_*`` function is one stage of a workflow, and they are meant
to be run in order, each picking up where the last left off:

1. :func:`run_openmm_relaxation` or :func:`run_openmm_relaxation_simple` --
   take the strain out of the starting structure,
2. :func:`run_openmm_heating` -- warm it to temperature under restraints,
3. :func:`run_openmm_npt` -- relax the box density,
4. :func:`run_openmm_prod` -- classical production, optionally biased,
5. :func:`run_openmm_rpmd_equilibration` then
   :func:`run_openmm_rpmd_prod` or :func:`run_openmm_rpmd_contracted`, or
   :func:`run_openmm_adqtb_eq` then :func:`run_openmm_adqtb_prod` -- the two
   routes to nuclear quantum effects.

Ring-polymer MD is the reference method and converges to the exact quantum
statistics with enough beads, but costs a force evaluation per bead;
the adaptive quantum thermal bath costs no more than a classical run but is
an approximation. :func:`run_openmm_steered` sits outside the sequence and
pulls a collective variable to generate a reference path (see
:mod:`reactiontools.tools_path`).

Every stage takes the same shape: build the system, optionally deuterate it,
attach a PLUMED bias and the reporters, run, then save restart data and a
structure for the next stage to start from. RPMD restart files contain every
bead rather than an ordinary single-Context checkpoint. The shared arguments
behave the same throughout -- *potential* with *ml_idx* runs an ML/MM mixed
system and forces the CUDA platform, *plumed_script_path* attaches a bias,
and *output_prefix* names every file the stage writes.
"""
import hashlib
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import openmm.unit as unit
from openmmml import MLPotential
from openmmplumed import PlumedForce

from openmm import openmm, app
from .reporters import (RPMDQuantumSpreadReporter,
                        RPMDBeadReporter,
                        RPMDCentroidReporter,
                        _validate_observable_indices,
                        )
from .tools import (deuterate_system, check_platform, init_beads,
                    centroid_positions, step_rpmd)


_RPMD_RESTART_KIND = "openmmnqe-rpmd-restart"
_RPMD_RESTART_VERSION = 2


def _validate_rpmd_n_beads(n_beads):
    """
    Require an RPMD bead count to be a positive, non-boolean integer.

    Parameters
    ----------
    n_beads : int
        Bead count to check.

    Returns
    -------
    int
        The bead count as a plain int.

    Raises
    ------
    ValueError
        If *n_beads* is a bool, not an integer, or not positive.
    """
    if (
        isinstance(n_beads, (bool, np.bool_))
        or not isinstance(n_beads, (int, np.integer))
        or n_beads <= 0
    ):
        raise ValueError("n_beads must be a positive integer")
    return int(n_beads)


def _validate_pdb_identity_name(name, description, max_length):
    """
    Reject topology identity names that PDB output cannot preserve.

    Parameters
    ----------
    name : str
        Atom or residue name to check.
    description : str
        What *name* labels, used in the error message.
    max_length : int
        Width of the PDB field the name has to fit, 4 for an atom and 3 for
        a residue.

    Raises
    ------
    ValueError
        If *name* is empty, over-long, non-ASCII, contains whitespace, or is
        not a string.
    """
    if (
        not isinstance(name, str)
        or not name
        or len(name) > max_length
        or not name.isascii()
        or any(character.isspace() for character in name)
    ):
        raise ValueError(
            f"Cannot create a PDB-stable RPMD restart: {description} name "
            f"{name!r} is not representable in a {max_length}-character "
            "PDB identity field"
        )


def _topology_identity_signature(topology):
    """
    Return a PDB-round-trip-stable ordered topology signature.

    PDB serialization is allowed to renumber atom, residue, and chain IDs and
    does not reliably preserve bond type/order metadata.  The restart identity
    therefore uses atom order, chemically meaningful names/elements, the
    ordered chain/residue grouping, and bond endpoints.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology to fingerprint.

    Returns
    -------
    str
        Hex SHA-256 digest of the signature, stable across a PDB round trip
        of the same structure.

    Raises
    ------
    ValueError
        If an atom or residue name would not survive PDB output.
    """
    chain_ordinals = {
        chain: ordinal for ordinal, chain in enumerate(topology.chains())
    }
    residue_ordinals = {
        residue: ordinal for ordinal, residue in enumerate(topology.residues())
    }
    for residue in topology.residues():
        _validate_pdb_identity_name(
            residue.name,
            f"residue {residue_ordinals[residue]}",
            3,
        )
    atoms = []
    for atom in topology.atoms():
        residue = atom.residue
        chain = residue.chain
        element = atom.element
        _validate_pdb_identity_name(atom.name, f"atom {atom.index}", 4)
        atoms.append({
            "name": atom.name,
            "element": None if element is None else element.symbol,
            "atomic_number": (
                None if element is None else element.atomic_number
            ),
            "residue_ordinal": residue_ordinals[residue],
            "residue_name": residue.name,
            "chain_ordinal": chain_ordinals[chain],
        })

    bonds = sorted(
        (
            min(bond.atom1.index, bond.atom2.index),
            max(bond.atom1.index, bond.atom2.index),
        )
        for bond in topology.bonds()
    )
    payload = json.dumps(
        {"atoms": atoms, "bonds": bonds},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _particle_masses_dalton(system):
    """
    Return ordered particle masses as finite, non-negative floats.

    Parameters
    ----------
    system : openmm.System
        System whose particles are read, in index order.

    Returns
    -------
    numpy.ndarray
        Masses in daltons, shaped ``(n_particles,)``.

    Raises
    ------
    ValueError
        If any mass is not finite and non-negative.
    """
    masses = np.asarray([
        system.getParticleMass(index).value_in_unit(unit.dalton)
        for index in range(system.getNumParticles())
    ], dtype=np.float64)
    if not np.isfinite(masses).all() or np.any(masses < 0.0):
        raise ValueError("RPMD System particle masses must be finite and non-negative")
    return masses


def _rpmd_temperature_kelvin(integrator):
    """
    Return an RPMD integrator's finite, positive temperature in kelvin.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        Integrator to read the temperature from.

    Returns
    -------
    float
        Temperature in kelvin.

    Raises
    ------
    ValueError
        If the temperature is not finite and positive.
    """
    temperature = float(integrator.getTemperature().value_in_unit(unit.kelvin))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("RPMDIntegrator temperature must be positive and finite")
    return temperature


def _restart_scalar(archive, name, scalar_type):
    """
    Read a restart scalar only when its shape and dtype match the schema.

    Parameters
    ----------
    archive : numpy.lib.npyio.NpzFile
        Opened restart archive.
    name : str
        Field to read.
    scalar_type : {"integer", "float", "boolean", "string"}
        Kind the field is required to hold.

    Returns
    -------
    int or float or bool or str
        The stored value, as a Python scalar.

    Raises
    ------
    ValueError
        If the field is not a scalar, or does not hold *scalar_type*.
    """
    value = archive[name]
    if value.shape != ():
        raise ValueError(f"RPMD restart field {name} must be a scalar")

    dtype_kind = value.dtype.kind
    expected_kinds = {
        "integer": {"i", "u"},
        "float": {"f"},
        "boolean": {"b"},
        "string": {"U"},
    }
    if dtype_kind not in expected_kinds[scalar_type]:
        raise ValueError(
            f"RPMD restart field {name} must be a scalar {scalar_type}"
        )
    return value.item()


class PreparedSystem:
    """
    Stand-in force field that hands a pre-built ``openmm.System`` to a stage.

    Every ``run_openmm_*`` stage builds its System by calling
    ``forcefield.createSystem(topology, **kwargs)``.  Wrapping an existing
    System in this class routes it through that seam unchanged, so a System
    prepared elsewhere -- for example a QM/MM System carrying an
    ``openmm.PythonForce`` -- runs under the classical, RPMD, and adQTB
    stages without any stage signature changing.

    Parameters
    ----------
    system : openmm.System
        The System to hand out.

    Raises
    ------
    TypeError
        If *system* is not an ``openmm.System``.

    Notes
    -----
    ``createSystem`` returns the held System as-is, never a copy: a copy
    would sever externally held references (an ``openmm.PythonForce``
    callback, say) from the System the stage actually runs.  Stage options
    that mutate the System -- ``deuterate``, a non-None ``barostat_freq``,
    ``plumed_script_path`` -- therefore mutate this instance too, and
    re-running a mutating stage against the same ``PreparedSystem``
    accumulates their forces.  Build a fresh System and ``PreparedSystem``
    per stage when using those options.

    A ``PreparedSystem`` can also serve as the MM base of an ML/MM mixed
    system by passing *potential* and *ml_idx* to a stage as usual.
    """

    def __init__(self, system):
        if not isinstance(system, openmm.System):
            raise TypeError(
                "PreparedSystem wraps an existing openmm.System, got "
                f"{type(system).__name__}. Build the System first, or pass "
                "a force field to the stage instead."
            )
        self._system = system

    @property
    def system(self):
        """
        The held System that :meth:`createSystem` returns.

        Returns
        -------
        openmm.System
            The System this instance was built around.
        """
        return self._system

    def createSystem(self, topology, **kwargs):
        """
        Return the held System after checking it matches *topology*.

        Parameters
        ----------
        topology : openmm.app.Topology
            The topology the stage is about to simulate.
        **kwargs
            System-construction options, accepted and ignored: they describe
            how to build a System, and this one is already built.

        Returns
        -------
        openmm.System
            The held System, as-is.

        Raises
        ------
        ValueError
            If the topology's atom count differs from the held System's
            particle count, meaning the System was prepared for a different
            structure than the stage was given.
        """
        num_atoms = topology.getNumAtoms()
        num_particles = self._system.getNumParticles()
        if num_atoms != num_particles:
            raise ValueError(
                f"PreparedSystem holds {num_particles} particles but the "
                f"stage topology has {num_atoms} atoms; the System was "
                "prepared for a different structure. Rebuild the System from "
                "the same structure the Modeller was built from."
            )
        return self._system


def _build_system(modeller, forcefield, platform_name, potential, ml_idx, calculator):
    """
    Construct the system and platform shared by every ``run_openmm_*`` driver.

    Three configurations are supported:

    * **Pure MM** -- *forcefield* is a plain ``openmm.app.ForceField`` and no
      ML arguments are given.
    * **Pure ML** -- *forcefield* is an ``openmmml.MLPotential`` standing in
      for a force field; *ml_idx* stays None because every atom is ML. An ASE
      *calculator* may be supplied for ``MLPotential('ase')`` and is forwarded
      to its ``createSystem``.
    * **ML/MM mixed** -- *potential* (or *calculator*, which implies
      ``MLPotential('ase')``) together with *ml_idx* promotes the MM system to
      a mixed system via ``createMixedSystem``, and the platform is forced to
      CUDA.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField or openmmml.MLPotential
        The force field used to parameterise the system, or an ML potential
        used in its place for a pure-ML system.
    platform_name : str or None
        OpenMM platform name. None auto-detects via ``check_platform``.
    potential : object or None
        ML potential object with a ``createMixedSystem`` method.
    ml_idx : list of int or None
        Atom indices for the ML region of a mixed system.
    calculator : object or None
        Optional ASE calculator. Only ``MLPotential('ase')`` consumes it;
        other potentials silently ignore it.

    Returns
    -------
    system : openmm.System
        The parameterised system.
    platform : openmm.Platform
        The platform to run it on.

    Raises
    ------
    ValueError
        If *potential* is given without *ml_idx*, or if *calculator* is given
        without *ml_idx* when *forcefield* is not an ``MLPotential``. In both
        cases the ML region is undefined, and ``ForceField.createSystem``
        absorbs unknown keywords into ``**args``, so passing a calculator
        through would quietly run the whole simulation at pure MM instead.
        Also raised when *ml_idx* is empty or is supplied without either an ML
        potential or calculator.
    """
    if potential is not None and ml_idx is None:
        raise ValueError(
            "An ML potential was given but ml_idx is None, so the ML region is "
            "undefined. Pass ml_idx with the indices of the atoms the ML potential "
            "should cover, or pass the MLPotential as forcefield to run pure ML."
        )
    if (calculator is not None and ml_idx is None
            and not isinstance(forcefield, MLPotential)):
        raise ValueError(
            "A calculator was given without ml_idx, and forcefield is not an "
            "MLPotential, so ForceField.createSystem would silently ignore it. "
            "Pass ml_idx to define an ML/MM region, or pass MLPotential('ase') "
            "as forcefield to run pure ML."
        )
    if ml_idx is not None and potential is None and calculator is None:
        raise ValueError(
            "ml_idx was given without an ML potential or calculator. Pass a "
            "potential/calculator for a mixed system, or omit ml_idx for pure MM."
        )
    if (potential is not None or calculator is not None) and ml_idx is not None:
        if len(ml_idx) == 0:
            raise ValueError("ml_idx is empty; a mixed system needs at least one ML atom.")

    run_mixed = ml_idx is not None and (potential is not None or calculator is not None)
    if run_mixed:
        print("Adding ML potential to the system...", flush=True)
        platform_name = 'CUDA'
        print("ML potential in use: forcing platform to CUDA.", flush=True)

        # An explicit potential wins; 'ase' is only the fallback that wraps a
        # bare calculator.
        if calculator is not None and potential is None:
            potential = MLPotential('ase')

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    system_kwargs = {
        'nonbondedMethod': app.PME if has_box else app.CutoffNonPeriodic,
        'nonbondedCutoff': 1.0 * unit.nanometer,
        'constraints': None,
        'rigidWater': False,
        'removeCMMotion': True,
        # RPMD and adQTB depend on the physical vibrational frequencies, so
        # hydrogen mass repartitioning must not be used.  Keep this explicit:
        # it is too important for the NQE workflows to rely on an OpenMM
        # default, and it also keeps the MM part of a mixed system unchanged.
        'hydrogenMass': None,
    }

    if not run_mixed:
        # Pure MM, or pure ML with the MLPotential standing in as forcefield.
        if calculator is not None:
            system_kwargs['calculator'] = calculator
        return forcefield.createSystem(modeller.topology, **system_kwargs), platform

    mm_system = forcefield.createSystem(modeller.topology, **system_kwargs)
    if calculator is not None:
        system_kwargs['calculator'] = calculator
    system = potential.createMixedSystem(modeller.topology, mm_system, ml_idx, **system_kwargs)

    return system, platform


def _maybe_deuterate(modeller, system, deuterate, deuterate_option):
    """
    Deuterate the system in place when requested.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller whose topology names the hydrogens to convert.
    system : openmm.System
        System whose particle masses are edited in place.
    deuterate : bool
        Whether to deuterate at all. False makes this a no-op.
    deuterate_option : str
        Selection passed through to :func:`openmmnqe.tools.deuterate_system`.
    """
    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)


def _load_plumed(system, plumed_script_path):
    """
    Attach a PLUMED bias force to the system if a script path is given.

    Parameters
    ----------
    system : openmm.System
        System the bias force is added to, in place.
    plumed_script_path : str or None
        Path to a PLUMED input script. None makes this a no-op.
    """
    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path, 'r') as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)


def _add_standard_reporters(simulation, output_prefix, n_report,
                            pdb_steps=False, stdout_volume=False,
                            checkpoint_interval=None):
    """
    Append the standard reporter set shared by the classical drivers.

    Order: optional PDB trajectory, stdout state data, ``.log`` state data,
    optional periodic checkpoint.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation the reporters are appended to.
    output_prefix : str
        Prefix for the files written, giving ``<prefix>.log`` and friends.
    n_report : int
        Interval between reports, in steps.
    pdb_steps : bool, optional
        Also write a ``<prefix>_steps.pdb`` trajectory. Default is False.
    stdout_volume : bool, optional
        Include box volume in the stdout report, which is worth having under
        a barostat. Default is False.
    checkpoint_interval : int or None, optional
        Interval between ``<prefix>.chk`` checkpoints. With None, no
        checkpoint reporter is added. Default is None.
    """
    if pdb_steps:
        simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True,
                                                      volume=stdout_volume))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))
    if checkpoint_interval is not None:
        simulation.reporters.append(app.CheckpointReporter(f'{output_prefix}.chk',
                                                           checkpoint_interval))


def _add_rpmd_progress_reporters(simulation, output_prefix, n_report):
    """
    Append Context-independent progress reporters for an RPMD run.

    An RPMD integrator's ordinary Context state is not a bead average and is
    not guaranteed to mirror any particular copy.  Step, time, speed, and box
    volume remain meaningful, but Context energy and kinetic-temperature
    fields do not, so they are deliberately omitted here.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation the reporters are appended to.
    output_prefix : str
        Prefix for the ``<prefix>.log`` progress log.
    n_report : int
        Interval between reports, in steps.
    """
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout,
        n_report,
        step=True,
        speed=True,
    ))
    simulation.reporters.append(app.StateDataReporter(
        f'{output_prefix}.log',
        n_report,
        step=True,
        time=True,
        speed=True,
        volume=True,
    ))


def _add_rpmd_reporters(simulation, topology, output_prefix, n_report, n_beads,
                        atoms_to_watch, expansion_metric="rms",
                        distance_pairs=None):
    """
    Append the RPMD reporter trio: optional spread, then centroid and beads.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation the reporters are appended to.
    topology : openmm.app.Topology
        Topology written into the PDB output and used to bounds-check the
        watched atoms.
    output_prefix : str
        Prefix for ``<prefix>_spread.log``, ``<prefix>_centroid.pdb`` and the
        per-bead ``<prefix>_bead_<i>.pdb`` files.
    n_report : int
        Interval between reports, in steps.
    n_beads : int
        Number of ring-polymer beads.
    atoms_to_watch : list of int or None
        Atoms whose expansion is logged. With None, no spread reporter is
        added and *distance_pairs* may not be given either.
    expansion_metric : {"rms", "mean"}, optional
        Spread metric recorded for *atoms_to_watch*. Default is ``"rms"``.
    distance_pairs : iterable of pair of int or None, optional
        Atom pairs whose centroid distance is logged alongside the expansion.
        Default is None.

    Raises
    ------
    ValueError
        If *distance_pairs* is given without *atoms_to_watch*, or an index
        lies outside *topology*.
    """
    if distance_pairs is not None and atoms_to_watch is None:
        raise ValueError("distance_pairs require atoms_to_watch")
    if atoms_to_watch is not None:
        atoms_to_watch, distance_pairs = _validate_observable_indices(
            atoms_to_watch,
            distance_pairs,
            n_atoms=topology.getNumAtoms(),
        )
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
            metric=expansion_metric,
            distance_pairs=distance_pairs,
        ))

    simulation.reporters.append(RPMDCentroidReporter(
        topology=topology,
        file_name=f"{output_prefix}_centroid.pdb",
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDBeadReporter(
        topology=topology,
        file_base_name=output_prefix,
        reportInterval=n_report,
        num_beads=n_beads,
    ))


def _save_rpmd_restart(simulation, checkpoint_file, n_beads):
    """
    Atomically save every RPMD copy to a portable restart archive.

    OpenMM's ordinary ``Context`` checkpoint only sees the copy currently
    mirrored into the Context.  The other copies live in private arrays owned
    by ``RPMDIntegrator``, so they must be collected through its copy-specific
    API.  Positions are stored in nanometres and velocities in nanometres per
    picosecond.  The archive also carries the ordered particle masses, atom
    and topology signature, source temperature, box, time, and step count
    needed to validate and continue in a new ``Simulation``.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``.
    checkpoint_file : str
        Path the ``.npz`` archive is written to. It is written via a
        temporary file and moved into place, so an interrupted save cannot
        leave a half-written restart behind.
    n_beads : int
        Number of ring-polymer beads, checked against the integrator.

    Raises
    ------
    ValueError
        If *n_beads* is not a positive integer or disagrees with the
        integrator, or the topology cannot be given a PDB-stable signature.
    """
    n_beads = _validate_rpmd_n_beads(n_beads)

    integrator = simulation.integrator
    actual_beads = integrator.getNumCopies()
    if actual_beads != n_beads:
        raise ValueError(
            f"n_beads={n_beads} does not match RPMDIntegrator copies={actual_beads}"
        )

    n_particles = simulation.system.getNumParticles()
    if simulation.topology.getNumAtoms() != n_particles:
        raise ValueError(
            "Cannot save RPMD restart: Topology atom count does not match "
            "System particle count"
        )
    particle_masses = _particle_masses_dalton(simulation.system)
    topology_signature = _topology_identity_signature(simulation.topology)
    temperature_kelvin = _rpmd_temperature_kelvin(integrator)
    positions = []
    velocities = []
    first_state = None
    for bead in range(n_beads):
        state = integrator.getState(
            bead,
            getPositions=True,
            getVelocities=True,
        )
        if first_state is None:
            first_state = state
        bead_positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer
        )
        bead_velocities = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        if bead_positions.shape != (n_particles, 3):
            raise ValueError(
                f"Bead {bead} has position shape {bead_positions.shape}; "
                f"expected {(n_particles, 3)}"
            )
        if bead_velocities.shape != (n_particles, 3):
            raise ValueError(
                f"Bead {bead} has velocity shape {bead_velocities.shape}; "
                f"expected {(n_particles, 3)}"
            )
        positions.append(np.asarray(bead_positions, dtype=np.float64))
        velocities.append(np.asarray(bead_velocities, dtype=np.float64))

    positions = np.stack(positions)
    velocities = np.stack(velocities)
    if not np.isfinite(positions).all():
        raise ValueError("Cannot save RPMD restart: bead positions are not finite")
    if not np.isfinite(velocities).all():
        raise ValueError("Cannot save RPMD restart: bead velocities are not finite")

    box_vectors = first_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
        unit.nanometer
    )
    time_ps = first_state.getTime().value_in_unit(unit.picosecond)
    step_count = simulation.currentStep
    periodic = simulation.system.usesPeriodicBoundaryConditions()

    checkpoint_file = os.fspath(checkpoint_file)
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_file))
    file_descriptor, temporary_file = tempfile.mkstemp(
        prefix=f".{os.path.basename(checkpoint_file)}.",
        suffix=".tmp",
        dir=checkpoint_dir,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.savez(
                handle,
                kind=np.asarray(_RPMD_RESTART_KIND),
                format_version=np.asarray(_RPMD_RESTART_VERSION, dtype=np.int64),
                num_beads=np.asarray(n_beads, dtype=np.int64),
                num_particles=np.asarray(n_particles, dtype=np.int64),
                particle_masses_dalton=particle_masses,
                topology_signature_sha256=np.asarray(topology_signature),
                temperature_kelvin=np.asarray(
                    temperature_kelvin, dtype=np.float64
                ),
                positions_nm=positions,
                velocities_nm_per_ps=velocities,
                periodic=np.asarray(periodic, dtype=np.bool_),
                box_vectors_nm=np.asarray(box_vectors, dtype=np.float64),
                time_ps=np.asarray(time_ps, dtype=np.float64),
                step_count=np.asarray(step_count, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_file, checkpoint_file)
    except BaseException:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)
        raise


def _read_rpmd_restart(checkpoint_file):
    """
    Read and validate the format-independent portion of an RPMD restart.

    Parameters
    ----------
    checkpoint_file : str
        Path to a ``.npz`` archive written by :func:`_save_rpmd_restart`.

    Returns
    -------
    dict
        The restart contents: ``num_beads``, ``num_particles``,
        ``particle_masses_dalton``, ``topology_signature_sha256``,
        ``temperature_kelvin``, ``positions_nm``, ``velocities_nm_per_ps``,
        ``periodic``, ``box_vectors_nm``, and the run's time and step count.

    Raises
    ------
    ValueError
        If the file is corrupt, is a generic OpenMM checkpoint rather than a
        bead-aware one, or fails the restart schema.
    """
    try:
        archive = np.load(checkpoint_file, allow_pickle=False)
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"RPMD restart {checkpoint_file} is corrupt or unreadable; "
            "rerun RPMD equilibration to create a new checkpoint."
        ) from exc
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(
            f"{checkpoint_file} is not a bead-aware openmmnqe RPMD restart. "
            "A generic OpenMM checkpoint cannot safely restore all RPMD copies; "
            "rerun RPMD equilibration to create a new checkpoint."
        ) from exc

    try:
        try:
            if not hasattr(archive, "files"):
                raise ValueError("restart is not an NPZ archive")
            header = {"kind", "format_version"}
            missing_header = header.difference(archive.files)
            if missing_header:
                raise ValueError(
                    "RPMD restart is missing fields: "
                    + ", ".join(sorted(missing_header))
                )
            kind = _restart_scalar(archive, "kind", "string")
            if kind != _RPMD_RESTART_KIND:
                raise ValueError("checkpoint is not an openmmnqe RPMD restart")
            version = _restart_scalar(archive, "format_version", "integer")
            if version == 1:
                raise ValueError(
                    "RPMD restart version 1 lacks particle identity, mass, and "
                    "temperature metadata. Rerun RPMD equilibration to create a "
                    "compatible checkpoint."
                )
            if version != _RPMD_RESTART_VERSION:
                raise ValueError(
                    f"Unsupported RPMD restart version {version}; "
                    f"expected {_RPMD_RESTART_VERSION}. Rerun RPMD equilibration "
                    "to create a compatible checkpoint."
                )

            required = header | {
                "num_beads",
                "num_particles",
                "particle_masses_dalton",
                "topology_signature_sha256",
                "temperature_kelvin",
                "positions_nm",
                "velocities_nm_per_ps",
                "periodic",
                "box_vectors_nm",
                "time_ps",
                "step_count",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    "RPMD restart is missing fields: "
                    + ", ".join(sorted(missing))
                )
            num_beads = _validate_rpmd_n_beads(
                _restart_scalar(archive, "num_beads", "integer")
            )
            num_particles = _restart_scalar(
                archive, "num_particles", "integer"
            )
            if num_particles < 0:
                raise ValueError(
                    "RPMD restart num_particles cannot be negative"
                )
            step_count = _restart_scalar(archive, "step_count", "integer")
            if step_count < 0:
                raise ValueError("RPMD restart step count cannot be negative")
            topology_signature = _restart_scalar(
                archive, "topology_signature_sha256", "string"
            )
            if (
                len(topology_signature) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in topology_signature
                )
            ):
                raise ValueError(
                    "RPMD restart topology signature must be 64 lowercase "
                    "hexadecimal characters"
                )
            return {
                "num_beads": num_beads,
                "num_particles": int(num_particles),
                "particle_masses_dalton": np.array(
                    archive["particle_masses_dalton"], dtype=np.float64
                ),
                "topology_signature_sha256": topology_signature,
                "temperature_kelvin": float(
                    _restart_scalar(archive, "temperature_kelvin", "float")
                ),
                "positions_nm": np.array(
                    archive["positions_nm"], dtype=np.float64
                ),
                "velocities_nm_per_ps": np.array(
                    archive["velocities_nm_per_ps"], dtype=np.float64
                ),
                "periodic": bool(
                    _restart_scalar(archive, "periodic", "boolean")
                ),
                "box_vectors_nm": np.array(
                    archive["box_vectors_nm"], dtype=np.float64
                ),
                "time_ps": float(
                    _restart_scalar(archive, "time_ps", "float")
                ),
                "step_count": int(step_count),
            }
        except (OSError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(
                f"RPMD restart {checkpoint_file} is corrupt or unreadable; "
                "rerun RPMD equilibration to create a new checkpoint."
            ) from exc
    finally:
        if hasattr(archive, "close"):
            archive.close()


def _load_rpmd_restart(simulation, checkpoint_file, n_beads):
    """
    Restore every RPMD copy before the integrator takes its first step.

    The restart is checked against the Simulation it is being loaded into:
    a mismatched bead count, particle count, mass ordering, topology
    signature, or temperature means the archive belongs to a different run
    and is refused rather than silently reinterpreted.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``, restored in place.
    checkpoint_file : str
        Path to a restart written by :func:`_save_rpmd_restart`.
    n_beads : int
        Number of ring-polymer beads the run expects.

    Raises
    ------
    ValueError
        If the restart is corrupt, or does not match the current bead count,
        ordered topology, particle masses, or temperature.
    """
    n_beads = _validate_rpmd_n_beads(n_beads)
    restart = _read_rpmd_restart(checkpoint_file)
    integrator = simulation.integrator
    integrator_beads = integrator.getNumCopies()
    n_particles = simulation.system.getNumParticles()
    periodic = simulation.system.usesPeriodicBoundaryConditions()
    if simulation.topology.getNumAtoms() != n_particles:
        raise ValueError(
            "Current Topology atom count does not match System particle count"
        )
    current_masses = _particle_masses_dalton(simulation.system)
    current_signature = _topology_identity_signature(simulation.topology)
    current_temperature = _rpmd_temperature_kelvin(integrator)

    if restart["num_beads"] != n_beads:
        raise ValueError(
            f"RPMD restart contains {restart['num_beads']} beads, but "
            f"n_beads={n_beads} was requested"
        )
    if integrator_beads != n_beads:
        raise ValueError(
            f"n_beads={n_beads} does not match RPMDIntegrator copies={integrator_beads}"
        )
    if restart["num_particles"] != n_particles:
        raise ValueError(
            f"RPMD restart contains {restart['num_particles']} particles, but "
            f"the current System contains {n_particles}"
        )
    if restart["particle_masses_dalton"].shape != (n_particles,):
        raise ValueError(
            "RPMD restart particle masses have shape "
            f"{restart['particle_masses_dalton'].shape}; expected {(n_particles,)}"
        )
    if not np.isfinite(restart["particle_masses_dalton"]).all():
        raise ValueError("RPMD restart particle masses contain non-finite values")
    if np.any(restart["particle_masses_dalton"] < 0.0):
        raise ValueError("RPMD restart particle masses cannot be negative")
    if not np.allclose(
        restart["particle_masses_dalton"],
        current_masses,
        rtol=1e-12,
        atol=1e-12,
    ):
        mismatch = int(np.flatnonzero(~np.isclose(
            restart["particle_masses_dalton"],
            current_masses,
            rtol=1e-12,
            atol=1e-12,
        ))[0])
        raise ValueError(
            f"RPMD restart particle mass {mismatch} does not match the current "
            "System; check atom ordering and deuteration settings"
        )
    if restart["topology_signature_sha256"] != current_signature:
        raise ValueError(
            "RPMD restart atom/topology identity does not match the current "
            "Topology; check atom ordering and source structure"
        )
    if not np.isfinite(restart["temperature_kelvin"]):
        raise ValueError("RPMD restart temperature is not finite")
    if not np.isclose(
        restart["temperature_kelvin"],
        current_temperature,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"RPMD restart temperature {restart['temperature_kelvin']} K does "
            f"not match the current RPMDIntegrator temperature "
            f"{current_temperature} K"
        )
    expected_shape = (n_beads, n_particles, 3)
    if restart["positions_nm"].shape != expected_shape:
        raise ValueError(
            f"RPMD restart positions have shape {restart['positions_nm'].shape}; "
            f"expected {expected_shape}"
        )
    if restart["velocities_nm_per_ps"].shape != expected_shape:
        raise ValueError(
            "RPMD restart velocities have shape "
            f"{restart['velocities_nm_per_ps'].shape}; expected {expected_shape}"
        )
    if restart["box_vectors_nm"].shape != (3, 3):
        raise ValueError(
            f"RPMD restart box has shape {restart['box_vectors_nm'].shape}; "
            "expected (3, 3)"
        )
    if restart["periodic"] != periodic:
        raise ValueError(
            "RPMD restart periodicity does not match the current System"
        )
    for name in ("positions_nm", "velocities_nm_per_ps", "box_vectors_nm"):
        if not np.isfinite(restart[name]).all():
            raise ValueError(f"RPMD restart field {name} contains non-finite values")
    if not np.isfinite(restart["time_ps"]):
        raise ValueError("RPMD restart time is not finite")

    if periodic:
        box_vectors = [
            openmm.Vec3(*row) * unit.nanometer
            for row in restart["box_vectors_nm"]
        ]
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setTime(restart["time_ps"] * unit.picosecond)
    simulation.currentStep = restart["step_count"]
    for bead in range(n_beads):
        integrator.setPositions(
            bead,
            restart["positions_nm"][bead] * unit.nanometer,
        )
        integrator.setVelocities(
            bead,
            restart["velocities_nm_per_ps"][bead]
            * unit.nanometer
            / unit.picosecond,
        )


def _save_final_state(simulation, output_prefix, pdb_suffix='.pdb', save_checkpoint=True,
                      n_beads=None):
    """
    Save an optional checkpoint and write the final structure to PDB.

    Pass *n_beads* for an RPMD run: all copy positions and velocities are
    saved in a bead-aware restart archive, and the final PDB positions are
    averaged over the copies via
    :func:`openmmnqe.tools.centroid_positions`. Without a bead count, an
    ordinary OpenMM checkpoint and Context structure are written.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation to save.
    output_prefix : str
        Prefix for the files written, giving ``<prefix>.chk`` and
        ``<prefix><pdb_suffix>``.
    pdb_suffix : str, optional
        Suffix for the final structure file. Default is ``'.pdb'``.
    save_checkpoint : bool, optional
        Whether to write a checkpoint alongside the structure. Default is
        True.
    n_beads : int or None, optional
        Number of ring-polymer beads for an RPMD run, or None for a
        classical one. Default is None.
    """
    if save_checkpoint:
        checkpoint_file = f'{output_prefix}.chk'
        if n_beads is None:
            simulation.saveCheckpoint(checkpoint_file)
        else:
            _save_rpmd_restart(simulation, checkpoint_file, n_beads)
    if n_beads is None:
        positions = simulation.context.getState(getPositions=True).getPositions()
    else:
        positions = centroid_positions(simulation,
                                       simulation.topology.getNumAtoms(),
                                       n_beads)
    with open(f'{output_prefix}{pdb_suffix}', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, positions, f)


def _load_checkpoint(simulation, checkpoint_file, n_beads=None):
    """
    Load equilibration restart data into *simulation*.

    With *n_beads*, require the bead-aware openmmnqe RPMD format and restore
    every copy. Without it, delegate to OpenMM's ordinary Context checkpoint
    loader.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation restored in place.
    checkpoint_file : str
        Path to the checkpoint written by the preceding stage.
    n_beads : int or None, optional
        Number of ring-polymer beads for an RPMD run, or None for a
        classical one. Default is None.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist. Returning quietly here would let a
        batch job exit successfully having run no simulation at all.
    ValueError
        If an RPMD restart is legacy, corrupt, or incompatible with the current
        bead count, ordered topology, particle masses, or temperature.
    """
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_file} not found. Run the equilibration stage first, "
            "or pass checkpoint_file with the path it was written to."
        )

    print(f"Loading state from {checkpoint_file}...", flush=True)
    if n_beads is None:
        simulation.loadCheckpoint(checkpoint_file)
    else:
        _load_rpmd_restart(simulation, checkpoint_file, n_beads)


def run_openmm_relaxation(modeller,
                          forcefield,
                          output_prefix='minimized',
                          temperature=300.0 * unit.kelvin,
                          gamma=1.0 / unit.picosecond,
                          time_step=1.0 * unit.femtoseconds,
                          n_1=1_000,
                          n_2=1_000,
                          n_3=2_000,
                          backbone_names=None,
                          ks_1=100.0,
                          ks_2=10.0,
                          ks_3=0.0,
                          platform_name=None,
                          potential=None,
                          ml_idx=None,
                          calculator=None,
                          ):
    """
    Minimise in stages, easing the backbone restraints as it goes.

    Three successive minimisation stages are executed with decreasing restraint
    spring constants on backbone atoms, allowing the structure to relax gently.
    An optional ML/MM mixed potential can be used.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for the output PDB file. Default is ``'minimized'``.
    temperature : openmm.unit.Quantity, optional
        Temperature for the Langevin integrator. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    n_1 : int, optional
        Maximum iterations for stage 1 (strong restraints). Default is 1000.
    n_2 : int, optional
        Maximum iterations for stage 2 (weak restraints). Default is 1000.
    n_3 : int, optional
        Maximum iterations for stage 3 (unrestrained). Default is 2000.
    backbone_names : list of str or None, optional
        Atom names considered backbone for restraints. If None, defaults to
        ``['CA', 'C', 'N', 'P', 'O3']``.
    ks_1 : float, optional
        Spring constant for stage 1 in kJ/mol/nm^2. Default is 100.0.
    ks_2 : float, optional
        Spring constant for stage 2 in kJ/mol/nm^2. Default is 10.0.
    ks_3 : float, optional
        Spring constant for stage 3 in kJ/mol/nm^2. Default is 0.0.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    current_positions = modeller.positions
    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 0.0)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    atom_indices = []
    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            pos = current_positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
            atom_indices.append(atom.index)

    system.addForce(restraint)
    print(f"Restraints applied to {len(atom_indices)} backbone atoms.", flush=True)
    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(current_positions)

    print(f"\n--- Stage 1: Strong Backbone Restraints ({ks_1} kJ/mol/nm^2) ---", flush=True)
    k_strong = ks_1 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_strong)
    simulation.minimizeEnergy(maxIterations=n_1)

    print(f"\n--- Stage 2: Weak Backbone Restraints ({ks_2} kJ/mol/nm^2) ---", flush=True)
    k_weak = ks_2 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_weak)
    simulation.minimizeEnergy(maxIterations=n_2)

    print(f"\n--- Stage 3: Unrestrained Relaxation ({ks_3} kJ/mol/nm^2) ---", flush=True)
    k_vweak = ks_3 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_vweak)
    simulation.minimizeEnergy(maxIterations=n_3)

    _save_final_state(simulation, output_prefix, save_checkpoint=False)
    print(f"\nProcess complete. Saved to {output_prefix}", flush=True)


def run_openmm_relaxation_simple(modeller,
                                 forcefield,
                                 output_prefix='minimized',
                                 temperature=300.0 * unit.kelvin,
                                 gamma=1.0 / unit.picosecond,
                                 time_step=1.0 * unit.femtoseconds,
                                 platform_name=None,
                                 potential=None,
                                 ml_idx=None,
                                 calculator=None,
                                 ):
    """
    Perform a simple, unrestrained energy minimisation.

    Sets up a system with a Langevin integrator, runs ``minimizeEnergy``,
    and saves the minimised structure and checkpoint. Optionally uses an
    ML/MM mixed potential.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for output files (PDB, checkpoint, log). Default is ``'minimized'``.
    temperature : openmm.unit.Quantity, optional
        Temperature for the Langevin integrator. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    """
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print("Minimizing energy", flush=True)
    simulation.minimizeEnergy()

    _save_final_state(simulation, output_prefix)
    print(f"\nProcess complete. Saved to {output_prefix}", flush=True)


def run_openmm_heating(modeller,
                       forcefield,
                       output_prefix='equilibrate',
                       k1=100.0,
                       backbone_names=None,
                       target_temp=300.0 * unit.kelvin,
                       temp_step=50.0 * unit.kelvin,
                       gamma=1.0 / unit.picosecond,
                       time_step=1.0 * unit.femtoseconds,
                       n_report=1_000,
                       steps_per_stage=5_000,
                       steps_final=5_000,
                       platform_name=None,
                       deuterate=False,
                       deuterate_option='water',
                       potential=None,
                       ml_idx=None,
                       calculator=None,
                       ):
    """
    Heat a system from 0 K to temperature, under backbone restraints.

    The temperature is incremented in steps of ``temp_step`` until
    ``target_temp`` is reached, followed by a final equilibration stage at the
    target temperature. Backbone atoms are harmonically restrained throughout.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for output files. Default is ``'equilibrate'``.
    k1 : float, optional
        Backbone restraint spring constant in kJ/mol/nm^2. Default is 100.0.
    backbone_names : list of str or None, optional
        Atom names considered backbone for restraints. If None, defaults to
        ``['CA', 'C', 'N', 'P', 'O3']``.
    target_temp : openmm.unit.Quantity, optional
        Target temperature. Default is 300.0 K.
    temp_step : openmm.unit.Quantity, optional
        Temperature increment per heating stage. Default is 50.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps_per_stage : int, optional
        Number of MD steps per heating stage. Default is 5000.
    steps_final : int, optional
        Number of MD steps for the final equilibration at target temperature.
        Default is 5000.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate (e.g. ``'water'``, ``'all'``).
        Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    print("Applying backbone restraints for heating...", flush=True)
    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", k1 * unit.kilojoules_per_mole / (unit.nanometer ** 2))
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            restraint.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(restraint)

    current_temp = 0 * unit.kelvin
    integrator = openmm.LangevinMiddleIntegrator(current_temp,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print(f"\n--- Starting Gentle Heating (0K -> {target_temp}) ---", flush=True)
    _add_standard_reporters(simulation, output_prefix, n_report, pdb_steps=True)

    # The ramp stops one step short of the target and the target gets a stage of
    # its own, so that a target which is not a whole multiple of temp_step is
    # still reached exactly rather than being left at the multiple below it.
    temp = temp_step
    while temp < target_temp:
        print(f"\n-> Heating to {temp}...", flush=True)
        integrator.setTemperature(temp)
        if temp == temp_step:
            simulation.context.setVelocitiesToTemperature(temp)
        simulation.step(steps_per_stage)
        temp += temp_step

    print(f"\n-> Heating to {target_temp}...", flush=True)
    integrator.setTemperature(target_temp)
    if target_temp <= temp_step:
        # The ramp never ran, so this stage is also where the velocities start.
        simulation.context.setVelocitiesToTemperature(target_temp)
    simulation.step(steps_per_stage)
    print("\n--- Heating Complete ---", flush=True)
    print(f"Running final equilibration at {target_temp} for {steps_final} steps...", flush=True)
    simulation.step(steps_final)

    _save_final_state(simulation, output_prefix)
    print(f"Saved equilibrated structure to {output_prefix}", flush=True)


def run_openmm_npt(modeller,
                   forcefield,
                   output_prefix='npt_equilibrate',
                   pressure=1.0 * unit.bar,
                   temperature=300.0 * unit.kelvin,
                   gamma=1.0 / unit.picosecond,
                   time_step=1.0 * unit.femtoseconds,
                   barostat_freq=50,
                   backbone_names=None,
                   k=10.0,
                   n_report=500,
                   n_1=5_000,
                   n_2=15_000,
                   platform_name=None,
                   deuterate=False,
                   deuterate_option='water',
                   potential=None,
                   ml_idx=None,
                   calculator=None,
                   ):
    """
    Run a two-phase NPT density equilibration.

    Phase 1 applies backbone restraints while relaxing the box density under a
    Monte Carlo barostat. Phase 2 removes restraints and continues the NPT
    simulation. An optional ML/MM mixed potential is supported.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for output files. Default is ``'npt_equilibrate'``.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the barostat. Default is 1.0 bar.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    barostat_freq : int or None, optional
        Barostat attempt frequency in steps. If None, no barostat is added.
        Default is 50.
    backbone_names : list of str or None, optional
        Atom names considered backbone for restraints. If None, defaults to
        ``['CA', 'C', 'N', 'P', 'O3']``.
    k : float, optional
        Backbone restraint spring constant in kJ/mol/nm^2. Default is 10.0.
    n_report : int, optional
        Reporter interval in steps. Default is 500.
    n_1 : int, optional
        Number of steps for restrained NPT (Phase 1). Default is 5000.
    n_2 : int, optional
        Number of steps for unrestrained NPT (Phase 2). Default is 15000.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", k * unit.kilojoules_per_mole / (unit.nanometer ** 2))
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    atom_indices = []
    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            restraint.addParticle(atom.index, modeller.positions[atom.index])
            atom_indices.append(atom.index)
    system.addForce(restraint)

    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    _add_standard_reporters(simulation, output_prefix, n_report, pdb_steps=True,
                            stdout_volume=True)

    print("\n--- Phase 1: Restrained NPT (Relaxing Density) ---", flush=True)
    simulation.step(n_1)

    print("\n--- Phase 2: Removing Restraints (Unrestrained NPT) ---", flush=True)
    simulation.context.setParameter("k", 0.0)
    simulation.step(n_2)

    _save_final_state(simulation, output_prefix)

    print(f"\nDensity equilibration complete. Saved to {output_prefix}", flush=True)


def run_openmm_prod(modeller,
                    forcefield,
                    plumed_script_path=None,
                    pressure=1.0 * unit.bar,
                    temperature=300.0 * unit.kelvin,
                    gamma=1.0 / unit.picosecond,
                    time_step=1.0 * unit.femtoseconds,
                    barostat_freq=50,
                    n_report=1_000,
                    steps=500_000,
                    output_prefix='prod',
                    platform_name=None,
                    deuterate=False,
                    deuterate_option='water',
                    potential=None,
                    ml_idx=None,
                    calculator=None,
                    ):
    """
    Run an NPT production MD simulation, optionally with PLUMED enhanced sampling.

    Sets up the system with a Langevin integrator and optional Monte Carlo
    barostat, loads an optional PLUMED bias script, and runs the production
    trajectory.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_script_path : str or None, optional
        Path to a PLUMED input script. If None, no PLUMED bias is applied.
        Default is None.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the barostat. Default is 1.0 bar.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    barostat_freq : int or None, optional
        Barostat attempt frequency in steps. If None, no barostat is added.
        Default is 50.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps : int, optional
        Total number of production MD steps. Default is 500000.
    output_prefix : str, optional
        Prefix for output files (PDB, checkpoint, log). Default is ``'prod'``.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    """
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    _load_plumed(system, plumed_script_path)
    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    _add_standard_reporters(simulation, output_prefix, n_report, pdb_steps=True,
                            checkpoint_interval=n_report * 10)
    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    _save_final_state(simulation, output_prefix)


def run_openmm_steered(modeller,
                       forcefield,
                       plumed_input,
                       steps,
                       output_prefix='smd',
                       temperature=300.0 * unit.kelvin,
                       gamma=1.0 / unit.picosecond,
                       time_step=0.5 * unit.femtoseconds,
                       n_report=100,
                       pressure=1.0 * unit.bar,
                       barostat_freq=None,
                       platform_name=None,
                       deuterate=False,
                       deuterate_option='water',
                       potential=None,
                       ml_idx=None,
                       calculator=None,
                       ):
    """
    Run a steered MD simulation, dragging a collective variable with PLUMED.

    This is :func:`run_openmm_prod` with the settings a pulling run wants: no
    barostat, a short time step, and frequent reporting so the trajectory has
    enough frames to pick a path out of. The PLUMED script comes from
    :func:`reactiontools.plumed_input_steered` or one of its wrappers, and
    the trajectory feeds :func:`reactiontools.path_from_steered_md`.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions, equilibrated at
        the reactant.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_input : str
        The PLUMED script itself, or the path to a file holding one. Anything
        containing a newline is taken to be a script and written to
        ``'{output_prefix}_plumed.dat'``.
    steps : int
        Number of MD steps to run. Use the step count returned alongside the
        script by the ``plumed_input_steered*`` builders, so the run covers the
        whole pulling schedule.
    output_prefix : str, optional
        Prefix for output files. Default is ``'smd'``.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 0.5 fs, since pulling a proton
        across a hydrogen bond is not gentle.
    n_report : int, optional
        Reporter interval in steps. Default is 100. Match this to the PLUMED
        ``PRINT`` stride so that every frame has a CV value.
    pressure : openmm.unit.Quantity, optional
        Target pressure, only used if a barostat is asked for. Default is
        1.0 bar.
    barostat_freq : int or None, optional
        Barostat attempt frequency in steps. Default is None, i.e. pull at
        constant volume.
    platform_name : str, optional
        OpenMM platform name. Default is None, which auto-detects.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.

    Returns
    -------
    str
        Path to the trajectory written by the run.
    """
    if '\n' in plumed_input:
        plumed_script_path = f'{output_prefix}_plumed.dat'
        with open(plumed_script_path, 'w') as f:
            f.write(plumed_input)
    else:
        plumed_script_path = plumed_input

    print(f"Starting steered MD for {steps} steps...", flush=True)
    run_openmm_prod(modeller,
                    forcefield,
                    plumed_script_path=plumed_script_path,
                    pressure=pressure,
                    temperature=temperature,
                    gamma=gamma,
                    time_step=time_step,
                    barostat_freq=barostat_freq,
                    n_report=n_report,
                    steps=steps,
                    output_prefix=output_prefix,
                    platform_name=platform_name,
                    deuterate=deuterate,
                    deuterate_option=deuterate_option,
                    potential=potential,
                    ml_idx=ml_idx,
                    calculator=calculator)

    traj_file = f'{output_prefix}_steps.pdb'
    print(f"Steered trajectory written to {traj_file}", flush=True)
    return traj_file


def _split_rpmd_seed(seed):
    """
    Derive independent NumPy and OpenMM seeds from one master seed.

    Bead initialisation and the PILE thermostat draw from separate streams,
    so one master seed is split rather than reused: sharing it would
    correlate the starting ring polymer with its thermostat noise.

    Parameters
    ----------
    seed : int or None
        Non-negative master seed, or None to leave both streams
        non-deterministic.

    Returns
    -------
    initialization_seed : int or None
        Seed for the NumPy generator that places the beads.
    thermostat_seed : int or None
        Seed for the OpenMM PILE thermostat. Never zero, which OpenMM reads
        as a request for a non-deterministic seed.

    Raises
    ------
    ValueError
        If *seed* is a bool, not an integer, or negative.
    """
    if seed is None:
        return None, None
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")

    initialization_sequence, thermostat_sequence = np.random.SeedSequence(
        int(seed)
    ).spawn(2)
    initialization_seed = int(
        initialization_sequence.generate_state(1, dtype=np.uint32)[0]
    )
    thermostat_seed = int(
        thermostat_sequence.generate_state(1, dtype=np.uint32)[0]
    ) % 2_147_483_647
    # OpenMM assigns a non-deterministic seed when given 0, so reserve zero for
    # the seed=None path and keep every explicit master seed reproducible.
    if thermostat_seed == 0:
        thermostat_seed = 1
    return initialization_seed, thermostat_seed


def run_openmm_rpmd_equilibration(modeller,
                                  forcefield,
                                  output_prefix='rpmd_ready',
                                  n_beads=32,
                                  temperature=300 * unit.kelvin,
                                  friction=1.0 / unit.picosecond,
                                  timestep=0.5 * unit.femtoseconds,
                                  n_report=1_000,
                                  n_1=1_000,
                                  n_2=5_000,
                                  platform_name=None,
                                  deuterate=False,
                                  deuterate_option='water',
                                  potential=None,
                                  ml_idx=None,
                                  calculator=None,
                                  atoms_to_watch=None,
                                  scale_factor=1.0,
                                  seed=None,
                                  expansion_metric="rms",
                                  distance_pairs_to_watch=None):
    """
    Equilibrate a ring-polymer molecular dynamics (RPMD) simulation.

    Performs a two-stage equilibration: stage 1 uses a reduced time step for
    gentle bead expansion, and stage 2 runs at the full time step. A
    checkpoint is saved at the end for use by subsequent production runs.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for output files. Default is ``'rpmd_ready'``.
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300 K.
    friction : openmm.unit.Quantity, optional
        RPMD friction coefficient. Default is 1.0 / ps.
    timestep : openmm.unit.Quantity, optional
        Integration time step. Default is 0.5 fs.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    n_1 : int, optional
        Number of steps for stage 1 (bead expansion). Default is 1000.
    n_2 : int, optional
        Number of steps for stage 2 (relaxation). Default is 5000.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    atoms_to_watch : list of int or None, optional
        Atom indices for quantum spread monitoring. Default is None.
    scale_factor : float, optional
        Multiplier applied to the exact free-ring-polymer normal-mode position
        amplitudes. Default is 1.0.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for initial positions/velocities and the PILE thermostat. If None,
        NumPy and OpenMM select independent seeds. Default is None.
    expansion_metric : {"rms", "mean"}, optional
        Spread metric written for *atoms_to_watch*. ``"mean"`` is the mean
        bead-centroid degree of expansion; ``"rms"`` preserves the existing
        radius-of-gyration output. Default is ``"rms"``.
    distance_pairs_to_watch : iterable of pair of int or None, optional
        Atom pairs whose centroid distances are written alongside the spread
        values. Requires *atoms_to_watch*. Default is None.
    """
    initialization_seed, thermostat_seed = _split_rpmd_seed(seed)
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep)
    if thermostat_seed is not None:
        integrator.setRandomNumberSeed(thermostat_seed)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    _add_rpmd_reporters(
        simulation,
        modeller.topology,
        output_prefix,
        n_report,
        n_beads,
        atoms_to_watch,
        expansion_metric=expansion_metric,
        distance_pairs=distance_pairs_to_watch,
    )
    _add_rpmd_progress_reporters(simulation, output_prefix, n_report)

    init_beads(
        modeller,
        simulation,
        n_beads,
        scale_factor=scale_factor,
        seed=initialization_seed,
    )

    print("\n--- Stage 1: Bead Expansion  ---", flush=True)
    integrator.setStepSize(timestep * 0.5)
    step_rpmd(simulation, n_1)

    print(f"\n--- Stage 2: Relaxation at full timestep ({timestep}) ---", flush=True)
    integrator.setStepSize(timestep)
    step_rpmd(simulation, n_2)

    print("\n--- Saving State ---", flush=True)
    # Not '_centroid.pdb': RPMDCentroidReporter owns that name and is still
    # holding it open, so writing here would truncate the trajectory it spent
    # the whole run building.
    _save_final_state(simulation, output_prefix, pdb_suffix='_final.pdb', n_beads=n_beads)
    print(f"Saved final centroid structure to {output_prefix}_final.pdb", flush=True)


def run_openmm_rpmd_contracted(modeller,
                               forcefield,
                               plumed_script_path=None,
                               checkpoint_file='rpmd_ready.chk',
                               output_prefix='rpmd_prod_contracted',
                               n_beads=32,
                               temperature=300 * unit.kelvin,
                               pressure=1.0 * unit.bar,
                               barostat_freq=50,
                               friction=1.0 / unit.picosecond,
                               timestep=0.5 * unit.femtoseconds,
                               steps=100_000,
                               n_report=1_000,
                               contractions=None,
                               platform_name=None,
                               deuterate=False,
                               deuterate_option='water',
                               potential=None,
                               ml_idx=None,
                               atoms_to_watch=None,
                               calculator=None,
                               expansion_metric="rms",
                               distance_pairs_to_watch=None):
    """
    Run a contracted ring-polymer MD (RPMD) production simulation.

    Uses the ring-polymer contraction scheme to evaluate expensive force
    components (e.g. PME reciprocal space) on fewer bead copies, reducing
    computational cost. A checkpoint from a prior RPMD equilibration is
    required. An optional PLUMED bias and ML/MM mixed potential are supported.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_script_path : str or None, optional
        Path to a PLUMED input script. If None, no bias is applied.
        Default is None.
    checkpoint_file : str, optional
        Path to the equilibration checkpoint. Default is ``'rpmd_ready.chk'``.
    output_prefix : str, optional
        Prefix for output files. Default is ``'rpmd_prod_contracted'``.
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300 K.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the RPMD barostat. Default is 1.0 bar.
    barostat_freq : int or None, optional
        RPMD barostat attempt frequency. If None, no barostat is added.
        Default is 50.
    friction : openmm.unit.Quantity, optional
        RPMD friction coefficient. Default is 1.0 / ps.
    timestep : openmm.unit.Quantity, optional
        Integration time step. Default is 0.5 fs.
    steps : int, optional
        Total number of production steps. Default is 100000.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    contractions : dict or None, optional
        Mapping of force group to the number of contracted copies. If None,
        defaults to ``{1: 8, 2: 1}``. Default is None.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    atoms_to_watch : list of int or None, optional
        Atom indices for quantum spread monitoring. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    expansion_metric : {"rms", "mean"}, optional
        Spread metric written for *atoms_to_watch*. Use ``"mean"`` for the
        Figure-7 degree of expansion. Default is ``"rms"``.
    distance_pairs_to_watch : iterable of pair of int or None, optional
        Atom pairs whose centroid distances are written alongside the spread
        values. Requires *atoms_to_watch*. Default is None.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist.
    ValueError
        If an ML potential or calculator is given without *ml_idx*.
    """
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    if contractions is None:
        # Each copy count must divide n_beads. Groups left out of the dict --
        # here group 0, the cheap bonded forces -- run on every bead.
        contractions = {
            1: 8,  # nonbonded direct space
            2: 1  # PME reciprocal space, on the centroid alone
        }

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.RPMDMonteCarloBarostat(pressure, barostat_freq))

    _load_plumed(system, plumed_script_path)

    # Contraction is keyed on force group, so the forces have to be sorted into
    # the groups `contractions` names: the costlier the force, the fewer beads
    # it should be evaluated on.
    print("Assigning force groups for contraction...", flush=True)

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            # One force object covering two costs, so its direct and reciprocal
            # halves are split across groups and contracted separately.
            force.setForceGroup(1)
            force.setReciprocalSpaceForceGroup(2)
            print(f"  - {force.__class__.__name__}: Direct->Group 1, Reciprocal->Group 2")

        elif isinstance(force, (openmm.HarmonicBondForce,
                                openmm.HarmonicAngleForce,
                                openmm.PeriodicTorsionForce,
                                openmm.RBTorsionForce,
                                openmm.CMAPTorsionForce)):
            force.setForceGroup(0)
            print(f"  - {force.__class__.__name__}: Group 0")

        else:
            force.setForceGroup(0)

    print(f"\nInitializing RPMDIntegrator with contractions: {contractions}", flush=True)
    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep, contractions)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    _load_checkpoint(simulation, checkpoint_file, n_beads=n_beads)

    _add_rpmd_reporters(
        simulation,
        modeller.topology,
        output_prefix,
        n_report,
        n_beads,
        atoms_to_watch,
        expansion_metric=expansion_metric,
        distance_pairs=distance_pairs_to_watch,
    )

    _add_rpmd_progress_reporters(simulation, output_prefix, n_report)

    print(f"\nStarting Production Run ({steps} steps)...")
    step_rpmd(simulation, steps)
    print("Done.", flush=True)

    print("\n--- Saving State ---", flush=True)
    # See run_openmm_rpmd_equilibration: '_centroid.pdb' belongs to the reporter.
    _save_final_state(simulation, output_prefix, pdb_suffix='_final.pdb', n_beads=n_beads)
    print(f"Saved final centroid structure to {output_prefix}_final.pdb", flush=True)


def run_openmm_rpmd_prod(modeller,
                         forcefield,
                         plumed_script_path=None,
                         checkpoint_file='rpmd_ready.chk',
                         output_prefix='rpmd_prod',
                         n_beads=32,
                         pressure=1.0 * unit.bar,
                         temperature=300.0 * unit.kelvin,
                         gamma=1.0 / unit.picosecond,
                         time_step=1.0 * unit.femtoseconds,
                         barostat_freq=50,
                         n_report=1_000,
                         steps=500_000,
                         platform_name=None,
                         deuterate=False,
                         deuterate_option='water',
                         potential=None,
                         ml_idx=None,
                         atoms_to_watch=None,
                         calculator=None,
                         expansion_metric="rms",
                         distance_pairs_to_watch=None):
    """
    Run a full ring-polymer MD (RPMD) production simulation.

    Loads a checkpoint from a prior RPMD equilibration and continues with a
    production run using the ``RPMDIntegrator``. An optional PLUMED bias,
    RPMD barostat, and ML/MM mixed potential are supported.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_script_path : str or None, optional
        Path to a PLUMED input script. If None, no bias is applied.
        Default is None.
    checkpoint_file : str, optional
        Path to the equilibration checkpoint. Default is ``'rpmd_ready.chk'``.
    output_prefix : str, optional
        Prefix for output files. Default is ``'rpmd_prod'``.
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the RPMD barostat. Default is 1.0 bar.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    barostat_freq : int or None, optional
        RPMD barostat attempt frequency. If None, no barostat is added.
        Default is 50.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps : int, optional
        Total number of production steps. Default is 500000.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    atoms_to_watch : list of int or None, optional
        Atom indices for quantum spread monitoring. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    expansion_metric : {"rms", "mean"}, optional
        Spread metric written for *atoms_to_watch*. Use ``"mean"`` for the
        Figure-7 degree of expansion. Default is ``"rms"``.
    distance_pairs_to_watch : iterable of pair of int or None, optional
        Atom pairs whose centroid distances are written alongside the spread
        values. Requires *atoms_to_watch*. Default is None.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist.
    ValueError
        If an ML potential or calculator is given without *ml_idx*.
    """
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.RPMDMonteCarloBarostat(pressure, barostat_freq))

    _load_plumed(system, plumed_script_path)
    integrator = openmm.RPMDIntegrator(n_beads,
                                       temperature,
                                       gamma,
                                       time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    _load_checkpoint(simulation, checkpoint_file, n_beads=n_beads)

    _add_rpmd_reporters(
        simulation,
        modeller.topology,
        output_prefix,
        n_report,
        n_beads,
        atoms_to_watch,
        expansion_metric=expansion_metric,
        distance_pairs=distance_pairs_to_watch,
    )

    _add_rpmd_progress_reporters(simulation, output_prefix, n_report)

    print(f"Starting production run for {steps} steps...", flush=True)
    step_rpmd(simulation, steps)
    print("Production run complete.", flush=True)

    _save_final_state(simulation, output_prefix, pdb_suffix='_final.pdb', n_beads=n_beads)


def run_openmm_adqtb_eq(modeller,
                        forcefield,
                        temperature=300.0 * unit.kelvin,
                        gamma=1.0 / unit.picosecond,
                        time_step=1.0 * unit.femtoseconds,
                        segment_length=0.5 * unit.picosecond,
                        adaptation_rate=0.5,
                        n_report=1_000,
                        steps=500_000,
                        output_prefix='adqtb_ready',
                        platform_name=None,
                        deuterate=False,
                        deuterate_option='water',
                        potential=None,
                        ml_idx=None,
                        calculator=None,
                        ):
    """
    Run an adaptive quantum thermal bath (adQTB) equilibration simulation.

    Uses the ``QTBIntegrator`` to thermalise the system with quantum thermal
    noise. A checkpoint is saved at the end for subsequent production runs.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    segment_length : openmm.unit.Quantity, optional
        Segment length for the QTB integrator. Default is 0.5 ps.
    adaptation_rate : float, optional
        Adaptation rate for the QTB integrator. Default is 0.5.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps : int, optional
        Total number of equilibration steps. Default is 500000.
    output_prefix : str, optional
        Prefix for output files. Default is ``'adqtb_ready'``.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    """
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    integrator = openmm.QTBIntegrator(temperature, gamma, time_step)
    integrator.setSegmentLength(segment_length)
    integrator.setDefaultAdaptationRate(adaptation_rate)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    _add_standard_reporters(simulation, output_prefix, n_report, pdb_steps=True,
                            checkpoint_interval=n_report * 10)
    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    _save_final_state(simulation, output_prefix)


def run_openmm_adqtb_prod(modeller,
                          forcefield,
                          plumed_script_path=None,
                          pressure=1.0 * unit.bar,
                          barostat_freq=50,
                          temperature=300.0 * unit.kelvin,
                          gamma=1.0 / unit.picosecond,
                          time_step=1.0 * unit.femtoseconds,
                          segment_length=0.5 * unit.picosecond,
                          adaptation_rate=0.5,
                          n_report=1_000,
                          steps=500_000,
                          output_prefix='adqtb_prod',
                          platform_name=None,
                          deuterate=False,
                          deuterate_option='water',
                          potential=None,
                          ml_idx=None,
                          calculator=None,
                          checkpoint_file='adqtb_ready.chk',
                          ):
    """
    Run an adaptive quantum thermal bath (adQTB) production simulation.

    Uses the ``QTBIntegrator`` with optional PLUMED enhanced-sampling bias
    and a Monte Carlo barostat for NPT conditions.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_script_path : str or None, optional
        Path to a PLUMED input script. If None, no bias is applied.
        Default is None.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the barostat. Default is 1.0 bar.
    barostat_freq : int or None, optional
        Barostat attempt frequency in steps. If None, no barostat is added.
        Default is 50.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    segment_length : openmm.unit.Quantity, optional
        Segment length for the QTB integrator. Default is 0.5 ps.
    adaptation_rate : float, optional
        Adaptation rate for the QTB integrator. Default is 0.5.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps : int, optional
        Total number of production steps. Default is 500000.
    output_prefix : str, optional
        Prefix for output files. Default is ``'adqtb_prod'``.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    checkpoint_file : str, optional
        Path to the adQTB equilibration checkpoint. The checkpoint contains
        the adapted friction spectrum as well as coordinates and velocities.
        Default is ``'adqtb_ready.chk'``.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist.
    """
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    _load_plumed(system, plumed_script_path)

    integrator = openmm.QTBIntegrator(temperature, gamma, time_step)
    integrator.setSegmentLength(segment_length)
    integrator.setDefaultAdaptationRate(adaptation_rate)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    _load_checkpoint(simulation, checkpoint_file)

    _add_standard_reporters(simulation, output_prefix, n_report, pdb_steps=True,
                            checkpoint_interval=n_report * 10)
    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    _save_final_state(simulation, output_prefix)
