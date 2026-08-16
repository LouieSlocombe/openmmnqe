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
import os
import sys
import tempfile

import numpy as np
import openmm.unit as unit
from openmmml import MLPotential
from openmmplumed import PlumedForce

from openmm import openmm, app
from .reporters import (RPMDQuantumSpreadReporter,
                        RPMDBeadReporter,
                        RPMDCentroidReporter,
                        )
from .tools import deuterate_system, check_platform, init_beads, centroid_positions


_RPMD_RESTART_KIND = "openmmnqe-rpmd-restart"
_RPMD_RESTART_VERSION = 1


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
    """Deuterate the system in place when requested."""
    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)


def _load_plumed(system, plumed_script_path):
    """Attach a PLUMED bias force to the system if a script path is given."""
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
    Append the standard reporter set shared by the drivers.

    Order: optional PDB trajectory, stdout state data, ``.log`` state data,
    optional periodic checkpoint.
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


def _add_rpmd_reporters(simulation, topology, output_prefix, n_report, n_beads,
                        atoms_to_watch):
    """Append the RPMD reporter trio (optional spread, centroid, beads)."""
    if atoms_to_watch is not None:
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
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
    """Atomically save every RPMD copy to a portable restart archive.

    OpenMM's ordinary ``Context`` checkpoint only sees the copy currently
    mirrored into the Context.  The other copies live in private arrays owned
    by ``RPMDIntegrator``, so they must be collected through its copy-specific
    API.  Positions are stored in nanometres and velocities in nanometres per
    picosecond.  The archive also carries the box, time, and step count needed
    to continue reporter schedules in a new ``Simulation``.
    """
    if n_beads <= 0:
        raise ValueError("n_beads must be positive")

    integrator = simulation.integrator
    actual_beads = integrator.getNumCopies()
    if actual_beads != n_beads:
        raise ValueError(
            f"n_beads={n_beads} does not match RPMDIntegrator copies={actual_beads}"
        )

    n_particles = simulation.system.getNumParticles()
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
    """Read and validate the format-independent portion of an RPMD restart."""
    try:
        archive = np.load(checkpoint_file, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(
            f"{checkpoint_file} is not a bead-aware openmmnqe RPMD restart. "
            "A generic OpenMM checkpoint cannot safely restore all RPMD copies; "
            "rerun RPMD equilibration to create a new checkpoint."
        ) from exc

    try:
        if not hasattr(archive, "files"):
            raise ValueError("restart is not an NPZ archive")
        required = {
            "kind",
            "format_version",
            "num_beads",
            "num_particles",
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
                "RPMD restart is missing fields: " + ", ".join(sorted(missing))
            )
        if str(archive["kind"].item()) != _RPMD_RESTART_KIND:
            raise ValueError("checkpoint is not an openmmnqe RPMD restart")
        version = int(archive["format_version"].item())
        if version != _RPMD_RESTART_VERSION:
            raise ValueError(
                f"Unsupported RPMD restart version {version}; "
                f"expected {_RPMD_RESTART_VERSION}"
            )
        return {
            "num_beads": int(archive["num_beads"].item()),
            "num_particles": int(archive["num_particles"].item()),
            "positions_nm": np.array(archive["positions_nm"], dtype=np.float64),
            "velocities_nm_per_ps": np.array(
                archive["velocities_nm_per_ps"], dtype=np.float64
            ),
            "periodic": bool(archive["periodic"].item()),
            "box_vectors_nm": np.array(
                archive["box_vectors_nm"], dtype=np.float64
            ),
            "time_ps": float(archive["time_ps"].item()),
            "step_count": int(archive["step_count"].item()),
        }
    finally:
        if hasattr(archive, "close"):
            archive.close()


def _load_rpmd_restart(simulation, checkpoint_file, n_beads):
    """Restore every RPMD copy before the integrator takes its first step."""
    restart = _read_rpmd_restart(checkpoint_file)
    integrator = simulation.integrator
    integrator_beads = integrator.getNumCopies()
    n_particles = simulation.system.getNumParticles()
    periodic = simulation.system.usesPeriodicBoundaryConditions()

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
    if restart["step_count"] < 0:
        raise ValueError("RPMD restart step count cannot be negative")
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

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist. Returning quietly here would let a
        batch job exit successfully having run no simulation at all.
    ValueError
        If an RPMD restart is legacy, corrupt, or incompatible with the current
        bead or particle count.
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
                                  atoms_to_watch=None):
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
    """
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    _add_rpmd_reporters(simulation, modeller.topology, output_prefix, n_report,
                        n_beads, atoms_to_watch)
    _add_standard_reporters(simulation, output_prefix, n_report)

    init_beads(modeller, simulation, n_beads)

    print("\n--- Stage 1: Bead Expansion  ---", flush=True)
    integrator.setStepSize(timestep * 0.5)
    simulation.step(n_1)

    print(f"\n--- Stage 2: Relaxation at full timestep ({timestep}) ---", flush=True)
    integrator.setStepSize(timestep)
    simulation.step(n_2)

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
                               calculator=None):
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

    _add_rpmd_reporters(simulation, modeller.topology, output_prefix, n_report,
                        n_beads, atoms_to_watch)

    _add_standard_reporters(simulation, output_prefix, n_report)

    print(f"\nStarting Production Run ({steps} steps)...")
    simulation.step(steps)
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
                         calculator=None):
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

    _add_rpmd_reporters(simulation, modeller.topology, output_prefix, n_report,
                        n_beads, atoms_to_watch)

    _add_standard_reporters(simulation, output_prefix, n_report)

    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
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
