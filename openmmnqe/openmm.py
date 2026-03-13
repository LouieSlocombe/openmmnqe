import os
import sys

import openmm.unit as unit
from openmm import openmm, app
from openmmplumed import PlumedForce

from .reporters import (RPMDQuantumSpreadReporter,
                        RPMDBeadReporter,
                        RPMDCentroidReporter,
                        )
from .tools import deuterate_system, check_platform


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
    Perform a staged energy minimisation with progressively weaker backbone restraints.

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
        Spring constant for stage 1 in kJ/mol/nm². Default is 100.0.
    ks_2 : float, optional
        Spring constant for stage 2 in kJ/mol/nm². Default is 10.0.
    ks_3 : float, optional
        Spring constant for stage 3 in kJ/mol/nm². Default is 0.0.
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.

    Returns
    -------
    None
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )

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

    print("\n--- Stage 2: Weak Backbone Restraints (10 kJ/mol/nm^2) ---", flush=True)
    k_weak = ks_2 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_weak)
    simulation.minimizeEnergy(maxIterations=n_2)

    print("\n--- Stage 3: Unrestrained Relaxation ---", flush=True)
    k_vweak = ks_3 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_vweak)
    simulation.minimizeEnergy(maxIterations=n_3)

    final_state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)
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
    platform_name : str, optional
        OpenMM platform name. Default is ``'CUDA'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.

    Returns
    -------
    None
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )

    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # Local energy minimization
    print("Minimizing energy", flush=True)
    simulation.minimizeEnergy()

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    final_state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)
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
                       steps_final=10_000,
                       platform_name=None,
                       deuterate=False,
                       deuterate_option='water',
                       potential=None,
                       ml_idx=None,
                       calculator=None,
                       ):
    """
    Gently heat a system from 0 K to the target temperature with backbone restraints.

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
        Backbone restraint spring constant in kJ/mol/nm². Default is 100.0.
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
        Default is 10000.
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate (e.g. ``'water'``, ``'all'``).
        Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.

    Returns
    -------
    None
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None
    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

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
    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    temp = temp_step
    while temp <= target_temp:
        print(f"\n-> Heating to {temp}...", flush=True)
        integrator.setTemperature(temp)
        if temp == temp_step:
            simulation.context.setVelocitiesToTemperature(temp)
        simulation.step(steps_per_stage)
        temp += temp_step
    print("\n--- Heating Complete ---", flush=True)
    print(f"Running final equilibration at {target_temp} for {steps_final} steps...", flush=True)
    simulation.step(steps_final)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
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
                   n_2=25_000,
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
        Backbone restraint spring constant in kJ/mol/nm². Default is 10.0.
    n_report : int, optional
        Reporter interval in steps. Default is 500.
    n_1 : int, optional
        Number of steps for restrained NPT (Phase 1). Default is 5000.
    n_2 : int, optional
        Number of steps for unrestrained NPT (Phase 2). Default is 25000.
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.

    Returns
    -------
    None
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

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

    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print("\n--- Phase 1: Restrained NPT (Relaxing Density) ---", flush=True)
    simulation.step(n_1)

    print("\n--- Phase 2: Removing Restraints (Unrestrained NPT) ---", flush=True)
    simulation.context.setParameter("k", 0.0)
    simulation.step(n_2)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)

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
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.

    Returns
    -------
    None
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path, 'r') as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)
    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    simulation.reporters.append(app.CheckpointReporter(f'{output_prefix}.chk', n_report * 10))
    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)


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
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
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

    Returns
    -------
    None
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    if atoms_to_watch is not None:
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
        ))

    simulation.reporters.append(RPMDCentroidReporter(
        topology=modeller.topology,
        file_name=f"{output_prefix}_centroid.pdb",
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name=output_prefix,
        reportInterval=n_report,
        num_beads=n_beads,
    ))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print("\n--- Stage 1: Bead Expansion  ---", flush=True)
    integrator.setStepSize(timestep * 0.5)
    simulation.step(n_1)

    print(f"\n--- Stage 2: Relaxation at full timestep ({timestep}) ---", flush=True)
    integrator.setStepSize(timestep)
    simulation.step(n_2)

    print("\n--- Saving State ---", flush=True)
    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}_centroid.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    print(f"Saved centroid visualization to {output_prefix}_centroid.pdb", flush=True)


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
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
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

    Returns
    -------
    None
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if contractions is None:
        # Note: NumCopies must be a divisor of num_beads (32).
        # Valid divisors for 32: 1, 2, 4, 8, 16, 32.
        contractions = {
            1: 8,  # Nonbonded Direct Space (calculate on every 4th bead)
            2: 1  # PME Reciprocal Space (calculate only on centroid)
        }
        # Group 0 is not in the dict, so it defaults to num_beads (32)

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.RPMDMonteCarloBarostat(pressure, barostat_freq))

    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path, 'r') as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)

    # We must assign specific forces to specific integer groups (0-31).
    # Group 0: Bonded Forces (Cheap) -> 32 copies (Implicit default)
    # Group 1: Nonbonded Direct Space (Expensive) -> 8 copies
    # Group 2: PME Reciprocal Space (Very Expensive) -> 1 copy

    print("Assigning force groups for contraction...", flush=True)

    for force in system.getForces():
        # Check for NonbondedForce (Handles VdW + Coulomb)
        if isinstance(force, openmm.NonbondedForce):
            # Set the Direct Space calculation to Group 1
            force.setForceGroup(1)
            # Set the Reciprocal Space (PME) calculation to Group 2
            force.setReciprocalSpaceForceGroup(2)
            print(f"  - {force.__class__.__name__}: Direct->Group 1, Reciprocal->Group 2")

        # Check for Bonded Forces (HarmonicBond, Angle, Torsion, etc.)
        elif isinstance(force, (openmm.HarmonicBondForce,
                                openmm.HarmonicAngleForce,
                                openmm.PeriodicTorsionForce,
                                openmm.RBTorsionForce,
                                openmm.CMAPTorsionForce)):
            force.setForceGroup(0)
            print(f"  - {force.__class__.__name__}: Group 0")

        # Barostat and others
        else:
            force.setForceGroup(0)

    print(f"\nInitializing RPMDIntegrator with contractions: {contractions}", flush=True)
    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep, contractions)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint {checkpoint_file} not found. Run equilibration first.", flush=True)
        return

    print(f"Loading state from {checkpoint_file}...", flush=True)
    simulation.loadCheckpoint(checkpoint_file)

    if atoms_to_watch is not None:
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
        ))

    simulation.reporters.append(RPMDCentroidReporter(
        topology=modeller.topology,
        file_name=f"{output_prefix}_centroid.pdb",
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name=output_prefix,
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print(f"\nStarting Production Run ({steps} steps)...")
    simulation.step(steps)
    print("Done.", flush=True)

    print("\n--- Saving State ---", flush=True)
    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}_centroid.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    print(f"Saved centroid visualization to {output_prefix}_centroid.pdb", flush=True)


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
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
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

    Returns
    -------
    None
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.RPMDMonteCarloBarostat(pressure, barostat_freq))

    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path, 'r') as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)
    integrator = openmm.RPMDIntegrator(n_beads,
                                       temperature,
                                       gamma,
                                       time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint {checkpoint_file} not found. Run equilibration first.", flush=True)
        return

    print(f"Loading state from {checkpoint_file}...", flush=True)
    simulation.loadCheckpoint(checkpoint_file)

    if atoms_to_watch is not None:
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
        ))

    simulation.reporters.append(RPMDCentroidReporter(
        topology=modeller.topology,
        file_name=f"{output_prefix}_centroid.pdb",
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name=output_prefix,
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)


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
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.

    Returns
    -------
    None
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    integrator = openmm.QTBIntegrator(temperature, gamma, time_step)
    integrator.setSegmentLength(segment_length)
    integrator.setDefaultAdaptationRate(adaptation_rate)
    # set_adqtb_particle_types_by_element(integrator,
    #                                     topology=modeller.topology,
    #                                     system=system)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    simulation.reporters.append(app.CheckpointReporter(f'{output_prefix}.chk', n_report * 10))
    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)


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
    platform_name : str, optional
        OpenMM platform name. Default is ``'CPU'``.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.

    Returns
    -------
    None
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
            calculator=calculator,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    if barostat_freq is not None:
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path, 'r') as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)

    integrator = openmm.QTBIntegrator(temperature, gamma, time_step)
    integrator.setSegmentLength(segment_length)
    integrator.setDefaultAdaptationRate(adaptation_rate)
    # set_adqtb_particle_types_by_element(integrator,
    #                                     topology=modeller.topology,
    #                                     system=system)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    simulation.reporters.append(app.CheckpointReporter(f'{output_prefix}.chk', n_report * 10))
    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
