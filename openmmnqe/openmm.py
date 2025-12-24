import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import openmm.unit as unit
from openmm import openmm, app
from openmmplumed import PlumedForce
from openmmtools.integrators import GeodesicBAOABIntegrator

from .plotting import n_plot
from .tools import deuterate_system


def md_workflow(file_in,
                ff='amber19-all.xml',  # charmm36_2024.xml amber19-all.xml
                water_model='amber19/opc3.xml',  # charmm36_2024/tip5p.xml amber19/opc3.xml
                padding=1.0,
                temperature=300.0,
                pressure=1.0,
                friction_coeff=1.0,
                time_step=0.004,
                report_pdb=1_000,
                report_std=1_000,
                report_data=100,
                file_out='output.pdb',
                data_out='md_log.txt',
                n_nvt=10_000,
                n_npt=50_000,
                box_shape='dodecahedron',
                gbaoab=True,
                platform='CPU',
                ):
    # Prepare system
    pdb = app.PDBFile(file_in)
    forcefield = app.ForceField(ff, water_model)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield)

    # Solvate
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    # Setup system and integrator
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=app.HBonds)

    if gbaoab:
        integrator = GeodesicBAOABIntegrator(temperature * unit.kelvin,
                                             friction_coeff / unit.picosecond,
                                             time_step * unit.picoseconds)
    else:
        integrator = openmm.LangevinIntegrator(temperature * unit.kelvin,
                                               friction_coeff / unit.picosecond,
                                               time_step * unit.picoseconds)

    platform = openmm.Platform.getPlatformByName(platform)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # Local energy minimization
    print("Minimizing energy", flush=True)
    simulation.minimizeEnergy()

    # Setup reporting
    simulation.reporters.append(app.PDBReporter(file_out, report_pdb))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      report_std,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True))
    simulation.reporters.append(app.StateDataReporter(data_out,
                                                      report_data,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    # NVT equilibration
    print("Running NVT", flush=True)
    simulation.step(n_nvt)

    # NPT production MD
    system.addForce(openmm.MonteCarloBarostat(pressure * unit.bar,
                                              temperature * unit.kelvin))
    simulation.context.reinitialize(preserveState=True)
    print("Running NPT", flush=True)
    simulation.step(n_npt)

    return None


def md_analysis(file_in='md_log.txt'):
    # Analysis
    data = np.loadtxt(file_in, delimiter=',')

    step = data[:, 0]
    time = data[:, 1]
    potential_energy = data[:, 2]
    kinetic_energy = data[:, 3]
    total_energy = data[:, 4]
    temperature = data[:, 5]
    volume = data[:, 6]

    plt.plot(time, potential_energy, lw=2)
    n_plot('Time (ps)', 'Potential Energy (kJ/mol)')
    plt.show()

    plt.plot(time, kinetic_energy, lw=2)
    n_plot('Time (ps)', 'Kinetic Energy (kJ/mol)')
    plt.show()

    plt.plot(time, total_energy, lw=2)
    n_plot('Time (ps)', 'Total Energy (kJ/mol)')
    plt.show()

    plt.plot(time, temperature, lw=2)
    n_plot('Time (ps)', 'Temperature (K)')
    plt.show()

    plt.plot(time, volume, lw=2)
    n_plot('Time (ps)', 'Volume (nm^3)')
    plt.show()


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
                          platform_name='CPU',
                          potential=None,
                          ml_idx=None,
                          ):
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
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
                                 n_report=1,
                                 platform_name='CUDA',
                                 potential=None,
                                 ml_idx=None,
                                 ):
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

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
                       platform_name='CPU',
                       deuterate=False,
                       deuterate_option='water',
                       potential=None,
                       ml_idx=None,
                       ):
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
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
                   output_prefix='npt_equilibrated',
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
                   platform_name='CPU',
                   deuterate=False,
                   deuterate_option='water',
                   potential=None,
                   ml_idx=None,
                   ):
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    print("Adding MonteCarloBarostat...", flush=True)
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
                    platform_name='CPU',
                    deuterate=False,
                    deuterate_option='water',
                    potential=None,
                    ml_idx=None,
                    ):
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

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
                                  platform_name='CPU',
                                  deuterate=False,
                                  deuterate_option='water',
                                  potential=None,
                                  ml_idx=None,
                                  atoms_to_watch=None):
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
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
                               output_prefix='prod_contracted',
                               n_beads=32,
                               temperature=300 * unit.kelvin,
                               pressure=1.0 * unit.bar,
                               barostat_freq=50,
                               friction=1.0 / unit.picosecond,
                               timestep=0.5 * unit.femtoseconds,
                               steps=100_000,
                               n_report=1_000,
                               contractions=None,
                               platform_name='CPU',
                               deuterate=False,
                               deuterate_option='water',
                               potential=None,
                               ml_idx=None,
                               atoms_to_watch=None):
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

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
                         output_prefix='prod',
                         n_beads=32,
                         pressure=1.0 * unit.bar,
                         temperature=300.0 * unit.kelvin,
                         gamma=1.0 / unit.picosecond,
                         time_step=1.0 * unit.femtoseconds,
                         barostat_freq=50,
                         n_report=1_000,
                         steps=500_000,
                         platform_name='CPU',
                         deuterate=False,
                         deuterate_option='water',
                         potential=None,
                         ml_idx=None,
                         atoms_to_watch=None):
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path, 'r') as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)
    integrator = openmm.RPMDMonteCarloBarostat(temperature,
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


def run_openmm_adqtb_eq(modeller,
                        forcefield,
                        temperature=300.0 * unit.kelvin,
                        gamma=1.0 / unit.picosecond,
                        time_step=1.0 * unit.femtoseconds,
                        segment_length=0.5 * unit.picosecond,
                        adaptation_rate=0.5,
                        n_report=1_000,
                        steps=500_000,
                        output_prefix='prod',
                        platform_name='CPU',
                        deuterate=False,
                        deuterate_option='water',
                        potential=None,
                        ml_idx=None,
                        ):
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    integrator = openmm.QTBIntegrator(temperature, gamma, time_step)
    integrator.setSegmentLength(segment_length)
    integrator.setDefaultAdaptationRate(adaptation_rate)

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
                          output_prefix='prod',
                          platform_name='CPU',
                          deuterate=False,
                          deuterate_option='water',
                          potential=None,
                          ml_idx=None,
                          ):
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'  # Force CUDA for mixed potential
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
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
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

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
