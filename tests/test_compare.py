import os
from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
import openmm.app as app
import openmm.unit as unit

import openmmnqe as nqe
from openmm import openmm


def compute_rdf(context, particles, box_size):
    bins = 100
    iterations = 2_000
    counts = [0] * bins
    for _ in range(iterations):
        # Run a few steps of dynamics.
        context.getIntegrator().step(20)

        # Count the number of distances in each bin.
        pos = context.getState(positions=True).getPositions().value_in_unit(unit.nanometer)
        for i in range(particles):
            for j in range(i):
                delta = pos[i] - pos[j]
                delta -= np.round(delta / box_size) * box_size  # Apply periodic boundary conditions
                dist = unit.norm(delta)
                counts[int(bins * dist / box_size)] += 1

    # Convert the histogram of distances to the RDF.
    scale = (box_size * box_size * box_size) / (iterations * 0.5 * particles * particles)
    rdf = []
    for i in range(bins // 2):
        r1 = i * box_size / bins
        r2 = (i + 1) * box_size / bins
        volume = (4.0 / 3.0) * np.pi * (r2 * r2 * r2 - r1 * r1 * r1)
        rdf.append(scale * counts[i] / volume)
    return rdf


def test_parahydrogen():
    particles = 32
    box_size = 1.1896
    temperature = 25 * unit.kelvin
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(box_size, 0, 0),
                                        openmm.Vec3(0, box_size, 0),
                                        openmm.Vec3(0, 0, box_size))
    force = openmm.CustomNonbondedForce(
        """2625.49963*(exp(1.713-1.5671*p-0.00993*p*p) -
                       (12.14/p^6+215.2/p^8-143.1/p^9+4813.9/p^10)*(step(rc-p)*exp(-(rc/p-1)^2)+1-step(rc-p)));
                       p=r/0.05291772108; rc=8.32""")
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(box_size / 2)
    system.addForce(force)
    for i in range(particles):
        system.addParticle(2.0 * unit.amu)
        force.addParticle()
    positions = np.random.rand(particles, 3) * box_size

    integrator = openmm.LangevinIntegrator(temperature,
                                           1.0 / unit.picosecond,
                                           1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(temperature)
    # Equilibrate before collecting data
    integrator.step(1_000)
    classical_rdf = compute_rdf(context, particles, box_size)

    n_beads = 4
    integrator = openmm.RPMDIntegrator(n_beads,
                                       temperature,
                                       1.0 / unit.picosecond,
                                       1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(1_000)  # Equilibrate before collecting data
    rpmd_rdf = compute_rdf(context, particles, box_size)

    integrator = openmm.QTBIntegrator(temperature,
                                      20.0 / unit.picosecond,
                                      1.0 * unit.femtosecond)
    integrator.setSegmentLength(0.5 * unit.picosecond)
    for i in range(particles):
        integrator.setParticleType(i, 0)
    integrator.setDefaultAdaptationRate(0.5)

    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(temperature)

    for i in range(5):
        integrator.step(10_000)
        plt.plot(integrator.getAdaptedFriction(0), label=f'{i}')
    plt.xlim([0, 200])
    plt.legend()
    plt.show()

    qtb_rdf = compute_rdf(context, particles, box_size)

    plt.plot(classical_rdf, label="Classical")
    plt.plot(rpmd_rdf, label="RPMD")
    plt.plot(qtb_rdf, label="adQTB")
    plt.legend()
    plt.show()


def test_smd():
    print(flush=True)
    pdb = app.PDBFile('tests/data/pdb/deca-ala.pdb')

    forcefield = app.ForceField('amber14-all.xml')

    # We have a single molecule in vacuum so we use no cutoff.
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=app.NoCutoff,
                                     constraints=app.HBonds,
                                     hydrogenMass=1.5 * unit.amu)

    integrator = openmm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                                 1 / unit.picosecond,
                                                 0.004 * unit.picoseconds)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # Minimize
    openmm.LocalEnergyMinimizer.minimize(simulation.context)

    simulation.reporters.append(app.DCDReporter('smd_traj.dcd', 1_000))
    simulation.reporters.append(app.PDBReporter('smd_traj.pdb', 1_000))
    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      10_000,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    # Equilibrate
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    simulation.step(100_000)

    by_serial = {int(a.id): a for a in simulation.topology.atoms()}

    a1 = by_serial[4]
    a2 = by_serial[94]

    i, j = a1.index, a2.index
    print("OpenMM indices:", i, j, "| names:", a1.name, a2.name)

    # Distance from the loaded coordinates
    pos = pdb.positions
    delta = (pos[i] - pos[j]).value_in_unit(unit.nanometer)
    dist = np.linalg.norm([delta.x, delta.y, delta.z]) * unit.nanometer
    print("Distance:", f"{dist.value_in_unit(unit.nanometer):.3g} nm")

    # define the CV as the distance between the CAs of the two end residues
    index1 = i
    index2 = j
    cv = openmm.CustomBondForce('r')
    cv.addBond(index1, index2)

    # starting value
    r0 = round(dist.value_in_unit(unit.nanometer), 1) * unit.nanometer
    r_start = r0.value_in_unit(unit.nanometer)
    r_end = 2.0
    n_windows = 24
    print("Starting distance r0 =", r_start, "nm")

    # force constant
    fc_pull = 1_000.0 * unit.kilojoules_per_mole / unit.nanometers ** 2

    # pulling speed
    v_pulling = 0.01 * unit.nanometers / unit.picosecond  # nm/ps

    # simulation time step
    dt = simulation.integrator.getStepSize()

    # total number of steps
    total_steps = 30_000  # 120ps

    # number of steps to run between incrementing r0 (1 makes the simulation slow)
    increment_steps = 10

    # define a harmonic restraint on the CV
    # the location of the restrain will be moved as we run the simulation
    # this is constant velocity steered MD
    pullingForce = openmm.CustomCVForce('0.5 * fc_pull * (cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0)
    pullingForce.addCollectiveVariable("cv", cv)
    system.addForce(pullingForce)
    simulation.context.reinitialize(preserveState=True)

    # define the windows
    # during the pulling loop we will save specific configurations corresponding to the windows
    windows = np.linspace(r_start, r_end, n_windows)
    window_coords = []
    window_index = 0

    # SMD pulling loop
    smd_cv_values = []
    for i in range(total_steps // increment_steps):
        simulation.step(increment_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)
        smd_cv_values.append([i, current_cv_value[0]])

        if (i * increment_steps) % 5_000 == 0:
            print("r0 = ", r0, "r = ", current_cv_value)

        # increment the location of the CV based on the pulling velocity
        r0 += v_pulling * dt * increment_steps
        simulation.context.setParameter('r0', r0)

        # check if we should save this config as a window starting structure
        if (window_index < len(windows) and current_cv_value >= windows[window_index]):
            window_coords.append(
                simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions())
            window_index += 1

    # save the window structures
    for i, coords in enumerate(window_coords):
        outfile = open(f'window_{i}.pdb', 'w')
        app.PDBFile.writeFile(simulation.topology, coords, outfile)
        outfile.close()

    # Save the simulation
    simulation.saveCheckpoint('eq.chk')

    smd_cv_values = np.array(smd_cv_values)
    plt.plot(smd_cv_values[:, 0] * increment_steps, smd_cv_values[:, 1])
    plt.xlabel("Steps")
    plt.ylabel("CV value (nm)")
    plt.show()

    def run_window(window_index):
        print('running window', window_index)

        # load in the starting configuration for this window
        pdb = app.PDBFile(f'window_{window_index}.pdb')

        # we can reuse the existing Simulation
        simulation.context.setPositions(pdb.positions)

        # Set the fixed location of the harmonic restraint for this window
        simulation.context.setParameter('r0', windows[window_index])

        # run short equilibration with new positions and r0
        simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
        simulation.step(1_000)

        # run the data collection

        # total number of steps
        total_steps = 100_000  # 400 ps

        # frequency to record the current CV value
        record_steps = 1_000

        # run the simulation and record the value of the CV.
        cv_values = []
        for i in range(total_steps // record_steps):
            simulation.step(record_steps)

            # get the current value of the cv
            current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)
            cv_values.append([i, current_cv_value[0]])

        # save the CV timeseries to a file so we can postprocess
        np.savetxt(f'cv_values_window_{window_index}.txt', np.array(cv_values))

        print('Completed window', window_index)

    for n in range(n_windows):
        run_window(n)

    # plot the histograms
    metafilelines = []
    for i in range(len(windows)):
        data = np.loadtxt(f'cv_values_window_{i}.txt')
        plt.hist(data[:, 1])
        metafileline = f'cv_values_window_{i}.txt {windows[i]} 1000\n'
        metafilelines.append(metafileline)

    plt.xlabel("r (nm)")
    plt.ylabel("count")
    plt.show()

    with open("metafile.txt", "w") as f:
        f.writelines(metafilelines)

    # execute WHAM to get the PMF using os
    # wget http://membrane.urmc.rochester.edu/sites/default/files/wham/wham-release-2.1.0.tgz
    # tar xf wham-release-2.0.11.tgz
    # cd wham/wham && make

    os.system(
        f'/home/louie/skunkworks/wham/wham/wham/wham {r_start} {r_end} 50 1e-6 300 0 metafile.txt pmf.txt > wham_log.txt')

    # plot the PMF
    pmf = np.loadtxt("pmf.txt")
    plt.plot(pmf[:, 0], pmf[:, 1])
    plt.xlabel("r (nm)")
    plt.ylabel("PMF (kJ/mol)")
    plt.show()

    # Clean up
    for i in range(len(windows)):
        nqe.remove_file(f'cv_values_window_{i}.txt')
        nqe.remove_file(f'window_{i}.pdb')
    nqe.remove_file("metafile.txt")
    nqe.remove_file("pmf.txt")
    nqe.remove_file("smd_traj.dcd")
    nqe.remove_file("smd_traj.pdb")
    nqe.remove_file("wham_log.txt")
