"""Long-running comparisons of classical and quantum simulation methods."""

import argparse
import os
from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
import openmm.app as app
import openmm.unit as unit

import openmmnqe as nqe
from openmm import openmm

# WHAM is built separately, see http://membrane.urmc.rochester.edu/sites/default/files/wham/
WHAM = os.environ.get("WHAM_PATH")


def _accumulate_rdf_counts(counts, positions, particles, box_size):
    """Add pair distances from one configuration to an RDF histogram."""
    bins = len(counts)
    for i in range(particles):
        for j in range(i):
            delta = positions[i] - positions[j]
            delta -= np.round(delta / box_size) * box_size
            distance = unit.norm(delta)
            bin_index = int(bins * distance / box_size)
            if bin_index < bins:
                counts[bin_index] += 1


def _normalise_rdf(counts, samples, particles, box_size):
    """Convert a pair-distance histogram to a radial distribution function."""
    bins = len(counts)
    scale = box_size ** 3 / (samples * 0.5 * particles ** 2)
    rdf = []
    for i in range(bins // 2):
        r1 = i * box_size / bins
        r2 = (i + 1) * box_size / bins
        volume = (4.0 / 3.0) * np.pi * (r2 ** 3 - r1 ** 3)
        rdf.append(scale * counts[i] / volume)
    return rdf


def compute_rdf(context, particles, box_size):
    """
    Compute the radial distribution function by sampling an OpenMM context.

    Parameters
    ----------
    context : openmm.Context
        Context to advance and sample positions from.
    particles : int
        Number of particles in the system.
    box_size : float
        Length of the (cubic) periodic box, in nanometers.

    Returns
    -------
    list of float
        RDF value for each of the first half of the histogram bins.
    """
    bins = 100
    iterations = 2_000
    counts = [0] * bins
    for _ in range(iterations):
        context.getIntegrator().step(20)
        positions = context.getState(positions=True).getPositions().value_in_unit(
            unit.nanometer
        )
        _accumulate_rdf_counts(counts, positions, particles, box_size)

    return _normalise_rdf(counts, iterations, particles, box_size)


def compute_rpmd_rdf(integrator, particles, box_size):
    """
    Compute a bead-averaged RDF from every copy of an RPMD integrator.

    RPMD state is retrieved directly from ``RPMDIntegrator`` because the
    ordinary Context state does not represent an individual ring-polymer
    copy.
    """
    bins = 100
    iterations = 2_000
    n_beads = integrator.getNumCopies()
    counts = [0] * bins
    for _ in range(iterations):
        integrator.step(20)
        for bead in range(n_beads):
            positions = integrator.getState(
                bead,
                getPositions=True,
            ).getPositions().value_in_unit(unit.nanometer)
            _accumulate_rdf_counts(counts, positions, particles, box_size)

    return _normalise_rdf(
        counts,
        iterations * n_beads,
        particles,
        box_size,
    )


def run_parahydrogen():
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
    for _ in range(particles):
        system.addParticle(2.0 * unit.amu)
        force.addParticle()
    positions = np.random.rand(particles, 3) * box_size

    topology = app.Topology()
    chain = topology.addChain()
    for particle in range(particles):
        residue = topology.addResidue("PH2", chain)
        topology.addAtom(f"PH2{particle + 1}", None, residue)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

    integrator = openmm.LangevinIntegrator(temperature,
                                           1.0 / unit.picosecond,
                                           1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(temperature)
    # Equilibrate before collecting data
    integrator.step(1_000)  # Equilibrate before collecting data
    classical_rdf = compute_rdf(context, particles, box_size)
    centroid_positions = context.getState(
        getPositions=True,
    ).getPositions()
    del context

    n_beads = 4
    integrator = openmm.RPMDIntegrator(n_beads,
                                       temperature,
                                       1.0 / unit.picosecond,
                                       1.0 * unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator)
    modeller = app.Modeller(topology, centroid_positions)
    nqe.init_beads(modeller, simulation, n_beads)
    integrator.step(1_000)  # Equilibrate before collecting data
    rpmd_rdf = compute_rpmd_rdf(integrator, particles, box_size)
    del simulation

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


def run_smd():
    if WHAM is None:
        raise RuntimeError("set WHAM_PATH to the WHAM executable")

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

    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    simulation.step(100_000)

    by_serial = {int(a.id): a for a in simulation.topology.atoms()}

    a1 = by_serial[4]
    a2 = by_serial[94]

    i, j = a1.index, a2.index
    print("OpenMM indices:", i, j, "| names:", a1.name, a2.name)

    pos = pdb.positions
    delta = (pos[i] - pos[j]).value_in_unit(unit.nanometer)
    dist = np.linalg.norm([delta.x, delta.y, delta.z]) * unit.nanometer
    print("Distance:", f"{dist.value_in_unit(unit.nanometer):.3g} nm")

    # define the CV as the distance between the CAs of the two end residues
    index1 = i
    index2 = j
    cv = openmm.CustomBondForce('r')
    cv.addBond(index1, index2)

    r0 = round(dist.value_in_unit(unit.nanometer), 1) * unit.nanometer
    r_start = r0.value_in_unit(unit.nanometer)
    r_end = 2.0
    n_windows = 24
    print("Starting distance r0 =", r_start, "nm")

    fc_pull = 1_000.0 * unit.kilojoules_per_mole / unit.nanometers ** 2
    v_pulling = 0.01 * unit.nanometers / unit.picosecond
    dt = simulation.integrator.getStepSize()
    total_steps = 30_000  # 120 ps
    increment_steps = 10  # steps between incrementing r0 (1 makes the simulation slow)

    # A harmonic restraint on the CV whose centre moves at a constant velocity
    # as the simulation runs -- constant velocity steered MD.
    pullingForce = openmm.CustomCVForce('0.5 * fc_pull * (cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0)
    pullingForce.addCollectiveVariable("cv", cv)
    system.addForce(pullingForce)
    simulation.context.reinitialize(preserveState=True)

    # Umbrella-sampling windows; the pulling loop below saves the configuration
    # closest to each one as that window's starting structure.
    windows = np.linspace(r_start, r_end, n_windows)
    window_coords = []
    window_index = 0

    smd_cv_values = []
    for i in range(total_steps // increment_steps):
        simulation.step(increment_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)
        smd_cv_values.append([i, current_cv_value[0]])

        if (i * increment_steps) % 5_000 == 0:
            print("r0 = ", r0, "r = ", current_cv_value)

        r0 += v_pulling * dt * increment_steps
        simulation.context.setParameter('r0', r0)

        # Save this configuration once the CV reaches the next window centre.
        if (window_index < len(windows) and current_cv_value >= windows[window_index]):
            window_coords.append(
                simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions())
            window_index += 1

    for i, coords in enumerate(window_coords):
        with open(f'window_{i}.pdb', 'w') as outfile:
            app.PDBFile.writeFile(simulation.topology, coords, outfile)

    simulation.saveCheckpoint('eq.chk')

    smd_cv_values = np.array(smd_cv_values)
    plt.plot(smd_cv_values[:, 0] * increment_steps, smd_cv_values[:, 1])
    plt.xlabel("Steps")
    plt.ylabel("CV value (nm)")
    plt.show()

    def run_window(window_index):
        print('running window', window_index)

        pdb = app.PDBFile(f'window_{window_index}.pdb')

        # Reuse the existing Simulation, just with this window's positions and r0.
        simulation.context.setPositions(pdb.positions)
        simulation.context.setParameter('r0', windows[window_index])

        simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
        simulation.step(1_000)

        total_steps = 100_000  # 400 ps
        record_steps = 1_000

        cv_values = []
        for i in range(total_steps // record_steps):
            simulation.step(record_steps)
            current_cv_value = pullingForce.getCollectiveVariableValues(simulation.context)
            cv_values.append([i, current_cv_value[0]])

        np.savetxt(f'cv_values_window_{window_index}.txt', np.array(cv_values))

        print('Completed window', window_index)

    for n in range(n_windows):
        run_window(n)

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

    # Run WHAM to get the PMF from the umbrella histograms. To build it:
    #   wget http://membrane.urmc.rochester.edu/sites/default/files/wham/wham-release-2.1.0.tgz
    #   tar xf wham-release-2.0.11.tgz
    #   cd wham/wham && make
    os.system(f'{WHAM} {r_start} {r_end} 50 1e-6 300 0 metafile.txt pmf.txt > wham_log.txt')

    pmf = np.loadtxt("pmf.txt")
    plt.plot(pmf[:, 0], pmf[:, 1])
    plt.xlabel("r (nm)")
    plt.ylabel("PMF (kJ/mol)")
    plt.show()

    for i in range(len(windows)):
        nqe.remove_file(f'cv_values_window_{i}.txt')
        nqe.remove_file(f'window_{i}.pdb')
    nqe.remove_file("metafile.txt")
    nqe.remove_file("pmf.txt")
    nqe.remove_file("smd_traj.dcd")
    nqe.remove_file("smd_traj.pdb")
    nqe.remove_file("wham_log.txt")


EXAMPLES = {
    "parahydrogen": run_parahydrogen,
    "smd": run_smd,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    args = parser.parse_args()
    EXAMPLES[args.example]()


if __name__ == "__main__":
    main()
