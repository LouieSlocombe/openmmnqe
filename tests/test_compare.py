import matplotlib.pyplot as plot
import numpy as np
import openmm.unit as unit
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
        plot.plot(integrator.getAdaptedFriction(0), label=f'{i}')
    plot.xlim([0, 200])
    plot.legend()
    plot.show()

    qtb_rdf = compute_rdf(context, particles, box_size)

    plot.plot(classical_rdf, label="Classical")
    plot.plot(rpmd_rdf, label="RPMD")
    plot.plot(qtb_rdf, label="adQTB")
    plot.legend()
    plot.show()
