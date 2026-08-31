"""Long-running comparisons of classical and quantum simulation methods."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm import openmm

import openmmnqe as nqe


def _accumulate_rdf_counts(counts: list[int], positions: unit.Quantity,
                           particles: int, box_size: float) -> None:
    """
    Add pair distances from one configuration to an RDF histogram.

    Parameters
    ----------
    counts : list of int
        Histogram updated in place, one bin per ``box_size / len(counts)``.
    positions : openmm.unit.Quantity
        Particle positions for this configuration.
    particles : int
        Number of particles to pair up.
    box_size : float
        Length of the cubic periodic box, in nanometres. Pair separations are
        minimum-imaged against it.
    """
    bins = len(counts)
    for i in range(particles):
        for j in range(i):
            delta = positions[i] - positions[j]
            delta -= np.round(delta / box_size) * box_size
            distance = unit.norm(delta)
            bin_index = int(bins * distance / box_size)
            if bin_index < bins:
                counts[bin_index] += 1


def _normalise_rdf(counts: list[int], samples: int, particles: int,
                   box_size: float) -> list[float]:
    """
    Convert a pair-distance histogram to a radial distribution function.

    Parameters
    ----------
    counts : list of int
        Pair-distance histogram accumulated by
        :func:`_accumulate_rdf_counts`.
    samples : int
        Number of configurations that went into *counts*.
    particles : int
        Number of particles in the system.
    box_size : float
        Length of the cubic periodic box, in nanometres.

    Returns
    -------
    list of float
        RDF value for each of the first half of the bins; beyond that the
        minimum image no longer samples a full shell.
    """
    bins = len(counts)
    scale = box_size ** 3 / (samples * 0.5 * particles ** 2)
    rdf = []
    for i in range(bins // 2):
        r1 = i * box_size / bins
        r2 = (i + 1) * box_size / bins
        volume = (4.0 / 3.0) * np.pi * (r2 ** 3 - r1 ** 3)
        rdf.append(scale * counts[i] / volume)
    return rdf


def compute_rdf(context: openmm.Context, particles: int,
                box_size: float) -> list[float]:
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


def compute_rpmd_rdf(integrator: openmm.RPMDIntegrator, particles: int,
                     box_size: float) -> list[float]:
    """
    Compute a bead-averaged RDF from every copy of an RPMD integrator.

    RPMD state is retrieved directly from ``RPMDIntegrator`` because the
    ordinary Context state does not represent an individual ring-polymer
    copy.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        Integrator to advance and sample every bead of.
    particles : int
        Number of particles in the system.
    box_size : float
        Length of the cubic periodic box, in nanometres.

    Returns
    -------
    list of float
        RDF value for each of the first half of the histogram bins, averaged
        over all beads.
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


def run_parahydrogen() -> None:
    """
    Compare classical and RPMD radial distribution functions for para-hydrogen.

    At 25 K the ring polymer spreads far enough to wash out the first peak
    that the classical simulation shows, which is the quantum effect the two
    curves are being compared for.
    """
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

    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(simulation.context)
    simulation.context.setVelocitiesToTemperature(temperature)

    # Log the adapted friction every segment, so the equilibration can be
    # shown to have converged rather than assumed to have.
    with nqe.track_adqtb_friction(simulation, 'compare_friction.log'):
        simulation.step(50_000)

    for label, verdict in nqe.adqtb_convergence('compare_friction.log').items():
        print(f"{label}: converged={verdict.converged} "
              f"drift ratio {verdict.drift_ratio:.2f}")
    nqe.plot_adqtb_friction_spectra('compare_friction.log', show=True)

    qtb_rdf = compute_rdf(simulation.context, particles, box_size)

    plt.plot(classical_rdf, label="Classical")
    plt.plot(rpmd_rdf, label="RPMD")
    plt.plot(qtb_rdf, label="adQTB")
    plt.legend()
    plt.show()


EXAMPLES = {
    "parahydrogen": run_parahydrogen,
}


def main() -> None:
    """Run whichever comparison is named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    args = parser.parse_args()
    EXAMPLES[args.example]()


if __name__ == "__main__":
    main()
