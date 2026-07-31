#!/usr/bin/env python3
import argparse
import random

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmplumed import PlumedForce


def build_toy_dimer():
    """
    Minimal 2-particle system so the example runs without external files.
    The PLUMED script must then use ATOMS=1,2 (PLUMED indexing).
    """
    system = mm.System()
    mass = 39.948 * unit.amu  # argon-ish
    system.addParticle(mass)
    system.addParticle(mass)

    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 0.55 * unit.nanometer,
                 2000 * unit.kilojoule_per_mole / unit.nanometer ** 2)
    system.addForce(bond)

    # Minimal topology
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("DUM", chain)
    a1 = top.addAtom("A1", app.element.argon, res)
    a2 = top.addAtom("A2", app.element.argon, res)
    top.addBond(a1, a2)

    positions = [mm.Vec3(0, 0, 0), mm.Vec3(0.60, 0, 0)] * unit.nanometer
    return top, system, positions


def load_pdb_system(pdb_path, ff_xmls, nonbonded_cutoff_nm=1.0):
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField(*ff_xmls)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=nonbonded_cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
    )
    return pdb.topology, system, pdb.positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walker-id", type=int, required=True)
    ap.add_argument("--n-walkers", type=int, required=True)
    ap.add_argument("--plumed", default="plumed.dat")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--dt-fs", type=float, default=2.0)
    ap.add_argument("--temp-k", type=float, default=300.0)
    ap.add_argument("--friction-ps", type=float, default=1.0)
    ap.add_argument("--pdb", default=None, help="Optional: run a real system from PDB")
    ap.add_argument("--ff", nargs="*", default=["amber14-all.xml", "amber14/tip3p.xml"])
    ap.add_argument("--platform", default=None, help="CUDA, OpenCL, CPU, Reference (optional)")
    args = ap.parse_args()

    # Distinct RNG per walker
    seed = (12345 + 1000 * args.walker_id) % 2 ** 31
    random.seed(seed)

    # Build system
    if args.pdb:
        topology, system, positions = load_pdb_system(args.pdb, args.ff)
        # Make sure your PLUMED ATOMS indices match the atoms in this PDB/topology.
        # Remember: PLUMED is 1-based, OpenMM is 0-based.
    else:
        topology, system, positions = build_toy_dimer()
        # For toy dimer, use in plumed.dat: DISTANCE ATOMS=1,2

    # Read & parameterize PLUMED script
    with open(args.plumed) as f:
        script = f.read()
    script = (script
              .replace("__WALKER_ID__", str(args.walker_id))
              .replace("__N_WALKERS__", str(args.n_walkers)))

    system.addForce(PlumedForce(script))

    # Integrator
    temperature = args.temp_k * unit.kelvin
    friction = args.friction_ps / unit.picosecond
    dt = args.dt_fs * unit.femtoseconds
    integrator = mm.LangevinMiddleIntegrator(temperature, friction, dt)
    integrator.setRandomNumberSeed(seed)

    # Platform selection (optional)
    if args.platform:
        platform = mm.Platform.getPlatformByName(args.platform)
        simulation = app.Simulation(topology, system, integrator, platform)
    else:
        simulation = app.Simulation(topology, system, integrator)

    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature, seed)

    # Per-walker outputs (avoid collisions)
    simulation.reporters.append(app.StateDataReporter(
        f"log_w{args.walker_id}.txt", 1000,
        step=True, time=True, temperature=True,
        potentialEnergy=True, speed=True, progress=True,
        remainingTime=True, totalSteps=args.steps
    ))
    simulation.reporters.append(app.DCDReporter(f"traj_w{args.walker_id}.dcd", 5000))

    print(f"Starting walker {args.walker_id}/{args.n_walkers} (seed={seed})")
    simulation.step(args.steps)
    print("Done.")


if __name__ == "__main__":
    main()
