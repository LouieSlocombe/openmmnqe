"""Run a short OPES metadynamics simulation through OpenMM-PLUMED."""

import os

import openmm.app as app
import openmm.unit as unit
import torch
from openmm import openmm
from openmmplumed import PlumedForce


def main():
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    modeller.addSolvent(
        forcefield,
        padding=1.5 * unit.nanometer,
        boxShape="cube",
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )
    plumed_script = """
    d1: DISTANCE ATOMS=1,31
    opes: OPES_METAD ARG=d1 PACE=500 BARRIER=40 TEMP=300
    PRINT STRIDE=500 ARG=d1,opes.bias FILE=COLVAR
    """
    system.addForce(PlumedForce(plumed_script))

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        1.0 * unit.femtosecond,
    )
    platform_name = "CUDA" if torch.cuda.is_available() else "CPU"
    platform = openmm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy()
    simulation.step(1_000)

    os.remove("COLVAR")
    os.remove("KERNELS")


if __name__ == "__main__":
    main()
