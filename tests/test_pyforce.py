import glob
import os

import numpy as np
import openmm.app as app
import openmm.unit as unit
import torch
from ase.calculators.orca import ORCA, OrcaProfile
from mace.calculators.foundations_models import mace_off
from openmmml import MLPotential
from openmmplumed import PlumedForce

from openmm import openmm

device = 'CUDA' if torch.cuda.is_available() else 'CPU'
print(f"Using device: {device}", flush=True)


def test_ase_mace():
    print(flush=True)
    pdb = app.PDBFile('tests/data/pdb/toluene.pdb')
    potential = MLPotential('ase')
    calculator = mace_off('small', default_dtype='float32')
    system = potential.createSystem(pdb.topology, calculator=calculator)
    platform = openmm.Platform.getPlatform(device)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions(pdb.getPositions(asNumpy=True))
    energy = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(energy, flush=True)
    assert np.isclose(-713468.6327560507, energy, rtol=1e-6)


def test_openmm_mlp():
    print(flush=True)
    pdb = app.PDBFile('tests/data/pdb/toluene.pdb')
    potential = MLPotential('mace-off23-small')
    system = potential.createSystem(pdb.topology, returnEnergyType='energy')
    platform = openmm.Platform.getPlatform(device)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions(pdb.getPositions(asNumpy=True))
    energy = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(energy, flush=True)
    assert np.isclose(-713468.6327560507, energy, rtol=1e-6)


def test_ase_orca():
    print(flush=True)
    pdb = app.PDBFile('tests/data/pdb/toluene.pdb')
    potential = MLPotential('ase')

    profile = OrcaProfile(command=os.environ['ORCA_PATH'])
    calculator = ORCA(profile=profile, orcasimpleinput='PBE def2-SVP RI D3BJ ENGRAD')  # PBE def2-SVP RI D3BJ or B97-3c

    system = potential.createSystem(pdb.topology, calculator=calculator)
    platform = openmm.Platform.getPlatform(device)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions(pdb.getPositions(asNumpy=True))
    energy = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(energy, flush=True)
    assert np.isclose(-711546.8260864686, energy, rtol=1e-6)
    [os.remove(file) for file in glob.glob('orca.*')]


def test_opes():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='cube')

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds
    )

    plumed_script = """
    d1: DISTANCE ATOMS=1,31
    opes: OPES_METAD ARG=d1 PACE=500 BARRIER=40 TEMP=300
    PRINT STRIDE=500 ARG=d1,opes.bias FILE=COLVAR
    """
    system.addForce(PlumedForce(plumed_script))

    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin,
                                                 1.0 / unit.picosecond,
                                                 1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatform(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print("Minimizing energy...")
    simulation.minimizeEnergy()

    print("Running OPES simulation...")
    simulation.step(1_000)

    os.remove("COLVAR")
    os.remove("KERNELS")
