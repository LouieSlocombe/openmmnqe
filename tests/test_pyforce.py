import os
import glob
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

from mace.calculators.foundations_models import mace_off

from ase.calculators.orca import OrcaProfile
from ase.calculators.orca import ORCA


def test_ase_mace():
    pdb = app.PDBFile('tests/data/pdb/malonaldehyde.pdb')
    potential = MLPotential('ase')
    calculator = mace_off('small', default_dtype='float32')
    system = potential.createSystem(pdb.topology, calculator=calculator)
    platform_ints = range(mm.Platform.getNumPlatforms())
    platform = mm.Platform.getPlatform(platform_ints[0])
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    context.setPositions(pdb.getPositions(asNumpy=True))
    energy = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(energy)
    assert abs(energy - -701887.9719313162) < 1e-3


def test_ase_orca():
    pdb = app.PDBFile('tests/data/pdb/malonaldehyde.pdb')
    potential = MLPotential('ase')

    profile = OrcaProfile(command=os.environ['ORCA_PATH'])
    calculator = ORCA(profile=profile, orcasimpleinput='PBE def2-SVP RI D3BJ ENGRAD')  # PBE def2-SVP RI D3BJ or B97-3c

    system = potential.createSystem(pdb.topology, calculator=calculator)
    platform_ints = range(mm.Platform.getNumPlatforms())
    platform = mm.Platform.getPlatform(platform_ints[0])
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    context.setPositions(pdb.getPositions(asNumpy=True))
    energy = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(energy)
    assert abs(energy - -700122.448099461) < 1e-3
    # remove all the orca.* files
    [os.remove(file) for file in glob.glob('orca.*')]
