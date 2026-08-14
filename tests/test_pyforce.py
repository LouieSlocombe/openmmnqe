import glob
import os

import numpy as np
import openmm.app as app
import openmm.unit as unit
import torch
from ase.calculators.orca import ORCA, OrcaProfile
from mace.calculators.foundations_models import mace_mp, mace_off
from openmmml import MLPotential
import pytest

from openmm import openmm

pytestmark = pytest.mark.pipeline

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


def test_ase_mace_mh1():
    print(flush=True)
    pdb = app.PDBFile('tests/data/pdb/toluene.pdb')
    potential = MLPotential('ase')
    # MACE-MH-1 is multi-head, 'omol' is the wB97M-VV10 organic chemistry head
    if 'MACE_MODELS' in os.environ:
        model = os.path.join(os.environ['MACE_MODELS'], 'mace-mh-1.model')
    else:
        model = 'mh-1'
    calculator = mace_mp(model=model,
                         default_dtype='float32',
                         device=device.lower(),
                         head='omol')
    system = potential.createSystem(pdb.topology, calculator=calculator)
    platform = openmm.Platform.getPlatform(device)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions(pdb.getPositions(asNumpy=True))
    energy = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(energy, flush=True)
    # Reference taken on CPU with float32 and no dispersion correction
    assert np.isclose(-712905.8092781925, energy, rtol=1e-5)


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


@pytest.mark.orca
@pytest.mark.skipif("ORCA_PATH" not in os.environ,
                    reason="needs an ORCA installation (ORCA_PATH unset)")
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
