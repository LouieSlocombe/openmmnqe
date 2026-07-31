import numpy as np
from ase.io import read

import openmmnqe as nqe


def test_orca_calc_preset():
    print(flush=True)
    atoms = read('tests/data/fad.xyz')
    calc = nqe.orca_calc_preset()
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    print(energy, flush=True)
    assert np.allclose(energy, -10325.045291755621)


def test_orca_optimise_atoms():
    print(flush=True)
    atoms = read('tests/data/fad.xyz')
    opt_atoms = nqe.orca_optimise_atoms(atoms)
    calc = nqe.orca_calc_preset()
    opt_atoms.set_calculator(calc)
    energy = opt_atoms.get_potential_energy()
    print(energy, flush=True)
    assert np.allclose(energy, -10326.977956847948)


def test_orca_calculate_goat():
    atoms = read('tests/data/fad.xyz')
    goat = nqe.orca_calculate_goat(atoms)
    print(goat)
