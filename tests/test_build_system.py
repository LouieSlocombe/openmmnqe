"""Regression tests for the ML argument handling in ``_build_system``.

Commit 85231d1 guarded against an ML potential arriving without ``ml_idx``,
but the guard also rejected the legitimate pure-ML configuration where an
``MLPotential`` stands in for the force field and an ASE calculator is
forwarded to its ``createSystem``.  These tests pin down all three outcomes
without a GPU or a MACE download: the guards raise before any system is
built, and the pure-ML build uses ASE's toy Lennard-Jones calculator.
"""
import openmm.app as app
import pytest
from ase.calculators.lj import LennardJones
from openmmml import MLPotential

from openmmnqe.openmm import _build_system

TOLUENE = 'tests/data/pdb/toluene.pdb'


def _toluene_modeller():
    pdb = app.PDBFile(TOLUENE)
    return app.Modeller(pdb.topology, pdb.positions)


def test_calculator_with_plain_forcefield_needs_ml_idx():
    """ForceField.createSystem would silently swallow the calculator."""
    modeller = _toluene_modeller()
    forcefield = app.ForceField('amber14-all.xml')
    with pytest.raises(ValueError, match="ml_idx"):
        _build_system(modeller, forcefield, 'CPU',
                      potential=None, ml_idx=None, calculator=LennardJones())


def test_potential_needs_ml_idx():
    """A mixed-system potential with no ML region is undefined."""
    modeller = _toluene_modeller()
    forcefield = app.ForceField('amber14-all.xml')
    with pytest.raises(ValueError, match="ml_idx"):
        _build_system(modeller, forcefield, 'CPU',
                      potential=MLPotential('ase'), ml_idx=None,
                      calculator=LennardJones())


def test_pure_ml_forcefield_with_calculator_builds_system():
    """MLPotential('ase') as forcefield + calculator is the pure-ML path."""
    modeller = _toluene_modeller()
    system, platform = _build_system(modeller, MLPotential('ase'), 'CPU',
                                     potential=None, ml_idx=None,
                                     calculator=LennardJones())
    assert system.getNumParticles() == modeller.topology.getNumAtoms()
    assert platform.getName() == 'CPU'
