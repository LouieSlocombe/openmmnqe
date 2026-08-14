"""Focused unit tests for lightweight simulation helpers."""

from types import SimpleNamespace

import numpy as np
import openmm.unit as unit

import openmmnqe as nqe


class _State:
    def __init__(self, positions, box_vectors):
        self._positions = np.asarray(positions, dtype=float) * unit.nanometer
        self._box_vectors = np.asarray(box_vectors, dtype=float) * unit.nanometer

    def getPositions(self, asNumpy=False):
        return self._positions

    def getPeriodicBoxVectors(self, asNumpy=False):
        return self._box_vectors


class _Integrator:
    def __init__(self, bead_positions, box_vectors):
        self._states = [_State(positions, box_vectors) for positions in bead_positions]

    def getState(self, bead, **kwargs):
        return self._states[bead]


class _System:
    def __init__(self, periodic):
        self._periodic = periodic

    def usesPeriodicBoundaryConditions(self):
        return self._periodic


def test_centroid_positions_unwraps_beads_across_box_boundary():
    simulation = SimpleNamespace(
        integrator=_Integrator(
            bead_positions=[[[0.1, 0.0, 0.0]], [[1.9, 0.0, 0.0]]],
            box_vectors=[[2.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.0, 0.0, 2.0]],
        ),
        system=_System(periodic=True),
    )

    centroid = nqe.centroid_positions(simulation, n_atoms=1, n_beads=2)
    centroid_nm = centroid.value_in_unit(unit.nanometer)

    assert np.allclose(centroid_nm[0], [0.0, 0.0, 0.0], atol=1e-12)


def test_centroid_positions_does_not_wrap_nonperiodic_systems():
    simulation = SimpleNamespace(
        integrator=_Integrator(
            bead_positions=[[[0.1, 0.0, 0.0]], [[1.9, 0.0, 0.0]]],
            box_vectors=[[2.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.0, 0.0, 2.0]],
        ),
        system=_System(periodic=False),
    )

    centroid = nqe.centroid_positions(simulation, n_atoms=1, n_beads=2)
    centroid_nm = centroid.value_in_unit(unit.nanometer)

    assert np.allclose(centroid_nm[0], [1.0, 0.0, 0.0])


