"""Small topology operations shared by structure and trajectory writers."""

from __future__ import annotations

from collections.abc import Iterable

import openmm.unit as unit
from openmm import app


def _subset_modeller(topology: app.Topology, positions: unit.Quantity,
                     atom_indices: Iterable[int]) -> app.Modeller:
    """Copy an atom subset, preserving topology order and connectivity.

    The input topology and positions are not modified. Indices are consumed
    once, duplicates select the same atom, and absent indices select nothing;
    callers that require bounds checks perform them before calling here.
    """
    modeller = app.Modeller(topology, positions)
    keep = frozenset(atom_indices)
    modeller.delete([
        atom for atom in modeller.topology.atoms() if atom.index not in keep
    ])
    return modeller
