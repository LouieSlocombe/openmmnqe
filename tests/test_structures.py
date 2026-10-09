"""Atom subsets retain topology structure and leave the input untouched."""

from __future__ import annotations

import numpy as np
import openmm.app as app
import openmm.unit as unit

from openmmnqe._structures import _subset_modeller


def test_subset_preserves_chains_residues_bonds_box_and_position_alignment() -> None:
    topology = app.Topology()
    first_residue = topology.addResidue("ONE", topology.addChain("A"), "10")
    second_residue = topology.addResidue("TWO", topology.addChain("B"), "20")
    atoms = [
        topology.addAtom(f"C{index}", app.element.carbon, residue)
        for index, residue in enumerate(
            [first_residue, first_residue, second_residue, second_residue]
        )
    ]
    topology.addBond(atoms[0], atoms[1])
    topology.addBond(atoms[1], atoms[2])
    topology.addBond(atoms[2], atoms[3])
    topology.setUnitCellDimensions(np.array([2.0, 3.0, 4.0]) * unit.nanometer)
    coordinates = np.arange(12.0).reshape(4, 3)
    positions = coordinates * unit.nanometer

    subset = _subset_modeller(topology, positions, iter([3, 2, 0, 2]))

    assert [atom.name for atom in subset.topology.atoms()] == ["C0", "C2", "C3"]
    assert [(chain.id, [res.id for res in chain.residues()])
            for chain in subset.topology.chains()] == [("A", ["10"]), ("B", ["20"])]
    assert [(a.name, b.name) for a, b in subset.topology.bonds()] == [("C2", "C3")]
    assert np.array_equal(
        subset.positions.value_in_unit(unit.nanometer), coordinates[[0, 2, 3]],
    )
    assert subset.topology.getPeriodicBoxVectors() == topology.getPeriodicBoxVectors()
    assert topology.getNumAtoms() == 4
    assert topology.getNumBonds() == 3
    assert np.array_equal(positions.value_in_unit(unit.nanometer), coordinates)
