"""Tests for structure editing and filesystem helpers."""

from io import StringIO

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3
from rdkit import Chem

import openmmnqe as nqe
import openmmnqe.io as nqe_io


def test_file_helpers_copy_list_and_remove(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("payload")
    destination = tmp_path / "destination"
    destination.mkdir()

    nqe.copy_and_rename_file(source, destination, "renamed.txt")

    copied = destination / "renamed.txt"
    assert copied.read_text() == "payload"
    assert nqe.list_files_with_pattern(destination, "*.txt") == [str(copied)]

    nqe.remove_file(copied)
    nqe.remove_file(copied)  # Missing files are intentionally harmless.
    assert not copied.exists()

    nested = destination / "nested"
    nested.mkdir()
    (nested / "file").write_text("data")
    nqe.remove_directory(destination)
    nqe.remove_directory(destination)
    assert not destination.exists()


def test_remove_file_pattern_only_removes_matches(tmp_path):
    matching = [tmp_path / "run-1.log", tmp_path / "run-2.log"]
    untouched = tmp_path / "run.txt"
    for path in [*matching, untouched]:
        path.write_text("data")

    nqe.remove_file_pattern(str(tmp_path / "*.log"))

    assert all(not path.exists() for path in matching)
    assert untouched.exists()


def test_xyz_to_sdf_writes_readable_molecule(data_dir, tmp_path):
    output = tmp_path / "gc.sdf"

    count = nqe.xyz_to_sdf(data_dir / "GC.xyz", output)
    molecules = [mol for mol in Chem.SDMolSupplier(str(output), removeHs=False) if mol]

    assert count == 1
    assert len(molecules) == 1
    assert molecules[0].GetNumAtoms() == 29
    assert molecules[0].GetNumBonds() > 0


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "No XYZ frames"),
        ("not-a-count\ncomment\n", "Expected atom count"),
        ("2\ncomment\nH 0 0 0\n", "Unexpected EOF"),
        ("1\ncomment\nH x 0 0\n", "Bad XYZ coordinates"),
    ],
)
def test_xyz_to_sdf_rejects_malformed_xyz(tmp_path, contents, message):
    source = tmp_path / "bad.xyz"
    source.write_text(contents)

    with pytest.raises(ValueError, match=message):
        nqe.xyz_to_sdf(source, tmp_path / "bad.sdf")


def test_relabel_residues_supports_paths_and_file_objects(data_dir, tmp_path):
    source = data_dir / "pdb" / "malonaldehyde.pdb"
    output = tmp_path / "renamed.pdb"

    returned = nqe.relabel_residues_in_pdb(source, {"LIG": "MAL"}, str(output))
    parsed = app.PDBFile(str(output))
    in_memory = StringIO()
    nqe.relabel_residues_in_pdb(source, {"LIG": "MAL"}, in_memory)

    assert {residue.name for residue in returned.topology.residues()} == {"MAL"}
    assert {residue.name for residue in parsed.topology.residues()} == {"MAL"}
    assert " MAL " in in_memory.getvalue()


def test_remove_residues_removes_only_requested_names(data_dir, tmp_path):
    output = tmp_path / "without_water.pdb"

    nqe.remove_residues_in_pdb(
        data_dir / "pdb" / "malformed.pdb",
        output,
        {"HOH"},
    )
    residues = [residue.name for residue in app.PDBFile(str(output)).topology.residues()]

    assert residues == ["AMM"]


def test_fix_pdb_runs_repair_steps_in_order(monkeypatch, tmp_path):
    calls = []

    class FakeFixer:
        topology = object()
        positions = object()

        def __init__(self, filename):
            calls.append(("init", filename))

        def findMissingResidues(self):
            calls.append("findMissingResidues")

        def findNonstandardResidues(self):
            calls.append("findNonstandardResidues")

        def replaceNonstandardResidues(self):
            calls.append("replaceNonstandardResidues")

        def removeHeterogens(self, keep_water):
            calls.append(("removeHeterogens", keep_water))

        def findMissingAtoms(self):
            calls.append("findMissingAtoms")

        def addMissingAtoms(self):
            calls.append("addMissingAtoms")

        def addMissingHydrogens(self, ph):
            calls.append(("addMissingHydrogens", ph))

    monkeypatch.setattr(nqe_io, "PDBFixer", FakeFixer)
    monkeypatch.setattr(
        nqe_io.app.PDBFile,
        "writeFile",
        lambda topology, positions, handle: calls.append("writeFile"),
    )

    nqe.fix_pdb("input.pdb", tmp_path / "output.pdb", ph=6.5, rm_heterogens=True)

    assert calls == [
        ("init", "input.pdb"),
        "findMissingResidues",
        "findNonstandardResidues",
        "replaceNonstandardResidues",
        ("removeHeterogens", True),
        "findMissingAtoms",
        "addMissingAtoms",
        ("addMissingHydrogens", 6.5),
        "writeFile",
    ]


def test_fix_pdb_can_keep_heterogens(monkeypatch, tmp_path):
    class FakeFixer:
        topology = object()
        positions = object()

        def __init__(self, filename):
            pass

        def __getattr__(self, name):
            if name == "removeHeterogens":
                pytest.fail("removeHeterogens should not be called")
            return lambda *args: None

    monkeypatch.setattr(nqe_io, "PDBFixer", FakeFixer)
    monkeypatch.setattr(nqe_io.app.PDBFile, "writeFile", lambda *args: None)

    nqe.fix_pdb("input.pdb", tmp_path / "output.pdb", rm_heterogens=False)


def test_convert_sdfs_to_pdb_preserves_molecules_and_bonds(data_dir, tmp_path):
    output = tmp_path / "combined.pdb"

    nqe.convert_sdfs_to_pdb(
        [data_dir / "CH4.sdf", data_dir / "H2O.sdf"],
        output,
    )
    topology = app.PDBFile(str(output)).topology

    assert topology.getNumAtoms() == 8
    assert topology.getNumBonds() == 6
    # OpenMM normalises the water alias H2O to its canonical PDB name HOH.
    assert [residue.name for residue in topology.residues()] == ["CH4", "HOH"]
    assert len(list(topology.chains())) == 2


def test_save_pdb_selection_preserves_requested_atom_order(data_dir, tmp_path):
    source = data_dir / "pdb" / "malonaldehyde.pdb"
    output = tmp_path / "selection.pdb"
    original = app.PDBFile(str(source))
    expected = [list(original.topology.atoms())[index].name for index in (0, 3, 8)]

    nqe.save_pdb_selection(source, [0, 3, 8], output)
    selected = app.PDBFile(str(output))

    assert [atom.name for atom in selected.topology.atoms()] == expected
    assert selected.topology.getNumAtoms() == 3


def test_move_pdb_to_origin_centres_coordinates(data_dir, tmp_path):
    output = tmp_path / "centred.pdb"

    nqe.move_pdb_to_origin(data_dir / "pdb" / "malonaldehyde.pdb", output)
    positions = app.PDBFile(str(output)).getPositions(asNumpy=True)
    centroid = positions.value_in_unit(unit.nanometer).mean(axis=0)

    assert np.allclose(centroid, 0.0, atol=5e-5)


def test_center_in_box_centres_orthorhombic_system():
    topology = app.Topology()
    residue = topology.addResidue("RES", topology.addChain())
    topology.addAtom("A1", app.Element.getByAtomicNumber(6), residue)
    topology.addAtom("A2", app.Element.getByAtomicNumber(6), residue)
    topology.setUnitCellDimensions(
        unit.Quantity((10.0, 10.0, 10.0), unit.nanometer)
    )
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * unit.nanometer
    modeller = app.Modeller(topology, positions)

    nqe.center_in_box(modeller)

    centred = modeller.positions.value_in_unit(unit.nanometer)
    assert np.allclose(centred.mean(axis=0), [5.0, 5.0, 5.0])
    assert np.allclose(centred[1] - centred[0], [3.0, 3.0, 3.0])


def test_center_in_box_uses_triclinic_vector_sum():
    topology = app.Topology()
    residue = topology.addResidue("RES", topology.addChain())
    topology.addAtom("A1", app.Element.getByAtomicNumber(6), residue)
    topology.setPeriodicBoxVectors(
        (
            Vec3(2.0, 0.0, 0.0),
            Vec3(0.5, 2.0, 0.0),
            Vec3(0.2, 0.3, 2.0),
        )
        * unit.nanometer
    )
    modeller = app.Modeller(topology, [Vec3(0.0, 0.0, 0.0)] * unit.nanometer)

    nqe.center_in_box(modeller)

    centred = modeller.positions.value_in_unit(unit.nanometer)
    assert np.allclose(centred[0], [1.35, 1.15, 1.0])


def test_center_in_box_leaves_nonperiodic_positions_unchanged():
    topology = app.Topology()
    residue = topology.addResidue("RES", topology.addChain())
    topology.addAtom("A1", app.Element.getByAtomicNumber(6), residue)
    original = [Vec3(1.0, 2.0, 3.0)] * unit.nanometer
    modeller = app.Modeller(topology, original)

    nqe.center_in_box(modeller)

    assert np.allclose(
        modeller.positions.value_in_unit(unit.nanometer),
        original.value_in_unit(unit.nanometer),
    )


def test_fix_pdb_chains_assigns_one_chain_per_residue(data_dir, tmp_path):
    output = tmp_path / "fixed.pdb"

    nqe.fix_pdb_chains(data_dir / "pdb" / "malformed.pdb", output)
    parsed = app.PDBFile(str(output))

    assert [chain.id for chain in parsed.topology.chains()] == ["A", "B"]
    assert [residue.name for residue in parsed.topology.residues()] == ["HOH", "AMM"]


def test_fix_pdb_chains_distinguishes_equal_ids(data_dir, tmp_path):
    output = tmp_path / "fixed.pdb"
    nqe.fix_pdb_chains(data_dir / "pdb" / "gc.pdb", output)

    chains_by_residue = {}
    for line in output.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            chains_by_residue.setdefault(line[17:20].strip(), set()).add(line[21])

    assert chains_by_residue == {"GGG": {"A"}, "CCC": {"B"}}


def test_fix_pdb_atom_labels_renumbers_and_makes_names_unique(data_dir, tmp_path):
    output = tmp_path / "fixed_atoms.pdb"
    nqe.fix_pdb_atom_labels(data_dir / "pdb" / "malformed.pdb", output)

    atom_lines = [
        line
        for line in output.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    serials = [int(line[6:11]) for line in atom_lines]
    names_by_residue = {}
    for line in atom_lines:
        names_by_residue.setdefault((line[21], line[22:27]), []).append(
            line[12:16].strip()
        )

    assert serials == list(range(1, len(atom_lines) + 1))
    assert all(len(names) == len(set(names)) for names in names_by_residue.values())
    assert all(not name[0].isdigit() for names in names_by_residue.values() for name in names)


def test_fix_pdb_atom_labels_restarts_names_for_equal_ids_in_different_chains(
    tmp_path,
):
    source = tmp_path / "two_chains.pdb"
    source.write_text(
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      2  C   LIG B   1       1.000   0.000   0.000  1.00  0.00           C  \n"
        "END\n"
    )
    output = tmp_path / "fixed.pdb"

    nqe.fix_pdb_atom_labels(source, output)
    names = [
        line[12:16].strip()
        for line in output.read_text().splitlines()
        if line.startswith("ATOM  ")
    ]

    assert names == ["C1", "C1"]


def test_save_only_index_atoms_does_not_mutate_modeller(data_dir, tmp_path):
    pdb = app.PDBFile(str(data_dir / "pdb" / "malonaldehyde.pdb"))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    output = tmp_path / "selection.pdb"

    nqe.save_only_index_atoms(modeller, [1, 4], file_idx=output)
    selected = app.PDBFile(str(output))

    assert modeller.topology.getNumAtoms() == 9
    expected = [list(modeller.topology.atoms())[index].name for index in (1, 4)]
    assert [atom.name for atom in selected.topology.atoms()] == expected
