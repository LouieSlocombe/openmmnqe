"""Tests for structure editing and filesystem helpers."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3
from rdkit import Chem

import openmmnqe as nqe
import openmmnqe.io as nqe_io


def test_file_helpers_copy_list_and_remove(tmp_path: Path) -> None:
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


def test_remove_file_pattern_only_removes_matches(tmp_path: Path) -> None:
    matching = [tmp_path / "run-1.log", tmp_path / "run-2.log"]
    untouched = tmp_path / "run.txt"
    for path in [*matching, untouched]:
        path.write_text("data")

    nqe.remove_file_pattern(str(tmp_path / "*.log"))

    assert all(not path.exists() for path in matching)
    assert untouched.exists()


def test_remove_file_helpers_propagate_non_missing_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected.log"
    protected.write_text("data")

    def deny_removal(path: str | Path) -> None:
        raise PermissionError(path)

    monkeypatch.setattr(nqe_io.os, "remove", deny_removal)

    with pytest.raises(PermissionError):
        nqe.remove_file(protected)
    with pytest.raises(PermissionError):
        nqe.remove_file_pattern(str(tmp_path / "*.log"))


def test_remove_file_pattern_tolerates_a_match_disappearing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    missing = tmp_path / "gone.log"
    remaining = tmp_path / "remaining.log"
    remaining.write_text("data")
    monkeypatch.setattr(
        nqe_io.glob, "glob", lambda pattern: [str(missing), str(remaining)],
    )

    nqe.remove_file_pattern(str(tmp_path / "*.log"))

    assert not remaining.exists()


def test_xyz_to_sdf_writes_readable_molecule(data_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "gc.sdf"

    count = nqe.xyz_to_sdf(data_dir / "GC.xyz", output)
    molecules = [mol for mol in Chem.SDMolSupplier(str(output), removeHs=False) if mol]

    assert count == 1
    assert len(molecules) == 1
    assert molecules[0].GetNumAtoms() == 29
    assert molecules[0].GetNumBonds() > 0


def test_xyz_to_sdf_parses_documented_charge_formats_across_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "charges.xyz"
    source.write_text(
        "\n".join(
            [
                "1",
                "anion charge=-1",
                "H 0 0 0",
                "1",
                "cation q: +2",
                "H 0 0 0",
                "1",
                "state +3 selected",
                "H 0 0 0",
                "1",
                "no charge here",
                "H 0 0 0",
                "1",
                "",
                "H 0 0 0",
            ]
        )
    )
    output = tmp_path / "charges.sdf"
    charges = []
    monkeypatch.setattr(
        nqe_io.rdDetermineBonds,
        "DetermineBonds",
        lambda molecule, charge: charges.append(charge),
    )

    count = nqe.xyz_to_sdf(
        source,
        output,
        default_charge=7,
        sanitize=False,
    )
    molecules = list(Chem.SDMolSupplier(str(output), removeHs=False, sanitize=False))

    assert count == 5
    assert charges == [-1, 2, 3, 7, 7]
    assert [molecule.GetProp("_Name") for molecule in molecules] == [
        "anion charge=-1",
        "cation q: +2",
        "state +3 selected",
        "no charge here",
        "charges_5",
    ]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "No XYZ frames"),
        ("not-a-count\ncomment\n", "Expected atom count"),
        ("2\ncomment\nH 0 0 0\n", "Unexpected EOF"),
        ("1\ncomment\nH x 0 0\n", "Bad XYZ coordinates"),
    ],
)
def test_xyz_to_sdf_rejects_malformed_xyz(tmp_path: Path, contents: str, message: str) -> None:
    source = tmp_path / "bad.xyz"
    source.write_text(contents)

    with pytest.raises(ValueError, match=message):
        nqe.xyz_to_sdf(source, tmp_path / "bad.sdf")


def test_xyz_to_sdf_closes_writer_when_conversion_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "molecule.xyz"
    source.write_text("1\nframe\nH 0 0 0\n")

    class RecordingWriter:
        closed = False

        def close(self) -> None:
            self.closed = True

    writer = RecordingWriter()
    monkeypatch.setattr(nqe_io.Chem, "SDWriter", lambda path: writer)

    def fail_bond_inference(molecule: Chem.Mol, charge: int) -> None:
        raise RuntimeError("bond inference failed")

    monkeypatch.setattr(
        nqe_io.rdDetermineBonds,
        "DetermineBonds",
        fail_bond_inference,
    )

    with pytest.raises(RuntimeError, match="bond inference failed"):
        nqe.xyz_to_sdf(source, tmp_path / "molecule.sdf")

    assert writer.closed


def test_relabel_residues_supports_paths_and_file_objects(data_dir: Path, tmp_path: Path) -> None:
    source = data_dir / "pdb" / "malonaldehyde.pdb"
    output = tmp_path / "renamed.pdb"

    returned = nqe.relabel_residues_in_pdb(source, {"LIG": "MAL"}, str(output))
    parsed = app.PDBFile(str(output))
    in_memory = StringIO()
    nqe.relabel_residues_in_pdb(source, {"LIG": "MAL"}, in_memory)

    assert {residue.name for residue in returned.topology.residues()} == {"MAL"}
    assert {residue.name for residue in parsed.topology.residues()} == {"MAL"}
    assert " MAL " in in_memory.getvalue()


def test_remove_residues_removes_only_requested_names(data_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "without_water.pdb"

    nqe.remove_residues_in_pdb(
        data_dir / "pdb" / "malformed.pdb",
        output,
        {"HOH"},
    )
    residues = [residue.name for residue in app.PDBFile(str(output)).topology.residues()]

    assert residues == ["AMM"]


def test_remove_residues_materializes_one_shot_name_iterable(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "without_residues.pdb"

    nqe.remove_residues_in_pdb(
        data_dir / "pdb" / "malformed.pdb",
        output,
        iter(("AMM", "HOH")),
    )

    atom_records = [
        line
        for line in output.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    assert atom_records == []


def test_remove_residues_accepts_one_name_as_a_string(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "without_water.pdb"

    nqe.remove_residues_in_pdb(
        data_dir / "pdb" / "malformed.pdb",
        output,
        "HOH",
    )

    residues = [
        residue.name
        for residue in app.PDBFile(str(output)).topology.residues()
    ]
    assert residues == ["AMM"]


def test_fix_pdb_runs_repair_steps_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    class FakeFixer:
        topology = object()
        positions = object()

        def __init__(self, filename: str) -> None:
            calls.append(("init", filename))

        def findMissingResidues(self) -> None:
            calls.append("findMissingResidues")

        def findNonstandardResidues(self) -> None:
            calls.append("findNonstandardResidues")

        def replaceNonstandardResidues(self) -> None:
            calls.append("replaceNonstandardResidues")

        def removeHeterogens(self, keep_water: bool) -> None:
            calls.append(("removeHeterogens", keep_water))

        def findMissingAtoms(self) -> None:
            calls.append("findMissingAtoms")

        def addMissingAtoms(self) -> None:
            calls.append("addMissingAtoms")

        def addMissingHydrogens(self, ph: float) -> None:
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


def test_fix_pdb_can_keep_heterogens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeFixer:
        topology = object()
        positions = object()

        def __init__(self, filename: str) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            if name == "removeHeterogens":
                pytest.fail("removeHeterogens should not be called")
            return lambda *args: None

    monkeypatch.setattr(nqe_io, "PDBFixer", FakeFixer)
    monkeypatch.setattr(nqe_io.app.PDBFile, "writeFile", lambda *args: None)

    nqe.fix_pdb("input.pdb", tmp_path / "output.pdb", rm_heterogens=False)


def test_convert_sdfs_to_pdb_preserves_molecules_and_bonds(data_dir: Path, tmp_path: Path) -> None:
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


@pytest.mark.parametrize("contents", ["", "not an SDF record\n$$$$\n"])
def test_convert_sdfs_to_pdb_rejects_empty_or_malformed_records_before_writing(
    data_dir: Path,
    tmp_path: Path,
    contents: str,
) -> None:
    bad_input = tmp_path / "bad.sdf"
    bad_input.write_text(contents)
    output = tmp_path / "combined.pdb"

    with pytest.raises(ValueError, match="empty or malformed|Malformed molecule"):
        nqe.convert_sdfs_to_pdb(
            [data_dir / "CH4.sdf", bad_input],
            output,
        )

    assert not output.exists()


def test_save_pdb_selection_preserves_requested_atom_order(data_dir: Path, tmp_path: Path) -> None:
    source = data_dir / "pdb" / "malonaldehyde.pdb"
    output = tmp_path / "selection.pdb"
    original = app.PDBFile(str(source))
    expected = [list(original.topology.atoms())[index].name for index in (0, 3, 8)]

    nqe.save_pdb_selection(source, [0, 3, 8], output)
    selected = app.PDBFile(str(output))

    assert [atom.name for atom in selected.topology.atoms()] == expected
    assert selected.topology.getNumAtoms() == 3


def test_move_pdb_to_origin_centres_coordinates(data_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "centred.pdb"

    nqe.move_pdb_to_origin(data_dir / "pdb" / "malonaldehyde.pdb", output)
    positions = app.PDBFile(str(output)).getPositions(asNumpy=True)
    centroid = positions.value_in_unit(unit.nanometer).mean(axis=0)

    assert np.allclose(centroid, 0.0, atol=5e-5)


def test_center_in_box_centres_orthorhombic_system() -> None:
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


def test_center_in_box_uses_triclinic_vector_sum() -> None:
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


def test_center_in_box_leaves_nonperiodic_positions_unchanged() -> None:
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


def test_fix_pdb_chains_assigns_one_chain_per_residue(data_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "fixed.pdb"

    nqe.fix_pdb_chains(data_dir / "pdb" / "malformed.pdb", output)
    parsed = app.PDBFile(str(output))

    assert [chain.id for chain in parsed.topology.chains()] == ["A", "B"]
    assert [residue.name for residue in parsed.topology.residues()] == ["HOH", "AMM"]


def test_fix_pdb_chains_distinguishes_equal_ids(data_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "fixed.pdb"
    nqe.fix_pdb_chains(data_dir / "pdb" / "gc.pdb", output)

    chains_by_residue = {}
    for line in output.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            chains_by_residue.setdefault(line[17:20].strip(), set()).add(line[21])

    assert chains_by_residue == {"GGG": {"A"}, "CCC": {"B"}}


def test_fix_pdb_atom_labels_renumbers_and_makes_names_unique(data_dir: Path, tmp_path: Path) -> None:
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
    tmp_path: Path,
) -> None:
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


def test_save_only_index_atoms_does_not_mutate_modeller(data_dir: Path, tmp_path: Path) -> None:
    pdb = app.PDBFile(str(data_dir / "pdb" / "malonaldehyde.pdb"))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    output = tmp_path / "selection.pdb"

    nqe.save_only_index_atoms(modeller, [1, 4], file_idx=output)
    selected = app.PDBFile(str(output))

    assert modeller.topology.getNumAtoms() == 9
    expected = [list(modeller.topology.atoms())[index].name for index in (1, 4)]
    assert [atom.name for atom in selected.topology.atoms()] == expected


def test_save_only_index_atoms_materializes_one_shot_index_iterable(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    pdb = app.PDBFile(str(data_dir / "pdb" / "malonaldehyde.pdb"))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    output = tmp_path / "selection.pdb"

    nqe.save_only_index_atoms(modeller, iter((0, 4)), file_idx=output)

    selected = app.PDBFile(str(output))
    expected = [list(modeller.topology.atoms())[index].name for index in (0, 4)]
    assert [atom.name for atom in selected.topology.atoms()] == expected


@pytest.mark.parametrize("indices", [(8, 3, 0, 3, -1, 99), (), (99,)])
def test_selection_writers_share_topology_order_and_ignore_absent_indices(
    data_dir: Path, tmp_path: Path, indices: tuple[int, ...],
) -> None:
    source = data_dir / "pdb" / "malonaldehyde.pdb"
    pdb = app.PDBFile(str(source))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    original_positions = np.array(
        modeller.positions.value_in_unit(unit.nanometer), copy=True,
    )
    from_path = tmp_path / "path.pdb"
    from_modeller = tmp_path / "modeller.pdb"

    nqe.save_pdb_selection(source, iter(indices), from_path)
    nqe.save_only_index_atoms(modeller, iter(indices), from_modeller)

    assert from_path.read_text() == from_modeller.read_text()
    kept = [atom for atom in pdb.topology.atoms() if atom.index in indices]
    records = [
        line for line in from_path.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    assert [line[12:16].strip() for line in records] == [atom.name for atom in kept]
    assert modeller.topology.getNumAtoms() == pdb.topology.getNumAtoms()
    assert np.array_equal(
        modeller.positions.value_in_unit(unit.nanometer), original_positions,
    )


def test_xyz_to_sdf_ignores_trailing_blank_lines(tmp_path: Path) -> None:
    source = tmp_path / "trailing.xyz"
    source.write_text(
        "3\nwater\nO 0 0 0\nH 0.9572 0 0\nH -0.24 0.927 0\n\n\n   \n"
    )

    count = nqe.xyz_to_sdf(source, tmp_path / "trailing.sdf")
    molecules = list(Chem.SDMolSupplier(str(tmp_path / "trailing.sdf"), removeHs=False))

    assert count == 1
    assert molecules[0].GetNumAtoms() == 3
    assert molecules[0].GetNumBonds() == 2


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("1\n", "Unexpected EOF after atom count"),
        ("1\ncomment\nH 0 0\n", "Bad XYZ atom line"),
        (
            "1\nfirst\nH 0 0 0\nnot-a-count\nsecond\nH 0 0 0\n",
            "Expected atom count at line 4",
        ),
    ],
)
def test_xyz_to_sdf_names_the_malformed_frame(
    tmp_path: Path, contents: str, message: str,
) -> None:
    source = tmp_path / "bad.xyz"
    source.write_text(contents)

    with pytest.raises(ValueError, match=message):
        nqe.xyz_to_sdf(source, tmp_path / "bad.sdf")


def test_xyz_to_sdf_kekulizes_aromatic_bonds_on_request(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # The MOL block always carries Kekule bond orders, so the written file
    # cannot show the difference; the SMILES the conversion prints can.
    def printed_smiles(**kwargs: Any) -> str:
        nqe.xyz_to_sdf(data_dir / "GC.xyz", tmp_path / "gc.sdf", **kwargs)
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("SMI:")]
        assert len(lines) == 1
        return lines[0].removeprefix("SMI:")

    aromatic = printed_smiles()
    kekulized = printed_smiles(kekulize=True)

    assert ":" in aromatic
    assert ":" not in kekulized
    assert "=" in kekulized


def test_xyz_to_sdf_tolerates_a_molecule_that_cannot_be_kekulized(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, tmp_path: Path,
) -> None:
    def refuse(molecule: Chem.Mol, clearAromaticFlags: bool = False) -> None:
        raise Chem.KekulizeException("no Kekule structure")

    monkeypatch.setattr(nqe_io.Chem, "Kekulize", refuse)

    count = nqe.xyz_to_sdf(data_dir / "GC.xyz", tmp_path / "gc.sdf", kekulize=True)

    assert count == 1
    assert (tmp_path / "gc.sdf").stat().st_size > 0


def test_xyz_to_sdf_falls_back_to_partial_sanitisation(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, tmp_path: Path,
) -> None:
    full_sanitize = nqe_io.Chem.SanitizeMol
    calls: list[Any] = []

    def sanitize(molecule: Chem.Mol, **kwargs: Any) -> Any:
        calls.append(kwargs.get("sanitizeOps"))
        if not kwargs:
            raise Chem.AtomValenceException("valence rejected")
        return full_sanitize(molecule, **kwargs)

    monkeypatch.setattr(nqe_io.Chem, "SanitizeMol", sanitize)

    count = nqe.xyz_to_sdf(data_dir / "GC.xyz", tmp_path / "gc.sdf")

    assert count == 1
    expected_partial = (
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
    )
    assert calls == [None, expected_partial]


def _write_three_residue_pdb(path: Path) -> None:
    """Two waters and an ammonia, so one relabel can hit twice and miss once."""
    path.write_text(
        "HETATM    1  O   HOH     1       0.000   0.000   0.000  1.00  0.00           O  \n"
        "HETATM    2  O   HOH     2       3.000   0.000   0.000  1.00  0.00           O  \n"
        "HETATM    3  N   AMM     3       6.000   0.000   0.000  1.00  0.00           N  \n"
        "END\n"
    )


def test_relabel_residues_counts_repeats_and_leaves_other_names_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "three.pdb"
    _write_three_residue_pdb(source)
    output = tmp_path / "relabelled.pdb"

    returned = nqe.relabel_residues_in_pdb(source, {"HOH": "LIG"}, output)

    assert [residue.name for residue in returned.topology.residues()] == ["LIG", "LIG", "AMM"]
    assert [
        residue.name for residue in app.PDBFile(str(output)).topology.residues()
    ] == ["LIG", "LIG", "AMM"]
    assert "Relabeled 2 residues from 'HOH' to 'LIG'" in capsys.readouterr().out


def test_relabel_residues_reports_when_nothing_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "three.pdb"
    _write_three_residue_pdb(source)
    output = tmp_path / "unchanged.pdb"

    nqe.relabel_residues_in_pdb(source, {"ZZZ": "YYY"}, output)

    assert [
        residue.name for residue in app.PDBFile(str(output)).topology.residues()
    ] == ["HOH", "HOH", "AMM"]
    assert "No residues found matching the relabel map" in capsys.readouterr().out


def test_remove_residues_writes_the_structure_unchanged_when_nothing_matches(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "unchanged.pdb"

    nqe.remove_residues_in_pdb(data_dir / "pdb" / "malformed.pdb", output, "ZZZ")

    residues = [residue.name for residue in app.PDBFile(str(output)).topology.residues()]
    assert residues == ["HOH", "AMM"]
    assert "No matching residues found to delete" in capsys.readouterr().out


def test_convert_sdfs_to_pdb_rejects_no_inputs_or_a_missing_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "combined.pdb"

    with pytest.raises(ValueError, match="No SDF input files"):
        nqe.convert_sdfs_to_pdb([], output)
    with pytest.raises(FileNotFoundError):
        nqe.convert_sdfs_to_pdb(tmp_path / "absent.sdf", output)

    assert not output.exists()


def test_convert_sdfs_to_pdb_explains_a_supplier_that_cannot_open_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "unreadable.sdf"
    source.write_text("")
    output = tmp_path / "combined.pdb"

    def refuse(path: str, **kwargs: Any) -> None:
        raise OSError("Invalid input file")

    monkeypatch.setattr(nqe_io.Chem, "SDMolSupplier", refuse)

    with pytest.raises(ValueError, match="empty or malformed"):
        nqe.convert_sdfs_to_pdb(source, output)

    assert not output.exists()


def test_convert_sdfs_to_pdb_rejects_a_file_with_no_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "blank.sdf"
    source.write_text("")
    output = tmp_path / "combined.pdb"
    monkeypatch.setattr(nqe_io.Chem, "SDMolSupplier", lambda path, **kwargs: iter(()))

    with pytest.raises(ValueError, match="no molecule records"):
        nqe.convert_sdfs_to_pdb(source, output)

    assert not output.exists()


def test_convert_sdfs_to_pdb_places_a_molecule_without_coordinates_at_the_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "flat.sdf"
    source.write_text("")
    output = tmp_path / "combined.pdb"
    flat = Chem.AddHs(Chem.MolFromSmiles("O"))
    assert flat.GetNumConformers() == 0
    monkeypatch.setattr(nqe_io.Chem, "SDMolSupplier", lambda path, **kwargs: iter([flat]))

    nqe.convert_sdfs_to_pdb(source, output)

    written = app.PDBFile(str(output))
    assert written.topology.getNumAtoms() == 3
    assert written.topology.getNumBonds() == 2
    assert np.array_equal(
        written.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        np.zeros((3, 3)),
    )
    assert "has no 3D coordinates" in capsys.readouterr().out


class _Vec3Like:
    """Attribute and index access like Vec3, but opaque to numpy.

    Without ``__len__`` numpy cannot size it, so ``np.asarray`` fails and
    ``center_in_box`` has to take its coordinate-by-coordinate fallback.
    """

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]


class _AttributePoint:
    """Only ``.x``, ``.y`` and ``.z``: indexing it raises."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class _IndexPoint:
    """Only ``__getitem__``, and no ``__len__`` for numpy to size it by."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self._values = (x, y, z)

    def __getitem__(self, index: int) -> float:
        return self._values[index]


@pytest.mark.parametrize(
    "point_type", [_Vec3Like, _AttributePoint, _IndexPoint],
    ids=["attributes and index", "attributes only", "index only"],
)
def test_center_in_box_falls_back_to_reading_odd_positions_one_coordinate_at_a_time(
    point_type: type,
) -> None:
    topology = app.Topology()
    topology.setUnitCellDimensions(unit.Quantity((4.0, 4.0, 4.0), unit.nanometer))
    points = [point_type(0.0, 0.0, 0.0), point_type(2.0, 2.0, 2.0)]
    with pytest.raises((TypeError, ValueError)):
        np.asarray(points, dtype=float)
    modeller = SimpleNamespace(
        topology=topology,
        positions=SimpleNamespace(value_in_unit=lambda target: list(points)),
    )

    nqe.center_in_box(modeller)

    centred = modeller.positions.value_in_unit(unit.nanometer)
    assert np.allclose(centred, [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])


def test_center_in_box_leaves_a_topology_without_box_support_alone() -> None:
    original = SimpleNamespace(
        value_in_unit=lambda target: [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
    )
    modeller = SimpleNamespace(topology=SimpleNamespace(), positions=original)

    nqe.center_in_box(modeller)

    assert modeller.positions is original


def test_fix_pdb_atom_labels_derives_the_element_from_the_name_when_absent(
    tmp_path: Path,
) -> None:
    # No element column: records trimmed short of column 77.
    source = tmp_path / "no_elements.pdb"
    source.write_text(
        "ATOM      9  CA  LIG A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      8  HB1 LIG A   1       1.000   0.000   0.000  1.00  0.00\n"
        "ATOM      7  1   LIG A   1       2.000   0.000   0.000  1.00  0.00\n"
        "END\n"
    )
    output = tmp_path / "fixed.pdb"

    nqe.fix_pdb_atom_labels(source, output)
    records = [
        line for line in output.read_text().splitlines() if line.startswith("ATOM  ")
    ]

    assert [int(line[6:11]) for line in records] == [1, 2, 3]
    assert [line[12:16] for line in records] == ["CA1 ", "HB1 ", " X1 "]
