import json
import os
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read
from mace.calculators.foundations_models import mace_off
from openmm import Vec3
from openmmml import MLPotential

import openmmnqe as nqe
import reactiontools as rt


@pytest.mark.pipeline
def test_openmm_ml():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield)
    nqe.remove_file_pattern('minimized*')


@pytest.mark.pipeline
def test_ase_mace():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")

    potential = MLPotential('ase')
    calculator = mace_off('small',
                          default_dtype='float32')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_relaxation_simple(modeller,
                                     potential,
                                     calculator=calculator)
    nqe.remove_file_pattern('minimized*')


@pytest.mark.orca
@pytest.mark.skipif("ORCA_PATH" not in os.environ,
                    reason="needs an ORCA installation (ORCA_PATH unset)")
def test_ase_orca():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/malonaldehyde.pdb")

    potential = MLPotential('ase')
    profile = OrcaProfile(command=os.environ['ORCA_PATH'])
    calculator = ORCA(profile=profile, orcasimpleinput='ENGRAD')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_relaxation_simple(modeller,
                                     potential,
                                     calculator=calculator)
    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('orca*')


@pytest.mark.pipeline
def test_openmm_ml_mixed_system():
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    padding = 1.5
    box_shape = 'dodecahedron'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    nqe.run_openmm_relaxation(modeller,
                              forcefield,
                              potential=potential,
                              ml_idx=ml_atoms)
    nqe.remove_file_pattern('minimized*')


def test_prepare_ligand_ff(tmp_path):
    print(flush=True)
    input_pdb = "tests/data/pdb/toluene.pdb"
    cache_name = str(tmp_path / "gaff-molecules.json")
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                          molecule,
                          gen_cache=True,
                          use_cache=False,
                          cache_name=cache_name)

    # Check that the cache files were created
    assert os.path.exists(cache_name)

    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule,
                                       gen_cache=False,
                                       use_cache=True,
                                       cache_name=cache_name)
    forcefield.createSystem(modeller.topology)


def test_nonstandard_ligand():
    print(flush=True)
    input_pdb = "tests/data/pdb/toluene.pdb"
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    pdb_topology = pdb_data.topology
    pdb_positions = pdb_data.positions
    modeller = app.Modeller(pdb_topology, pdb_positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

    nqe.run_openmm_relaxation(modeller,
                              forcefield,
                              platform_name='CPU')
    nqe.remove_file_pattern('minimized*')


def test_prepare_ligand_ff_multiple():
    print(flush=True)
    # A guanine/cytosine pair: two different residues, neither of which any
    # standard force field knows, so both have to be parameterised by GAFF.
    input_pdb = "tests/data/pdb/gc.pdb"
    # Deliberately not the cache test_prepare_ligand_ff leaves behind, so
    # neither test can seed the other's parameters.
    cache_name = "gaff-molecules-multiple.json"
    nqe.remove_file(cache_name)

    pdb_data, molecules = nqe.prepare_lig_system(input_pdb)
    assert len(molecules) == 2, "both residues should be picked up as ligands"
    assert {mol.name for mol in molecules} == {'GGG', 'CCC'}

    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecules,
                                       gen_cache=True,
                                       use_cache=False,
                                       cache_name=cache_name)
    forcefield.createSystem(modeller.topology)

    # The cache file appears as soon as it is opened, so its existence proves
    # nothing -- what matters is that both ligands are in it.
    with open(cache_name) as f:
        cache = json.load(f)
    cached_smiles = {record['smiles']
                     for table in cache.values()
                     for record in table.values()}
    assert cached_smiles == {mol.to_smiles() for mol in molecules}

    # The cache alone must be enough to rebuild the same system
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecules,
                                       gen_cache=False,
                                       use_cache=True,
                                       cache_name=cache_name)
    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     platform_name='CPU')

    nqe.remove_file(cache_name)
    nqe.remove_file_pattern('minimized*')


def test_prepare_lig_system_deduplicates_repeated_ligand_copies(tmp_path,
                                                                monkeypatch):
    source = Path("tests/data/pdb/toluene.pdb").resolve()
    pdb = app.PDBFile(str(source))
    receptor_topology = app.Topology()
    receptor_chain = receptor_topology.addChain()
    receptor_residue = receptor_topology.addResidue("ALA", receptor_chain)
    receptor_topology.addAtom(
        "CA", app.Element.getBySymbol("C"), receptor_residue
    )
    modeller = app.Modeller(
        receptor_topology, [Vec3(-1.0, 0.0, 0.0)] * unit.nanometer
    )
    modeller.add(pdb.topology, pdb.positions)
    shift = Vec3(1.0, 0.0, 0.0) * unit.nanometer
    modeller.add(pdb.topology, [position + shift for position in pdb.positions])

    repeated_pdb = tmp_path / "two_toluenes.pdb"
    with open(repeated_pdb, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle)

    monkeypatch.chdir(tmp_path)
    pdb_data, molecule = nqe.prepare_lig_system(str(repeated_pdb), lig_names="MBN")

    assert not isinstance(molecule, list)
    assert molecule.n_atoms == 15
    residues = list(pdb_data.topology.residues())
    assert [residue.name for residue in residues].count("MBN") == 2
    assert len(residues) == 3
    assert pdb_data.topology.getNumAtoms() == 31


def _get_total_mass(system):
    total_mass = 0.0 * unit.dalton
    for i in range(system.getNumParticles()):
        total_mass += system.getParticleMass(i)
    return total_mass


def test_deuterate_system():
    print(flush=True)
    pdb = app.PDBFile('tests/data/pdb/input.pdb')
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield,
                        model='tip3p',
                        padding=1.0 * unit.nanometer)

    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     constraints=app.HBonds,
                                     rigidWater=True)

    mass_before = _get_total_mass(system)

    # Count the same way deuterate_system does, so the two cannot drift apart
    h_count = sum(1 for atom in modeller.topology.atoms()
                  if atom.element and atom.element.symbol == 'H')

    print(f"--- Applying deuteration to {h_count} hydrogens (option='all') ---")
    nqe.deuterate_system(modeller, system, option='all')

    mass_after = _get_total_mass(system)
    expected_increase = (app.element.deuterium.mass - app.element.hydrogen.mass) * h_count
    actual_increase = mass_after - mass_before

    print(f"{'Mass before':<20} | {mass_before.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Mass after':<20} | {mass_after.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Actual increase':<20} | {actual_increase.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Expected increase':<20} | {expected_increase.value_in_unit(unit.dalton):.4f} Da")

    # Every hydrogen, and only the hydrogens, should have gained the H -> D mass
    # difference. The tolerance is relative: element-mass tables drift at the
    # ~1e-7 level between OpenMM releases, while one miscounted hydrogen is
    # ~2.5e-5 of the total here, so whole-atom errors are still caught.
    diff = abs(actual_increase - expected_increase).value_in_unit(unit.dalton)
    assert diff < 1e-5 * expected_increase.value_in_unit(unit.dalton)


def test_get_atoms_in_residue():
    print(flush=True)
    input_pdb = 'tests/data/pdb/input.pdb'
    indexes = nqe.get_atoms_in_residue(input_pdb, 0)
    print(indexes, flush=True)
    ref_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert indexes == ref_indexes


def test_count_dna_and_estimate_charge():
    topology = app.Topology()
    chain = topology.addChain()
    for residue_name in ("DA", "DC", "DG", "DT", "ALA"):
        topology.addResidue(residue_name, chain)

    assert nqe.count_dna_and_estimate_charge(topology) == -4


def test_atom_indices_from_vmd_picks():
    print(flush=True)
    input_pdb = 'tests/data/pdb/input.pdb'
    pdb = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    vmd_pick_str = ['PHE6:HE2', 'LYS29:CG']
    indexes = nqe.atom_indices_from_vmd_picks(modeller, vmd_pick_str)
    print(indexes, flush=True)
    ref_indexes = [86, 461]
    assert indexes == ref_indexes


def test_distance_between_atoms():
    print(flush=True)
    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    pdb = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    indexes = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O1', 'LIG1:H5'])
    distance = nqe.distance_between_atoms(modeller, indexes[0], indexes[1])
    nm = distance.value_in_unit(unit.nanometer)
    print(f"Distance between atoms: {nm:.4f} nm", flush=True)
    assert nm == pytest.approx(0.0991984, abs=1e-6)


def test_distance_between_atoms_carries_its_units():
    """Guards against the unit being stripped off the return value.

    Building the norm out of ``dr.x`` and friends silently loses it, because
    indexing a Quantity that wraps a Vec3 hands back the bare component. The
    result then reads as a plain number of nanometres, which is right up until
    someone asks it for another unit -- or calls ``value_in_unit`` and gets an
    AttributeError, as every caller in ``reactiontools.tools_cv`` does.
    """
    pdb = app.PDBFile('tests/data/pdb/malonaldehyde.pdb')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    indexes = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O1', 'LIG1:H5'])

    distance = nqe.distance_between_atoms(modeller, indexes[0], indexes[1])

    assert unit.is_quantity(distance)
    assert distance.unit.is_compatible(unit.nanometer)
    # The same length, asked for two ways.
    assert distance.value_in_unit(unit.angstrom) == pytest.approx(
        distance.value_in_unit(unit.nanometer) * 10.0)


def test_angle_between_atoms():
    print(flush=True)
    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    pdb = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    indexes = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O1', 'LIG1:H5', 'LIG1:O2'])
    angle = nqe.angle_between_atoms(modeller, indexes[0], indexes[1], indexes[2], degrees=True)
    print(f"Angle between atoms: {angle:.4f} degrees", flush=True)
    assert angle == pytest.approx(146.601, abs=0.01)


def test_fix_pdb_chains():
    print(flush=True)
    input_pdb = 'tests/data/pdb/malformed.pdb'
    nqe.fix_pdb_chains(input_pdb, "fixed.pdb")
    # Assert that the pdb file was created
    assert os.path.isfile('fixed.pdb')
    # Assert that there are two chains in the fixed pdb
    with open('fixed.pdb') as f:
        chains = set()
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                chain_id = line[21]
                chains.add(chain_id)
        assert len(chains) == 2
    os.remove('fixed.pdb')


def test_fix_pdb_chains_distinguishes_equal_ids_in_source_chains(tmp_path):
    output = tmp_path / "fixed.pdb"
    nqe.fix_pdb_chains("tests/data/pdb/gc.pdb", str(output))

    chains_by_residue = {}
    with open(output) as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                chains_by_residue.setdefault(line[17:20].strip(), set()).add(line[21])

    assert chains_by_residue["GGG"] != chains_by_residue["CCC"]


def test_fix_pdb_atom_labels():
    print(flush=True)
    input_pdb = 'tests/data/pdb/malformed.pdb'
    nqe.fix_pdb_atom_labels(input_pdb, "fixed_atoms.pdb")
    # Assert that the pdb file was created
    assert os.path.isfile('fixed_atoms.pdb')
    # Assert that there are no atom labels starting with a digit
    with open('fixed_atoms.pdb') as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                atom_label = line[12:16].strip()
                assert not atom_label[0].isdigit(), f"Atom label '{atom_label}' starts with a digit"
    os.remove('fixed_atoms.pdb')


def test_convert_xyz_to_pdb():
    # Define your file names
    input_file = 'tests/data/GC.xyz'
    output_file = 'output.pdb'
    rt.convert_xyz_to_pdb(input_file, output_file, cutoff_multiplier=1.1)
    # Assert that the pdb file was created
    assert os.path.isfile(output_file)
    os.remove(output_file)


def test_convert_pdb_to_xyz():
    print(flush=True)
    input_file = 'tests/data/pdb/malonaldehyde.pdb'
    output_file = 'output.xyz'
    n_frames = rt.convert_pdb_to_xyz(input_file, output_file)
    assert os.path.isfile(output_file)
    assert n_frames == 1

    # The atom count and elements must survive the conversion
    with open(input_file) as f:
        n_pdb_atoms = sum(1 for line in f if line.startswith(("ATOM  ", "HETATM")))
    atoms = read(output_file)
    assert len(atoms) == n_pdb_atoms
    assert 'X' not in atoms.get_chemical_symbols()
    os.remove(output_file)


def test_convert_xyz_to_pdb_round_trip():
    print(flush=True)
    input_file = 'tests/data/GC.xyz'
    pdb_file = 'round_trip.pdb'
    xyz_file = 'round_trip.xyz'

    # A GC base pair is two separate molecules
    n_clusters = rt.convert_xyz_to_pdb(input_file, pdb_file, cutoff_multiplier=1.1)
    assert n_clusters == 2
    rt.convert_pdb_to_xyz(pdb_file, xyz_file)

    original = read(input_file)
    recovered = read(xyz_file)
    assert sorted(recovered.get_chemical_symbols()) == sorted(original.get_chemical_symbols())

    # PDB coordinates carry three decimal places, so compare at that precision
    def key_set(atoms):
        return {(s, tuple(np.round(p, 3)))
                for s, p in zip(atoms.get_chemical_symbols(), atoms.positions, strict=True)}

    assert key_set(recovered) == key_set(original)
    os.remove(pdb_file)
    os.remove(xyz_file)


def test_move_pdb_to_origin():
    print(flush=True)
    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    nqe.move_pdb_to_origin(input_pdb, 'tmp.pdb')
    # Assert that the pdb file was created
    assert os.path.isfile('tmp.pdb')
    os.remove('tmp.pdb')


def test_convert_sdfs_to_pdb():
    nqe.convert_sdfs_to_pdb("tests/data/CH4.sdf", "single_ligand.pdb")
    assert os.path.isfile('single_ligand.pdb')
    os.remove('single_ligand.pdb')

    nqe.convert_sdfs_to_pdb(["tests/data/CH4.sdf", "tests/data/H2O.sdf"], "combined_ligands.pdb")
    assert os.path.isfile('combined_ligands.pdb')
    os.remove('combined_ligands.pdb')


def test_center_in_box():
    print(flush=True)
    # Create a simple topology and positions
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("RES", chain)
    topology.addAtom("A1", app.Element.getByAtomicNumber(6), residue)
    topology.addAtom("A2", app.Element.getByAtomicNumber(6), residue)
    # Set box dimensions (e.g., 10x10x10 nm)
    topology.setUnitCellDimensions(unit.Quantity((10.0, 10.0, 10.0), unit.nanometer))
    # Place atoms at arbitrary positions
    positions = unit.Quantity(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), unit.nanometer)
    modeller = app.Modeller(topology, positions)

    # Call the function under test
    nqe.center_in_box(modeller)

    # Check that the centroid is at the box center
    centered_positions = modeller.positions.value_in_unit(unit.nanometer)
    centroid = np.mean(centered_positions, axis=0)
    box_center = np.array([5.0, 5.0, 5.0])  # Half of box dimensions
    assert np.allclose(centroid, box_center, atol=1e-6), f"Centroid {centroid} not at box center {box_center}"


def test_center_in_box_uses_triclinic_vector_sum():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("RES", chain)
    topology.addAtom("A1", app.Element.getByAtomicNumber(6), residue)
    box_vectors = (
        Vec3(2.0, 0.0, 0.0),
        Vec3(0.5, 2.0, 0.0),
        Vec3(0.2, 0.3, 2.0),
    ) * unit.nanometer
    topology.setPeriodicBoxVectors(box_vectors)
    modeller = app.Modeller(
        topology, [Vec3(0.0, 0.0, 0.0)] * unit.nanometer
    )

    nqe.center_in_box(modeller)

    centered = modeller.positions.value_in_unit(unit.nanometer)
    assert np.allclose(centered[0], [1.35, 1.15, 1.0])
