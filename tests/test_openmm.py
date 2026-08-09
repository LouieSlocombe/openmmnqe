import json
import os
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import openmm.app as app
import openmm.unit as unit
import pandas as pd
import pytest
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read
from mace.calculators.foundations_models import mace_off
from openmmml import MLPotential
from scipy.stats import linregress

import openmmnqe as nqe


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


def test_prepare_ligand_ff():
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"
    cache_name = "gaff-molecules.json"
    rm_ions = ['Na+',
               'Cl-',
               'NA']
    residue_map = {'DGN': 'DG',
                   'DTN': 'DT',
                   'GTP': 'LIG'}
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb,
                                                rm_ions=rm_ions,
                                                residue_map=residue_map,
                                                lig_names='LIG',
                                                rm_files=False)
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
    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"

    rm_ions = ['Na+',
               'Cl-',
               'NA']
    residue_map = {'DGN': 'DG',
                   'DTN': 'DT',
                   'GTP': 'LIG'}
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb, rm_ions=rm_ions, residue_map=residue_map, lig_names='LIG')
    pdb_topology = pdb_data.topology
    pdb_positions = pdb_data.positions
    modeller = app.Modeller(pdb_topology, pdb_positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

    nqe.run_openmm_relaxation(modeller,
                              forcefield)
    nqe.remove_file_pattern('minimized*')


def test_prepare_ligand_ff_multiple():
    print(flush=True)
    # A guanine/cytosine pair: two different residues, neither of which any
    # standard force field knows, so both have to be parameterised by GAFF.
    # The G-T wobble PDBs cannot be used here -- their DGN/DTN deoxynucleosides
    # are standard amber14 residues, so amber14-all.xml matches them on graph
    # (residue names are not consulted) and the ligand path is never reached.
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
                                     forcefield)

    nqe.remove_file(cache_name)
    nqe.remove_file_pattern('minimized*')


def test_ligand_already_in_force_field_warns():
    print(flush=True)
    # DGN and DTN are free deoxynucleosides: non-standard names as far as the
    # ligand scan is concerned, but amber14-all.xml carries templates for both.
    # OpenMM matches residue templates on the molecular graph and never looks at
    # residue names, so the standard template wins and every bit of GAFF work
    # done for these two is discarded -- which is what the warnings say.
    with pytest.warns(UserWarning, match="standard AMBER residue names"):
        _, molecules = nqe.prepare_lig_system("tests/data/pdb/gt_wob_solv_clean.pdb")
    assert {mol.name for mol in molecules} == {'DGN', 'DTN'}

    with pytest.warns(UserWarning, match="already matched by residue template"):
        nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                              molecules)

    # The negative control: GGG/CCC are genuine ligands, so neither warning may
    # fire for them, or the check would be worthless.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, gc_molecules = nqe.prepare_lig_system("tests/data/pdb/gc.pdb")
        nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                              gc_molecules)
    assert not [w for w in caught
                if "standard AMBER residue names" in str(w.message)
                or "already matched by residue template" in str(w.message)]


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

    # Every hydrogen, and only the hydrogens, should have gained the H -> D mass difference
    assert abs(actual_increase - expected_increase).value_in_unit(unit.dalton) < 1e-3


def test_get_atoms_in_residue():
    print(flush=True)
    input_pdb = 'tests/data/pdb/input.pdb'
    indexes = nqe.get_atoms_in_residue(input_pdb, 0)
    print(indexes, flush=True)
    ref_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert indexes == ref_indexes


def test_count_dna_and_estimate_charge():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/gt_wob_pol.pdb")
    est_charge = nqe.count_dna_and_estimate_charge(pdb.topology)
    print(f"Estimated net charge: {est_charge}")
    assert est_charge == -6


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
    input_pdb = 'tests/data/pdb/gt_wob_solv_clean.pdb'
    pdb = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    indexes = nqe.atom_indices_from_vmd_picks(modeller, ['DTN1:H3', 'DGN1:O6'])
    distance = nqe.distance_between_atoms(modeller, indexes[0], indexes[1])
    nm = distance.value_in_unit(unit.nanometer)
    print(f"Distance between atoms: {nm:.4f} nm", flush=True)
    ref_distance = 0.179  # in nm
    assert abs(nm - ref_distance) < 0.01


def test_distance_between_atoms_carries_its_units():
    """Guards against the unit being stripped off the return value.

    Building the norm out of ``dr.x`` and friends silently loses it, because
    indexing a Quantity that wraps a Vec3 hands back the bare component. The
    result then reads as a plain number of nanometres, which is right up until
    someone asks it for another unit -- or calls ``value_in_unit`` and gets an
    AttributeError, as every caller in ``openmmnqe.plumed`` does.
    """
    pdb = app.PDBFile('tests/data/pdb/gt_wob_solv_clean.pdb')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    indexes = nqe.atom_indices_from_vmd_picks(modeller, ['DTN1:H3', 'DGN1:O6'])

    distance = nqe.distance_between_atoms(modeller, indexes[0], indexes[1])

    assert unit.is_quantity(distance)
    assert distance.unit.is_compatible(unit.nanometer)
    # The same length, asked for two ways.
    assert distance.value_in_unit(unit.angstrom) == pytest.approx(
        distance.value_in_unit(unit.nanometer) * 10.0)


def test_angle_between_atoms():
    print(flush=True)
    input_pdb = 'tests/data/pdb/gt_wob_solv_clean.pdb'
    pdb = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    indexes = nqe.atom_indices_from_vmd_picks(modeller, ['DTN1:N3', 'DTN1:H3', 'DGN1:O6'])
    angle = nqe.angle_between_atoms(modeller, indexes[0], indexes[1], indexes[2], degrees=True)
    print(f"Angle between atoms: {angle:.4f} degrees", flush=True)
    ref_angle = 149.89  # in degrees
    assert abs(angle - ref_angle) < 0.01


def ase_distance_between_atoms(atoms, index1, index2):
    """
    Calculate the distances between two atoms across all frames in a trajectory.

    Parameters
    ----------
    atoms : list of ase.Atoms
        Trajectory frames.
    index1 : int
        Index of the first atom.
    index2 : int
        Index of the second atom.

    Returns
    -------
    list of float
        Distance between the two specified atoms for each frame.
    """
    distances = []
    for frame in atoms:
        pos1 = frame[index1].position
        pos2 = frame[index2].position
        distance = np.linalg.norm(pos1 - pos2)
        distances.append(distance)
    return distances


def ase_angle_between_atoms(atoms, index1, index2, index3):
    """
    Calculate the angles formed by three atoms across all frames in a trajectory.

    Parameters
    ----------
    atoms : list of ase.Atoms
        Trajectory frames.
    index1 : int
        Index of the first atom.
    index2 : int
        Index of the second atom (vertex of the angle).
    index3 : int
        Index of the third atom.

    Returns
    -------
    list of float
        Angle in degrees formed by the three specified atoms for each frame.
    """
    angles = []
    for frame in atoms:
        pos1 = frame[index1].position
        pos2 = frame[index2].position
        pos3 = frame[index3].position
        v1 = pos1 - pos2
        v2 = pos3 - pos2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        angles.append(angle)
    return angles


def test_pca_distances():
    from sklearn.decomposition import PCA
    images = read('tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj', index=':')
    n_atoms = len(images[0])
    n_frames = len(images)

    # Upper-triangle indices, so dist(1,2) and dist(2,1) aren't counted twice.
    triu_i, triu_j = np.triu_indices(n_atoms, k=1)

    def get_all_distances(atoms):
        pos = atoms.get_positions()
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        return dist_matrix[triu_i, triu_j]

    print(f"Processing {n_frames} frames with {len(triu_i)} distances each...")
    feature_matrix = np.array([get_all_distances(img) for img in images])

    pca = PCA(n_components=1)
    pca.fit(feature_matrix)
    pc1_weights = pca.components_[0]

    results = []
    for idx, weight in enumerate(pc1_weights):
        results.append({
            'atom_i': triu_i[idx],
            'atom_j': triu_j[idx],
            'symbol_i': images[0].get_chemical_symbols()[triu_i[idx]],
            'symbol_j': images[0].get_chemical_symbols()[triu_j[idx]],
            'weight': weight,
            'abs_weight': abs(weight)
        })

    df = pd.DataFrame(results).sort_values(by='abs_weight', ascending=False)

    print(f"\nPC1 Explained Variance: {pca.explained_variance_ratio_[0] * 100:.2f}%")
    print("\n--- Top 10 Contributing Distances (The ones you should put in PLUMED) ---")
    print(df[['atom_i', 'symbol_i', 'atom_j', 'symbol_j', 'weight']].head(10))


def test_pca_angles():
    from sklearn.decomposition import PCA
    images = read('tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj', index=':')
    n_atoms = len(images[0])
    n_frames = len(images)
    symbols = images[0].get_chemical_symbols()

    # Every 3-atom combination gives 3 distinct angles depending on which atom
    # is the vertex, so each combo contributes all three (i-j-k, j-i-k, i-k-j).
    triplets = []
    for combo in combinations(range(n_atoms), 3):
        i, j, k = combo
        triplets.append((i, j, k))  # Angle i-j-k
        triplets.append((j, i, k))  # Angle j-i-k
        triplets.append((i, k, j))  # Angle i-k-j

    def get_all_angles(atoms, triplet_list):
        pos = atoms.get_positions()
        angles = []
        for i, j, k in triplet_list:
            v_ji = pos[i] - pos[j]
            v_jk = pos[k] - pos[j]

            norm_product = np.linalg.norm(v_ji) * np.linalg.norm(v_jk)
            if norm_product == 0:
                angles.append(0.0)
                continue

            cosine_angle = np.dot(v_ji, v_jk) / norm_product
            # Clip to guard against floating-point drift outside [-1, 1].
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angles.append(angle)
        return np.array(angles)

    print(f"Calculating {len(triplets)} angles for {n_frames} frames...")
    feature_matrix = np.array([get_all_angles(img, triplets) for img in images])

    pca = PCA(n_components=1)
    pca.fit(feature_matrix)
    pc1_weights = pca.components_[0]

    results = []
    for idx, weight in enumerate(pc1_weights):
        i, j, k = triplets[idx]
        results.append({
            'triplet': f"{symbols[i]}{i}-{symbols[j]}{j}-{symbols[k]}{k}",
            'indices': (i, j, k),
            'weight': weight,
            'abs_weight': abs(weight)
        })

    df = pd.DataFrame(results).sort_values(by='abs_weight', ascending=False)

    print(f"\nPC1 (Angles) Explained Variance: {pca.explained_variance_ratio_[0] * 100:.2f}%")
    print("\n--- Top 10 Contributing Angles ---")
    print(df[['triplet', 'weight']].head(10))


def check_linear_relationship(y, x=None, f_print=True):
    """
    Check the linear relationship between two variables using the coefficient of determination (R^2).

    Parameters
    ----------
    y : array_like
        The dependent variable (response data).
    x : array_like, optional
        The independent variable (predictor data). If None, defaults to a
        range of integers from 0 to the length of `y`.
    f_print : bool, optional
        If True, prints the calculated R^2 value. Default is True.

    Returns
    -------
    float
        The coefficient of determination (R^2), which indicates the strength
        of the linear relationship between `x` and `y`. A value closer to 1
        indicates a strong linear relationship.
    """
    if x is None:
        x = np.arange(len(y))
    _, _, r_value, _, _ = linregress(x, y)
    r2 = r_value ** 2
    if f_print:
        print(f"R2: {r2:.4f}")
    return r2


def test_ase_rc1():
    atoms = read('tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj', index=':')

    # How linear is the proton-transfer distance difference along the path?
    a1_d = 21
    a2_d = 30
    a3_d = 18
    d1 = np.array(ase_distance_between_atoms(atoms, a1_d, a2_d))
    d2 = np.array(ase_distance_between_atoms(atoms, a3_d, a2_d))
    r1 = d1 - d2
    check_linear_relationship(r1)

    # ... and the angle that opens up as it transfers?
    a1_a = 8
    a2_a = 5
    a3_a = 21
    angles = ase_angle_between_atoms(atoms, a1_a, a2_a, a3_a)
    check_linear_relationship(angles)

    rr = check_linear_relationship(r1, x=angles)
    plt.plot(angles, r1)
    plt.xlabel(f'Angle {a1_a}-{a2_a}-{a3_a}')
    plt.ylabel(f'Distance {a1_d}-{a2_d} and {a3_d}-{a2_d}')
    plt.title(f'Angle vs Distance Difference, R2={rr:.4f}')
    plt.show()


def test_ase_rc2():
    atoms = read('tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj', index=':')

    # The N1-O2 distance on its own, as a candidate coordinate
    a1_d = 6
    a2_d = 18
    d1 = np.array(ase_distance_between_atoms(atoms, a1_d, a2_d))
    rr = check_linear_relationship(d1)
    plt.plot(d1)
    plt.xlabel('Frame')
    plt.ylabel('Distance (Angstrom)')
    plt.title(f'r1, Distance Difference between {a1_d}-{a2_d}, R2={rr:.4f}')
    plt.show()


def test_ase_load():
    print(flush=True)
    name = 'tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj'
    atoms = read(name, index=':')
    print(atoms)
    assert len(atoms) > 0


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
    input_file = 'tests/data/G_T_wob.traj'
    output_file = 'output.pdb'
    nqe.convert_xyz_to_pdb(input_file, output_file, cutoff_multiplier=1.1)
    # Assert that the pdb file was created
    assert os.path.isfile(output_file)
    os.remove(output_file)


def test_convert_pdb_to_xyz():
    print(flush=True)
    input_file = 'tests/data/pdb/malonaldehyde.pdb'
    output_file = 'output.xyz'
    n_frames = nqe.convert_pdb_to_xyz(input_file, output_file)
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
    n_clusters = nqe.convert_xyz_to_pdb(input_file, pdb_file, cutoff_multiplier=1.1)
    assert n_clusters == 2
    nqe.convert_pdb_to_xyz(pdb_file, xyz_file)

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
    nqe.convert_sdfs_to_pdb("tests/data/DGN.sdf", "single_ligand.pdb")
    assert os.path.isfile('single_ligand.pdb')
    os.remove('single_ligand.pdb')

    nqe.convert_sdfs_to_pdb(["tests/data/DGN.sdf", "tests/data/DTN.sdf"], "combined_ligands.pdb")
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


def test_temperature_to_kbt():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    kbt = nqe.temperature_to_kbt(temperature)
    print(kbt, flush=True)
    assert np.allclose(kbt, 2.494338785445972, atol=1e-6)


def to_fraction(d):
    return (d - d[0]) / d[0] + 1


def test_check_neb_distances():
    print(flush=True)
    name = 'tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj'
    atoms = read(name, index=':')

    o6 = 5
    o4 = 21
    n3 = 19
    n1 = 6
    o2 = 18
    n2 = 8

    d_o6_o4 = np.array(ase_distance_between_atoms(atoms, o6, o4))
    d_o6_n3 = np.array(ase_distance_between_atoms(atoms, o6, n3))

    d_n1_n3 = np.array(ase_distance_between_atoms(atoms, n1, n3))
    d_n1_o2 = np.array(ase_distance_between_atoms(atoms, n1, o2))
    d_n2_o2 = np.array(ase_distance_between_atoms(atoms, n2, o2))

    o6_o4 = to_fraction(d_o6_o4)
    o6_n3 = to_fraction(d_o6_n3)
    n1_n3 = to_fraction(d_n1_n3)
    n1_o2 = to_fraction(d_n1_o2)
    n2_o2 = to_fraction(d_n2_o2)
    labels = ['o6_o4', 'o6_n3', 'n1_n3', 'n1_o2', 'n2_o2']
    dist = [d_o6_o4, d_o6_n3, d_n1_n3, d_n1_o2, d_n2_o2]
    items = [o6_o4, o6_n3, n1_n3, n1_o2, n2_o2]
    for i, item in enumerate(items):
        print(f"{labels[i]}: Start: {dist[i][0]:.2f} Min: {np.min(item):.2f}, Max: {np.max(item):.2f}", flush=True)
        tmp = labels[i].split('_')
        line = f'{labels[i]}: DISTANCE ATOMS=' + '{' + tmp[0] + '},' + '{' + tmp[1] + '}'
        print(line, flush=True)

        line = f'UPPER_WALLS ARG={labels[i]} AT={0.1 * dist[i][0] * np.max(item):.2f} ' + ' KAPPA={kappa}'
        print(line, flush=True)
        line = f'LOWER_WALLS ARG={labels[i]} AT={0.1 * dist[i][0] * np.min(item):.2f} ' + ' KAPPA={kappa}'
        print(line, flush=True)
        print(flush=True)
