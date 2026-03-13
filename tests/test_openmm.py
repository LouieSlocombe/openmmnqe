import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import openmm.app as app
import openmm.unit as unit
import pandas as pd
from ase.io import read
from ase.visualize import view
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

    nqe.run_openmm_relaxation(modeller,
                              forcefield)
    nqe.remove_file('minimized.pdb')


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
    nqe.remove_file('minimized.pdb')


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

    # nqe.remove_file(cache_name)


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
    nqe.remove_file('minimized.pdb')


def test_prepare_ligand_ff_multiple():
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_solv_clean.pdb"
    cache_name = "gaff-molecules.json"
    rm_ions = ['Na+',
               'Cl-',
               'NA']

    pdb_data, molecule = nqe.prepare_lig_system(input_pdb, rm_ions=rm_ions, rm_files=False, rm_lig_sdf=False)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # write the pdb_data to a file for inspection
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open('prepared_system.pdb', 'w'))

    # from openff.toolkit import Molecule
    # from openmmforcefields.generators import GAFFTemplateGenerator
    # from openmm.app import ForceField
    # from openmm.app import PDBFile
    # smis = ['[H][O][C]([H])([H])[C@@]1([H])[O][C@@]([H])([N]2[C]([H])=[N][C]3=[C]2[N]=[C]([N]([H])[H])[N]([H])[C]3=[O])[C]([H])([H])[C@]1([H])[O][H]',
    #         '[H][O][C]([H])([H])[C@@]1([H])[O][C@@]([H])([N]2[C](=[O])[N]([H])[C](=[O])[C]([C]([H])([H])[H])=[C]2[H])[C]([H])([H])[C@]1([H])[O][H]']
    # molecule = [Molecule.from_smiles(smi) for smi in smis]
    # molecule = [Molecule.from_file(f) for f in ["DGN.sdf", "DTN.sdf"]]
    # gaff = GAFFTemplateGenerator(molecules=molecule)
    # forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    # forcefield.registerTemplateGenerator(gaff.generator)

    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule,
                                       gen_cache=True,
                                       use_cache=False,
                                       cache_name=cache_name)
    forcefield.createSystem(modeller.topology)
    # Check that the cache files were created
    # assert os.path.exists(cache_name)
    #
    # forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
    #                                    molecule,
    #                                    gen_cache=False,
    #                                    use_cache=False,
    #                                    cache_name=cache_name)
    # forcefield.createSystem(modeller.topology)
    #
    # nqe.remove_file(cache_name)

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield)


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

    h_count = 0
    for atom in modeller.topology.atoms():
        if atom.element.symbol == 'H':
            h_count += 1

    print(f"--- Applying Deuteration to {h_count} Hydrogens (Option='water') ---")
    nqe.deuterate_system(modeller, system, option='all')

    mass_after = _get_total_mass(system)
    print(f"{'Mass Before':<20} | {mass_before.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Mass After':<20} | {mass_after.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Number of Hydrogens':<20} | {h_count}")

    mass_H = app.element.hydrogen.mass
    mass_D = app.element.deuterium.mass
    mass_delta_per_atom = mass_D - mass_H
    expected_increase = mass_delta_per_atom * h_count

    print(f"\n{'METRIC':<20} | {'VALUE':<15}")
    print("-" * 40)
    print(f"{'Mass Before':<20} | {mass_before.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Mass After':<20} | {mass_after.value_in_unit(unit.dalton):.4f} Da")
    print("-" * 40)
    print(f"{'Actual Increase':<20} | {(mass_after - mass_before).value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Expected Increase':<20} | {expected_increase.value_in_unit(unit.dalton):.4f} Da")

    # Assertion Check
    tolerance = 1e-3
    diff = abs((mass_after - mass_before) - expected_increase).value_in_unit(unit.dalton)

    if diff < tolerance:
        print("\n[SUCCESS] The system mass increased exactly as expected.")
    else:
        print("\n[FAILURE] The mass change did not match theoretical expectations.")


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
    print(f"Distance between atoms: {distance:.4f} nm", flush=True)
    ref_distance = 0.179  # in nm
    assert abs(distance - ref_distance) < 0.01


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

    Parameters:
        atoms (list): A list of ASE `Atoms` objects representing the trajectory frames.
        index1 (int): The index of the first atom.
        index2 (int): The index of the second atom.

    Returns:
        list: A list of distances (float) between the two specified atoms for each frame.
    """
    distances = []
    for frame in atoms:
        pos1 = frame[index1].position  # Position of the first atom in the current frame
        pos2 = frame[index2].position  # Position of the second atom in the current frame
        distance = np.linalg.norm(pos1 - pos2)  # Calculate the Euclidean distance
        distances.append(distance)  # Append the distance to the result list
    return distances


def ase_angle_between_atoms(atoms, index1, index2, index3):
    """
    Calculate the angles formed by three atoms across all frames in a trajectory.

    Parameters:
        atoms (list): A list of ASE `Atoms` objects representing the trajectory frames.
        index1 (int): The index of the first atom.
        index2 (int): The index of the second atom (vertex of the angle).
        index3 (int): The index of the third atom.

    Returns:
        list: A list of angles (float, in degrees) formed by the three specified atoms for each frame.
    """
    angles = []
    for frame in atoms:
        pos1 = frame[index1].position  # Position of the first atom
        pos2 = frame[index2].position  # Position of the second atom (vertex)
        pos3 = frame[index3].position  # Position of the third atom
        v1 = pos1 - pos2  # Vector from the vertex to the first atom
        v2 = pos3 - pos2  # Vector from the vertex to the third atom
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  # Cosine of the angle
        angle = np.arccos(cos_angle) * 180.0 / np.pi  # Convert the angle to degrees
        angles.append(angle)  # Append the calculated angle to the result list
    return angles


def test_pca_distances():
    from sklearn.decomposition import PCA
    # 1. Load the NEB path
    images = read('tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj', index=':')
    n_atoms = len(images[0])
    n_frames = len(images)

    # 2. Get indices for all unique pairs (the upper triangle of the distance matrix)
    # This ensures we don't count dist(1,2) and dist(2,1) separately
    triu_i, triu_j = np.triu_indices(n_atoms, k=1)

    def get_all_distances(atoms):
        pos = atoms.get_positions()
        # Efficiently calculate all pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        return dist_matrix[triu_i, triu_j]

    # 3. Build the Feature Matrix
    print(f"Processing {n_frames} frames with {len(triu_i)} distances each...")
    feature_matrix = np.array([get_all_distances(img) for img in images])

    # 4. Run PCA
    pca = PCA(n_components=1)
    pca.fit(feature_matrix)

    # 5. Extract the "Recipe" (The Weights/Loadings)
    pc1_weights = pca.components_[0]

    # 6. Map weights back to atom pairs for easy reading
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
    # 1. Load the NEB path
    images = read('tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj', index=':')
    n_atoms = len(images[0])
    n_frames = len(images)
    symbols = images[0].get_chemical_symbols()

    # 2. Generate all unique triplets (i, j, k) where j is the vertex
    # We use combinations for indices, but any atom can be the center (vertex)
    # To get all unique angles, we pick 3 atoms and realize one is the vertex.
    # Actually, to be thorough, for every 3 atoms, there are 3 possible angles.
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
            # Vectors: ba = a - b, bc = c - b (j is the vertex)
            v_ji = pos[i] - pos[j]
            v_jk = pos[k] - pos[j]

            # Cosine rule
            norm_product = np.linalg.norm(v_ji) * np.linalg.norm(v_jk)
            if norm_product == 0:
                angles.append(0.0)
                continue

            cosine_angle = np.dot(v_ji, v_jk) / norm_product
            # Clip to avoid floating point errors outside [-1, 1]
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angles.append(angle)
        return np.array(angles)

    # 3. Build the Feature Matrix
    print(f"Calculating {len(triplets)} angles for {n_frames} frames...")
    feature_matrix = np.array([get_all_angles(img, triplets) for img in images])

    # 4. Run PCA
    pca = PCA(n_components=1)
    pca.fit(feature_matrix)
    pc1_weights = pca.components_[0]

    # 5. Extract and Map Results
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
    Check the linear relationship between two variables using the coefficient of determination (R²).

    Parameters:
        y (array-like): The dependent variable (response data).
        x (array-like, optional): The independent variable (predictor data). If None, defaults to a range of integers
                                  from 0 to the length of `y`.
        f_print (bool, optional): If True, prints the calculated R² value. Defaults to True.

    Returns:
        float: The coefficient of determination (R²), which indicates the strength of the linear relationship
               between `x` and `y`. A value closer to 1 indicates a strong linear relationship.
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
    # view(atoms)

    # plot the distance between two atoms over the trajectory

    # atom1_index = 21
    # atom2_index = 30
    #
    # distances = np.array(ase_distance_between_atoms(atoms, atom1_index, atom2_index))
    #
    # # distances -= distances[0]  # Normalize to the first frame
    # # # turn it into percent change
    # # distances = (distances / distances[0]) * 100
    #
    # # plt.plot(distances)
    # # plt.xlabel('Frame')
    # # plt.ylabel('Distance (Angstrom)')
    # # plt.title(f'Distance between Atom {atom1_index} and Atom {atom2_index}')
    # # plt.show()
    #
    a1_d = 21
    a2_d = 30
    a3_d = 18
    d1 = np.array(ase_distance_between_atoms(atoms, a1_d, a2_d))
    d2 = np.array(ase_distance_between_atoms(atoms, a3_d, a2_d))
    r1 = d1 - d2
    rr = check_linear_relationship(r1)
    # plt.plot(r1)
    # plt.xlabel('Frame')
    # plt.ylabel('Distance (Angstrom)')
    # plt.title(f'r1, Distance Difference between {a1_d}-{a2_d} and {a3_d}-{a2_d}, R2={rr:.4f}')
    # plt.show()

    a1_a = 8
    a2_a = 5
    a3_a = 21

    angles = ase_angle_between_atoms(atoms, a1_a, a2_a, a3_a)
    rr = check_linear_relationship(angles)

    # plt.plot(angles)
    # plt.xlabel('Frame')
    # plt.ylabel('Angle (degrees)')
    # plt.title(f'Angle between Atoms {a1_a}-{a2_a}-{a3_a}, R2={rr:.4f}')
    # plt.show()
    #

    rr = check_linear_relationship(r1, x=angles)
    plt.plot(angles, r1)
    plt.xlabel(f'Angle {a1_a}-{a2_a}-{a3_a}')
    plt.ylabel(f'Distance {a1_d}-{a2_d} and {a3_d}-{a2_d}')
    plt.title(f'Angle vs Distance Difference, R2={rr:.4f}')
    plt.show()


def test_ase_rc2():
    atoms = read('tests/data/G_T_wob-G_T_enol_ML-NEB_B3LYP_GOLD_IMPSOL_NWC.traj', index=':')
    view(atoms)

    # a1_d = 21
    # a2_d = 30
    # a3_d = 18
    # dd1 = np.array(ase_distance_between_atoms(atoms, a1_d, a2_d))
    # dd2 = np.array(ase_distance_between_atoms(atoms, a3_d, a2_d))
    # # r1 = d1 + d2
    # # rr = check_linear_relationship(r1)
    # # plt.plot(r1)
    # # plt.xlabel('Frame')
    # # plt.ylabel('Distance (Angstrom)')
    # # plt.title(f'r1, Distance Difference between {a1_d}-{a2_d} and {a3_d}-{a2_d}, R2={rr:.4f}')
    # # plt.show()
    #
    # d1 = np.array(ase_distance_between_atoms(atoms, 5, 21))
    # d2 = np.array(ase_distance_between_atoms(atoms, 5, 19))
    # d3 = np.array(ase_distance_between_atoms(atoms, 6, 19))
    # d4 = np.array(ase_distance_between_atoms(atoms, 6, 18))
    # d5 = np.array(ase_distance_between_atoms(atoms, 8, 18))
    # r1 = (d1 - d2) + d3 - d4 + d5  # + dd1 - dd2
    #
    # n3 = 19
    # h3 = 30
    # o6 = 5
    # o4 = 21
    # n1 = 6
    # h1 = 13
    # o2 = 18
    # n2 = 8
    # n3_h3 = np.array(ase_distance_between_atoms(atoms, n3, h3))
    # o6_h3 = np.array(ase_distance_between_atoms(atoms, o6, h3))
    # o4_h3 = np.array(ase_distance_between_atoms(atoms, o4, h3))
    # n1_h1 = np.array(ase_distance_between_atoms(atoms, n1, h1))
    # n3_h1 = np.array(ase_distance_between_atoms(atoms, n3, h1))
    # n1_o2 = np.array(ase_distance_between_atoms(atoms, n1, o2))
    # n2_o2 = np.array(ase_distance_between_atoms(atoms, n2, o2))
    # n1_n3 = np.array(ase_distance_between_atoms(atoms, n1, n3))
    #
    # z1 = n3_h3 - o6_h3
    # z2 = o6_h3 - o4_h3
    # z3 = 0  # n1_h1 - n3_h1
    # z4 = n1_o2 - n2_o2
    # z5 = n2_o2 - n1_n3
    #
    # r1 = z1 + z2 + z3 + z4 + z5
    # rr = check_linear_relationship(r1)
    # plt.plot(r1)
    # plt.xlabel('Frame')
    # plt.ylabel('Distance (Angstrom)')
    # plt.title(f'R2={rr:.4f}')
    # plt.show()

    a1_d = 6
    a2_d = 18
    d1 = np.array(ase_distance_between_atoms(atoms, a1_d, a2_d))
    rr = check_linear_relationship(d1)
    plt.plot(d1)
    plt.xlabel('Frame')
    plt.ylabel('Distance (Angstrom)')
    plt.title(f'r1, Distance Difference between {a1_d}-{a2_d}, R2={rr:.4f}')
    plt.show()

    # a1_a = 0
    # a2_a = 5
    # a3_a = 16
    #
    # angles = ase_angle_between_atoms(atoms, a1_a, a2_a, a3_a)
    # rr = check_linear_relationship(angles)
    #
    # plt.plot(angles)
    # plt.xlabel('Frame')
    # plt.ylabel('Angle (degrees)')
    # plt.title(f'Angle between Atoms {a1_a}-{a2_a}-{a3_a}, R2={rr:.4f}')
    # plt.show()

    # rr = check_linear_relationship(r1, x=angles)
    # plt.plot(angles, r1)
    # plt.xlabel(f'Angle {a1_a}-{a2_a}-{a3_a}')
    # plt.ylabel(f'Distance {a1_d}-{a2_d} and {a3_d}-{a2_d}')
    # plt.title(f'Angle vs Distance Difference, R2={rr:.4f}')
    # plt.show()


def test_ase_load():
    print(flush=True)
    # name = 'tests/data/G_T_wob.traj'
    # atoms = read(name, index=':')
    # write(name.replace('.traj', '.xyz'), atoms[0])
    # print(atoms)
    # view(atoms)
    input_file = 'tests/data/G_enol_T.traj'
    output_file = 'tests/data/pdb/G_enol_T.pdb'
    nqe.convert_xyz_to_pdb(input_file, output_file, cutoff_multiplier=1.2)

    input_file = 'tests/data/G_T_enol.traj'
    output_file = 'tests/data/pdb/G_T_enol.pdb'
    nqe.convert_xyz_to_pdb(input_file, output_file, cutoff_multiplier=1.2)

    input_file = 'tests/data/G_T_wob.traj'
    output_file = 'tests/data/pdb/G_T_wob.pdb'
    nqe.convert_xyz_to_pdb(input_file, output_file, cutoff_multiplier=1.2)


def test_fix_pdb_chains():
    print(flush=True)
    input_pdb = 'tests/data/pdb/malformed.pdb'
    nqe.fix_pdb_chains(input_pdb, "fixed.pdb")
    # Assert that the pdb file was created
    assert os.path.isfile('fixed.pdb')
    # Assert that there are two chains in the fixed pdb
    with open('fixed.pdb', 'r') as f:
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
    with open('fixed_atoms.pdb', 'r') as f:
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
    atom1 = topology.addAtom("A1", app.Element.getByAtomicNumber(6), residue)
    atom2 = topology.addAtom("A2", app.Element.getByAtomicNumber(6), residue)
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
