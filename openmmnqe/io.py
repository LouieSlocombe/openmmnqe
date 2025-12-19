import glob
import os
import re
import shutil
from typing import List

import numpy as np
import openmm.unit as unit
from openmm.app import PDBFile, Modeller
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.constants import physical_constants as const

# Conversion factor from Bohr to Angstrom
bohr_to_angstrom = const["Bohr radius"][0] * 1e10
# Conversion factor between Angstrom and nm
A_to_nm = 1.0e-1  # Å to nm
# Conversion factor from eV to J
eV_to_J = const["electron volt-joule relationship"][0]
# Conversion factor from J to kJ
J_to_kJ = 1.0e-3  # 1 J = 0.001 kJ
# Avogadro's number
avo_num = const["Avogadro constant"][0]  # mol^-1

# Conversion factor from eV to 1 kJ/mol
eV_to_kJpermol = eV_to_J * J_to_kJ * avo_num
# Conversion factor from eV/Å² to 1 kJ/(mol·nm²)
eVperA2_to_kJpermolpernm2 = eV_to_kJpermol / A_to_nm ** 2


def remove_directory(directory):
    """
    Removes a directory if it exists, otherwise prints a message.

    Parameters:
    directory (str): The path to the directory to be removed.

    Returns:
    None
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    else:
        pass
    return None


def copy_and_rename_file(src, dst_dir, new_name):
    """
    Copies a file to a new directory and renames it.

    Parameters:
    src (str): The path to the source file.
    dst_dir (str): The path to the destination directory.
    new_name (str): The new name for the copied file.

    Returns:
    None
    """
    shutil.copy(src, os.path.join(dst_dir, new_name))
    return None


def list_files_with_pattern(directory, pattern):
    """
    Lists files in a directory that match a given pattern.

    Parameters:
    directory (str): The path to the directory to search in.
    pattern (str): The pattern to match files against.

    Returns:
    list: A list of file paths that match the given pattern.
    """
    return glob.glob(os.path.join(directory, pattern))


def search_fes_files(target_directory: str) -> list[str]:
    """
    Searches for files in the target directory with names matching the pattern 'FES*', where '*' is an integer.

    Parameters:
    target_directory (str): The directory to search in.

    Returns:
    list[str]: A list of matching file names.
    """
    pattern = re.compile(r'^FES\d+\.dat$')
    matching_files = []

    for filename in os.listdir(target_directory):
        if pattern.match(filename):
            matching_files.append(filename)

    return matching_files


def load_fes_data(directory: str, bins: int) -> list[np.ndarray]:
    """
    Find FES files in the directory and load their data into numpy arrays.

    Parameters:
    directory (str): Directory containing the FES files.
    bins (int): Number of bins for reshaping the data.

    Returns:
    list[np.ndarray]: List of numpy arrays, each containing the transformed FES data.
    """
    fes_files = search_fes_files(directory)
    fes_arrays = []
    bins += 1

    for file in sorted(fes_files):
        file_path = os.path.join(directory, file)

        # Read first line to detect number of CVs
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            n_cv = len(first_line.split()[2:])  # Skip #! FIELDS and time
        print(f"Loading {file_path} with {n_cv} FIELDS")
        data = np.loadtxt(file_path, comments="#")
        if n_cv == 2:
            transformed_data = np.array([1.0, 1.0 / eV_to_kJpermol])[:, np.newaxis] * data[:, :2].T
        else:
            transformed_data = np.array([1.0, 1.0 / eV_to_kJpermol])[:, np.newaxis] * data[:, :2].T
        fes_arrays.append(transformed_data)

    return fes_arrays


def xyz_to_sdf(xyz_path, sdf_path, default_charge=0, sanitize=True, kekulize=False):
    """
    Converts an XYZ file to an SDF file, optionally inferring bonds, sanitizing, and kekulizing the molecules.

    This function reads molecular structures from an XYZ file, processes them into RDKit molecule objects,
    and writes them to an SDF file. It supports optional charge parsing, bond inference, molecule sanitization,
    and kekulization.

    Parameters
    ----------
    xyz_path : str
        Path to the input XYZ file.
    sdf_path : str
        Path to the output SDF file.
    default_charge : int, optional
        Default charge to use for bond inference if no charge is specified in the XYZ file. Default is 0.
    sanitize : bool, optional
        If True, sanitizes the RDKit molecule objects before writing. Default is True.
    kekulize : bool, optional
        If True, kekulizes the RDKit molecule objects before writing. Default is False.

    Returns
    -------
    int
        The number of molecules successfully written to the SDF file.
    """

    def _parse_charge_from_comment(comment, fallback):
        """
        Extracts an integer charge from the comment line of an XYZ frame.

        Parameters
        ----------
        comment : str
            The comment line from the XYZ file.
        fallback : int
            The fallback charge to use if no charge is found in the comment.

        Returns
        -------
        int
            The extracted charge or the fallback value.
        """
        if comment is None:
            return fallback
        m = re.search(r'charge\s*[:=]?\s*([+-]?\d+)', comment, flags=re.I)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        m = re.search(r'(?:q\s*[:=])\s*([+-]?\d+)', comment, flags=re.I)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        m = re.search(r'(^|\s)([+-]\d+)(\s|$)', comment)
        if m:
            try:
                return int(m.group(2))
            except ValueError:
                pass
        return fallback

    def _read_xyz_frames(path):
        """
        Reads molecular frames from an XYZ file.

        Parameters
        ----------
        path : str
            Path to the XYZ file.

        Returns
        -------
        list of tuple
            A list of tuples, each containing the comment line and atom block for a frame.
        """
        frames = []
        with open(path, 'r', encoding='utf-8') as fh:
            lines = [ln.rstrip('\n') for ln in fh]
        i = 0
        n_total = len(lines)
        while i < n_total:
            while i < n_total and not lines[i].strip():
                i += 1
            if i >= n_total:
                break
            try:
                n = int(lines[i].strip())
            except ValueError:
                raise ValueError(f"Expected atom count at line {i + 1}, got: {lines[i]!r}")
            i += 1
            if i >= n_total:
                raise ValueError("Unexpected EOF after atom count.")
            comment = lines[i]
            i += 1
            if i + n > n_total:
                raise ValueError("Unexpected EOF in atom coordinate block.")
            block = lines[i:i + n]
            i += n
            frames.append((comment, block))
        if not frames:
            raise ValueError("No XYZ frames found.")
        return frames

    def _frame_to_mol(comment, coord_lines, name_fallback):
        """
        Converts a single XYZ frame to an RDKit molecule object.

        Parameters
        ----------
        comment : str
            The comment line from the XYZ frame.
        coord_lines : list of str
            The atom coordinate lines from the XYZ frame.
        name_fallback : str
            A fallback name for the molecule if no name is found in the comment.

        Returns
        -------
        rdkit.Chem.Mol
            The RDKit molecule object.
        """
        rw = Chem.RWMol()
        conf = Chem.Conformer(len(coord_lines))
        symbols = []
        for idx, line in enumerate(coord_lines):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad XYZ atom line (needs 'El x y z'): {line!r}")
            sym = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                raise ValueError(f"Bad XYZ coordinates on line: {line!r}")
            a = Chem.Atom(sym)
            atom_idx = rw.AddAtom(a)
            conf.SetAtomPosition(atom_idx, (x, y, z))
            symbols.append(sym)

        mol = rw.GetMol()
        conf.Set3D(True)
        mol.AddConformer(conf, assignId=True)

        title = (comment or "").strip() or name_fallback
        if title:
            mol.SetProp("_Name", title)

        return mol

    frames = _read_xyz_frames(xyz_path)
    base_name = os.path.splitext(os.path.basename(xyz_path))[0]

    writer = Chem.SDWriter(sdf_path)
    if writer is None:
        raise IOError(f"Could not open SDF writer for: {sdf_path}")

    n_written = 0
    for idx, (comment, coord_lines) in enumerate(frames, start=1):
        name_fallback = f"{base_name}_{idx}" if len(frames) > 1 else base_name
        mol = _frame_to_mol(comment, coord_lines, name_fallback)

        total_charge = _parse_charge_from_comment(comment, default_charge)

        rdDetermineBonds.DetermineBonds(mol, charge=total_charge)

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                )

        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass

        writer.write(mol)
        smi = Chem.MolToSmiles(mol, allBondsExplicit=True, allHsExplicit=True)
        print(f"SMI: {smi}")
        n_written += 1

    writer.close()
    return n_written


def extract_nonstandard_res(pdb_file_path: str,
                            output_dir: str = ".",
                            sdf: bool = False) -> list:
    """
    Extracts non-standard residues from a PDB file and saves them as XYZ files.

    This function identifies residues in a PDB file that are not part of a predefined
    set of standard residues. Each non-standard residue is saved as a separate XYZ file
    in the specified output directory. Optionally, the XYZ files can be converted to SDF format.

    Parameters
    ----------
    pdb_file_path : str
        Path to the input PDB file.
    output_dir : str, optional
        Directory where the extracted residue files will be saved. Default is the current directory.
    sdf : bool, optional
        If True, converts the extracted XYZ files to SDF format. Default is False.

    Returns
    -------
    list
        A list of file paths for the generated XYZ or SDF files.
    """
    pdb = PDBFile(pdb_file_path)

    # Extract topology and positions from the PDB file
    topology = pdb.getTopology()
    positions_quantity = pdb.getPositions(asNumpy=True)
    positions_angstrom = positions_quantity.value_in_unit(unit.angstrom)

    # Define a set of standard residues to ignore
    manual_standard_residues = {
        # Standard 20 protein residues
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        # Standard DNA residues (desoxy)
        'DA', 'DC', 'DG', 'DT',
        # Standard RNA residues (ribo)
        'A', 'C', 'G', 'U', 'RA', 'RC', 'RG', 'RU',
        # Common alternative protonation states for Histidine
        'HID', 'HIE', 'HIP',
        # Common synonyms
        'ADE', 'CYT', 'GUA', 'THY', 'URA',
        # Water
        'HOH', 'WAT', 'SOL'
    }

    residues_to_ignore = manual_standard_residues.copy()
    generated_files = []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through residues in the topology
    for residue in topology.residues():
        if residue.name not in residues_to_ignore:

            # Extract residue details
            res_name = residue.name
            res_id = residue.id
            chain_id = residue.chain.id

            # Generate a safe filename for the residue
            safe_res_name = "".join(c for c in res_name if c.isalnum())
            filename = f"{safe_res_name}_{chain_id}_{res_id}.xyz"
            output_path = os.path.join(output_dir, filename)

            # Get atoms in the residue
            atoms_in_residue = list(residue.atoms())
            num_atoms = len(atoms_in_residue)

            # Skip residues with 1 or fewer atoms
            if num_atoms <= 1:
                continue

            # Log the found non-standard residue
            print(f"Found non-standard residue: {res_name} (Chain {chain_id}, ResID {res_id})", flush=True)

            # Prepare XYZ file content
            xyz_content = [str(num_atoms)]
            comment = f"Residue: {res_name}, Chain: {chain_id}, ResID: {res_id}, Source: {os.path.basename(pdb_file_path)}"
            xyz_content.append(comment)

            for atom in atoms_in_residue:
                element = atom.element.symbol
                pos = positions_angstrom[atom.index]
                xyz_line = f"{element:<2}   {pos[0]:>12.6f} {pos[1]:>12.6f} {pos[2]:>12.6f}"
                xyz_content.append(xyz_line)

            # Write the XYZ file
            with open(output_path, 'w') as f:
                f.write("\n".join(xyz_content))
                f.write("\n")

            generated_files.append(output_path)
            print(f"Successfully wrote {num_atoms} atoms to {os.path.splitext(output_path)[0]}", flush=True)

    # Optionally convert XYZ files to SDF format
    if sdf:
        for xyz_file in generated_files:
            sdf_file = os.path.splitext(xyz_file)[0] + ".sdf"
            xyz_to_sdf(xyz_file, sdf_file, sanitize=True, kekulize=False)
            os.remove(xyz_file)
        generated_files = [os.path.splitext(f)[0] + ".sdf" for f in generated_files]

    return generated_files


def get_non_standard_residues(pdb_file):
    """
    Identifies non-standard residues in a PDB file.

    This function reads a PDB file, splits it into residues, and compares each residue
    against a predefined set of standard residues. Residues not in the standard set
    are considered non-standard and are returned as RDKit molecule objects.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.

    Returns
    -------
    list
        A list of RDKit molecule objects representing non-standard residues.
    """
    # Define a set of standard residues, including protein, DNA, RNA, water, and common ions
    standard_residues = {
        # Standard 20 protein residues
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        # Standard DNA residues (desoxy)
        'DA', 'DC', 'DG', 'DT',
        # Standard RNA residues (ribo)
        'A', 'C', 'G', 'U', 'RA', 'RC', 'RG', 'RU',
        # Common alternative protonation states for Histidine
        'HID', 'HIE', 'HIP',
        # Common synonyms
        'ADE', 'CYT', 'GUA', 'THY', 'URA',
        # Water
        'HOH', 'WAT', 'SOL',
        # Ions
        'NA', 'CL', 'K', 'MG', 'CA',
        'Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+'
    }

    # Load the PDB file without sanitization or hydrogen removal
    mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
    # Split the molecule into residues
    mols_by_residue = Chem.SplitMolByPDBResidues(mol)

    print(f"\n--- Found {len(mols_by_residue)} total residue fragments ---")

    non_standard_mols = []
    # Iterate through residues and identify non-standard ones
    for residue_key, fragment_mol in mols_by_residue.items():
        # Extract the residue name from the residue key
        res_name = residue_key.split('_')[0].strip()
        if res_name not in standard_residues:
            # Log non-standard residues
            print(f"  > Found non-standard residue: {residue_key}")
            print(Chem.MolToSmiles(fragment_mol))
            non_standard_mols.append(fragment_mol)
        else:
            # Log standard residues being skipped
            print(f"  - Skipping standard residue: {residue_key}")

    return non_standard_mols


def list_non_standard_residues(pdb_file):
    """
    Identifies and lists non-standard residues in a PDB file.

    This function reads a PDB file, splits it into residues, and compares each residue
    against a predefined set of standard residues. Residues not in the standard set
    are considered non-standard and are returned.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.

    Returns
    -------
    list
        A list of residue keys representing non-standard residues.
    """
    standard_residues = {
        # Standard 20 protein residues
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        # Standard DNA residues (desoxy)
        'DA', 'DC', 'DG', 'DT',
        # Standard RNA residues (ribo)
        'A', 'C', 'G', 'U', 'RA', 'RC', 'RG', 'RU',
        # Common alternative protonation states for Histidine
        'HID', 'HIE', 'HIP',
        # Common synonyms
        'ADE', 'CYT', 'GUA', 'THY', 'URA',
        # Water
        'HOH', 'WAT', 'SOL',
        # Ions
        'NA', 'CL', 'K', 'MG', 'CA',
        'Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+'
    }

    # Load the PDB file without sanitization or hydrogen removal
    mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
    # Split the molecule into residues
    mols_by_residue = Chem.SplitMolByPDBResidues(mol)

    non_standard_mols = []
    # Iterate through residues and identify non-standard ones
    for residue_key, fragment_mol in mols_by_residue.items():
        res_name = residue_key.split('_')[0].strip()
        if res_name not in standard_residues:
            non_standard_mols.append(residue_key)
    return non_standard_mols


def clean_ions_in_pdb(pdb_input_path: str, ions_to_remove: List[str], pdb_output_path: str) -> List[str]:
    """
    Removes specified ion residues from a PDB file and saves the cleaned structure.

    This function identifies ion residues in a PDB file based on their names and removes
    those that match the provided list. The cleaned PDB structure is then saved to the
    specified output file. A list of all ion types found in the input file is returned.

    Parameters
    ----------
    pdb_input_path : str
        Path to the input PDB file.
    ions_to_remove : list of str
        A list of ion residue names to remove from the PDB file.
    pdb_output_path : str
        Path to save the cleaned PDB file.

    Returns
    -------
    list of str
        A sorted list of all ion residue types found in the input PDB file.
    """
    pdb = PDBFile(pdb_input_path)
    modeller = Modeller(pdb.topology, pdb.positions)
    ions_to_remove_upper = {ion.upper() for ion in ions_to_remove}
    all_found_ion_types = set()
    residues_to_delete = []

    # Iterate through residues in the topology
    for res in modeller.topology.residues():
        res_name_upper = res.name.upper()
        # Skip water residues
        if res_name_upper in ['HOH', 'WAT']:
            continue
        # Identify single-atom residues as potential ions
        if len(list(res.atoms())) == 1:
            all_found_ion_types.add(res.name)
            if res_name_upper in ions_to_remove_upper:
                residues_to_delete.append(res)

    # Log the found ion types and residues to remove
    print(f"-> Found all potential ion types: {sorted(list(all_found_ion_types))}")
    print(f"-> Will remove {len(residues_to_delete)} residues matching: {ions_to_remove}")

    # Remove the identified residues
    if residues_to_delete:
        modeller.delete(residues_to_delete)
        print(f"Successfully removed {len(residues_to_delete)} ion residues.")
    else:
        print("No matching ion residues found to remove.")

    # Save the cleaned PDB structure to the output file
    with open(pdb_output_path, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print(f"Cleaned PDB saved to: {pdb_output_path}")

    return sorted(list(all_found_ion_types))


def relabel_residues_in_pdb(pdb_file_path, relabel_map, output_file):
    """
    Relabels residues in a PDB file based on a provided mapping and saves the modified structure.

    This function reads a PDB file, updates the residue names according to the `relabel_map`,
    and writes the modified structure to an output file. A summary of changes is printed.

    Parameters
    ----------
    pdb_file_path : str
        Path to the input PDB file.
    relabel_map : dict
        A dictionary mapping original residue names (keys) to new residue names (values).
    output_file : str or file-like object
        Path to the output PDB file or a file-like object where the modified structure will be saved.

    Returns
    -------
    PDBFile
        The modified PDB structure.
    """
    pdb = PDBFile(pdb_file_path)
    topology = pdb.topology
    positions = pdb.positions
    changed_residues = {}

    # Iterate through residues and relabel if they match the relabel_map
    for residue in topology.residues():
        if residue.name in relabel_map:
            original_name = residue.name
            new_name = relabel_map[original_name]
            residue.name = new_name
            change_key = (original_name, new_name)
            if change_key not in changed_residues:
                changed_residues[change_key] = 0
            changed_residues[change_key] += 1

    # Print a summary of changes
    if changed_residues:
        print("Relabeling complete. Summary of changes:")
        for (old, new), count in changed_residues.items():
            print(f"  - Relabeled {count} residues from '{old}' to '{new}'")
    else:
        print("No residues found matching the relabel map. Topology is unchanged.")

    # Save the modified topology and positions to the output file
    print(f"Saving modified topology and positions...")
    if isinstance(output_file, str):
        with open(output_file, 'w') as f:
            PDBFile.writeFile(topology, positions, f)
        print(f"Successfully saved modified PDB to: {output_file}")
    else:
        PDBFile.writeFile(topology, positions, output_file)
        print(f"Successfully wrote modified PDB to file-like object.")

    return pdb


def remove_residues_in_pdb(input_pdb, output_pdb, names):
    """
    Removes specific residues from a PDB file and writes the modified structure to a new file.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Path to the output PDB file where the modified structure will be saved.
    names : list of str
        A list of residue names to be removed from the PDB file.

    Returns
    -------
    None
    """
    # Load the PDB file
    pdb = PDBFile(input_pdb)
    modeller = Modeller(pdb.topology, pdb.positions)

    # Identify residues to delete based on the provided names
    residues_to_delete = [res for res in modeller.topology.residues()
                          if res.name in names]

    print(f"Found {len(residues_to_delete)} residues to delete.")

    # Delete the identified residues if any are found
    if residues_to_delete:
        modeller.delete(residues_to_delete)
        print("Successfully deleted residues.")
    else:
        print("No matching residues found to delete.")

    # Write the modified structure to the output PDB file
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)


def remove_water_residues_in_pdb(input_pdb, output_pdb, water_names=None):
    """Removes water residues from a PDB file.

    This function identifies and removes residues considered to be water from a PDB file
    and writes the resulting structure to a new PDB file. It is a convenience wrapper
    around `remove_residues_in_pdb`.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Path to the output PDB file for the cleaned structure.
    water_names : set of str, optional
        A set of residue names to be treated as water. If None, defaults to
        `{"HOH", "WAT"}`.

    Returns
    -------
    None
    """
    if water_names is None:
        water_names = {"HOH", "WAT"}
    print(f"Removing water residues.")
    remove_residues_in_pdb(input_pdb, output_pdb, water_names)
