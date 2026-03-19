import glob
import os
import re
import shutil
import string
from collections import defaultdict
from typing import List

import MDAnalysis as mda
import numpy as np
import openmm.unit as unit
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from openff.toolkit import Molecule, Topology
from openmm import app, Vec3
from openmm.app import PDBFile, Topology, Element, Modeller
from openmmforcefields.generators import GAFFTemplateGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.constants import physical_constants as const
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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
    Removes a directory if it exists.

    This function checks if the specified directory exists. If it does, the directory
    and all its contents are removed. If the directory does not exist, the function
    does nothing.

    Parameters
    ----------
    directory : str
        The path to the directory to be removed.

    Returns
    -------
    None
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    else:
        pass
    return None


def copy_and_rename_file(src, dst_dir, new_name):
    """
    Copies a file to a specified directory and renames it.

    This function takes the path to a source file, copies it to a destination directory,
    and renames the copied file to the specified new name.

    Parameters
    ----------
    src : str
        The path to the source file.
    dst_dir : str
        The path to the destination directory.
    new_name : str
        The new name for the copied file.

    Returns
    -------
    None
    """
    shutil.copy(src, os.path.join(dst_dir, new_name))
    return None


def list_files_with_pattern(directory, pattern):
    """
    Lists files in a directory that match a given pattern.

    This function searches the specified directory for files that match the given pattern
    and returns a list of their file paths. The pattern matching is performed using the
    `glob` module.

    Parameters
    ----------
    directory : str
        The path to the directory to search in.
    pattern : str
        The pattern to match files against (e.g., '*.txt' for all text files).

    Returns
    -------
    list
        A list of file paths that match the given pattern.
    """
    return glob.glob(os.path.join(directory, pattern))


def search_fes_files(target_directory: str) -> list[str]:
    """
    Searches for files in the target directory with names matching the pattern 'FES*', where '*' is an integer.

    This function scans the specified directory for files whose names match the pattern 'FES\d+\.dat',
    where '\d+' represents one or more digits. It returns a list of all matching file names.

    Parameters
    ----------
    target_directory : str
        The directory to search in.

    Returns
    -------
    list[str]
        A list of matching file names.
    """
    # Compile a regular expression pattern to match file names like 'FES<number>.dat'
    pattern = re.compile(r'^FES\d+\.dat$')
    matching_files = []

    # Iterate through all files in the target directory
    for filename in os.listdir(target_directory):
        # Check if the file name matches the pattern
        if pattern.match(filename):
            matching_files.append(filename)

    # Return the list of matching file names
    return matching_files


def load_fes_data(directory: str, bins: int) -> list[np.ndarray]:
    """
    Find FES files in the directory and load their data into numpy arrays.

    This function searches for FES files in the specified directory, reads their data,
    and transforms it into numpy arrays. The transformation involves scaling the data
    based on predefined conversion factors. The number of bins is incremented by 1
    before processing.

    Parameters
    ----------
    directory : str
        Directory containing the FES files.
    bins : int
        Number of bins for reshaping the data.

    Returns
    -------
    list[np.ndarray]
        A list of numpy arrays, each containing the transformed FES data.
    """
    # Search for FES files in the specified directory
    fes_files = search_fes_files(directory)
    fes_arrays = []
    bins += 1  # Increment the number of bins

    # Process each FES file
    for file in sorted(fes_files):
        file_path = os.path.join(directory, file)

        # Read the first line to detect the number of collective variables (CVs)
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            n_cv = len(first_line.split()[2:])  # Skip #! FIELDS and time
        print(f"Loading {file_path} with {n_cv} FIELDS")

        # Load the data from the file, ignoring lines starting with '#'
        data = np.loadtxt(file_path, comments="#")

        # Transform the data based on the number of CVs
        if n_cv == 2:
            transformed_data = np.array([1.0, 1.0 / eV_to_kJpermol])[:, np.newaxis] * data[:, :2].T
        else:
            transformed_data = np.array([1.0, 1.0 / eV_to_kJpermol])[:, np.newaxis] * data[:, :2].T

        # Append the transformed data to the result list
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


def fix_pdb(file_in, file_out, ph=7.0, rm_heterogens=True):
    """Fixes a PDB file using PDBFixer.

    This function processes a PDB file to correct common issues like
    missing residues, non-standard residues, missing atoms, and missing
    hydrogens.

    Parameters
    ----------
    file_in : str
        Path to the input PDB file.
    file_out : str
        Path to write the fixed PDB file.
    ph : float, optional
        The pH to use when adding missing hydrogens. Default is 7.0.
    rm_heterogens : bool, optional
        If True, remove heterogen atoms like water, ions, and ligands.
        Default is True.

    Returns
    -------
    None
    """
    fixer = PDBFixer(filename=file_in)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    if rm_heterogens:
        fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    app.PDBFile.writeFile(fixer.topology, fixer.positions, open(file_out, 'w'))
    return None


def make_sdf(pdb_file, lig_name='LIG'):
    """
    Converts a ligand from a PDB file to an SDF file.

    This function reads a PDB file, extracts the ligand specified by its residue name,
    and writes it to an SDF file. The ligand's atomic elements are guessed and added
    to the topology before conversion.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    lig_name : str, optional
        Residue name of the ligand to extract. Default is 'LIG'.

    Returns
    -------
    None
    """
    u = mda.Universe(pdb_file)
    elements = mda.topology.guessers.guess_types(u.atoms.names)
    u.add_TopologyAttr('elements', elements)
    lig = u.select_atoms(f"resname {lig_name}")
    mol = lig.convert_to("RDKIT")
    # write to sdf file
    Chem.MolToMolFile(mol, f"{lig_name}.sdf", kekulize=False)
    return None


def pdb_patcher(pdb_file, lig_name='LIG'):
    """
    Modifies a PDB file to replace placeholder residue names and characters.

    This function reads a PDB file, replaces occurrences of the character 'x' with a space,
    and changes the residue name 'UNK' to the specified ligand name. The modified PDB
    content is then written back to the same file.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file to be modified.
    lig_name : str, optional
        The new residue name to replace 'UNK'. Default is 'LIG'.

    Returns
    -------
    None
    """
    with open(pdb_file, 'r') as f:
        pdb_data = f.read()
    pdb_data = pdb_data.replace('x', ' ')
    pdb_data = pdb_data.replace('UNK', lig_name)
    with open(pdb_file, 'w') as f:
        f.write(pdb_data)
    return None


def combine_sdf_pdb(input_pdb, lig_name='LIG', patch=True):
    """
    Combine a ligand SDF file with a receptor PDB file into a single PDB.

    Reads the ligand from ``<lig_name>.sdf``, converts it to an OpenMM
    topology, appends it to the receptor topology loaded from *input_pdb*,
    and overwrites *input_pdb* with the combined structure. Optionally
    patches residue labels via :func:`pdb_patcher`.

    Parameters
    ----------
    input_pdb : str
        Path to the receptor PDB file. The combined output is written back
        to this file.
    lig_name : str, optional
        Residue name (and SDF filename stem) of the ligand. Default is ``'LIG'``.
    patch : bool, optional
        If True, run :func:`pdb_patcher` on the output to fix residue names.
        Default is True.

    Returns
    -------
    None
    """
    # Combine ligand and receptor into one pdb
    pdb = app.PDBFile(input_pdb)
    molecule = Molecule.from_file(f'{lig_name}.sdf')
    ligand_ff_topology = molecule.to_topology()
    ligand_omm_topology = ligand_ff_topology.to_openmm()
    ligand_positions = ligand_ff_topology.get_positions().to_openmm()
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.add(ligand_omm_topology, ligand_positions)
    with open(input_pdb, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    if patch:
        pdb_patcher(input_pdb, lig_name=lig_name)
    return None


def convert_sdfs_to_pdb(input_files, output_filename="combined_output.pdb"):
    """
    Convert one or more SDF files into a single combined PDB file.

    Each molecule in each SDF file is assigned a unique chain, residue ID,
    and three-letter residue name. Bonds and 3-D coordinates are preserved.

    Parameters
    ----------
    input_files : str or list of str
        Path(s) to the input SDF file(s).
    output_filename : str, optional
        Path for the output PDB file. Default is ``'combined_output.pdb'``.

    Returns
    -------
    None
    """
    if isinstance(input_files, str):
        input_files = [input_files]
    all_mols = []
    for sdf_path in input_files:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        for mol in suppl:
            if mol is not None:
                all_mols.append(mol)
    combined_topology = Topology()
    combined_positions = []
    for i, mol in enumerate(all_mols):
        res_name = input_files[i].split('.')[0]
        residue = combined_topology.addResidue(res_name, combined_topology.addChain())
        rdkit_idx_to_atom = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            element = Element.getBySymbol(symbol)
            omm_atom = combined_topology.addAtom(symbol, element, residue)
            rdkit_idx_to_atom[atom.GetIdx()] = omm_atom

        for bond in mol.GetBonds():
            atom1 = rdkit_idx_to_atom[bond.GetBeginAtomIdx()]
            atom2 = rdkit_idx_to_atom[bond.GetEndAtomIdx()]
            combined_topology.addBond(atom1, atom2)

        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for j in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(j)
                combined_positions.append(Vec3(pos.x, pos.y, pos.z) * 0.1)
        else:
            print(f"Warning: Molecule {res_name} has no 3D coordinates.", flush=True)
            combined_positions.extend([Vec3(0, 0, 0)] * mol.GetNumAtoms())

    with open(output_filename, 'w') as f:
        PDBFile.writeFile(combined_topology, combined_positions * unit.nanometers, f)


def prepare_lig_system(input_pdb,
                       combined_pdb='combined_system.pdb',
                       clean_pdb='cleaned.pdb',
                       rm_ions=None,
                       residue_map=None,
                       rm_files=True,
                       rm_lig_sdf=True,
                       lig_names=None):
    """
    Prepare a protein–ligand system from a raw PDB file.

    Removes water and (optionally) ions, relabels residues, identifies
    non-standard (ligand) residues, generates SDF files, fixes the PDB,
    and combines ligand and receptor topologies into one PDB ready for
    force-field parameterisation.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    combined_pdb : str, optional
        Path for the intermediate combined PDB file. Default is
        ``'combined_system.pdb'``.
    clean_pdb : str, optional
        Path for the intermediate cleaned PDB file. Default is
        ``'cleaned.pdb'``.
    rm_ions : list of str or None, optional
        Ion residue names to remove. If None, no ions are removed.
        Default is None.
    residue_map : dict or None, optional
        Mapping of old residue names to new names for relabelling.
        If None, no relabelling is performed. Default is None.
    rm_files : bool, optional
        If True, remove intermediate files after completion. Default is True.
    rm_lig_sdf : bool, optional
        If True, remove generated ligand SDF files after completion.
        Default is True.
    lig_names : str, list of str, or None, optional
        Ligand residue name(s). If None, non-standard residues are
        auto-detected. Default is None.

    Returns
    -------
    pdb_data : openmm.app.PDBFile
        The final combined PDB data.
    molecules : openff.toolkit.Molecule or list of openff.toolkit.Molecule
        The ligand molecule(s). A single ``Molecule`` is returned when only
        one ligand is present; otherwise a list.
    """
    remove_water_residues_in_pdb(input_pdb, clean_pdb)

    if rm_ions is not None:
        clean_ions_in_pdb(clean_pdb, rm_ions, clean_pdb)
    if residue_map is not None:
        relabel_residues_in_pdb(clean_pdb, residue_map, clean_pdb)

    if lig_names is None:
        non_std_residues = list_non_standard_residues(clean_pdb)
        lig_names_list = list(set([key.split('_')[0].strip() for key in non_std_residues]))
        print(f"Identified ligands: {lig_names_list}", flush=True)
    elif isinstance(lig_names, str):
        lig_names_list = [lig_names]
    else:
        lig_names_list = list(lig_names)

    molecules = []
    generated_sdfs = []

    for lig_name in lig_names_list:
        sdf_filename = f'{lig_name}.sdf'
        make_sdf(clean_pdb, lig_name=lig_name)
        generated_sdfs.append(sdf_filename)
        mol = Molecule.from_file(sdf_filename)
        mol.name = lig_name
        molecules.append(mol)

    pdb_temp = app.PDBFile(clean_pdb)
    residues = list(pdb_temp.topology.residues())
    lig_count = sum(1 for r in residues if r.name in lig_names_list)
    total_count = len(residues)
    print(f"Total residues: {total_count}, Ligand residues: {lig_count}", flush=True)
    is_ligand_only = (total_count > 0 and lig_count == total_count)

    if is_ligand_only:
        print('Only ligand residues found in PDB.', flush=True)
        combined_pdb = clean_pdb
        # convert_sdfs_to_pdb(generated_sdfs, output_filename=combined_pdb)
        # fix_pdb(clean_pdb, combined_pdb, rm_heterogens=False)
        for lig_name in lig_names_list:
            pdb_patcher(combined_pdb, lig_name=lig_name)
    else:
        fix_pdb(clean_pdb, combined_pdb, rm_heterogens=False)
        remove_residues_in_pdb(combined_pdb, combined_pdb, names=lig_names_list)
        for lig_name in lig_names_list:
            print('Patching PDB for ligand:', lig_name, flush=True)
            combine_sdf_pdb(combined_pdb, lig_name=lig_name, patch=True)

    pdb_data = app.PDBFile(combined_pdb)
    if rm_files:
        if os.path.exists(clean_pdb):
            os.remove(clean_pdb)
        if os.path.exists(combined_pdb):
            os.remove(combined_pdb)

    if rm_lig_sdf:
        for sdf_file in generated_sdfs:
            if os.path.exists(sdf_file):
                os.remove(sdf_file)

    for mol, lig_name in zip(molecules, lig_names_list):
        mol.name = lig_name

    if len(molecules) == 1:
        return pdb_data, molecules[0]
    else:
        return pdb_data, molecules


def prepare_ligand_ff(standard_ff,
                      molecule,
                      gen_cache=False,
                      use_cache=False,
                      cache_name="gaff-molecules.json",
                      n_conf=10,
                      pc_method='mmff94',
                      gaff_ver='gaff-2.11'):
    """
    Build an OpenMM ForceField that includes GAFF parameters for ligand(s).

    Generates conformers and partial charges for each ligand molecule (unless
    a cache is used), registers a ``GAFFTemplateGenerator`` with the force
    field, and optionally populates a parameter cache for later reuse.

    Parameters
    ----------
    standard_ff : str or list of str
        Standard force field XML file name(s) (e.g. ``'amber14-all.xml'``).
    molecule : openff.toolkit.Molecule or list of openff.toolkit.Molecule
        The ligand molecule(s) to parameterise.
    gen_cache : bool, optional
        If True, trigger parameterisation to populate the JSON cache.
        Default is False.
    use_cache : bool, optional
        If True, load parameters from an existing cache file instead of
        recomputing. Default is False.
    cache_name : str, optional
        Filename for the GAFF parameter cache. Default is
        ``'gaff-molecules.json'``.
    n_conf : int, optional
        Number of conformers to generate per molecule. Default is 10.
    pc_method : str, optional
        Partial-charge method name (e.g. ``'mmff94'``, ``'am1bcc'``).
        Default is ``'mmff94'``.
    gaff_ver : str, optional
        GAFF force field version string. Default is ``'gaff-2.11'``.

    Returns
    -------
    forcefield : openmm.app.ForceField
        An OpenMM ForceField with a registered GAFF template generator for
        the supplied ligand molecule(s).
    """
    if not isinstance(molecule, list):
        molecules = [molecule]
    else:
        molecules = molecule

    if isinstance(standard_ff, str):
        standard_ff = [standard_ff]

    if not use_cache:
        print(f'Pre-calculating conformers and charges ({pc_method})...', flush=True)
        for mol in molecules:
            print(f'  - Processing molecule: {mol}', flush=True)
            if mol.n_conformers == 0:
                mol.generate_conformers(n_conformers=n_conf)
            if mol.partial_charges is None:
                mol.assign_partial_charges(partial_charge_method=pc_method,
                                           use_conformers=mol.conformers)

    active_cache = cache_name if (use_cache or gen_cache) else None

    print(f'Initializing GAFF generator (Cache: {active_cache})...', flush=True)
    gaff = GAFFTemplateGenerator(molecules=molecules,
                                 cache=active_cache,
                                 forcefield=gaff_ver)

    forcefield = app.ForceField(*standard_ff)
    forcefield.registerTemplateGenerator(gaff.generator)

    if gen_cache:
        print('Triggering parameterization to populate cache...', flush=True)
        for mol in molecules:
            omm_topology = mol.to_topology().to_openmm()
            forcefield.createSystem(omm_topology)

    return forcefield


def save_pdb_selection(input_pdb_path, atom_indices, output_pdb_path):
    """
    Saves a subset of atoms from a PDB file to a new PDB file.

    This function reads a PDB file, selects a subset of atoms based on their indices,
    and writes the selected atoms to a new PDB file. Atoms not in the specified indices
    are removed from the output.

    Parameters
    ----------
    input_pdb_path : str
        Path to the input PDB file.
    atom_indices : list of int
        A list of atom indices to keep in the output PDB file.
    output_pdb_path : str
        Path to save the output PDB file containing the selected atoms.

    Returns
    -------
    None

    Notes
    -----
    - If the selection is empty, a warning is printed, and the output PDB file will be empty.
    - The atom indices should correspond to the indices in the input PDB file.
    """
    pdb = app.PDBFile(input_pdb_path)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    keep_indices = set(atom_indices)
    atoms_to_delete = []
    all_atoms = list(modeller.topology.atoms())

    for atom in all_atoms:
        if atom.index not in keep_indices:
            atoms_to_delete.append(atom)

    num_deleted = len(atoms_to_delete)
    if num_deleted == len(all_atoms):
        print("Warning: Your selection is empty! The output PDB will be empty.")

    modeller.delete(atoms_to_delete)

    print(f"Writing selection ({len(all_atoms) - num_deleted} atoms) to {output_pdb_path}...")
    with open(output_pdb_path, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)


def remove_file_pattern(pattern: str):
    """
    Remove all files matching a specific glob pattern.

    Parameters
    ----------
    pattern : str
        The glob pattern to match files (e.g., "*.txt" for all text files).

    Returns
    -------
    None
        This function does not return a value.
    """
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except OSError:
            pass


def remove_file(file_path: str):
    """
    Remove a file if it exists.

    Parameters
    ----------
    file_path : str
        The path to the file to be removed.

    Returns
    -------
    None
        This function does not return a value.
    """
    try:
        os.remove(file_path)
    except OSError:
        pass


def move_pdb_to_origin(input_pdb, output_filename):
    """
    Moves the atomic positions in a PDB file to the origin.

    This function reads a PDB file, calculates the geometric center (centroid)
    of all atomic positions, shifts all positions so that the centroid is at
    the origin, and writes the modified structure to a new PDB file.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_filename : str
        Path to the output PDB file where the modified structure will be saved.

    Returns
    -------
    None
        The function writes the modified PDB structure to the specified output file.
    """
    # Load the PDB file
    pdb = PDBFile(input_pdb)
    # Get atomic positions as a NumPy array
    positions = pdb.getPositions(asNumpy=True)
    # Calculate the geometric center (centroid) of the positions
    center = np.mean(positions, axis=0)
    # Shift all positions so that the centroid is at the origin
    new_positions = positions - center

    # Write the modified structure to the output file
    with open(output_filename, 'w') as f:
        PDBFile.writeFile(pdb.topology, new_positions, f)


def center_in_box(modeller):
    """
    Centers the atomic positions of a molecular system within the simulation box.

    This function calculates the centroid of the atomic positions in the modeller object
    and shifts the positions so that the centroid is at the center of the simulation box.
    The box dimensions are determined from the modeller's topology.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The Modeller object containing the topology and positions of the molecular system.

    Returns
    -------
    None
        The function modifies the positions of the modeller object in place.
    """
    # Extract positions in nanometers
    pos_list = modeller.positions.value_in_unit(unit.nanometer)

    # Ensure positions are a NumPy array of shape (N, 3)
    try:
        pos_nm = np.asarray(pos_list, dtype=float)
        if pos_nm.ndim != 2 or pos_nm.shape[1] != 3:
            raise ValueError
    except Exception:
        # Handle cases where positions are not directly convertible to NumPy array
        pos_nm = np.array([[getattr(p, 'x', p[0]),
                            getattr(p, 'y', p[1]),
                            getattr(p, 'z', p[2])] for p in pos_list], dtype=float)

    # Calculate the centroid of the positions
    centroid = pos_nm.mean(axis=0)

    # Initialize box dimensions
    box_vec = None

    # Check if the topology has unit cell dimensions
    if hasattr(modeller.topology, 'getUnitCellDimensions'):
        dims = modeller.topology.getUnitCellDimensions()
        if dims is not None:
            box_vec = np.array([dims.x, dims.y, dims.z])

    # If unit cell dimensions are not available, check for periodic box vectors
    if box_vec is None and hasattr(modeller.topology, 'getPeriodicBoxVectors'):
        vecs = modeller.topology.getPeriodicBoxVectors()
        if vecs is not None:
            box_vec = np.array([vecs[0].x, vecs[1].y, vecs[2].z])

    # If box dimensions are available, center the positions
    if box_vec is not None:
        box_center = box_vec / 2.0
        shift = box_center - centroid
        new_pos = pos_nm + shift
        modeller.positions = unit.Quantity(new_pos, unit.nanometer)


def fix_pdb_chains(input_file, output_file):
    """
    Assign unique chain IDs to each residue in a PDB file.

    Reads the PDB line by line and replaces the chain-ID column so that
    every new residue receives the next available chain letter/digit
    (A–Z, a–z, 0–9, cycling).

    Parameters
    ----------
    input_file : str
        Path to the input PDB file.
    output_file : str
        Path to the output PDB file with corrected chain IDs.

    Returns
    -------
    None
    """
    chain_chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    current_residue = None
    chain_index = -1
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith(("ATOM  ", "HETATM")):
                res_id = line[22:27]
                if res_id != current_residue:
                    current_residue = res_id
                    chain_index += 1
                chain_id = chain_chars[chain_index % len(chain_chars)]
                new_line = line[:21] + chain_id + line[22:]
                outfile.write(new_line)
            else:
                outfile.write(line)


def fix_pdb_atom_labels(input_file, output_file):
    """
    Regenerate unique atom names and serial numbers in a PDB file.

    Within each residue, atoms are renamed ``<Element><count>`` (e.g. C1,
    C2, H1) and atom serial numbers are renumbered sequentially from 1.

    Parameters
    ----------
    input_file : str
        Path to the input PDB file.
    output_file : str
        Path to the output PDB file with corrected atom labels.

    Returns
    -------
    None
    """
    current_residue = None
    element_counts = {}
    global_atom_serial = 1
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith(("ATOM  ", "HETATM")):
                res_id = line[22:27]
                if res_id != current_residue:
                    current_residue = res_id
                    element_counts = {}
                element = line[76:78].strip().upper()
                if not element:
                    old_name = line[12:16].strip()
                    element = ''.join([c for c in old_name if c.isalpha()])[:2]
                    if not element:
                        element = "X"
                element_counts[element] = element_counts.get(element, 0) + 1
                count = element_counts[element]
                raw_new_name = f"{element}{count}"
                if len(element) == 1:
                    new_atom_name = f" {raw_new_name:<3}"[:4]
                else:
                    new_atom_name = f"{raw_new_name:<4}"[:4]
                new_serial = f"{global_atom_serial:>5}"[:5]
                global_atom_serial += 1
                new_line = line[:6] + new_serial + line[11:12] + new_atom_name + line[16:]
                outfile.write(new_line)
            else:
                outfile.write(line)


def convert_xyz_to_pdb(input_file: str, output_file: str, cutoff_multiplier: float = 1.1) -> int:
    """
    Convert an XYZ file to a PDB file with connectivity and residue assignment.

    Molecules (clusters) are identified using distance-based connectivity and
    assigned unique chain IDs, residue IDs, and three-letter residue names.
    Atoms within each cluster are reordered following Hill-system convention
    (C, H, then remaining elements alphabetically). CONECT records are written
    for all bonds.

    Parameters
    ----------
    input_file : str
        Path to the input XYZ file.
    output_file : str
        Path to the output PDB file.
    cutoff_multiplier : float, optional
        Multiplier applied to natural covalent-radius cutoffs when
        determining bonded neighbours. Default is 1.1.

    Returns
    -------
    int
        The number of molecular clusters (connected components) found.
    """
    # 1. Load the original structure
    original_atoms = read(input_file)
    n_atoms = len(original_atoms)

    # 2. Initial connectivity pass to identify molecules (clusters)
    base_cutoffs = natural_cutoffs(original_atoms)
    cutoffs = [c * cutoff_multiplier for c in base_cutoffs]
    i, j = neighbor_list('ij', original_atoms, cutoffs)

    adjacency_matrix = csr_matrix((np.ones_like(i), (i, j)), shape=(n_atoms, n_atoms))
    n_clusters, original_labels = connected_components(csgraph=adjacency_matrix, directed=False)

    # 3. Canonicalise Atom Ordering: Group by cluster, then Hill system (C, H, others)
    def get_sort_key(cluster_id, symbol):
        """
        Return a sort key for ordering atoms by cluster then Hill system.

        Parameters
        ----------
        cluster_id : int
            The cluster (molecule) index.
        symbol : str
            The atomic element symbol.

        Returns
        -------
        tuple
            A 3-tuple ``(cluster_id, priority, symbol)`` where *priority*
            is 0 for C, 1 for H, and 2 for all other elements.
        """
        if symbol == 'C':
            return (cluster_id, 0, symbol)
        elif symbol == 'H':
            return (cluster_id, 1, symbol)
        else:
            return (cluster_id, 2, symbol)

    sort_data = [(get_sort_key(original_labels[idx], atom.symbol), idx) for idx, atom in enumerate(original_atoms)]
    sort_data.sort(key=lambda x: x[0])
    sorted_indices = [x[1] for x in sort_data]

    # 4. Create the cleanly reordered Atoms object
    atoms = original_atoms[sorted_indices]
    sorted_labels = [original_labels[idx] for idx in sorted_indices]

    # 5. Map sorted labels to Chain and Residue IDs
    unique_labels = []
    for lbl in sorted_labels:
        if not unique_labels or unique_labels[-1] != lbl:
            unique_labels.append(lbl)

    available_chains = string.ascii_uppercase + string.ascii_lowercase + string.digits
    num_chains = len(available_chains)

    label_to_chain = {}
    label_to_resid = {}
    label_to_resname = {}

    for cluster_idx, lbl in enumerate(unique_labels):
        # Chain ID wraps around every 62 clusters
        label_to_chain[lbl] = available_chains[cluster_idx % num_chains]

        # Residue ID increments only when the chain ID wraps around
        label_to_resid[lbl] = (cluster_idx // num_chains) + 1

        # Unique three-letter residue name (AAA, AAB, ..., AAZ, ABA, ..., ZZZ)
        a = cluster_idx // (26 * 26) % 26
        b = (cluster_idx // 26) % 26
        c = cluster_idx % 26
        label_to_resname[lbl] = chr(65 + a) + chr(65 + b) + chr(65 + c)

    chain_ids = [label_to_chain[lbl] for lbl in sorted_labels]
    res_ids = [label_to_resid[lbl] for lbl in sorted_labels]
    res_names = [label_to_resname[lbl] for lbl in sorted_labels]

    # 6. Re-calculate connectivity on the newly sorted atoms
    base_cutoffs = natural_cutoffs(atoms)
    cutoffs = [c * cutoff_multiplier for c in base_cutoffs]
    i, j = neighbor_list('ij', atoms, cutoffs)

    # 7. Generate unique atom names
    atom_names = []
    element_counts_per_cluster = defaultdict(int)

    for idx, atom in enumerate(atoms):
        chain = chain_ids[idx]
        resid = res_ids[idx]
        sym = atom.symbol

        # Track the element count specifically within this unique chain/residue combo
        tracking_key = f"{chain}_{resid}_{sym}"
        element_counts_per_cluster[tracking_key] += 1
        count = element_counts_per_cluster[tracking_key]

        unique_name = f"{sym}{count}"

        if len(sym) == 1 and len(unique_name) < 4:
            formatted_name = f" {unique_name:<3}"
        else:
            formatted_name = f"{unique_name[:4]:<4}"

        atom_names.append(formatted_name)

    # 8. Write the properly grouped and ordered PDB file
    with open(output_file, 'w') as f:
        for idx, atom in enumerate(atoms):
            serial = idx + 1
            sym = atom.symbol
            x, y, z = atom.position

            line = (f"ATOM  {serial:>5} {atom_names[idx]} {res_names[idx]:>3} {chain_ids[idx]}{res_ids[idx]:>4}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {sym:>2}\n")
            f.write(line)

        conect_dict = defaultdict(list)
        for a1, a2 in zip(i, j):
            conect_dict[a1].append(a2)

        for atom_idx in sorted(conect_dict.keys()):
            a1_pdb = atom_idx + 1
            bonded_atoms = sorted([x + 1 for x in conect_dict[atom_idx]])

            for chunk_start in range(0, len(bonded_atoms), 4):
                chunk = bonded_atoms[chunk_start:chunk_start + 4]
                line = f"CONECT{a1_pdb:>5}"
                for b in chunk:
                    line += f"{b:>5}"
                f.write(line + "\n")

    return n_clusters


def convert_xyz_to_plumed_ref(xyz_file, template_pdb, output_file, atom_line='HETATM'):
    """
    Converts a multi-frame XYZ trajectory to a PLUMED-compatible multi-model PDB file using a template PDB.

    This function reads a template PDB file and a multi-frame XYZ file, then writes a new PDB file
    where each frame from the XYZ is inserted as a model, using the atom records from the template.
    The coordinates in each model are replaced with those from the corresponding XYZ frame.
    Terminal entries (TER) are preserved in their original locations.
    The output is formatted for use as a reference structure in PLUMED metadynamics simulations.

    Parameters
    ----------
    xyz_file : str
        Path to the input XYZ file containing one or more frames.
    template_pdb : str
        Path to the template PDB file whose atom records will be used as a base.
    output_file : str
        Path to the output PDB file to be written in multi-model format.
    atom_line : str, tuple, optional
        The record type to match in the template PDB (default: 'HETATM'). Can also be ('ATOM', 'HETATM').

    Returns
    -------
    None
    """
    # 1. Read template and keep BOTH atom lines and TER lines
    with open(template_pdb, 'r') as f:
        template_lines = [line for line in f if line.startswith(atom_line) or line.startswith('TER')]

    # 2. Parse the XYZ file
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        return

    num_atoms = int(lines[0].strip())
    frames = []
    for i in range(0, len(lines), num_atoms + 2):
        frame_coords = lines[i + 2: i + num_atoms + 2]
        frames.append([l.split()[1:] for l in frame_coords])

    # 3. Write out the PLUMED-formatted multi-model PDB
    with open(output_file, 'w') as f:
        f.write("REMARK TYPE=MULTI-ST-PDB\n")
        f.write("REMARK ARG=path.s,path.z\n")

        for i, frame in enumerate(frames):
            f.write(f"REMARK NUMBER={i + 1}\n")
            f.write(f"REMARK STEP={i}\n")

            coord_idx = 0  # Independent counter for the XYZ coordinates

            for t_line in template_lines:
                if t_line.startswith('TER'):
                    # Write TER lines exactly as they appear in the template
                    f.write(t_line)
                else:
                    # Inject XYZ coordinates into the ATOM/HETATM lines
                    coords = frame[coord_idx]
                    new_line = (t_line[:30] +
                                f"{float(coords[0]):8.3f}{float(coords[1]):8.3f}{float(coords[2]):8.3f}" +
                                t_line[54:])
                    f.write(new_line)
                    coord_idx += 1  # Only advance coordinate index for actual atoms

            f.write("ENDMDL\n")
    pdb_remove_ter_index(template_pdb, template_pdb)
    pdb_remove_ter_index(output_file, output_file)


def save_only_index_atoms(modeller, idx_list, file_idx='index_atoms.pdb'):
    """
    Saves only the atoms with specified indices from a Modeller object to a PDB file.

    This function creates a new Modeller object from the provided modeller's topology and positions,
    selects atoms whose indices are in the given idx_list, deletes all other atoms, and writes the
    resulting structure to a PDB file.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The Modeller object containing the molecular system.
    idx_list : list of int
        List of atom indices to retain in the output PDB file.
    file_idx : str, optional
        Filename for the output PDB file. Default is 'index_atoms.pdb'.

    Returns
    -------
    None
        The function writes the selected atoms to the specified PDB file.
    """
    modeller_new = app.Modeller(modeller.topology, modeller.positions)
    atoms_to_keep = [atom for atom in modeller_new.topology.atoms() if atom.index in idx_list]
    modeller_new.delete([atom for atom in modeller_new.topology.atoms() if atom not in atoms_to_keep])
    with open(file_idx, 'w') as f:
        app.PDBFile.writeFile(modeller_new.topology, modeller_new.positions, f)


def pdb_remove_ter_index(input_path, output_path):
    """
    Renumber atom serial indices in a PDB file and normalize TER/CONECT records.

    This function reads a PDB-like file, rewrites atom serial numbers sequentially,
    and updates `CONECT` records to remain consistent with the new numbering. It also:
    - resets numbering at each new model marker (`MODEL` or `REMARK NUMBER=`),
    - updates any `TER` record to share the serial index of the following atom,
    - preserves all non-target lines unchanged.

    Parameters
    ----------
    input_path : str
        Path to the input PDB file.
    output_path : str
        Path to write the cleaned/reindexed PDB file.

    Returns
    -------
    None
        The processed content is written to `output_path`.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    clean_lines = []
    atom_serial = 1
    # Maps old atom serials (as strings) to new right-aligned 5-char serial fields.
    index_map = {}

    for line in lines:
        if line.startswith(("MODEL", "REMARK NUMBER=")):
            # Start a fresh atom numbering sequence for each model.
            atom_serial = 1
            clean_lines.append(line)

        elif line.startswith(("ATOM  ", "HETATM")):
            # Reindex ATOM/HETATM serial field (columns 7-11 in PDB format).
            old_serial = line[6:11].strip()
            index_map.setdefault(old_serial, f"{atom_serial:>5}")
            new_line = line[:6] + f"{atom_serial:5d}" + line[11:]
            clean_lines.append(new_line)
            atom_serial += 1

        elif line.startswith("TER"):
            # Preserve the TER line and update its serial number (columns 7-11).
            # We explicitly do NOT increment `atom_serial` so the next ATOM
            # line receives this exact same index.
            if len(line) >= 11:
                new_line = line[:6] + f"{atom_serial:5d}" + line[11:]
            else:
                # Fallback just in case the TER line is truncated
                new_line = f"{line.strip():<6}{atom_serial:5d}\n"
            clean_lines.append(new_line)

        elif line.startswith("CONECT"):
            # Rewrite CONECT atom references using the reindex map.
            new_conect = line[:6]
            for i in range(6, len(line.strip()), 5):
                old_idx = line[i:i + 5].strip()
                new_conect += index_map.get(old_idx, line[i:i + 5])
            clean_lines.append(new_conect.rstrip() + "\n")

        else:
            # Preserve unrelated records (e.g., REMARK, END, CRYST1).
            clean_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(clean_lines)
