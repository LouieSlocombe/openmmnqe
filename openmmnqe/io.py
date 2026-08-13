"""Structure preparation and file conversion, upstream of every simulation.

Getting a raw structure to the point where OpenMM will build a system from it
is most of the work of setting up a run, and it is what this module does.  The
entry point for a protein-ligand system is :func:`prepare_lig_system`, which
strips the water and ions, works out which residues are ligands, and hands
back a combined PDB and the ligand molecules; :func:`prepare_ligand_ff` then
parameterises those ligands with GAFF and returns a force field that covers
the whole system.  The rest are the smaller operations those two are built
from and that the workflows call directly -- relabelling residues, repairing
a PDB with PDBFixer, centring a structure in its box, and converting between
XYZ, SDF and PDB.

Two of those conversions matter more than the others:
:func:`convert_xyz_to_plumed_ref` turns a reaction path into the multi-model
PDB that PLUMED's ``PATHMSD`` reads, and :func:`save_only_index_atoms` writes
the ``index_atoms.pdb`` that the path is aligned against.

A note on residue names. Whether a residue counts as a ligand is decided by
name against :data:`STANDARD_RESIDUE_NAMES`, but OpenMM itself matches
residue templates on the molecular graph and ignores names entirely. The two
can disagree, and when they do the GAFF parameters generated for a "ligand"
are silently discarded in favour of the standard template -- hence the
warnings raised by :func:`_warn_ff_named_ligands` and
:func:`_warn_ff_matched_molecules`.
"""
import glob
import os
import re
import shutil
import string
import warnings
from collections import defaultdict
from typing import List

import MDAnalysis as mda
import numpy as np
import openmm.unit as unit
from ase.data import chemical_symbols
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from openff.toolkit import Molecule
from openmm.app import PDBFile, Topology, Element, Modeller
from openmmforcefields.generators import GAFFTemplateGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.constants import physical_constants as const
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from openmm import app, Vec3
from reactiontools import as_fes

# Conversion factor from Bohr to Angstrom
bohr_to_angstrom = const["Bohr radius"][0] * 1e10
# Conversion factor from Angstrom to nm
A_to_nm = 1.0e-1
# Conversion factor from eV to J
eV_to_J = const["electron volt-joule relationship"][0]
# Conversion factor from J to kJ
J_to_kJ = 1.0e-3
# Avogadro's number, mol^-1
avo_num = const["Avogadro constant"][0]

# Conversion factor from eV to kJ/mol
eV_to_kJpermol = eV_to_J * J_to_kJ * avo_num
# Conversion factor from eV/Angstrom^2 to kJ/(mol*nm^2)
eVperA2_to_kJpermolpernm2 = eV_to_kJpermol / A_to_nm ** 2

# Two-letter element symbols that are also common prefixes of hydrogen atom
# names, e.g. "HG21" is a gamma hydrogen rather than mercury. These are only
# consulted when a PDB line has no element column to fall back on.
_HYDROGEN_LOOKALIKES = {'He', 'Hf', 'Hg', 'Ho', 'Hs'}

# Residue names the non-standard residue scans treat as already parameterised.
# Anything else in a PDB is assumed to be a ligand needing its own parameters.
STANDARD_RESIDUE_NAMES = frozenset({
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
})

# Residue names that amber14-all.xml ships templates for but which are
# deliberately absent from STANDARD_RESIDUE_NAMES: the chain-terminal and free
# nucleoside variants, and the peptide capping groups. Calling them standard
# would leave a PDB made only of, say, two free deoxynucleosides with no
# residues at all to hand back from prepare_lig_system, so they stay detectable
# as ligands and instead trigger the warning in _warn_ff_named_ligands.
FORCE_FIELD_RESIDUE_NAMES = frozenset({
    # DNA 5'/3' termini and free deoxynucleosides
    'DA5', 'DA3', 'DAN', 'DC5', 'DC3', 'DCN',
    'DG5', 'DG3', 'DGN', 'DT5', 'DT3', 'DTN',
    # RNA 5'/3' termini and free nucleosides
    'A5', 'A3', 'AN', 'C5', 'C3', 'CN',
    'G5', 'G3', 'GN', 'U5', 'U3', 'UN',
    # Peptide capping groups
    'ACE', 'NME', 'NHE'
})


def _element_from_pdb_line(line):
    """
    Determine the element symbol for a PDB ``ATOM``/``HETATM`` line.

    The element column (77-78) is used when present. Otherwise the symbol is
    inferred from the atom name in columns 13-16, which by convention starts in
    column 13 for two-character symbols and column 14 for one-character ones.

    Parameters
    ----------
    line : str
        A single ``ATOM`` or ``HETATM`` record.

    Returns
    -------
    str
        The element symbol, or ``'X'`` if none could be determined.

    Notes
    -----
    The atom-name fallback relies on the name being correctly aligned, which
    is what distinguishes calcium (``CA  ``) from an alpha carbon (`` CA ``).
    Hydrogen names that collide with two-letter symbols (see
    `_HYDROGEN_LOOKALIKES`) are resolved as hydrogen.
    """
    symbol = line[76:78].strip()
    if symbol:
        return symbol.capitalize()

    # No element column, so fall back to the atom name
    name = line[12:16]
    letters = ''.join(c for c in name if c.isalpha())
    if not letters:
        return 'X'

    # A name indented or prefixed by a digit carries a one-character symbol
    if not name[:1].isalpha():
        return letters[0] if letters[0] in chemical_symbols else 'X'

    symbol = letters[:2].capitalize()
    if symbol in _HYDROGEN_LOOKALIKES:
        return 'H'
    if symbol in chemical_symbols:
        return symbol
    # Two-letter guess is not an element, so treat it as a single-letter symbol
    return symbol[0] if symbol[0] in chemical_symbols else 'X'


def _write_xyz_frame(fh, symbols, positions, comment=''):
    """
    Write a single frame to an open XYZ file handle.

    Parameters
    ----------
    fh : file-like object
        An open, writable text handle.
    symbols : sequence of str
        Element symbol for each atom.
    positions : sequence of sequence of float
        Cartesian coordinates in Angstrom, one ``(x, y, z)`` triple per atom.
    comment : str, optional
        Text for the frame's comment line. Default is an empty string.
    """
    fh.write(f"{len(symbols)}\n")
    fh.write(f"{comment}\n")
    for symbol, (x, y, z) in zip(symbols, positions):
        fh.write(f"{symbol:<2}   {x:>12.6f} {y:>12.6f} {z:>12.6f}\n")


def _format_pdb_atom_name(symbol, count):
    """
    Format a unique atom name for the PDB atom-name field (columns 13-16).

    Single-character elements are indented by one column, following the PDB
    convention, and names longer than four characters are truncated so that
    column alignment is preserved.

    Parameters
    ----------
    symbol : str
        The element symbol.
    count : int
        Occurrence index of this element within its residue.

    Returns
    -------
    str
        A four-character atom name.
    """
    name = f"{symbol}{count}"
    if len(symbol) == 1 and len(name) < 4:
        return f" {name:<3}"
    return f"{name:<4}"[:4]


def remove_directory(directory):
    """
    Remove a directory and its contents, if it exists.

    Parameters
    ----------
    directory : str
        Path to the directory. A path that does not exist is not an error.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)


def copy_and_rename_file(src, dst_dir, new_name):
    """
    Copy a file into a directory under a new name.

    Parameters
    ----------
    src : str
        Path to the source file.
    dst_dir : str
        Directory to copy it into.
    new_name : str
        Name the copy is given.
    """
    shutil.copy(src, os.path.join(dst_dir, new_name))
    return None


def list_files_with_pattern(directory, pattern):
    """
    List the files in a directory matching a glob pattern.

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str
        Glob pattern to match names against, e.g. ``'*.txt'``.

    Returns
    -------
    list of str
        Matching paths, in whatever order :mod:`glob` returns them.
    """
    return glob.glob(os.path.join(directory, pattern))


def search_fes_files(target_directory: str, pattern: str = r'^fes_?(\d+)\.dat$') -> list[str]:
    r"""
    Find the numbered FES files in a directory, in index order.

    By default this matches the names written by ``plumed sum_hills --stride``
    and by the bundled OPES ``FES_from_*`` scripts, i.e. ``FES1.dat`` and
    ``fes_1.dat``, case-insensitively.  Results are ordered by the number they
    carry rather than alphabetically, so ``fes_2.dat`` precedes ``fes_10.dat``.

    Parameters
    ----------
    target_directory : str
        Directory to search.
    pattern : str, optional
        Regular expression matched against each file name.  Its first capture
        group, if present, is used as the sort key. Default matches
        ``fes_<n>.dat`` and ``fes<n>.dat``.

    Returns
    -------
    list of str
        Matching file names, sorted by their index.
    """
    regex = re.compile(pattern, re.IGNORECASE)
    matches = []

    for filename in os.listdir(target_directory):
        found = regex.match(filename)
        if found:
            # Sort on the captured index when the pattern provides one.
            key = int(found.group(1)) if found.groups() else filename
            matches.append((key, filename))

    return [filename for _, filename in sorted(matches)]


def load_fes_data(directory: str,
                  bins: int | None = None,
                  energy_unit: str = "eV",
                  source_unit: str = "kJ/mol",
                  pattern: str = r'^fes_?(\d+)\.dat$',
                  verbose: bool = True) -> list:
    """
    Find the numbered FES files in a directory and load them in order.

    Each file is parsed with :func:`reactiontools.as_fes`, so 1-D and 2-D
    surfaces are both handled and the result can be handed straight to
    :func:`reactiontools.plot_fes_1d` or
    :func:`reactiontools.plot_fes_2d` to show how a surface converges.

    Parameters
    ----------
    directory : str
        Directory containing the FES files.
    bins : int, optional
        Unused; kept so existing positional calls keep working.
    energy_unit : str, optional
        Unit to convert the free energies to (default is ``"eV"``).
    source_unit : str, optional
        Unit the files are written in (default is ``"kJ/mol"``, which is what
        PLUMED writes when driven from OpenMM).
    pattern : str, optional
        Regular expression selecting the files; see :func:`search_fes_files`.
    verbose : bool, optional
        Whether to report each file as it is loaded (default is True).

    Returns
    -------
    list of reactiontools.FES
        One surface per file, ordered by file index.
    """
    fes_files = search_fes_files(directory, pattern=pattern)
    fes_arrays = []

    for filename in fes_files:
        file_path = os.path.join(directory, filename)
        fes = as_fes(file_path, energy_unit=energy_unit, source_unit=source_unit)
        if verbose:
            print(f"Loading {file_path} with {fes.ndim} CV(s)")
        fes_arrays.append(fes)

    return fes_arrays


def xyz_to_sdf(xyz_path, sdf_path, default_charge=0, sanitize=True, kekulize=False):
    """
    Convert an XYZ file to SDF, inferring the bonds as it goes.

    XYZ carries only elements and coordinates, so the bonds and bond orders
    an SDF needs are worked out from the geometry by RDKit.  That inference
    depends on the total charge, which is read from each frame's comment line
    if it says something like ``charge=-1`` or ``q: -1``, and taken from
    *default_charge* otherwise.

    Parameters
    ----------
    xyz_path : str
        Path to the input XYZ file. May hold more than one frame.
    sdf_path : str
        Path to the output SDF file.
    default_charge : int, optional
        Total charge assumed for any frame whose comment does not give one.
        Default is 0.
    sanitize : bool, optional
        Whether to sanitise each molecule before writing, falling back to a
        partial sanitisation if the full one fails. Default is True.
    kekulize : bool, optional
        Whether to write explicit alternating bonds rather than aromatic
        ones. Default is False.

    Returns
    -------
    int
        The number of molecules written.

    Raises
    ------
    ValueError
        If the XYZ file is malformed or holds no frames.
    IOError
        If *sdf_path* cannot be opened for writing.
    """

    def _parse_charge_from_comment(comment, fallback):
        """
        Extract a total charge from the comment line of an XYZ frame.

        Three spellings are recognised, in decreasing order of confidence:
        ``charge=-1``, ``q: -1``, and a bare signed integer standing alone.

        Parameters
        ----------
        comment : str or None
            The frame's comment line.
        fallback : int
            Charge to return when the comment gives none.

        Returns
        -------
        int
            The charge found, or *fallback*.
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
        Split an XYZ file into its frames.

        Parameters
        ----------
        path : str
            Path to the XYZ file.

        Returns
        -------
        list of tuple
            One ``(comment, atom_lines)`` pair per frame.

        Raises
        ------
        ValueError
            If a frame is truncated, its atom count is unreadable, or the
            file holds no frames at all.
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
        Build an RDKit molecule from one XYZ frame.

        The molecule comes back with atoms and 3-D coordinates but no bonds;
        those are inferred by the caller.

        Parameters
        ----------
        comment : str
            The frame's comment line, used as the molecule name.
        coord_lines : list of str
            The frame's atom lines, each ``El x y z``.
        name_fallback : str
            Name to use when the comment is empty.

        Returns
        -------
        rdkit.Chem.Mol
            The molecule, with one conformer.

        Raises
        ------
        ValueError
            If an atom line is short or its coordinates do not parse.
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
    Write each non-standard residue of a PDB file out to its own file.

    A residue counts as non-standard when its name is absent from
    :data:`STANDARD_RESIDUE_NAMES`, which in practice means a ligand needing
    parameters of its own.  Single-atom residues are skipped: those are ions,
    and there is nothing to parameterise.

    Parameters
    ----------
    pdb_file_path : str
        Path to the input PDB file.
    output_dir : str, optional
        Directory to write the residue files to, created if needed. Default
        is the current directory.
    sdf : bool, optional
        Whether to convert the files to SDF, with bonds inferred, and delete
        the XYZ originals. Default is False.

    Returns
    -------
    list of str
        Paths to the files written, one per residue.
    """
    pdb = PDBFile(pdb_file_path)

    topology = pdb.getTopology()
    positions_quantity = pdb.getPositions(asNumpy=True)
    positions_angstrom = positions_quantity.value_in_unit(unit.angstrom)

    generated_files = []

    os.makedirs(output_dir, exist_ok=True)

    for residue in topology.residues():
        if residue.name not in STANDARD_RESIDUE_NAMES:
            res_name = residue.name
            res_id = residue.id
            chain_id = residue.chain.id

            safe_res_name = "".join(c for c in res_name if c.isalnum())
            filename = f"{safe_res_name}_{chain_id}_{res_id}.xyz"
            output_path = os.path.join(output_dir, filename)

            atoms_in_residue = list(residue.atoms())
            num_atoms = len(atoms_in_residue)

            if num_atoms <= 1:
                continue

            print(f"Found non-standard residue: {res_name} (Chain {chain_id}, ResID {res_id})", flush=True)

            comment = f"Residue: {res_name}, Chain: {chain_id}, ResID: {res_id}, Source: {os.path.basename(pdb_file_path)}"
            with open(output_path, 'w') as f:
                _write_xyz_frame(f,
                                 [atom.element.symbol for atom in atoms_in_residue],
                                 [positions_angstrom[atom.index] for atom in atoms_in_residue],
                                 comment)

            generated_files.append(output_path)
            print(f"Successfully wrote {num_atoms} atoms to {os.path.splitext(output_path)[0]}", flush=True)

    if sdf:
        for xyz_file in generated_files:
            sdf_file = os.path.splitext(xyz_file)[0] + ".sdf"
            xyz_to_sdf(xyz_file, sdf_file, sanitize=True, kekulize=False)
            os.remove(xyz_file)
        generated_files = [os.path.splitext(f)[0] + ".sdf" for f in generated_files]

    return generated_files


def get_non_standard_residues(pdb_file):
    """
    Collect the non-standard residues of a PDB file as RDKit molecules.

    Membership of :data:`STANDARD_RESIDUE_NAMES` decides what counts as
    standard. Each residue found and each one skipped is printed, with its
    SMILES, which is the quick way to see what a structure is carrying.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.

    Returns
    -------
    list of rdkit.Chem.Mol
        One unsanitised molecule per non-standard residue.

    See Also
    --------
    list_non_standard_residues : Returns just the residue keys.
    """
    mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
    mols_by_residue = Chem.SplitMolByPDBResidues(mol)

    print(f"\n--- Found {len(mols_by_residue)} total residue fragments ---")

    non_standard_mols = []
    for residue_key, fragment_mol in mols_by_residue.items():
        res_name = residue_key.split('_')[0].strip()
        if res_name not in STANDARD_RESIDUE_NAMES:
            print(f"  > Found non-standard residue: {residue_key}")
            print(Chem.MolToSmiles(fragment_mol))
            non_standard_mols.append(fragment_mol)
        else:
            print(f"  - Skipping standard residue: {residue_key}")

    return non_standard_mols


def list_non_standard_residues(pdb_file):
    """
    List the keys of the non-standard residues in a PDB file.

    The quiet counterpart to :func:`get_non_standard_residues`: it names the
    residues without building molecules or printing anything, which is what
    :func:`prepare_lig_system` uses to decide what its ligands are.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.

    Returns
    -------
    list of str
        Residue keys, each ``'<RESNAME>_<CHAIN><RESID>'`` as RDKit spells
        them.

    See Also
    --------
    get_non_standard_residues : Returns the molecules themselves.
    """
    mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
    mols_by_residue = Chem.SplitMolByPDBResidues(mol)

    non_standard_mols = []
    for residue_key, fragment_mol in mols_by_residue.items():
        res_name = residue_key.split('_')[0].strip()
        if res_name not in STANDARD_RESIDUE_NAMES:
            non_standard_mols.append(residue_key)
    return non_standard_mols


def clean_ions_in_pdb(pdb_input_path: str, ions_to_remove: List[str], pdb_output_path: str) -> List[str]:
    """
    Remove named ion residues from a PDB file.

    Anything that is a single-atom residue and is not water is taken to be
    an ion.  Every such type found is reported, not only the ones removed,
    so a first pass with an empty *ions_to_remove* is a way of finding out
    what a structure contains.

    Parameters
    ----------
    pdb_input_path : str
        Path to the input PDB file.
    ions_to_remove : list of str
        Residue names to delete, matched case-insensitively.
    pdb_output_path : str
        Path to write the cleaned PDB to. May be the input path.

    Returns
    -------
    list of str
        Every ion residue type found in the input, sorted.
    """
    pdb = PDBFile(pdb_input_path)
    modeller = Modeller(pdb.topology, pdb.positions)
    ions_to_remove_upper = {ion.upper() for ion in ions_to_remove}
    all_found_ion_types = set()
    residues_to_delete = []

    for res in modeller.topology.residues():
        res_name_upper = res.name.upper()
        if res_name_upper in ['HOH', 'WAT']:
            continue
        # A single-atom residue is treated as an ion.
        if len(list(res.atoms())) == 1:
            all_found_ion_types.add(res.name)
            if res_name_upper in ions_to_remove_upper:
                residues_to_delete.append(res)

    print(f"-> Found all potential ion types: {sorted(list(all_found_ion_types))}")
    print(f"-> Will remove {len(residues_to_delete)} residues matching: {ions_to_remove}")

    if residues_to_delete:
        modeller.delete(residues_to_delete)
        print(f"Successfully removed {len(residues_to_delete)} ion residues.")
    else:
        print("No matching ion residues found to remove.")

    with open(pdb_output_path, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print(f"Cleaned PDB saved to: {pdb_output_path}")

    return sorted(list(all_found_ion_types))


def relabel_residues_in_pdb(pdb_file_path, relabel_map, output_file):
    """
    Rename residues in a PDB file according to a mapping.

    Renaming is how a residue is moved between the standard force field and
    GAFF, since :func:`prepare_lig_system` decides what is a ligand by name.
    A summary of what changed is printed.

    Parameters
    ----------
    pdb_file_path : str
        Path to the input PDB file.
    relabel_map : dict
        Mapping from old residue name to new. Names not in it are left be.
    output_file : str or file-like object
        Path to write to, or an open handle. May be the input path.

    Returns
    -------
    openmm.app.PDBFile
        The relabelled structure.
    """
    pdb = PDBFile(pdb_file_path)
    topology = pdb.topology
    positions = pdb.positions
    changed_residues = {}

    for residue in topology.residues():
        if residue.name in relabel_map:
            original_name = residue.name
            new_name = relabel_map[original_name]
            residue.name = new_name
            change_key = (original_name, new_name)
            if change_key not in changed_residues:
                changed_residues[change_key] = 0
            changed_residues[change_key] += 1

    if changed_residues:
        print("Relabeling complete. Summary of changes:")
        for (old, new), count in changed_residues.items():
            print(f"  - Relabeled {count} residues from '{old}' to '{new}'")
    else:
        print("No residues found matching the relabel map. Topology is unchanged.")

    print("Saving modified topology and positions...")
    if isinstance(output_file, str):
        with open(output_file, 'w') as f:
            PDBFile.writeFile(topology, positions, f)
        print(f"Successfully saved modified PDB to: {output_file}")
    else:
        PDBFile.writeFile(topology, positions, output_file)
        print("Successfully wrote modified PDB to file-like object.")

    return pdb


def remove_residues_in_pdb(input_pdb, output_pdb, names):
    """
    Remove every residue with one of the given names from a PDB file.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Path to write the result to. May be the input path.
    names : iterable of str
        Residue names to delete, matched exactly.
    """
    pdb = PDBFile(input_pdb)
    modeller = Modeller(pdb.topology, pdb.positions)

    residues_to_delete = [res for res in modeller.topology.residues()
                          if res.name in names]

    print(f"Found {len(residues_to_delete)} residues to delete.")

    if residues_to_delete:
        modeller.delete(residues_to_delete)
        print("Successfully deleted residues.")
    else:
        print("No matching residues found to delete.")

    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)


def remove_water_residues_in_pdb(input_pdb, output_pdb, water_names=None):
    """
    Remove the water residues from a PDB file.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Path to write the result to. May be the input path.
    water_names : set of str or None, optional
        Residue names to treat as water. If None, ``{"HOH", "WAT"}`` is
        used. Default is None.

    See Also
    --------
    remove_residues_in_pdb : The general form this wraps.
    """
    if water_names is None:
        water_names = {"HOH", "WAT"}
    print("Removing water residues.")
    remove_residues_in_pdb(input_pdb, output_pdb, water_names)


def fix_pdb(file_in, file_out, ph=7.0, rm_heterogens=True):
    """
    Repair a PDB file with PDBFixer.

    Rebuilds missing residues and atoms, substitutes non-standard residues
    for their standard equivalents, and adds hydrogens.  Worth running on a
    raw crystal structure; note that the substitution step is a silent
    structural edit, so it is not something to apply to a structure that is
    already equilibrated.

    Parameters
    ----------
    file_in : str
        Path to the input PDB file.
    file_out : str
        Path to write the repaired PDB to.
    ph : float, optional
        pH the hydrogens are added at, which decides the protonation of
        titratable residues. Default is 7.0.
    rm_heterogens : bool, optional
        Whether to drop water, ions and ligands. Default is True.
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
    with open(file_out, 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return None


def make_sdf(pdb_file, lig_name='LIG'):
    """
    Extract ligand residues from a PDB file into ``<lig_name>.sdf``.

    Elements are guessed from the atom names, since a PDB written by some
    tools carries no element column and MDAnalysis needs one to hand RDKit a
    molecule. Each residue is written as a separate SDF record so repeated
    copies of the same ligand do not become one disconnected molecule.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    lig_name : str, optional
        Residue name of the ligand, which also names the output file.
        Default is ``'LIG'``.
    """
    u = mda.Universe(pdb_file)
    elements = mda.topology.guessers.guess_types(u.atoms.names)
    u.add_TopologyAttr('elements', elements)
    residues = u.select_atoms(f"resname {lig_name}").residues
    if len(residues) == 0:
        raise ValueError(f"No residues named {lig_name!r} found in {pdb_file}")

    with Chem.SDWriter(f"{lig_name}.sdf") as writer:
        writer.SetKekulize(False)
        for residue in residues:
            writer.write(residue.atoms.convert_to("RDKIT"))
    return None


def pdb_patcher(pdb_file, lig_name='LIG'):
    """
    Repair the placeholder names a round-trip through OpenFF leaves behind.

    A ligand that has been through ``Molecule.to_topology()`` comes back
    named ``UNK``, with ``x`` where the atom names should have padding, so
    both are patched in place.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file, rewritten in place.
    lig_name : str, optional
        Residue name to put in place of ``UNK``. Default is ``'LIG'``.
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
    """
    pdb = app.PDBFile(input_pdb)
    molecules = Molecule.from_file(f'{lig_name}.sdf')
    if not isinstance(molecules, list):
        molecules = [molecules]

    modeller = app.Modeller(pdb.topology, pdb.positions)
    for molecule in molecules:
        ligand_ff_topology = molecule.to_topology()
        ligand_omm_topology = ligand_ff_topology.to_openmm()
        ligand_positions = ligand_ff_topology.get_positions().to_openmm()
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
    """
    if isinstance(input_files, str):
        input_files = [input_files]
    all_mols = []
    for sdf_path in input_files:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        for mol in suppl:
            if mol is not None:
                all_mols.append((mol, sdf_path))
    combined_topology = Topology()
    combined_positions = []
    for mol, sdf_path in all_mols:
        res_name = os.path.splitext(os.path.basename(sdf_path))[0]
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


def _warn_ff_named_ligands(lig_names_list):
    """
    Warn about "ligands" whose residue names belong to a standard force field.

    A name-only heads-up, raised as early as possible so the caller does not
    pay for SDF extraction and charge generation before finding out. It only
    knows about the names in :data:`FORCE_FIELD_RESIDUE_NAMES`;
    :func:`_warn_ff_matched_molecules` is the authoritative check, since it
    asks the force field itself.

    Parameters
    ----------
    lig_names_list : list of str
        Residue names about to be treated as ligands.
    """
    known = sorted(name for name in lig_names_list if name in FORCE_FIELD_RESIDUE_NAMES)
    if known:
        warnings.warn(
            f"Residue(s) {', '.join(known)} are being treated as ligands, but they are "
            "standard AMBER residue names (amber14-all.xml ships templates for all the "
            "terminal and free-nucleoside variants). If you go on to parameterise them "
            "with GAFF, OpenMM will match the standard template instead and that work "
            "will be discarded. Pass lig_names explicitly, or relabel them via "
            "residue_map, to leave them to the standard force field.",
            UserWarning,
            stacklevel=3)


def _warn_ff_matched_molecules(forcefield, molecules):
    """
    Warn about ligands the standard force field already has a template for.

    OpenMM matches residue templates on the molecular graph -- residue names
    are never consulted -- so a "ligand" that happens to be a standard residue
    is parameterised from the standard force field and the GAFF template
    generator is simply never called for it. The GAFF conformers, charges and
    cache entries produced for such a molecule are silently discarded, which is
    what this warning exists to make visible.

    ``forcefield`` must not have the GAFF generator registered yet, or every
    molecule would match.

    Parameters
    ----------
    forcefield : openmm.app.ForceField
        Force field built from the standard XML files alone.
    molecules : list of openff.toolkit.Molecule
        The ligand molecules about to be parameterised.
    """
    for mol in molecules:
        try:
            templates = forcefield.getMatchingTemplates(mol.to_topology().to_openmm())
        except ValueError:
            # No template for this molecule: GAFF is genuinely needed.
            continue
        template_names = ', '.join(sorted({template.name for template in templates}))
        warnings.warn(
            f"Ligand '{mol.name}' is already matched by residue template(s) "
            f"{template_names} in the supplied standard force field. OpenMM matches "
            "templates on the molecular graph and ignores residue names, so the GAFF "
            "parameters generated here will never be used and no cache entry will be "
            "written for it. Drop it from the ligand list unless it is covalently "
            "bonded to the rest of the system, where the match may not hold.",
            UserWarning,
            stacklevel=3)


def prepare_lig_system(input_pdb,
                       combined_pdb='combined_system.pdb',
                       clean_pdb='cleaned.pdb',
                       rm_ions=None,
                       residue_map=None,
                       rm_files=True,
                       rm_lig_sdf=True,
                       lig_names=None,
                       fix_receptor=False):
    """
    Prepare a protein-ligand system from a raw PDB file.

    Removes water and (optionally) ions, relabels residues, identifies
    non-standard (ligand) residues, generates SDF files, optionally
    repairs the receptor, and combines ligand and receptor topologies
    into one PDB ready for force-field parameterisation.

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
    fix_receptor : bool, optional
        If True, run :func:`fix_pdb` on the receptor before the ligand is
        re-attached, rebuilding any missing residues and atoms and adding
        hydrogens at pH 7. This is worth enabling for raw crystal
        structures, but PDBFixer also rewrites any residue in its
        substitution table to the standard equivalent, discarding that
        residue's hydrogens and re-adding them at pH 7. Those are silent
        structural edits, and they are pure overhead for an input that is
        already equilibrated, so it is off by default. The ligand is
        unaffected either way -- it is re-attached from its SDF after the
        fixer runs. Has no effect on ligand-only systems, which have no
        receptor to repair. Default is False.

    Returns
    -------
    pdb_data : openmm.app.PDBFile
        The final combined PDB data.
    molecules : openff.toolkit.Molecule or list of openff.toolkit.Molecule
        The ligand molecule(s). A single ``Molecule`` is returned when only
        one ligand is present; otherwise a list.

    Warns
    -----
    UserWarning
        If a residue treated as a ligand is named after a residue a standard
        force field already provides a template for (see
        :data:`FORCE_FIELD_RESIDUE_NAMES`).
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

    _warn_ff_named_ligands(lig_names_list)

    molecules = []
    generated_sdfs = []

    for lig_name in lig_names_list:
        sdf_filename = f'{lig_name}.sdf'
        make_sdf(clean_pdb, lig_name=lig_name)
        generated_sdfs.append(sdf_filename)
        ligand_molecules = Molecule.from_file(sdf_filename)
        if not isinstance(ligand_molecules, list):
            ligand_molecules = [ligand_molecules]

        for mol in ligand_molecules:
            mol.name = lig_name
            if not any(Molecule.are_isomorphic(mol, known)[0]
                       for known in molecules):
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
        for lig_name in lig_names_list:
            pdb_patcher(combined_pdb, lig_name=lig_name)
    else:
        if fix_receptor:
            fix_pdb(clean_pdb, combined_pdb, rm_heterogens=False)
            remove_residues_in_pdb(combined_pdb, combined_pdb, names=lig_names_list)
        else:
            remove_residues_in_pdb(clean_pdb, combined_pdb, names=lig_names_list)
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

    Warns
    -----
    UserWarning
        If ``standard_ff`` already has a residue template matching one of the
        molecules. OpenMM matches templates on the molecular graph rather than
        by residue name, so the standard template wins and the GAFF parameters
        for that molecule are never used.
    """
    if not isinstance(molecule, list):
        molecules = [molecule]
    else:
        molecules = molecule

    if isinstance(standard_ff, str):
        standard_ff = [standard_ff]

    forcefield = app.ForceField(*standard_ff)
    # Before any charges are computed, since anything the standard force field
    # already covers is work thrown away.
    _warn_ff_matched_molecules(forcefield, molecules)

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

    forcefield.registerTemplateGenerator(gaff.generator)

    if gen_cache:
        print('Triggering parameterization to populate cache...', flush=True)
        for mol in molecules:
            omm_topology = mol.to_topology().to_openmm()
            forcefield.createSystem(omm_topology)

    return forcefield


def save_pdb_selection(input_pdb_path, atom_indices, output_pdb_path):
    """
    Write out only the chosen atoms of a PDB file.

    Parameters
    ----------
    input_pdb_path : str
        Path to the input PDB file.
    atom_indices : iterable of int
        0-based indices of the atoms to keep, as numbered in the input.
    output_pdb_path : str
        Path to write the selection to.

    Notes
    -----
    A selection matching nothing is warned about rather than raised, and
    produces an empty PDB.
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
    """
    try:
        os.remove(file_path)
    except OSError:
        pass


def move_pdb_to_origin(input_pdb, output_filename):
    """
    Shift a structure so its centroid sits at the origin.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_filename : str
        Path to write the shifted structure to.

    See Also
    --------
    center_in_box : Centres in a periodic box rather than on the origin.
    """
    pdb = PDBFile(input_pdb)
    positions = pdb.getPositions(asNumpy=True)
    center = np.mean(positions, axis=0)
    new_positions = positions - center

    with open(output_filename, 'w') as f:
        PDBFile.writeFile(pdb.topology, new_positions, f)


def center_in_box(modeller):
    """
    Shift a system so its centroid sits at the centre of its periodic box.

    A solute left near a box face interacts with its own periodic image, so
    this is worth doing before solvating.  If the topology carries no box at
    all the positions are left alone.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller whose positions are shifted in place.

    See Also
    --------
    move_pdb_to_origin : Centres on the origin rather than in a box.
    """
    pos_list = modeller.positions.value_in_unit(unit.nanometer)

    try:
        pos_nm = np.asarray(pos_list, dtype=float)
        if pos_nm.ndim != 2 or pos_nm.shape[1] != 3:
            raise ValueError
    except Exception:
        # Fallback for position sequences that don't convert directly, e.g. Vec3 objects.
        pos_nm = np.array([[getattr(p, 'x', p[0]),
                            getattr(p, 'y', p[1]),
                            getattr(p, 'z', p[2])] for p in pos_list], dtype=float)

    centroid = pos_nm.mean(axis=0)
    box_vec = None

    if hasattr(modeller.topology, 'getUnitCellDimensions'):
        dims = modeller.topology.getUnitCellDimensions()
        if dims is not None:
            box_vec = np.array([dims.x, dims.y, dims.z])

    # Fall back to the periodic box vectors if unit cell dimensions aren't set.
    if box_vec is None and hasattr(modeller.topology, 'getPeriodicBoxVectors'):
        vecs = modeller.topology.getPeriodicBoxVectors()
        if vecs is not None:
            box_vec = np.array([vecs[0].x, vecs[1].y, vecs[2].z])

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
    (A-Z, a-z, 0-9, cycling).

    Parameters
    ----------
    input_file : str
        Path to the input PDB file.
    output_file : str
        Path to the output PDB file with corrected chain IDs.
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
                new_atom_name = _format_pdb_atom_name(element, element_counts[element])
                new_serial = f"{global_atom_serial:>5}"[:5]
                global_atom_serial += 1
                new_line = line[:6] + new_serial + line[11:12] + new_atom_name + line[16:]
                outfile.write(new_line)
            else:
                outfile.write(line)


def convert_xyz_to_pdb(input_file: str, output_file: str, cutoff_multiplier: float = 1.1,
                       index: int = -1) -> int:
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
        Path to the input XYZ file, or any other structure file ASE can read.
    output_file : str
        Path to the output PDB file.
    cutoff_multiplier : float, optional
        Multiplier applied to natural covalent-radius cutoffs when
        determining bonded neighbours. Default is 1.1.
    index : int, optional
        Which frame to convert when the input holds a trajectory. Default is
        -1 (the last frame). Only a single frame is written.

    Returns
    -------
    int
        The number of molecular clusters (connected components) found.

    See Also
    --------
    convert_pdb_to_xyz : The inverse conversion.
    """
    original_atoms = read(input_file, index=index)
    n_atoms = len(original_atoms)

    cutoffs = [c * cutoff_multiplier for c in natural_cutoffs(original_atoms)]
    i, j = neighbor_list('ij', original_atoms, cutoffs)

    adjacency_matrix = csr_matrix((np.ones_like(i), (i, j)), shape=(n_atoms, n_atoms))
    n_clusters, labels = connected_components(csgraph=adjacency_matrix, directed=False)

    # Group atoms by cluster, then order each cluster by the Hill system (C, H, others).
    hill_priority = {'C': 0, 'H': 1}
    symbols = original_atoms.get_chemical_symbols()
    order = sorted(range(n_atoms),
                   key=lambda idx: (labels[idx], hill_priority.get(symbols[idx], 2), symbols[idx]))

    atoms = original_atoms[order]
    sorted_labels = [labels[idx] for idx in order]

    # Re-index the connectivity found above onto the new ordering, rather than
    # paying for a second neighbour-list pass over the sorted atoms.
    old_to_new = np.empty(n_atoms, dtype=int)
    old_to_new[order] = np.arange(n_atoms)
    i, j = old_to_new[i], old_to_new[j]

    unique_labels = list(dict.fromkeys(sorted_labels))
    available_chains = string.ascii_uppercase + string.ascii_lowercase + string.digits
    num_chains = len(available_chains)
    cluster_ids = {}

    for cluster_idx, lbl in enumerate(unique_labels):
        # Chain ID wraps around every 62 clusters, and the residue ID
        # increments only when the chain ID wraps
        chain = available_chains[cluster_idx % num_chains]
        resid = (cluster_idx // num_chains) + 1

        # Unique three-letter residue name (AAA, AAB, ..., AAZ, ABA, ..., ZZZ)
        resname = ''.join(chr(65 + (cluster_idx // 26 ** p) % 26) for p in (2, 1, 0))

        cluster_ids[lbl] = (chain, resid, resname)

    element_counts_per_cluster = defaultdict(int)

    with open(output_file, 'w') as f:
        for idx, atom in enumerate(atoms):
            chain, resid, resname = cluster_ids[sorted_labels[idx]]
            sym = atom.symbol
            x, y, z = atom.position

            # Atom names are made unique within each chain/residue combination
            element_counts_per_cluster[(chain, resid, sym)] += 1
            name = _format_pdb_atom_name(sym, element_counts_per_cluster[(chain, resid, sym)])

            f.write(f"ATOM  {idx + 1:>5} {name} {resname:>3} {chain}{resid:>4}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {sym:>2}\n")

        conect_dict = defaultdict(list)
        for a1, a2 in zip(i, j):
            conect_dict[a1].append(a2)

        for atom_idx in sorted(conect_dict):
            bonded_atoms = sorted(b + 1 for b in conect_dict[atom_idx])

            # CONECT records hold at most four bonded partners each
            for chunk_start in range(0, len(bonded_atoms), 4):
                chunk = ''.join(f"{b:>5}" for b in bonded_atoms[chunk_start:chunk_start + 4])
                f.write(f"CONECT{atom_idx + 1:>5}{chunk}\n")

        f.write("END\n")

    return n_clusters


def convert_pdb_to_xyz(input_file: str, output_file: str, comment: str | None = None) -> int:
    """
    Convert a PDB file to an XYZ file.

    Every ``ATOM`` and ``HETATM`` record contributes one atom, in file order.
    Multi-model PDB files (``MODEL``/``ENDMDL``) produce one XYZ frame per
    model; files without model records produce a single frame. Coordinates are
    passed through unchanged, both formats using Angstrom.

    Parameters
    ----------
    input_file : str
        Path to the input PDB file.
    output_file : str
        Path to the output XYZ file.
    comment : str, optional
        Text for the comment line of every frame. If None, a comment naming
        the source file (and frame number, when there is more than one) is
        generated. Default is None.

    Returns
    -------
    int
        The number of frames written.

    Raises
    ------
    ValueError
        If the input file contains no ``ATOM`` or ``HETATM`` records.

    See Also
    --------
    convert_xyz_to_pdb : The inverse conversion.
    """
    frames = []
    symbols = []
    positions = []

    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                symbols.append(_element_from_pdb_line(line))
                positions.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            elif line.startswith("ENDMDL") and symbols:
                frames.append((symbols, positions))
                symbols, positions = [], []

    # Catch the trailing model, and the single-frame case with no model records
    if symbols:
        frames.append((symbols, positions))

    if not frames:
        raise ValueError(f"No ATOM or HETATM records found in {input_file!r}.")

    source = os.path.basename(input_file)
    with open(output_file, 'w') as f:
        for frame_idx, (frame_symbols, frame_positions) in enumerate(frames, start=1):
            if comment is not None:
                text = comment
            elif len(frames) > 1:
                text = f"Frame: {frame_idx} of {len(frames)}, Source: {source}"
            else:
                text = f"Source: {source}"
            _write_xyz_frame(f, frame_symbols, frame_positions, text)

    print(f"Wrote {len(frames)} frame(s) to {output_file}", flush=True)
    return len(frames)


def convert_xyz_to_plumed_ref(xyz_file, template_pdb, output_file, atom_line='HETATM'):
    """
    Convert a reaction path from XYZ into the reference PLUMED's PATHMSD reads.

    Each XYZ frame becomes one model of a multi-model PDB, written by taking
    the template's atom records and substituting the frame's coordinates into
    the columns that hold them.  Going through a template rather than writing
    fresh records is what keeps the atom names, residues and numbering
    identical to the structure the simulation will be aligned against; PLUMED
    matches the two by atom serial, and any disagreement is fatal.  Both files
    are renumbered on the way out to guarantee that.

    Parameters
    ----------
    xyz_file : str
        Path to the input XYZ file, holding one frame per path image.
    template_pdb : str
        Path to the template PDB, normally the ``index_atoms.pdb`` written by
        :func:`save_only_index_atoms`. Renumbered in place.
    output_file : str
        Path to write the multi-model PDB to.
    atom_line : str or tuple of str, optional
        Record type the template's atoms are written under; only these lines
        are carried into the output. Default is ``'HETATM'``, which is what
        OpenMM writes for ligand-like residues.

    See Also
    --------
    openmmnqe.path.path_from_steered_md : Builds the path from steered MD.
    """
    with open(template_pdb, 'r') as f:
        template_lines = [line for line in f if line.startswith(atom_line) or line.startswith('TER')]

    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        return

    num_atoms = int(lines[0].strip())
    frames = []
    for i in range(0, len(lines), num_atoms + 2):
        frame_coords = lines[i + 2: i + num_atoms + 2]
        frames.append([l.split()[1:] for l in frame_coords])

    with open(output_file, 'w') as f:
        f.write("REMARK TYPE=MULTI-ST-PDB\n")
        f.write("REMARK ARG=path.s,path.z\n")

        for i, frame in enumerate(frames):
            f.write(f"REMARK NUMBER={i + 1}\n")
            f.write(f"REMARK STEP={i}\n")

            # Tracks position within `frame` separately from `template_lines`,
            # since TER lines consume the latter but not the former.
            coord_idx = 0

            for t_line in template_lines:
                if t_line.startswith('TER'):
                    f.write(t_line)
                else:
                    coords = frame[coord_idx]
                    new_line = (t_line[:30] +
                                f"{float(coords[0]):8.3f}{float(coords[1]):8.3f}{float(coords[2]):8.3f}" +
                                t_line[54:])
                    f.write(new_line)
                    coord_idx += 1

            f.write("ENDMDL\n")
    pdb_remove_ter_index(template_pdb, template_pdb)
    pdb_remove_ter_index(output_file, output_file)


def save_only_index_atoms(modeller, idx_list, file_idx='index_atoms.pdb'):
    """
    Write out only the chosen atoms of a Modeller.

    This is how the alignment template for a path collective variable is
    made: the atoms written here are the ones ``FIT_TO_TEMPLATE`` and
    ``PATHMSD`` see, and the file is what :func:`convert_xyz_to_plumed_ref`
    and :func:`openmmnqe.path.path_from_steered_md` take their atom records
    from. The input Modeller is left untouched.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the full system.
    idx_list : iterable of int
        0-based indices of the atoms to keep.
    file_idx : str, optional
        Path to write to. Default is ``'index_atoms.pdb'``, which is the
        name the PLUMED inputs in :mod:`openmmnqe.plumed` reference.
    """
    modeller_new = app.Modeller(modeller.topology, modeller.positions)
    atoms_to_keep = [atom for atom in modeller_new.topology.atoms() if atom.index in idx_list]
    modeller_new.delete([atom for atom in modeller_new.topology.atoms() if atom not in atoms_to_keep])
    with open(file_idx, 'w') as f:
        app.PDBFile.writeFile(modeller_new.topology, modeller_new.positions, f)


def pdb_remove_ter_index(input_path, output_path):
    """
    Renumber the atom serials of a PDB file, keeping TER and CONECT in step.

    Atoms are renumbered sequentially from 1, restarting at each model, and
    ``CONECT`` records are rewritten to point at the new serials.  PLUMED
    insists that a reference and its template agree on numbering, which an
    edited PDB usually does not, so both are put through this before being
    handed over.  Records other than these are copied out unchanged.

    Parameters
    ----------
    input_path : str
        Path to the input PDB file.
    output_path : str
        Path to write the renumbered PDB to. May be the input path.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    clean_lines = []
    atom_serial = 1
    # Old serial (as written) -> new right-aligned 5-character serial field
    index_map = {}

    for line in lines:
        if line.startswith(("MODEL", "REMARK NUMBER=")):
            atom_serial = 1
            clean_lines.append(line)

        elif line.startswith(("ATOM  ", "HETATM")):
            old_serial = line[6:11].strip()
            index_map.setdefault(old_serial, f"{atom_serial:>5}")
            new_line = line[:6] + f"{atom_serial:5d}" + line[11:]
            clean_lines.append(new_line)
            atom_serial += 1

        elif line.startswith("TER"):
            # A TER shares the serial of the atom that follows it, so this
            # deliberately does not advance the counter.
            if len(line) >= 11:
                new_line = line[:6] + f"{atom_serial:5d}" + line[11:]
            else:
                new_line = f"{line.strip():<6}{atom_serial:5d}\n"
            clean_lines.append(new_line)

        elif line.startswith("CONECT"):
            new_conect = line[:6]
            for i in range(6, len(line.strip()), 5):
                old_idx = line[i:i + 5].strip()
                new_conect += index_map.get(old_idx, line[i:i + 5])
            clean_lines.append(new_conect.rstrip() + "\n")

        else:
            clean_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(clean_lines)


def strip_hydrogens_keep_indices(input_pdb, output_pdb, keep=None):
    """
    Remove hydrogen atoms from a PDB file, except for a chosen subset.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Path to write the filtered PDB file.
    keep : iterable of int, optional
        0-based atom indices (matching PDB serial number minus one) of
        hydrogens to retain even though they would otherwise be stripped.
        If None, all hydrogens are removed. Default is None.
    """
    if keep is None:
        keep = set()
    else:
        keep = set([i + 1 for i in keep])
    with open(input_pdb, 'r') as fin, open(output_pdb, 'w') as fout:
        for line in fin:
            if line.startswith("MODEL"):
                fout.write(line)
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                element = line[76:78].strip().upper()
                atom_idx = int(line[6:11].strip())
                if not element:
                    atom_name = line[12:16].strip()
                    if atom_name.startswith('H') or (atom_name[0].isdigit() and 'H' in atom_name[:2]):
                        element = 'H'
                is_hydrogen = (element == 'H')
                if not is_hydrogen:
                    fout.write(line)
                else:
                    if atom_idx in keep:
                        fout.write(line)
            else:
                fout.write(line)
