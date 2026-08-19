"""Structure edits and file conversion, upstream of every simulation.

The operations that get a structure into the shape a run needs, and that the
workflows call directly: repairing a raw PDB with PDBFixer
(:func:`fix_pdb`), relabelling residues (:func:`relabel_residues_in_pdb`),
deleting residues (:func:`remove_residues_in_pdb`), centring a structure in
its box (:func:`center_in_box`), writing out a selection, and converting
between XYZ, SDF and PDB.

**Ligand parameters are not here.** Turning the non-standard residues of a
structure into an OpenMM force field is
`forcefill <https://github.com/LouieSlocombe/forcefill>`_'s job, and it is a
dependency::

    import forcefill as ff
    import openmm.app as app

    names = ("amber14-all.xml", "amber14/tip3pfb.xml")
    result = ff.build_forcefield_xml(input_pdb, "ligands.xml", base_forcefield=names)
    if result.skipped:
        raise RuntimeError(f"forcefill skipped residues: {result.skipped}")
    extra = [] if result.forcefield_xml is None else [result.forcefield_xml]

    pdb_data = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    forcefield = app.ForceField(*names, *extra)

A skipped residue would otherwise surface later, as a template error at
``createSystem``, and ``forcefield_xml`` is ``None`` when the base files
already matched every residue, which ``ForceField()`` will not take.

forcefill asks the base force field which residues it cannot match, rather
than deciding by residue name, so a structure has to be *repaired* before it
is parameterised -- an unrepaired crystal structure is missing its hydrogens,
which makes every protein residue unmatched too.  :func:`fix_pdb` is that
step, and it stays here because forcefill is subtractive only and will not
repair anything.  Its counterpart for stripping water, buffer ions and
crystallization additives is ``forcefill.clean_pdb``, which knows more residue
names than :func:`remove_residues_in_pdb` does but refuses to touch a standard
residue or write over its own input.

:func:`save_only_index_atoms` is the one writer here that belongs to the
reaction side rather than the preparation side: it writes the
``index_atoms.pdb`` a path collective variable is aligned against. It stays
because deleting atoms from a topology needs OpenMM. The conversions that go
with it -- turning a path into the multi-model PDB ``PATHMSD`` reads, and the
XYZ/PDB handling around it -- live in :mod:`reactiontools.tools_io`.
"""
import glob
import os
import re
import shutil
import string

import numpy as np
import openmm.unit as unit
from openmm.app import PDBFile, Topology, Element, Modeller
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from openmm import app, Vec3
from reactiontools import format_pdb_atom_name


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


def relabel_residues_in_pdb(pdb_file_path, relabel_map, output_file):
    """
    Rename residues in a PDB file according to a mapping.

    Renaming is the only way to move a residue between the standard force
    field and GAFF.  ``forcefill.build_forcefield_xml`` parameterises whatever
    the base force field cannot match and takes no list of ligand names, so a
    residue is forced into GAFF by giving it a name no standard template
    claims, and left to the standard force field by giving it one that does.
    A summary of what changed is printed.

    Parameters
    ----------
    pdb_file_path : str or os.PathLike
        Path to the input PDB file.
    relabel_map : dict
        Mapping from old residue name to new. Names not in it are left be.
    output_file : str, os.PathLike, or file-like object
        Path to write to, or an open handle. May be the input path.

    Returns
    -------
    openmm.app.PDBFile
        The relabelled structure.
    """
    pdb = PDBFile(os.fspath(pdb_file_path))
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
    if isinstance(output_file, (str, os.PathLike)):
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

    The blunt form, and the one to reach for when you know exactly what you
    want gone.  ``forcefill.clean_pdb`` is the considered form: it knows the
    water, bulk-ion and crystallization-additive names by category, keeps
    structural metals, and refuses to delete a standard residue, a residue
    covalently bonded to its neighbours, or to write over its own input.  This
    function does none of that and will delete anything you name.

    Parameters
    ----------
    input_pdb : str or os.PathLike
        Path to the input PDB file.
    output_pdb : str or os.PathLike
        Path to write the result to. May be the input path.
    names : iterable of str
        Residue names to delete, matched exactly.
    """
    pdb = PDBFile(os.fspath(input_pdb))
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


def fix_pdb(file_in, file_out, ph=7.0, rm_heterogens=True):
    """
    Repair a PDB file with PDBFixer.

    Rebuilds missing residues and atoms, substitutes non-standard residues
    for their standard equivalents, and adds hydrogens.  Worth running on a
    raw crystal structure; note that the substitution step is a silent
    structural edit, so it is not something to apply to a structure that is
    already equilibrated.

    **Run this before parameterising, not after.**
    ``forcefill.build_forcefield_xml`` decides what needs parameters by asking
    the base force field what it cannot match, and a protein missing its
    hydrogens matches nothing.  On a bare protein that is fatal: every residue
    is standard-but-unmatched, none can be auto-parameterised, and forcefill
    raises.  With a free ligand alongside, it instead comes back with the
    protein residues in ``result.skipped``, which also suppresses forcefill's
    whole-structure check.  forcefill is subtractive only and will not repair
    anything itself.  The order for a raw structure is repair, then
    ``forcefill.clean_pdb``, then ``build_forcefield_xml``, each writing its
    own file.

    One thing this does *not* reproduce: PDBFixer runs over the whole
    structure, ligand included, so ``replaceNonstandardResidues`` and the pH
    re-protonation apply to the ligand too.  To protect a ligand from that,
    protonate it separately and splice it back -- see forcefill's
    ``examples/prepare_trypsin_ben.py``.

    Parameters
    ----------
    file_in : str or os.PathLike
        Path to the input PDB file.
    file_out : str or os.PathLike
        Path to write the repaired PDB to.
    ph : float, optional
        pH the hydrogens are added at, which decides the protonation of
        titratable residues. Default is 7.0.
    rm_heterogens : bool, optional
        Whether to drop water, ions and ligands. Default is True.
    """
    fixer = PDBFixer(filename=os.fspath(file_in))
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


def convert_sdfs_to_pdb(input_files, output_filename="combined_output.pdb"):
    """
    Convert one or more SDF files into a single combined PDB file.

    Each molecule in each SDF file is assigned a unique chain, residue ID,
    and three-letter residue name. Bonds and 3-D coordinates are preserved.

    Parameters
    ----------
    input_files : str, os.PathLike, or sequence thereof
        Path(s) to the input SDF file(s).
    output_filename : str or os.PathLike, optional
        Path for the output PDB file. Default is ``'combined_output.pdb'``.
    """
    if isinstance(input_files, (str, os.PathLike)):
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
        element_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            element = Element.getBySymbol(symbol)
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
            atom_name = format_pdb_atom_name(symbol, element_counts[symbol])
            omm_atom = combined_topology.addAtom(atom_name, element, residue)
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


def save_pdb_selection(input_pdb_path, atom_indices, output_pdb_path):
    """
    Write out only the chosen atoms of a PDB file.

    Parameters
    ----------
    input_pdb_path : str or os.PathLike
        Path to the input PDB file.
    atom_indices : iterable of int
        0-based indices of the atoms to keep, as numbered in the input.
    output_pdb_path : str or os.PathLike
        Path to write the selection to.

    Notes
    -----
    A selection matching nothing is warned about rather than raised, and
    produces an empty PDB.
    """
    pdb = app.PDBFile(os.fspath(input_pdb_path))
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
    input_pdb : str or os.PathLike
        Path to the input PDB file.
    output_filename : str or os.PathLike
        Path to write the shifted structure to.

    See Also
    --------
    center_in_box : Centres in a periodic box rather than on the origin.
    """
    pdb = PDBFile(os.fspath(input_pdb))
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
    box_center = None
    if hasattr(modeller.topology, 'getPeriodicBoxVectors'):
        vecs = modeller.topology.getPeriodicBoxVectors()
        if vecs is not None:
            box_vectors = np.asarray(vecs.value_in_unit(unit.nanometer), dtype=float)
            box_center = np.sum(box_vectors, axis=0) / 2.0

    if box_center is not None:
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
                res_id = (line[21], line[22:27])
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
                res_id = (line[21], line[22:27])
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
                new_atom_name = format_pdb_atom_name(element, element_counts[element])
                new_serial = f"{global_atom_serial:>5}"[:5]
                global_atom_serial += 1
                new_line = line[:6] + new_serial + line[11:12] + new_atom_name + line[16:]
                outfile.write(new_line)
            else:
                outfile.write(line)


def save_only_index_atoms(modeller, idx_list, file_idx='index_atoms.pdb'):
    """
    Write out only the chosen atoms of a Modeller.

    This is how the alignment template for a path collective variable is
    made: the atoms written here are the ones ``FIT_TO_TEMPLATE`` and
    ``PATHMSD`` see, and the file is what
    :func:`reactiontools.convert_xyz_to_plumed_ref` and
    :func:`reactiontools.path_from_steered_md` take their atom records from.
    The input Modeller is left untouched.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the full system.
    idx_list : iterable of int
        0-based indices of the atoms to keep.
    file_idx : str, optional
        Path to write to. Default is ``'index_atoms.pdb'``, which is the
        name the PLUMED inputs in :mod:`reactiontools.tools_cv` reference.
    """
    modeller_new = app.Modeller(modeller.topology, modeller.positions)
    atoms_to_keep = [atom for atom in modeller_new.topology.atoms() if atom.index in idx_list]
    modeller_new.delete([atom for atom in modeller_new.topology.atoms() if atom not in atoms_to_keep])
    with open(file_idx, 'w') as f:
        app.PDBFile.writeFile(modeller_new.topology, modeller_new.positions, f)
