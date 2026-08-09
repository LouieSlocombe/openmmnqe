"""Reference paths for path collective variables, estimated from steered MD.

A ``PATHMSD`` simulation needs a reference path: an ordered set of frames
walking the system from reactant to product.  The rest of this package builds
that path with a nudged elastic band on the ML potential, which is accurate but
needs both endpoints optimised up front.  The functions here take the cheaper
route of dragging a collective variable across with a moving harmonic restraint
(see :func:`openmmnqe.plumed.plumed_input_steered`) and picking frames out of
the trajectory that leaves behind.  The result is a path in the full simulation
environment -- solvent included -- rather than one interpolated in vacuum.

The typical sequence is

1. :func:`openmmnqe.plumed.plumed_input_steered_pt` writes the pulling script,
2. :func:`openmmnqe.openmm.run_openmm_steered` runs it,
3. :func:`path_from_steered_md` selects the frames and writes ``neb_path.pdb``,
4. :func:`openmmnqe.plumed.plumed_input_neb_path` biases along it.

Note that the path atoms should belong to a single molecule.  A trajectory is
written with molecules wrapped into the periodic box, so a path spanning two
molecules -- the two bases of a pair, say -- can have them wrapped to opposite
sides of the box, which no amount of alignment repairs.
"""
import os

import mdtraj as md
import numpy as np
from reactiontools import read_plumed_file

from .io import _write_xyz_frame, convert_xyz_to_plumed_ref
from .plumed import estimate_path_lambda


def cv_from_colvar(colvar_file, n_frames, cv_name=None):
    """
    Read a CV from a PLUMED COLVAR file, one value per trajectory frame.

    PLUMED and OpenMM disagree about when to write: ``PRINT`` fires at step 0
    and every stride thereafter, while a reporter first fires one interval in.
    So a COLVAR written with the same stride as the reporter holds exactly one
    row more than the trajectory has frames, and its first row is dropped
    here.  Any other length is resampled onto the frames by assuming both
    cover the same span of time at a constant rate.

    Parameters
    ----------
    colvar_file : str
        Path to the COLVAR file written by the steered run.
    n_frames : int
        Number of frames in the trajectory the values are wanted for.
    cv_name : str or None, optional
        Field name of the CV, e.g. ``'pt_cv'``. If None, the first column
        after ``time`` is used. Default is None.

    Returns
    -------
    numpy.ndarray
        CV value for each trajectory frame, of length *n_frames*.
    """
    colvar = read_plumed_file(colvar_file)
    cv = colvar.column(cv_name if cv_name is not None else 1)

    if cv.size == n_frames:
        return cv
    if cv.size == n_frames + 1:
        # The extra row is the one PLUMED wrote at step 0, before the
        # trajectory had a frame to go with it.
        return cv[1:]

    print(f"COLVAR has {cv.size} rows for {n_frames} frames, resampling.", flush=True)
    # Frame i sits at (i + 1) / n_frames of the way through the run.
    frame_fraction = (np.arange(n_frames) + 1.0) / n_frames
    colvar_fraction = np.arange(cv.size) / (cv.size - 1.0)
    return np.interp(frame_fraction, colvar_fraction, cv)


def _nearest_monotone(series, targets):
    """
    Pick the entry of *series* closest to each target, never going backwards.

    Searching the whole series for each target would let a noisy trajectory
    hand back frames out of order, which is no use as a path.  Each search
    therefore starts after the previously chosen entry, and stops early enough
    to leave one entry for every target still to come.

    Parameters
    ----------
    series : array_like
        Values to search, one per frame.
    targets : array_like
        Values to match, in the order they should appear along the path.

    Returns
    -------
    numpy.ndarray
        Strictly increasing indices into *series*, one per target.

    Raises
    ------
    ValueError
        If *series* is shorter than *targets*.
    """
    series = np.asarray(series, dtype=float)
    n_targets = len(targets)
    if series.size < n_targets:
        raise ValueError(f"Cannot pick {n_targets} frames from a series of {series.size}.")

    picks = []
    low = 0
    for i, target in enumerate(targets):
        # Leave one frame behind for each target that has not been placed yet
        high = series.size - (n_targets - i - 1)
        window = series[low:high]
        picks.append(low + int(np.argmin(np.abs(window - target))))
        low = picks[-1] + 1
    return np.asarray(picks)


def select_frames_by_cv(cv, n_images, cv_start=None, cv_stop=None):
    """
    Choose the frames that are evenly spaced along a collective variable.

    Parameters
    ----------
    cv : array_like
        CV value for each frame of the trajectory.
    n_images : int
        Number of frames to select.
    cv_start, cv_stop : float or None, optional
        Ends of the CV range to cover. If None, the first and last values in
        *cv* are used. Default is None.

    Returns
    -------
    numpy.ndarray
        Indices of the selected frames, in path order.
    """
    cv = np.asarray(cv, dtype=float)
    start = cv[0] if cv_start is None else cv_start
    stop = cv[-1] if cv_stop is None else cv_stop
    return _nearest_monotone(cv, np.linspace(start, stop, n_images))


def select_frames_by_msd(xyz, n_images):
    """
    Choose the frames that are evenly spaced along the trajectory itself.

    Distance is measured as the RMSD between consecutive frames accumulated
    along the trajectory, which is the spacing ``PATHMSD`` cares about.  Use
    this when there is no COLVAR to select on, and bear in mind that thermal
    jitter inflates the arc length of a noisy trajectory.

    Parameters
    ----------
    xyz : numpy.ndarray
        Coordinates with shape ``(n_frames, n_atoms, 3)``, already aligned.
    n_images : int
        Number of frames to select.

    Returns
    -------
    numpy.ndarray
        Indices of the selected frames, in path order.
    """
    step = np.sqrt(np.mean(np.sum(np.diff(xyz, axis=0) ** 2, axis=-1), axis=-1))
    arc = np.concatenate(([0.0], np.cumsum(step)))
    return _nearest_monotone(arc, np.linspace(0.0, arc[-1], n_images))


def _smooth_frames(xyz, picks, window):
    """
    Average each selected frame with its neighbours to damp thermal noise.

    Parameters
    ----------
    xyz : numpy.ndarray
        Aligned coordinates with shape ``(n_frames, n_atoms, 3)``.
    picks : array_like of int
        Indices of the selected frames.
    window : int
        Number of frames either side to average over. Zero returns the
        selected frames untouched.

    Returns
    -------
    numpy.ndarray
        Coordinates of the path, with shape ``(len(picks), n_atoms, 3)``.
    """
    if window <= 0:
        return xyz[picks]

    smoothed = np.empty((len(picks),) + xyz.shape[1:])
    for i, frame in enumerate(picks):
        low = max(0, frame - window)
        high = min(xyz.shape[0], frame + window + 1)
        smoothed[i] = xyz[low:high].mean(axis=0)
    return smoothed


def path_from_steered_md(traj_file,
                         template_pdb='index_atoms.pdb',
                         output_file='neb_path.pdb',
                         colvar_file='COLVAR_SMD',
                         cv_name=None,
                         n_images=15,
                         atom_indices=None,
                         top=None,
                         cv_start=None,
                         cv_stop=None,
                         smooth=0,
                         align=True,
                         atom_line='HETATM'):
    """
    Estimate a path collective variable from a steered MD trajectory.

    Frames evenly spaced along the CV are pulled out of the trajectory,
    aligned, and written as the multi-model PDB that ``PATHMSD`` reads, in the
    same format :func:`openmmnqe.io.convert_xyz_to_plumed_ref` produces for a
    NEB path.  An XYZ copy is written alongside it for viewing.

    Parameters
    ----------
    traj_file : str
        Trajectory from the steered run, e.g. the ``smd_steps.pdb`` written by
        :func:`openmmnqe.openmm.run_openmm_steered`.
    template_pdb : str, optional
        PDB holding the path atoms, as written by
        :func:`openmmnqe.io.save_only_index_atoms`. Its atom records are the
        template for the output. Default is ``'index_atoms.pdb'``. Note that
        this file is renumbered in place so that it and the path agree on
        atom numbering, which is what PLUMED expects of them.
    output_file : str, optional
        Multi-model PDB to write the path to. Default is ``'neb_path.pdb'``,
        which is what the ``PATHMSD`` inputs in :mod:`openmmnqe.plumed`
        reference.
    colvar_file : str or None, optional
        COLVAR file from the steered run. If None, frames are spaced by RMSD
        along the trajectory instead of by CV. Default is ``'COLVAR_SMD'``.
    cv_name : str or None, optional
        Field name of the CV in *colvar_file*. If None, the first column after
        ``time`` is used. Default is None.
    n_images : int, optional
        Number of frames in the path. Default is 15; ``PATHMSD`` behaves best
        with 15 to 30.
    atom_indices : list of int or None, optional
        Atoms of the full system that make up the path, normally the same
        indices passed to :func:`openmmnqe.io.save_only_index_atoms`. If None,
        every atom in the trajectory is used. Default is None.
    top : str or None, optional
        Topology file for trajectory formats that carry none, such as DCD.
        Default is None.
    cv_start, cv_stop : float or None, optional
        Ends of the CV range the path should span. If None, the first and last
        values in the COLVAR are used, i.e. the whole pull. Default is None.
    smooth : int, optional
        Number of neighbouring frames either side to average each path frame
        with. Default is 0, which keeps the frames as they were sampled. Try 2
        or 3 if the path comes out jagged.
    align : bool, optional
        Whether to superpose every frame on the first before selecting.
        Default is True.
    atom_line : str or tuple of str, optional
        Record type the template's atoms are written under. Default is
        ``'HETATM'``, which is what OpenMM writes for the ligand-like
        residues these paths normally cover.

    Returns
    -------
    float
        The LAMBDA value recommended for this path, as reported by
        :func:`openmmnqe.plumed.estimate_path_lambda`.

    Raises
    ------
    ValueError
        If the path atoms and the template PDB do not match, or the trajectory
        holds fewer frames than the path needs.
    """
    if atom_indices is not None:
        # Sorting keeps the frames in topology order, which is the order
        # save_only_index_atoms wrote the template in.
        atom_indices = np.asarray(sorted(atom_indices), dtype=int)

    # Only formats that carry no topology of their own accept `top`
    load_kwargs = {'top': top} if top is not None else {}
    traj = md.load(traj_file, atom_indices=atom_indices, **load_kwargs)
    n_template = md.load(template_pdb).n_atoms
    if traj.n_atoms != n_template:
        raise ValueError(f"Path has {traj.n_atoms} atoms but {template_pdb} has {n_template}. "
                         "Pass the atom indices the template was written from.")
    with open(template_pdb, 'r') as handle:
        n_records = sum(1 for line in handle if line.startswith(atom_line))
    if n_records != n_template:
        # Only lines of this record type make it into the output, so a
        # mismatch here would quietly write a path with atoms missing.
        raise ValueError(f"{template_pdb} holds {n_template} atoms but {n_records} {atom_line} "
                         "records. Set atom_line to the record type it uses.")
    if traj.n_frames < n_images:
        raise ValueError(f"Trajectory has {traj.n_frames} frames, too few for {n_images} images. "
                         "Report the steered run more often, or ask for fewer images.")

    if align:
        traj.superpose(traj, 0)

    if colvar_file is not None:
        cv = cv_from_colvar(colvar_file, traj.n_frames, cv_name=cv_name)
        picks = select_frames_by_cv(cv, n_images, cv_start=cv_start, cv_stop=cv_stop)
        print(f"Path spans CV {cv[picks[0]]:.3f} to {cv[picks[-1]]:.3f}", flush=True)
    else:
        picks = select_frames_by_msd(traj.xyz, n_images)

    print(f"Selected frames {picks.tolist()} of {traj.n_frames}", flush=True)
    positions = _smooth_frames(traj.xyz, picks, smooth) * 10.0  # nm to Angstrom

    symbols = [atom.element.symbol if atom.element is not None else atom.name[:2]
               for atom in traj.topology.atoms]
    xyz_file = f"{os.path.splitext(output_file)[0]}.xyz"
    with open(xyz_file, 'w') as handle:
        for i, frame in enumerate(positions):
            _write_xyz_frame(handle, symbols, frame, comment=f"steered MD path image {i + 1}")

    convert_xyz_to_plumed_ref(xyz_file, template_pdb, output_file, atom_line=atom_line)
    print(f"Wrote {n_images} path images to {output_file}", flush=True)
    return estimate_path_lambda(output_file)
