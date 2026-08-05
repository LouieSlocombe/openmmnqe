import copy
import os
import re
import tempfile
from pathlib import Path
from typing import Union

import geodesic_interpolate as gi
import numpy as np
import pandas as pd
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS
from scipy.interpolate import CubicSpline


def orca_calc_preset(orca_path=None,
                     directory=None,
                     calc_type='DFT',
                     xc='r2SCAN-3c',
                     charge=0,
                     multiplicity=1,
                     basis_set='',
                     n_procs=1,
                     f_solv=False,
                     f_disp=False,
                     atom_list=None,
                     calc_extra=None,
                     blocks_extra=None,
                     scf_option=None):
    """
    Build an ASE ORCA calculator from a small set of common presets.

    Assembles the ORCA "simple input" line and block section for one of a
    few common calculation types, so callers do not have to hand-write ORCA
    input syntax for routine DFT, MP2, CCSD(T) or QM/XTB2 jobs.

    Parameters
    ----------
    orca_path : str or None, optional
        Path to the ORCA executable. If None, read from the ``ORCA_PATH``
        environment variable. Default is None.
    directory : str or None, optional
        Working directory for ORCA's input/output files. If None, a new
        temporary directory is created. Default is None.
    calc_type : str, optional
        One of ``'DFT'``, ``'MP2'``, ``'CCSD'`` or ``'QM/XTB2'``, each
        building the corresponding ORCA method keyword(s). Any other value
        is passed straight through as the ORCA method keyword. Default is
        ``'DFT'``.
    xc : str, optional
        Exchange-correlation functional, used for ``'DFT'`` and as the QM
        region's method for ``'QM/XTB2'``. Default is ``'r2SCAN-3c'``.
    charge : int, optional
        Total charge. Default is 0.
    multiplicity : int, optional
        Spin multiplicity. Values above 1 switch ``'DFT'``/``'QM/XTB2'`` to
        a ``UKS`` reference and ``'MP2'``/``'CCSD'`` to a ``UKS`` reference
        as well. Default is 1.
    basis_set : str, optional
        Basis set keyword, appended to the simple input line (and, for
        ``'MP2'``/``'CCSD'``, also used as the auxiliary ``/C`` basis).
        Default is ``''``.
    n_procs : int, optional
        Number of MPI processes requested via ``%pal``. Default is 1.
    f_solv : bool or str, optional
        Implicit solvation via CPCM/SMD. ``True`` uses water; a string
        names an explicit SMD solvent; ``False``/``None`` disables
        solvation. Default is False.
    f_disp : bool or str, optional
        Dispersion correction. ``True`` uses ``D4``; a string is used as
        the dispersion keyword directly; ``False``/``None`` disables it.
        Default is False.
    atom_list : str or None, optional
        ORCA atom-selection string for the QM region (without braces),
        used only when *calc_type* is ``'QM/XTB2'``. Default is None.
    calc_extra : str or None, optional
        Extra text appended to the simple input line, e.g. ``'TIGHTOPT'``.
        Default is None.
    blocks_extra : str or None, optional
        Extra ORCA block text appended after ``%pal``/``%CPCM``. Ignored
        when *calc_type* is ``'QM/XTB2'``, which builds its own blocks
        instead. Default is None.
    scf_option : str or None, optional
        Extra SCF-related keyword appended to the simple input line.
        Default is None.

    Returns
    -------
    ase.calculators.orca.ORCA
        A configured ORCA calculator requesting an energy and gradient
        (``EnGrad``).
    """
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    if directory is None:
        directory = os.path.join(tempfile.mkdtemp(), 'orca')

    profile = OrcaProfile(command=orca_path)
    if n_procs > 1:
        inpt_procs = '%pal nprocs {} end'.format(n_procs)
    else:
        inpt_procs = ''

    if f_solv is not None and f_solv is not False:
        if f_solv:
            f_solv = 'WATER'
        inpt_solv = '''
                                              %CPCM SMD TRUE
                                                  SMDSOLVENT "{}"
                                              END'''.format(f_solv)
    else:
        inpt_solv = ''

    if f_disp is None or f_disp is False:
        inpt_disp = ''
    else:
        if f_disp:
            f_disp = 'D4'
        inpt_disp = f_disp

    if atom_list is not None and calc_type == 'QM/XTB2':
        atom_list = '{' + atom_list + '}'
        inpt_xtb = f'''
        %QMMM QMATOMS {atom_list} END END
                   '''
    else:
        inpt_xtb = ''

    if blocks_extra is None:
        blocks_extra = ''

    inpt_blocks = inpt_procs + inpt_solv + blocks_extra

    if calc_type == 'DFT':
        inpt_simple = '{} {} {}'.format(xc, inpt_disp, basis_set)
    elif calc_type == 'MP2':
        inpt_simple = 'DLPNO-{} {} {}/C'.format(calc_type, basis_set, basis_set)
    elif calc_type == 'CCSD':
        inpt_simple = 'DLPNO-{}(T) {} {}/C'.format(calc_type, basis_set, basis_set)
    elif calc_type == 'QM/XTB2':
        inpt_simple = '{} {} {} {}'.format(calc_type, xc, inpt_disp, basis_set)
        inpt_blocks = inpt_procs + inpt_solv + inpt_xtb
    else:
        inpt_simple = '{} {}'.format(calc_type, basis_set)

    if multiplicity > 1:
        if calc_type == 'DFT' or calc_type == 'QM/XTB2':
            inpt_simple = 'UKS  ' + inpt_simple
        elif calc_type == 'MP2' or calc_type == 'CCSD':
            inpt_simple = 'UKS ' + inpt_simple

    if scf_option is not None:
        inpt_simple += ' ' + scf_option

    if calc_extra is not None:
        inpt_simple += ' ' + calc_extra

    calc = ORCA(
        profile=profile,
        charge=charge,
        mult=multiplicity,
        directory=directory,
        orcasimpleinput=inpt_simple + ' EnGrad',
        orcablocks=inpt_blocks
    )
    return calc


def orca_optimise_atoms(atoms,
                        charge=0,
                        multiplicity=1,
                        orca_path=None,
                        xc='r2SCAN-3c',
                        basis_set='',
                        tight_opt=True,
                        tight_scf=False,
                        f_solv=False,
                        f_disp=False,
                        n_procs=1):
    """
    Optimise a geometry at the DFT level with ORCA.

    Builds a DFT calculator via :func:`orca_calc_preset` with an
    ``OPT``/``TIGHTOPT`` keyword, runs it in a scratch directory, and reads
    back the final geometry from ORCA's own optimisation trajectory.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry. Its ``calc`` is set to the ORCA calculator as a
        side effect.
    charge : int, optional
        Total charge. Default is 0.
    multiplicity : int, optional
        Spin multiplicity. Default is 1.
    orca_path : str or None, optional
        Path to the ORCA executable. If None, read from the ``ORCA_PATH``
        environment variable. Default is None.
    xc : str, optional
        Exchange-correlation functional. Default is ``'r2SCAN-3c'``.
    basis_set : str, optional
        Basis set keyword. Default is ``''``.
    tight_opt : bool, optional
        If True, use ``TIGHTOPT`` instead of ``OPT``. Default is True.
    tight_scf : bool, optional
        If True, add ``TIGHTSCF`` to the optimisation keywords. Default is
        False.
    f_solv : bool or str, optional
        Implicit solvation; see :func:`orca_calc_preset`. Default is False.
    f_disp : bool or str, optional
        Dispersion correction; see :func:`orca_calc_preset`. Default is
        False.
    n_procs : int, optional
        Number of MPI processes requested via ``%pal``. Default is 1.

    Returns
    -------
    ase.Atoms
        The optimised geometry, read from ORCA's ``orca.xyz`` output.
    """
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    else:
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        opt_option = 'TIGHTOPT'
    else:
        opt_option = 'OPT'

    if tight_scf:
        calc_extra = f'{opt_option} TIGHTSCF'
    else:
        calc_extra = f'{opt_option}'

    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, "orca.xyz")
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra)
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        return read(orca_file, format="xyz")


def _extract_conformer_info(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Parse the conformer ensemble table from an ORCA GOAT output file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the ORCA ``.out`` file from a GOAT run.

    Returns
    -------
    pandas.DataFrame
        One row per conformer, with columns ``Conformer`` (integer index),
        ``Energy_kcal_mol`` and ``Percent_total``.

    Raises
    ------
    ValueError
        If no ensemble table could be found in the file.
    """
    line_pat = re.compile(
        r"""^\s*
            (?P<conformer>\d+)\s+          # integer index
            (?P<energy>-?\d+\.\d+)\s+      # energy in kcal/mol
            \d+\s+                         # degeneracy (ignored)
            (?P<ptotal>\d+\.\d+)\s+        # % total
            \d+\.\d+\s*?$                  # % cumulative (ignored)
        """,
        re.VERBOSE,
    )
    header_pat = re.compile(r"Conformer\s+Energy.*% total", re.I)
    rows = []
    in_table = False
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not in_table and header_pat.search(line):
                in_table = True
                continue

            if in_table:
                if line.strip() == "" or line.strip().startswith("Conformers"):
                    break
                m = line_pat.match(line)
                if m:
                    rows.append(
                        (
                            int(m["conformer"]),
                            float(m["energy"]),
                            float(m["ptotal"]),
                        )
                    )
    if not rows:
        raise ValueError(
            "Could not locate ensemble table. Check that the file is complete."
        )

    return pd.DataFrame(
        rows, columns=["Conformer", "Energy_kcal_mol", "Percent_total"]
    )


def orca_calculate_goat(atoms,
                        charge=0,
                        multiplicity=1,
                        orca_path=None,
                        n_procs=1):
    """
    Run ORCA's GOAT conformer search and collect the resulting ensemble.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry for the conformer search.
    charge : int, optional
        Total charge. Default is 0.
    multiplicity : int, optional
        Spin multiplicity. Default is 1.
    orca_path : str or None, optional
        Path to the ORCA executable. If None, read from the ``ORCA_PATH``
        environment variable. Default is None.
    n_procs : int, optional
        Number of MPI processes requested via ``%pal``. Default is 1.

    Returns
    -------
    atoms : list of ase.Atoms
        Every conformer in the final ensemble, in the order ORCA wrote
        them.
    df : pandas.DataFrame
        Conformer energies and populations; see
        :func:`_extract_conformer_info`.
    """
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    else:
        orca_path = os.path.abspath(orca_path)
    profile = OrcaProfile(command=orca_path)
    if n_procs > 1:
        inpt_procs = '%pal nprocs {} end'.format(n_procs)
    else:
        inpt_procs = ''

    with tempfile.TemporaryDirectory() as temp_dir:
        calc = ORCA(
            profile=profile,
            charge=charge,
            mult=multiplicity,
            directory=temp_dir,
            orcasimpleinput='GOAT',
            orcablocks=inpt_procs
        )
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        xyz_file = os.path.join(temp_dir, "orca.finalensemble.xyz")
        orca_file = os.path.join(temp_dir, "orca.out")

        df = _extract_conformer_info(orca_file)
        atoms = read(xyz_file, format="xyz", index=':')
        return atoms, df


def get_neb_path(images):
    """
    Compute the cumulative arc length along a sequence of structures.

    The distance between consecutive images is the Frobenius norm of the
    difference between their Cartesian coordinate arrays; the cumulative sum
    of these steps gives a reaction-coordinate-like x-axis for plotting an
    NEB or IRC energy profile.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Structures along the path, e.g. NEB or IRC images.

    Returns
    -------
    numpy.ndarray
        Cumulative arc length at each image, starting at 0 for the first
        image.
    """
    positions = [atoms.positions for atoms in images]
    path = [0] + [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    return np.cumsum(path)


def stitch_path(path1, path2, f_reverse_path=False):
    """
    Join two half-paths that share a common endpoint into one path.

    Typical use is combining two IRC (intrinsic reaction coordinate) runs
    that both start from the transition state: *path1* is reversed so it
    runs away from the shared point, and *path2* is appended after dropping
    its first frame (the same point *path1* now ends on), giving a single
    trajectory from one end to the other through the shared point.

    Parameters
    ----------
    path1 : sequence
        First path, given in the order that starts at the shared point.
    path2 : sequence
        Second path, also given in the order that starts at the shared
        point.
    f_reverse_path : bool, optional
        If True, reverse the entire stitched path before returning it.
        Default is False.

    Returns
    -------
    list
        The stitched path, ``list(path1)[::-1] + list(path2)[1:]`` (or its
        reverse).
    """
    irc = list(path1)[::-1] + list(path2)[1:]
    if f_reverse_path:
        irc = irc[::-1]
    return irc


def resample_path(path, n_resample):
    """
    Resample a path of structures onto evenly spaced points along its arc length.

    Cubic-spline-interpolates the Cartesian coordinates of every atom as a
    function of cumulative arc length (see :func:`get_neb_path`), then
    evaluates the spline at *n_resample* evenly spaced points. The first and
    last images are copied through unchanged; only the interior images are
    interpolated.

    Parameters
    ----------
    path : sequence of ase.Atoms
        Structures to resample.
    n_resample : int
        Number of images in the resampled path, including the endpoints.

    Returns
    -------
    list of ase.Atoms
        The resampled path, each interior image a copy of ``path[0]`` with
        interpolated positions.
    """
    path_distance = get_neb_path(path)
    path_interp = np.linspace(0, path_distance[-1], n_resample)
    positions = np.array([image.positions for image in path])
    positions_interp = CubicSpline(path_distance, positions)(path_interp)
    irc_resampled = [path[0]]
    for ii in range(1, n_resample - 1):
        atoms = path[0].copy()
        atoms.positions = positions_interp[ii, :, :]
        irc_resampled.append(atoms)
    irc_resampled.append(path[-1])
    return irc_resampled


def optimise_geom(atoms, calc,
                  fmax=0.01,
                  steps=1000,
                  opti_traj='opti.traj'):
    """
    Relax a structure to its nearest local minimum with BFGS.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry. Not modified in place; a copy is optimised.
    calc : ase.calculators.calculator.Calculator
        ASE calculator providing energies and forces.
    fmax : float, optional
        Force convergence threshold, in eV/Angstrom. Default is 0.01.
    steps : int, optional
        Maximum number of optimiser steps. Default is 1000.
    opti_traj : str, optional
        Path for the optimisation trajectory file; removed after the final
        geometry is read back. Default is ``'opti.traj'``.

    Returns
    -------
    ase.Atoms
        The relaxed geometry.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    BFGS(atoms, trajectory=opti_traj).run(fmax=fmax, steps=steps)
    atoms = read(opti_traj, index=-1)
    os.remove(opti_traj)
    return atoms


def optimise_reactant_product(reactant, product, calc,
                              fmax=0.01,
                              steps=1000,
                              reactant_opti='reactant_opti.traj',
                              product_opti='product_opti.traj'):
    """
    Relax a reactant and a product geometry with the same calculator.

    Parameters
    ----------
    reactant, product : ase.Atoms
        Starting geometries to relax.
    calc : ase.calculators.calculator.Calculator
        ASE calculator providing energies and forces.
    fmax : float, optional
        Force convergence threshold, in eV/Angstrom. Default is 0.01.
    steps : int, optional
        Maximum number of optimiser steps. Default is 1000.
    reactant_opti, product_opti : str, optional
        Trajectory file paths passed to :func:`optimise_geom` for each
        structure. Default is ``'reactant_opti.traj'`` and
        ``'product_opti.traj'``.

    Returns
    -------
    reactant, product : ase.Atoms
        The relaxed geometries.
    """
    print('Optimising reactant...', flush=True)
    reactant = optimise_geom(reactant, calc,
                             fmax=fmax,
                             steps=steps,
                             opti_traj=reactant_opti)

    print('Optimizing product...', flush=True)
    product = optimise_geom(product, calc,
                            fmax=fmax,
                            steps=steps,
                            opti_traj=product_opti)
    return reactant, product


def prepare_neb(reactant, product, calc,
                n_images=5,
                climb=True,
                rm_ro_trans=True,
                geo_int=True,
                k=2.0):
    """
    Build an ASE NEB band of images between a reactant and a product.

    Intermediate images are generated either by geodesic interpolation
    (``geodesic_interpolate.geodesic_interpolate``), which tends to give a
    better starting band for bond-breaking/forming reactions, or by ASE's
    own linear-plus-IDPP interpolation. Every image is given its own copy of
    *calc* and has its energy evaluated once before the band is returned, so
    the caller can inspect it before running an optimiser.

    Parameters
    ----------
    reactant, product : ase.Atoms
        Endpoint geometries for the band.
    calc : ase.calculators.calculator.Calculator
        ASE calculator to attach (via ``copy.copy``) to every image.
    n_images : int, optional
        Total number of images in the band, including the endpoints.
        Default is 5.
    climb : bool, optional
        Whether to enable the climbing-image variant once the band is
        optimised. Default is True.
    rm_ro_trans : bool, optional
        Whether the NEB should remove overall rotation and translation
        between images. Default is True.
    geo_int : bool, optional
        If True, interpolate the intermediate images with
        ``geodesic_interpolate``. If False, use ASE's ``NEB.interpolate``
        followed by an IDPP refinement. Default is True.
    k : float, optional
        Spring constant between images. Default is 2.0.

    Returns
    -------
    ase.mep.NEB
        The prepared (but not yet optimised) NEB band.
    """
    neb_images = [reactant]
    for ii in range(n_images - 2):
        neb_images.append(reactant.copy())
    neb_images.append(product)

    if geo_int:
        neb_images = gi.geodesic_interpolate(neb_images, n_images=n_images)

    for image in neb_images:
        image.calc = copy.copy(calc)
        image.get_potential_energy()

    neb = NEB(neb_images,
              climb=climb,
              remove_rotation_and_translation=rm_ro_trans,
              k=k)
    if not geo_int:
        neb.interpolate()
        neb.interpolate("idpp")
    return neb


def optimise_neb(neb,
                 fmax=0.01,
                 steps=1000,
                 ts_traj='ts.traj',
                 n_images=5):
    """
    Optimise a prepared NEB band with BFGS.

    Parameters
    ----------
    neb : ase.mep.NEB
        The band to optimise, e.g. from :func:`prepare_neb`.
    fmax : float, optional
        Force convergence threshold, in eV/Angstrom. Default is 0.01.
    steps : int, optional
        Maximum number of optimiser steps. Default is 1000.
    ts_traj : str, optional
        Path for the optimisation trajectory file. Default is
        ``'ts.traj'``.
    n_images : int, optional
        Number of images in *neb*; used to read back exactly the final band
        (the last *n_images* frames of the trajectory) rather than every
        step of every image. Default is 5.

    Returns
    -------
    list of ase.Atoms
        The optimised band images.
    """
    BFGS(neb, trajectory=ts_traj).run(fmax=fmax, steps=steps)
    return read(ts_traj, index=f"-{n_images}:")


def get_ts_image(neb_images, calc):
    """
    Pick the highest-energy image of a band as the transition-state guess.

    Parameters
    ----------
    neb_images : sequence of ase.Atoms
        Images making up the band, e.g. an optimised NEB.
    calc : ase.calculators.calculator.Calculator
        ASE calculator to attach (via ``copy.copy``) to every image before
        evaluating its energy.

    Returns
    -------
    ase.Atoms
        The image with the highest potential energy.
    """
    for image in neb_images:
        image.calc = copy.copy(calc)
    index = np.argmax([image.get_potential_energy() for image in neb_images])
    return neb_images[index]


def quick_guess_path(reactant, product, n_images=25):
    """
    Geodesic-interpolate a quick path guess between two structures.

    A cheap first look at a reaction path with no optimisation; see
    :func:`prepare_neb` for a band that can then be refined with NEB.

    Parameters
    ----------
    reactant, product : ase.Atoms
        Endpoint geometries.
    n_images : int, optional
        Number of images in the path, including the endpoints. Default is
        25.

    Returns
    -------
    list of ase.Atoms
        The interpolated path.
    """
    return gi.geodesic_interpolate([reactant, product], n_images=n_images)


def quick_guess_ts(reactant, product, n_images=25):
    """
    Geodesic-interpolate a quick transition-state guess between two structures.

    Takes the middle image of :func:`quick_guess_path`, which is a cheap
    first guess at the transition state before any real optimisation.

    Parameters
    ----------
    reactant, product : ase.Atoms
        Endpoint geometries.
    n_images : int, optional
        Number of images the interpolated path is built with before its
        midpoint is taken. Default is 25.

    Returns
    -------
    ase.Atoms
        The middle image of the interpolated path.
    """
    atoms_ts = gi.geodesic_interpolate([reactant, product], n_images=n_images)
    atoms_ts = atoms_ts[n_images // 2]
    return atoms_ts
