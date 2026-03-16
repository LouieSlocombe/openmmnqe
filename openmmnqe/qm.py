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
    positions = [atoms.positions for atoms in images]
    path = [0] + [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    return np.cumsum(path)


def stitch_path(path1, path2, f_reverse_path=False):
    irc = list(path1)[::-1] + list(path2)[1:]
    if f_reverse_path:
        irc = irc[::-1]
    return irc


def resample_path(path, n_resample):
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
    BFGS(neb, trajectory=ts_traj).run(fmax=fmax, steps=steps)
    return read(ts_traj, index=f"-{n_images}:")


def get_ts_image(neb_images, calc):
    for image in neb_images:
        image.calc = copy.copy(calc)
    index = np.argmax([image.get_potential_energy() for image in neb_images])
    return neb_images[index]


def quick_guess_path(reactant, product, n_images=25):
    return gi.geodesic_interpolate([reactant, product], n_images=n_images)


def quick_guess_ts(reactant, product, n_images=25):
    atoms_ts = gi.geodesic_interpolate([reactant, product], n_images=n_images)
    atoms_ts = atoms_ts[n_images // 2]
    return atoms_ts
