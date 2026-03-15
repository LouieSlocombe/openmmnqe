import os
import re
import tempfile
from pathlib import Path
from typing import Union

import pandas as pd
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.io import read


def orca_calc_preset(orca_path=None,
                     directory=None,
                     calc_type='DFT',
                     xc='wB97X',
                     charge=0,
                     multiplicity=1,
                     basis_set='def2-SVP',
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
                        basis_set='def2-QZVP',
                        tight_opt=False,
                        tight_scf=False,
                        f_solv=False,
                        f_disp=False,
                        n_procs=1):
    # Determine the ORCA path
    if orca_path is None:
        # Try to read the path from the environment variable
        orca_path = os.environ.get('ORCA_PATH')
    else:
        # Convert the provided path to an absolute path
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        # Set up geometry optimization and frequency calculation parameters
        opt_option = 'TIGHTOPT'
    else:
        # Set up frequency calculation parameters only
        opt_option = 'OPT'

    if tight_scf:
        # Set up tight SCF convergence parameters
        calc_extra = f'{opt_option} TIGHTSCF'
    else:
        # Use default SCF convergence parameters
        calc_extra = f'{opt_option}'

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, "orca.xyz")

        # Set up the ORCA calculator with the specified parameters
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
        # Assign the calculator to the molecule
        atoms.calc = calc

        # Trigger the calculation to optimise the geometry
        _ = atoms.get_potential_energy()

        # Load the optimised geometry from the ORCA output file
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
            orcasimpleinput='GOAT XTB',
            orcablocks=inpt_procs
        )
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        xyz_file = os.path.join(temp_dir, "orca.finalensemble.xyz")
        orca_file = os.path.join(temp_dir, "orca.out")

        df = _extract_conformer_info(orca_file)
        atoms = read(xyz_file, format="xyz", index=':')
        return atoms, df
