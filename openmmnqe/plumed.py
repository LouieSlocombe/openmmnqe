import os

import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

from .tools import atom_indices_to_plumed


def plumed_input_1pt(idx,
                     temperature,
                     r_0=0.14,  # nm
                     wall=0.4,  # nm
                     pace=500,
                     height=50.0,  # kJ/mol
                     sigma=0.05,  # nm
                     bias=20.0,
                     grid_min=-1.1,
                     grid_max=1.1,
                     grid_bin=200):
    idx = atom_indices_to_plumed(idx)
    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    c_d: COORDINATION GROUPA={idx[0]} GROUPB={idx[1]} R_0={r_0}
    c_a: COORDINATION GROUPA={idx[2]} GROUPB={idx[1]} R_0={r_0}
    pt_cv: COMBINE ARG=c_d,c_a COEFFICIENTS=1,-1 PERIODIC=NO

    dist_da: DISTANCE ATOMS={idx[2]},{idx[0]}
    uwall: UPPER_WALLS ARG=dist_da AT={wall} KAPPA=3000

    metad: METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}
    PRINT ARG=c_d,c_a,pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input


def plumed_input_2pt_1d(idx1,
                        idx2,
                        temperature,
                        r_0=0.14,  # nm
                        wall=0.4,  # nm
                        pace=500,
                        height=50.0,  # kJ/mol
                        sigma=0.05,  # nm
                        bias=20.0,
                        grid_min=-1.1,
                        grid_max=1.1,
                        grid_bin=200):
    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    c_d1: COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r_0}
    c_a1: COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r_0}
    cv_diff1: COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1,-1 PERIODIC=NO

    dist_da_1: DISTANCE ATOMS={idx1[2]},{idx1[0]}
    u_wall_1: UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA=3000

    c_d2: COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r_0}
    c_a2: COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r_0}
    cv_diff2: COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1,-1 PERIODIC=NO

    dist_da_2: DISTANCE ATOMS={idx2[2]},{idx2[0]}
    u_wall_2: UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA=3000

    pt_cv: COMBINE ARG=cv_diff1,cv_diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

    metad: METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}
    PRINT ARG=pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input


def plumed_input_2pt_2d(idx1,
                        idx2,
                        temperature,
                        r_0=0.14,  # nm
                        wall=0.4,  # nm
                        pace=500,
                        height=50.0,  # kJ/mol
                        sigma=0.05,  # nm
                        bias=20.0,
                        grid_min=-1.1,
                        grid_max=1.1,
                        grid_bin=200):
    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    c_d1: COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r_0}
    c_a1: COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r_0}
    cv_diff1: COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1,-1 PERIODIC=NO

    dist_da_1: DISTANCE ATOMS={idx1[2]},{idx1[0]}
    u_wall_1: UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA=3000

    c_d2: COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r_0}
    c_a2: COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r_0}
    cv_diff2: COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1,-1 PERIODIC=NO

    dist_da_2: DISTANCE ATOMS={idx2[2]},{idx2[0]}
    u_wall_2: UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA=3000

    metad: METAD ARG=cv_diff1,cv_diff2 PACE={pace} HEIGHT={height} SIGMA={sigma},{sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min},{grid_min} GRID_MAX={grid_max},{grid_max} GRID_BIN={grid_bin},{grid_bin}
    PRINT ARG=cv_diff1,cv_diff2,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min},{grid_min} --max {grid_max},{grid_max} --bin {grid_bin},{grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input
