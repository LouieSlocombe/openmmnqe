import os
import numpy as np
import openmm.unit as unit

from .tools import atom_indices_to_plumed, distance_between_atoms, temperature_to_kbt

fes_cmd = os.path.join(os.path.dirname(os.path.realpath(__file__)), "opes", "FES_from_State.py")


def plumed_input_1pt(modeller,
                     idx,
                     temperature,
                     r_0=1.1,
                     wall=1.5,
                     angle_lim=130.0,
                     pace=500,
                     height=15.0,  # kJ/mol
                     sigma=0.05,  # nm
                     bias=20.0,
                     grid_min=-1.1,
                     grid_max=1.1,
                     grid_bin=200,
                     kappa=500.0,
                     f_opes=False):
    # Distances
    r_01 = distance_between_atoms(modeller, idx[0], idx[1])
    r_21 = distance_between_atoms(modeller, idx[2], idx[1])
    r_02 = distance_between_atoms(modeller, idx[0], idx[2])
    r_0 = np.round(min(r_01, r_21) * r_0, decimals=2)

    # Limits
    wall = np.round(r_02 * wall, decimals=2)
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    # Convert atom indices to PLUMED format
    idx = atom_indices_to_plumed(idx)

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    if f_opes:
        metad_line = f"metad:      OPES_METAD ARG=pt_cv PACE={pace} BARRIER={height} SIGMA={sigma} TEMP={temperature_str} STATE_WFILE=STATE STATE_WSTRIDE={pace}"
        sum_hills_input = f'python3 {fes_cmd} --state STATE --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
    else:
        metad_line = f"metad:      METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}"
        sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'

    plumed_input = f"""
# Proton transfer
c_d:        COORDINATION GROUPA={idx[0]} GROUPB={idx[1]} R_0={r_0}
c_a:        COORDINATION GROUPA={idx[2]} GROUPB={idx[1]} R_0={r_0}
pt_cv:      COMBINE ARG=c_d,c_a COEFFICIENTS=1,-1 PERIODIC=NO

# Limits
dist_da:    DISTANCE ATOMS={idx[2]},{idx[0]}
dist_wall:  UPPER_WALLS ARG=dist_da AT={wall} KAPPA={kappa}
ang_1:      ANGLE ATOMS={idx[2]},{idx[1]},{idx[0]}
ang_wall:   LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA={kappa}

# Metadynamics
{metad_line}
PRINT       ARG=c_d,c_a,pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    return plumed_input, sum_hills_input


def plumed_input_2pt_1d(modeller,
                        idx1,
                        idx2,
                        temperature,
                        r_0=1.1,
                        wall=1.5,
                        angle_lim=130.0,
                        pace=500,
                        height=15.0,  # kJ/mol
                        sigma=0.05,  # nm
                        bias=20.0,
                        grid_min=-1.1,
                        grid_max=1.1,
                        grid_bin=200,
                        kappa=500.0,
                        f_opes=False):
    # Proton transfer 1
    r1_01 = distance_between_atoms(modeller, idx1[0], idx1[1])
    r1_21 = distance_between_atoms(modeller, idx1[2], idx1[1])
    r1_02 = distance_between_atoms(modeller, idx1[0], idx1[2])
    r1_0 = np.round(min(r1_01, r1_21) * r_0, decimals=2)

    # Proton transfer 2
    r2_01 = distance_between_atoms(modeller, idx2[0], idx2[1])
    r2_21 = distance_between_atoms(modeller, idx2[2], idx2[1])
    r2_02 = distance_between_atoms(modeller, idx2[0], idx2[2])
    r2_0 = np.round(min(r2_01, r2_21) * r_0, decimals=2)

    # Limits
    wall = np.round(max(r1_02, r2_02) * wall, decimals=2)
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    # Convert atom indices to PLUMED format
    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    if f_opes:
        metad_line = f"metad:      OPES_METAD ARG=pt_cv PACE={pace} BARRIER={height} SIGMA={sigma} TEMP={temperature_str} STATE_WFILE=STATE STATE_WSTRIDE={pace}"
        sum_hills_input = f'python3 {fes_cmd} --state STATE --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
    else:
        metad_line = f"metad:      METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}"
        sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'

    plumed_input = f"""
# Proton transfer 1
c_d1:       COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r1_0}
c_a1:       COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r1_0}
cv_diff1:   COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1,-1 PERIODIC=NO

# Limits
dist_da_1:  DISTANCE ATOMS={idx1[2]},{idx1[0]}
u_wall_1:   UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA={kappa}
ang_1:      ANGLE ATOMS={idx1[2]},{idx1[1]},{idx1[0]}
w_1:        LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA={kappa}

# Proton transfer 2
c_d2:       COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r2_0}
c_a2:       COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r2_0}
cv_diff2:   COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1,-1 PERIODIC=NO

# Limits
dist_da_2:  DISTANCE ATOMS={idx2[2]},{idx2[0]}
u_wall_2:   UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA={kappa}
ang_2:      ANGLE ATOMS={idx2[2]},{idx2[1]},{idx2[0]}
w_2:        LOWER_WALLS ARG=ang_2 AT={angle_lim} KAPPA={kappa}

# Combine the two proton transfers into a single CV
pt_cv:      COMBINE ARG=cv_diff1,cv_diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

# Metadynamics
{metad_line}
PRINT       ARG=pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    return plumed_input, sum_hills_input


def plumed_input_2pt_2d(modeller,
                        idx1,
                        idx2,
                        temperature,
                        r_0=1.1,
                        wall=1.5,
                        angle_lim=130.0,
                        pace=500,
                        height=15.0,  # kJ/mol
                        sigma=0.05,  # nm
                        bias=20.0,
                        grid_min=-1.1,
                        grid_max=1.1,
                        grid_bin=200,
                        kappa=500.0,
                        f_opes=False):
    # Proton transfer 1
    r1_01 = distance_between_atoms(modeller, idx1[0], idx1[1])
    r1_21 = distance_between_atoms(modeller, idx1[2], idx1[1])
    r1_02 = distance_between_atoms(modeller, idx1[0], idx1[2])
    r1_0 = np.round(min(r1_01, r1_21) * r_0, decimals=2)

    # Proton transfer 2
    r2_01 = distance_between_atoms(modeller, idx2[0], idx2[1])
    r2_21 = distance_between_atoms(modeller, idx2[2], idx2[1])
    r2_02 = distance_between_atoms(modeller, idx2[0], idx2[2])
    r2_0 = np.round(min(r2_01, r2_21) * r_0, decimals=2)

    # Limits
    wall = np.round(max(r1_02, r2_02) * wall, decimals=2)
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    # Convert atom indices to PLUMED format
    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    if f_opes:
        metad_line = f"metad:      OPES_METAD ARG=cv_diff1,cv_diff2 PACE={pace} BARRIER={height} SIGMA={sigma},{sigma} TEMP={temperature_str} STATE_WFILE=STATE STATE_WSTRIDE={pace}"
        sum_hills_input = f'python3 {fes_cmd} --state STATE --min {grid_min},{grid_min} --max {grid_max},{grid_max} --bin {grid_bin},{grid_bin} --kt {kt_str}'
    else:
        metad_line = f"metad:      METAD ARG=cv_diff1,cv_diff2 PACE={pace} HEIGHT={height} SIGMA={sigma},{sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min},{grid_min} GRID_MAX={grid_max},{grid_max} GRID_BIN={grid_bin},{grid_bin}"
        sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min},{grid_min} --max {grid_max},{grid_max} --bin {grid_bin},{grid_bin} --kt {kt_str}'

    plumed_input = f"""
# Proton transfer 1
c_d1:       COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r1_0}
c_a1:       COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r1_0}
cv_diff1:   COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1,-1 PERIODIC=NO

# Limits
dist_da_1:  DISTANCE ATOMS={idx1[2]},{idx1[0]}
u_wall_1:   UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA={kappa}
ang_1:      ANGLE ATOMS={idx1[2]},{idx1[1]},{idx1[0]}
w_1:        LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA={kappa}

# Proton transfer 2
c_d2:       COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r2_0}
c_a2:       COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r2_0}
cv_diff2:   COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1,-1 PERIODIC=NO

# Limits
dist_da_2:  DISTANCE ATOMS={idx2[2]},{idx2[0]}
u_wall_2:   UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA={kappa}
ang_2:      ANGLE ATOMS={idx2[2]},{idx2[1]},{idx2[0]}
w_2:        LOWER_WALLS ARG=ang_2 AT={angle_lim} KAPPA={kappa}

# Metadynamics
{metad_line}
PRINT       ARG=cv_diff1,cv_diff2,metad.bias STRIDE={pace} FILE=COLVAR
        """
    return plumed_input, sum_hills_input


def plumed_input_wob_1(idx1,
                       idx2,
                       temperature,
                       r_0=0.14,  # nm
                       wall=0.4,  # nm
                       angle_lim=100.0,
                       pace=500,
                       height=15.0,  # kJ/mol
                       sigma=0.05,  # nm
                       bias=20.0,
                       grid_min=-1.1,
                       grid_max=1.1,
                       grid_bin=200):
    # Convert atom indices to PLUMED format
    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)
    # Convert angle limit to radians and round
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)
    plumed_input = f"""
c_d1: COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r_0}
c_a1: COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r_0}
cv_diff1: COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1.0,-1.0 PERIODIC=NO

dist_da_1: DISTANCE ATOMS={idx1[2]},{idx1[0]}
u_wall_1: UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA=500

ang_1: ANGLE ATOMS={idx1[2]},{idx1[1]},{idx1[0]}
w_1: LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA=500

c_d2: COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r_0}
c_a2: COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r_0}
cv_diff2: COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1.0,-1.0 PERIODIC=NO

dist_da_2: DISTANCE ATOMS={idx2[2]},{idx2[0]}
u_wall_2: UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA=500

ang_2: ANGLE ATOMS={idx2[2]},{idx2[1]},{idx2[0]}
w_2: LOWER_WALLS ARG=ang_2 AT={angle_lim} KAPPA=500

pt_cv: COMBINE ARG=cv_diff1,cv_diff2 COEFFICIENTS=1.0,-1.0 PERIODIC=NO

metad: METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}
PRINT ARG=pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input


def plumed_input_wob_2(modeller,
                       idx,
                       temperature,
                       r_0=1.1,
                       wall=4.0,
                       pace=500,
                       height=15.0,  # kJ/mol
                       sigma=0.05,  # nm
                       bias=20.0,
                       grid_min=-1.1,
                       grid_max=1.1,
                       grid_bin=200):
    # Unpack indices
    idx_n3, idx_h3, idx_o6, idx_o4, idx_n1, idx_h1, idx_o2, idx_n2 = idx

    # Calculate r_0 for each distance
    r_1 = np.round(distance_between_atoms(modeller, idx_n3, idx_h3) * r_0, decimals=2)
    r_2 = np.round(distance_between_atoms(modeller, idx_o6, idx_h3) * r_0, decimals=2)
    r_3 = np.round(distance_between_atoms(modeller, idx_o6, idx_h3) * r_0, decimals=2)
    r_4 = np.round(distance_between_atoms(modeller, idx_o4, idx_h3) * r_0, decimals=2)
    r_5 = np.round(distance_between_atoms(modeller, idx_n1, idx_h1) * r_0, decimals=2)
    r_6 = np.round(distance_between_atoms(modeller, idx_n3, idx_h1) * r_0, decimals=2)
    r_7 = np.round(distance_between_atoms(modeller, idx_n1, idx_o2) * r_0, decimals=2)
    r_8 = np.round(distance_between_atoms(modeller, idx_n2, idx_o2) * r_0, decimals=2)
    r_9 = np.round(distance_between_atoms(modeller, idx_n2, idx_o2) * r_0, decimals=2)
    r_10 = np.round(distance_between_atoms(modeller, idx_n3, idx_h3) * r_0, decimals=2)

    # # Convert atom indices to PLUMED format
    # idx_n3 = atom_indices_to_plumed(idx_n3)
    # idx_h3 = atom_indices_to_plumed(idx_h3)
    # idx_o6 = atom_indices_to_plumed(idx_o6)
    # idx_o4 = atom_indices_to_plumed(idx_o4)
    # idx_n1 = atom_indices_to_plumed(idx_n1)
    # idx_h1 = atom_indices_to_plumed(idx_h1)
    # idx_o2 = atom_indices_to_plumed(idx_o2)
    # idx_n2 = atom_indices_to_plumed(idx_n2)

    idx_n3 += 1
    idx_h3 += 1
    idx_o6 += 1
    idx_o4 += 1
    idx_n1 += 1
    idx_h1 += 1
    idx_o2 += 1
    idx_h1 += 1
    idx_n2 += 1

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)
    plumed_input = f"""
# z1: top PT reaction coordinate
c_1: COORDINATION GROUPA={idx_n3} GROUPB={idx_h3} R_0={r_1}
c_2: COORDINATION GROUPA={idx_o6} GROUPB={idx_h3} R_0={r_2}
# c_1: DISTANCE ATOMS={idx_n3},{idx_h3}
# c_2: DISTANCE ATOMS={idx_o6},{idx_h3}
z1: COMBINE ARG=c_1,c_2 COEFFICIENTS=1,-1 PERIODIC=NO

# z2: top PT reaction coordinate
c_3: COORDINATION GROUPA={idx_o6} GROUPB={idx_h3} R_0={r_3}
c_4: COORDINATION GROUPA={idx_o4} GROUPB={idx_h3} R_0={r_4}
# c_3: DISTANCE ATOMS={idx_o6},{idx_h3}
# c_4: DISTANCE ATOMS={idx_o4},{idx_h3}
z2: COMBINE ARG=c_3,c_4 COEFFICIENTS=1,-1 PERIODIC=NO

# z3: second PT reaction coordinate
c_5: COORDINATION GROUPA={idx_n1} GROUPB={idx_h1} R_0={r_5}
c_6: COORDINATION GROUPA={idx_n3} GROUPB={idx_h1} R_0={r_6}
# c_5: DISTANCE ATOMS={idx_n1},{idx_h1}
# c_6: DISTANCE ATOMS={idx_n3},{idx_h1}
z3: COMBINE ARG=c_5,c_6 COEFFICIENTS=1,-1 PERIODIC=NO
    
# z4
c_7: COORDINATION GROUPA={idx_n1} GROUPB={idx_o2} R_0={r_7}
c_8: COORDINATION GROUPA={idx_n2} GROUPB={idx_o2} R_0={r_8}
# c_7: DISTANCE ATOMS={idx_n1},{idx_o2}
# c_8: DISTANCE ATOMS={idx_n2},{idx_o2}
z4: COMBINE ARG=c_7,c_8 COEFFICIENTS=1,-1 PERIODIC=NO

# z5
c_9: COORDINATION GROUPA={idx_n2} GROUPB={idx_o2} R_0={r_9}
c_10: COORDINATION GROUPA={idx_n1} GROUPB={idx_n3} R_0={r_10}
# c_9: DISTANCE ATOMS={idx_n2},{idx_o2} 
# c_10: DISTANCE ATOMS={idx_n1},{idx_n3}
z5: COMBINE ARG=c_9,c_10 COEFFICIENTS=1,-1 PERIODIC=NO

z: COMBINE ARG=z1,z2,z3,z4,z5 COEFFICIENTS=1,1,1,1,1 PERIODIC=NO

d1: DISTANCE ATOMS={idx_o6},{idx_o4} 
d2: DISTANCE ATOMS={idx_n1},{idx_n3}
d3: DISTANCE ATOMS={idx_n2},{idx_o2}

uw1: UPPER_WALLS ARG=d1 AT={wall} KAPPA=500
uw2: UPPER_WALLS ARG=d2 AT={wall} KAPPA=500
uw3: UPPER_WALLS ARG=d3 AT={wall} KAPPA=500

metad: METAD ARG=z PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS
PRINT ARG=z,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --bin {grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input


def plumed_input_wob_3(modeller,
                       idx_o4,
                       idx_h3,
                       idx_o2,
                       idx_o6,
                       idx_n2,
                       idx_nr1,
                       idx_nr2,
                       temperature,
                       r_0=1.1,
                       wall=1.1,
                       pace=500,
                       height=15.0,  # kJ/mol
                       sigma=0.05,  # nm
                       bias=20.0,
                       grid_min=-1.1,
                       grid_max=1.1,
                       grid_bin=200):
    wall_u = np.round(distance_between_atoms(modeller, idx_nr1[0], idx_nr2[0]) * wall, decimals=2)
    wall_l = np.round(distance_between_atoms(modeller, idx_nr1[0], idx_nr2[0]) * (2.0 - wall), decimals=2)

    wall_1 = np.round(distance_between_atoms(modeller, idx_o6[0], idx_o4[0]) * wall, decimals=2)
    wall_2 = np.round(distance_between_atoms(modeller, idx_n2[0], idx_o2[0]) * wall, decimals=2)

    # Proton-transfer CVs
    idx_o4 = atom_indices_to_plumed(idx_o4)[0]
    idx_h3 = atom_indices_to_plumed(idx_h3)[0]
    idx_o2 = atom_indices_to_plumed(idx_o2)[0]

    idx_o6 = atom_indices_to_plumed(idx_o6)[0]
    idx_n2 = atom_indices_to_plumed(idx_n2)[0]

    idx_nr1 = atom_indices_to_plumed(idx_nr1)[0]
    idx_nr2 = atom_indices_to_plumed(idx_nr2)[0]

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)
    plumed_input = f"""
# z1: PT reaction coordinate
c_1: DISTANCE ATOMS={idx_o4},{idx_h3}
c_2: DISTANCE ATOMS={idx_o2},{idx_h3}
z1: COMBINE ARG=c_1,c_2 COEFFICIENTS=1,-1 PERIODIC=NO

# z2: base-pair wobble coordinate
z2: ANGLE ATOMS={idx_n2},{idx_o6},{idx_o4}

# Constraint to bound the sliding of the bases
w_a1: LOWER_WALLS ARG=z2 AT={np.round(np.deg2rad(80.0), decimals=2)} KAPPA=500
w_a2: UPPER_WALLS ARG=z2 AT={np.round(np.deg2rad(150.0), decimals=2)} KAPPA=500

# Constraint to R-groups to keep bases together
d_rr: DISTANCE ATOMS={idx_nr1},{idx_nr2}
w_d1: UPPER_WALLS ARG=d_rr AT={wall_u} KAPPA=500
w_d2: LOWER_WALLS ARG=d_rr AT={wall_l} KAPPA=500

# Constraint to prevent excessive opening of the base pair
d_oo: DISTANCE ATOMS={idx_o6},{idx_o4}
w_oo: UPPER_WALLS ARG=d_oo AT={wall_1} KAPPA=500
d_no: DISTANCE ATOMS={idx_n2},{idx_o2}
w_no: UPPER_WALLS ARG=d_no AT={wall_2} KAPPA=500

z: COMBINE ARG=z1,z2 COEFFICIENTS=1,1 PERIODIC=NO

metad: METAD ARG=z PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS
PRINT ARG=z,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --bin {grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input


def plumed_input_wob_4(modeller,
                       idx,
                       temperature,
                       r_0=1.1,
                       wall=1.1,
                       pace=500,
                       height=15.0,  # kJ/mol
                       sigma=0.05,  # nm
                       bias=20.0,
                       grid_bin=200,
                       kappa=1000.0,
                       f_opes=False):
    rr_u = np.round(distance_between_atoms(modeller, idx[-1], idx[-2]) * wall, decimals=2)
    rr_l = np.round(distance_between_atoms(modeller, idx[-1], idx[-2]) * (2.0 - wall), decimals=2)

    # Convert atom indices to PLUMED format
    idx = atom_indices_to_plumed(idx)
    # Unpack indices
    n3, h3, o6, o4, n1, h1, o2, n2, nr1, nr2 = idx
    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    if f_opes:
        metad_line = f"metad:      OPES_METAD ARG=z PACE={pace} BARRIER={height} SIGMA={sigma} TEMP={temperature_str} STATE_WFILE=STATE STATE_WSTRIDE={pace}"
        sum_hills_input = f'python3 {fes_cmd} --state STATE --bin {grid_bin} --kt {kt_str}'
    else:
        metad_line = f"metad:      METAD ARG=z PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS"
        sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --bin {grid_bin} --kt {kt_str}'

    plumed_input = f"""
# Get the distances
n3_h3: DISTANCE ATOMS={n3},{h3}
o6_h3: DISTANCE ATOMS={o6},{h3}
o4_h3: DISTANCE ATOMS={o4},{h3}
n1_h1: DISTANCE ATOMS={n1},{h1}
n3_h1: DISTANCE ATOMS={n3},{h1}
n1_o2: DISTANCE ATOMS={n1},{o2}
n2_o2: DISTANCE ATOMS={n2},{o2}
n1_n3: DISTANCE ATOMS={n1},{n3}

# Define the CVs
z1: COMBINE ARG=n3_h3,o6_h3 COEFFICIENTS=1,-1 PERIODIC=NO
z2: COMBINE ARG=o6_h3,o4_h3 COEFFICIENTS=1,-1 PERIODIC=NO
z3: COMBINE ARG=n1_h1,n3_h1 COEFFICIENTS=1,-1 PERIODIC=NO
z4: COMBINE ARG=n1_o2,n2_o2 COEFFICIENTS=1,-1 PERIODIC=NO
z5: COMBINE ARG=n1_o2,n1_n3 COEFFICIENTS=1,-1 PERIODIC=NO

# Combine into a single CV
z: COMBINE ARG=z1,z2,z3,z4,z5 COEFFICIENTS=1,1,1,1,1 PERIODIC=NO

# # Constraints to keep the bases together
# nr_dist: DISTANCE ATOMS={nr1},{nr2}
# uw_nr: UPPER_WALLS ARG=nr_dist AT={rr_u} KAPPA={kappa}
# lw_nr: LOWER_WALLS ARG=nr_dist AT={rr_l} KAPPA={kappa}
# 
# # rise restraint
# base1: GROUP ATOMS={nr1},{o6},{n1},{n2}
# base2: GROUP ATOMS={nr2},{o4},{n3},{o2}
# com1: COM ATOMS=base1
# com2: COM ATOMS=base2
# dist: DISTANCE ATOMS=com1,com2 COMPONENTS
# RESTRAINT ARG=dist.z AT=0.0 KAPPA={kappa}

# # Constraints to prevent excessive opening of the base pair
# nr_ang: ANGLE ATOMS={nr1},{o6},{nr2}
# w_a1: LOWER_WALLS ARG=nr_ang AT={np.round(np.deg2rad(120.0), decimals=2)} KAPPA={kappa}
# w_a2: UPPER_WALLS ARG=nr_ang AT={np.round(np.deg2rad(140.0), decimals=2)} KAPPA={kappa}

{metad_line}
PRINT ARG=z,metad.bias STRIDE={pace} FILE=COLVAR
        """
    return plumed_input, sum_hills_input


def plumed_input_neb_path(temperature,
                          wall=0.1,
                          pace=500,
                          height=15.0,  # kJ/mol
                          sigma=0.1,  # nm
                          bias=5.0,
                          grid_min=12.0,
                          grid_max=14.0,
                          grid_bin=200,
                          kappa=500.0,
                          lambda_val=250.0,
                          neigh_size=4,
                          f_opes=False):
    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    if f_opes:
        metad_line = f"metad: OPES_METAD ARG=path.sss PACE={pace} BARRIER={height} SIGMA={sigma} TEMP={temperature_str} STATE_WFILE=STATE STATE_WSTRIDE={pace}"
        sum_hills_input = f'python3 {fes_cmd} --state STATE --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
    else:
        metad_line = f"metad: METAD ARG=path.sss PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}"
        sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'

    plumed_input = f'''
WHOLEMOLECULES ENTITY0=1-9
FIT_TO_TEMPLATE REFERENCE=index_atoms.pdb TYPE=OPTIMAL
path: PATHMSD REFERENCE=neb_path.pdb LAMBDA={lambda_val} NEIGH_SIZE={neigh_size}
{metad_line}
path_limit: UPPER_WALLS ARG=path.zzz AT={wall} KAPPA={kappa}
PRINT ARG=path.sss,path.zzz,metad.bias STRIDE={pace} FILE=COLVAR
        '''
    return plumed_input, sum_hills_input
