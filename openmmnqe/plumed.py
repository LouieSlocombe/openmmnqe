import numpy as np
import openmm.unit as unit

from .tools import atom_indices_to_plumed, distance_between_atoms


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
                     grid_bin=200):
    """
    Generate a PLUMED input script for a single proton-transfer CV with metadynamics.

    Constructs a 1-D coordination-difference collective variable for a
    donor–proton–acceptor triplet, adds upper-wall and lower-angle restraints,
    and configures well-tempered metadynamics.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions, used to
        compute inter-atomic distances for setting ``R_0`` and wall values.
    idx : list of int
        Three 0-based atom indices ``[donor, proton, acceptor]``.
    temperature : openmm.unit.Quantity
        Simulation temperature (with units of kelvin).
    r_0 : float, optional
        Multiplicative factor applied to the shortest donor/acceptor–proton
        distance to obtain the COORDINATION ``R_0``. Default is 1.1.
    wall : float, optional
        Multiplicative factor applied to the donor–acceptor distance to set
        the upper-wall position. Default is 1.5.
    angle_lim : float, optional
        Lower-wall angle limit in degrees. Default is 130.0.
    pace : int, optional
        METAD deposition pace in steps. Default is 500.
    height : float, optional
        Initial Gaussian height in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width in nm. Default is 0.05.
    bias : float, optional
        Bias factor for well-tempered metadynamics. Default is 20.0.
    grid_min : float, optional
        Minimum grid value for the CV. Default is -1.1.
    grid_max : float, optional
        Maximum grid value for the CV. Default is 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.

    Returns
    -------
    plumed_input : str
        The PLUMED input script as a string.
    sum_hills_input : str
        The ``plumed sum_hills`` command line for post-processing.
    """
    r_01 = distance_between_atoms(modeller, idx[0], idx[1])
    r_21 = distance_between_atoms(modeller, idx[2], idx[1])
    r_02 = distance_between_atoms(modeller, idx[0], idx[2])
    r_0 = np.round(min(r_01, r_21) * r_0, decimals=2)
    wall = np.round(r_02 * wall, decimals=2)

    # Convert atom indices to PLUMED format
    idx = atom_indices_to_plumed(idx)

    # Convert angle limit to radians and round
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    c_d: COORDINATION GROUPA={idx[0]} GROUPB={idx[1]} R_0={r_0}
    c_a: COORDINATION GROUPA={idx[2]} GROUPB={idx[1]} R_0={r_0}
    pt_cv: COMBINE ARG=c_d,c_a COEFFICIENTS=1,-1 PERIODIC=NO

    dist_da: DISTANCE ATOMS={idx[2]},{idx[0]}
    uwall: UPPER_WALLS ARG=dist_da AT={wall} KAPPA=3000
    
    ang_1: ANGLE ATOMS={idx[2]},{idx[1]},{idx[0]}
    w_1: LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA=500.0

    metad: METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}
    PRINT ARG=c_d,c_a,pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
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
                        grid_bin=200):
    """
    Generate a PLUMED input for two proton-transfer sites combined into a 1-D CV.

    Two coordination-difference CVs are computed (one per donor–proton–acceptor
    triplet) and averaged into a single 1-D collective variable. Upper-wall
    and lower-angle restraints are applied to each site. Well-tempered
    metadynamics is configured on the combined CV.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    idx1 : list of int
        Three 0-based atom indices ``[donor, proton, acceptor]`` for the
        first proton-transfer site.
    idx2 : list of int
        Three 0-based atom indices ``[donor, proton, acceptor]`` for the
        second proton-transfer site.
    temperature : openmm.unit.Quantity
        Simulation temperature (with units of kelvin).
    r_0 : float, optional
        Factor for computing COORDINATION ``R_0``. Default is 1.1.
    wall : float, optional
        Factor for the upper-wall distance. Default is 1.5.
    angle_lim : float, optional
        Lower-wall angle limit in degrees. Default is 130.0.
    pace : int, optional
        METAD deposition pace in steps. Default is 500.
    height : float, optional
        Initial Gaussian height in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width in nm. Default is 0.05.
    bias : float, optional
        Bias factor for well-tempered metadynamics. Default is 20.0.
    grid_min : float, optional
        Minimum grid value. Default is -1.1.
    grid_max : float, optional
        Maximum grid value. Default is 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.

    Returns
    -------
    plumed_input : str
        The PLUMED input script as a string.
    sum_hills_input : str
        The ``plumed sum_hills`` command line for post-processing.
    """
    r1_01 = distance_between_atoms(modeller, idx1[0], idx1[1])
    r1_21 = distance_between_atoms(modeller, idx1[2], idx1[1])
    r1_02 = distance_between_atoms(modeller, idx1[0], idx1[2])
    r1_0 = np.round(min(r1_01, r1_21) * r_0, decimals=2)

    r2_01 = distance_between_atoms(modeller, idx2[0], idx2[1])
    r2_21 = distance_between_atoms(modeller, idx2[2], idx2[1])
    r2_02 = distance_between_atoms(modeller, idx2[0], idx2[2])
    r2_0 = np.round(min(r2_01, r2_21) * r_0, decimals=2)

    wall = np.round(max(r1_02, r2_02) * wall, decimals=2)

    # Convert atom indices to PLUMED format
    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)
    # Convert angle limit to radians and round
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    # Proton transfer 1
    c_d1: COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r1_0}
    c_a1: COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r1_0}
    cv_diff1: COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1,-1 PERIODIC=NO
    # Limits
    dist_da_1: DISTANCE ATOMS={idx1[2]},{idx1[0]}
    u_wall_1: UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA=3000
    ang_1: ANGLE ATOMS={idx1[2]},{idx1[1]},{idx1[0]}
    w_1: LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA=500.0

    # Proton transfer 2
    c_d2: COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r2_0}
    c_a2: COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r2_0}
    cv_diff2: COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1,-1 PERIODIC=NO
    # Limits
    dist_da_2: DISTANCE ATOMS={idx2[2]},{idx2[0]}
    u_wall_2: UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA=3000
    ang_2: ANGLE ATOMS={idx2[2]},{idx2[1]},{idx2[0]}
    w_2: LOWER_WALLS ARG=ang_2 AT={angle_lim} KAPPA=500.0

    pt_cv: COMBINE ARG=cv_diff1,cv_diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

    metad: METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}
    PRINT ARG=pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
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
                        grid_bin=200):
    """
    Generate a PLUMED input for two proton-transfer sites as a 2-D metadynamics CV.

    Two independent coordination-difference CVs are computed (one per
    donor–proton–acceptor triplet) and used as the two dimensions of a 2-D
    well-tempered metadynamics simulation. Upper-wall and lower-angle
    restraints are applied to each site.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    idx1 : list of int
        Three 0-based atom indices ``[donor, proton, acceptor]`` for the
        first proton-transfer site.
    idx2 : list of int
        Three 0-based atom indices ``[donor, proton, acceptor]`` for the
        second proton-transfer site.
    temperature : openmm.unit.Quantity
        Simulation temperature (with units of kelvin).
    r_0 : float, optional
        Factor for computing COORDINATION ``R_0``. Default is 1.1.
    wall : float, optional
        Factor for the upper-wall distance. Default is 1.5.
    angle_lim : float, optional
        Lower-wall angle limit in degrees. Default is 130.0.
    pace : int, optional
        METAD deposition pace in steps. Default is 500.
    height : float, optional
        Initial Gaussian height in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width in nm. Default is 0.05.
    bias : float, optional
        Bias factor for well-tempered metadynamics. Default is 20.0.
    grid_min : float, optional
        Minimum grid value for each CV dimension. Default is -1.1.
    grid_max : float, optional
        Maximum grid value for each CV dimension. Default is 1.1.
    grid_bin : int, optional
        Number of grid bins per dimension. Default is 200.

    Returns
    -------
    plumed_input : str
        The PLUMED input script as a string.
    sum_hills_input : str
        The ``plumed sum_hills`` command line for post-processing.
    """
    r1_01 = distance_between_atoms(modeller, idx1[0], idx1[1])
    r1_21 = distance_between_atoms(modeller, idx1[2], idx1[1])
    r1_02 = distance_between_atoms(modeller, idx1[0], idx1[2])
    r1_0 = np.round(min(r1_01, r1_21) * r_0, decimals=2)

    r2_01 = distance_between_atoms(modeller, idx2[0], idx2[1])
    r2_21 = distance_between_atoms(modeller, idx2[2], idx2[1])
    r2_02 = distance_between_atoms(modeller, idx2[0], idx2[2])
    r2_0 = np.round(min(r2_01, r2_21) * r_0, decimals=2)

    wall = np.round(max(r1_02, r2_02) * wall, decimals=2)

    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)
    # Convert angle limit to radians and round
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    # Proton transfer 1
    c_d1: COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r1_0}
    c_a1: COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r1_0}
    cv_diff1: COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1,-1 PERIODIC=NO
    # Limits
    dist_da_1: DISTANCE ATOMS={idx1[2]},{idx1[0]}
    u_wall_1: UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA=3000
    ang_1: ANGLE ATOMS={idx1[2]},{idx1[1]},{idx1[0]}
    w_1: LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA=500.0
    
    # Proton transfer 2
    c_d2: COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r2_0}
    c_a2: COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r2_0}
    cv_diff2: COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1,-1 PERIODIC=NO
    dist_da_2: DISTANCE ATOMS={idx2[2]},{idx2[0]}
    u_wall_2: UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA=3000
    ang_2: ANGLE ATOMS={idx2[2]},{idx2[1]},{idx2[0]}
    w_2: LOWER_WALLS ARG=ang_2 AT={angle_lim} KAPPA=500.0

    metad: METAD ARG=cv_diff1,cv_diff2 PACE={pace} HEIGHT={height} SIGMA={sigma},{sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min},{grid_min} GRID_MAX={grid_max},{grid_max} GRID_BIN={grid_bin},{grid_bin}
    PRINT ARG=cv_diff1,cv_diff2,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min},{grid_min} --max {grid_max},{grid_max} --bin {grid_bin},{grid_bin} --kt {kt_str}'
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
    """
    Generate a PLUMED input for a Wobble (wob) base-pair proton-transfer CV (variant 1).

    Constructs coordination-difference CVs for two donor–proton–acceptor
    triplets using fixed ``R_0`` values (in nm). Upper-wall and lower-angle
    restraints are applied. The combined CV is biased with well-tempered
    metadynamics.

    Parameters
    ----------
    idx1 : list of int
        Three 0-based atom indices ``[donor, proton, acceptor]`` for the
        first proton-transfer site.
    idx2 : list of int
        Three 0-based atom indices ``[donor, proton, acceptor]`` for the
        second proton-transfer site.
    temperature : openmm.unit.Quantity
        Simulation temperature (with units of kelvin).
    r_0 : float, optional
        COORDINATION ``R_0`` in nm. Default is 0.14.
    wall : float, optional
        Upper-wall distance in nm. Default is 0.4.
    angle_lim : float, optional
        Lower-wall angle limit in degrees. Default is 100.0.
    pace : int, optional
        METAD deposition pace in steps. Default is 500.
    height : float, optional
        Initial Gaussian height in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width in nm. Default is 0.05.
    bias : float, optional
        Bias factor for well-tempered metadynamics. Default is 20.0.
    grid_min : float, optional
        Minimum grid value. Default is -1.1.
    grid_max : float, optional
        Maximum grid value. Default is 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.

    Returns
    -------
    plumed_input : str
        The PLUMED input script as a string.
    sum_hills_input : str
        The ``plumed sum_hills`` command line for post-processing.
    """
    # Convert atom indices to PLUMED format
    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)
    # Convert angle limit to radians and round
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    c_d1: COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r_0}
    c_a1: COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r_0}
    cv_diff1: COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1.0,-1.0 PERIODIC=NO

    dist_da_1: DISTANCE ATOMS={idx1[2]},{idx1[0]}
    u_wall_1: UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA=3000

    ang_1: ANGLE ATOMS={idx1[2]},{idx1[1]},{idx1[0]}
    w_1: LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA=500.0

    c_d2: COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r_0}
    c_a2: COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r_0}
    cv_diff2: COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1.0,-1.0 PERIODIC=NO

    dist_da_2: DISTANCE ATOMS={idx2[2]},{idx2[0]}
    u_wall_2: UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA=3000

    ang_2: ANGLE ATOMS={idx2[2]},{idx2[1]},{idx2[0]}
    w_2: LOWER_WALLS ARG=ang_2 AT={angle_lim} KAPPA=500.0

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
    """
    Generate a PLUMED input for a multi-site Wobble proton-transfer CV (variant 2).

    Constructs five coordination-difference sub-CVs (z1–z5) from eight atom
    indices representing a multi-step proton relay. The sub-CVs are summed
    into a single collective variable ``z`` and biased with well-tempered
    metadynamics. Upper-wall distance restraints are applied to key
    heavy-atom pairs.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions, used to
        compute inter-atomic distances for ``R_0`` values.
    idx : list of int
        Eight 0-based atom indices in the order
        ``[N3, H3, O6, O4, N1, H1, O2, N2]``.
    temperature : openmm.unit.Quantity
        Simulation temperature (with units of kelvin).
    r_0 : float, optional
        Multiplicative factor for computing COORDINATION ``R_0`` from
        inter-atomic distances. Default is 1.1.
    wall : float, optional
        Upper-wall distance in nm for heavy-atom pairs. Default is 4.0.
    pace : int, optional
        METAD deposition pace in steps. Default is 500.
    height : float, optional
        Initial Gaussian height in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width in nm. Default is 0.05.
    bias : float, optional
        Bias factor for well-tempered metadynamics. Default is 20.0.
    grid_min : float, optional
        Minimum grid value. Default is -1.1.
    grid_max : float, optional
        Maximum grid value. Default is 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.

    Returns
    -------
    plumed_input : str
        The PLUMED input script as a string.
    sum_hills_input : str
        The ``plumed sum_hills`` command line for post-processing.
    """
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

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
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

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
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
                       grid_bin=200):
    # Convert atom indices to PLUMED format
    idx = atom_indices_to_plumed(idx)
    # Unpack indices
    n3, h3, o6, o4, n1, h1, o2, n2 = idx
    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
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

metad: METAD ARG=z PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS
PRINT ARG=z,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --bin {grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input
