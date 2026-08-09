import os

import mdtraj as md
import numpy as np
import openmm.unit as unit

from .tools import atom_indices_to_plumed, distance_between_atoms, temperature_to_kbt

fes_cmd = os.path.join(os.path.dirname(os.path.realpath(__file__)), "opes", "FES_from_State.py")


def _metad_and_sumhills(f_opes, arg, pace, height, sigma, bias, temperature_str,
                        kt_str, grid_bin, grid_min=None, grid_max=None,
                        label='metad:      '):
    """
    Build the metadynamics bias line and the matching FES-reconstruction command.

    Parameters
    ----------
    f_opes : bool
        If True, bias with ``OPES_METAD`` (reconstructed with the bundled OPES
        ``FES_from_State.py``); otherwise use well-tempered ``METAD`` with
        ``plumed sum_hills``.
    arg : str
        The PLUMED ``ARG=`` value the bias acts on.
    pace, height, sigma, bias
        Values interpolated into the bias line. For multi-dimensional biases
        pass pre-joined strings (e.g. ``f'{sigma},{sigma}'``).
    temperature_str, kt_str
        Temperature in kelvin and the kBT string for the FES command.
    grid_bin
        Grid bin count (pre-joined string for multi-dimensional biases).
    grid_min, grid_max : optional
        Grid bounds; omitted from both commands when None.
    label : str, optional
        Prefix of the bias line (label plus alignment spacing).

    Returns
    -------
    tuple of (str, str)
        The ``metad_line`` and ``sum_hills_input`` strings.
    """
    fes_grid = f' --min {grid_min} --max {grid_max}' if grid_min is not None else ''
    if f_opes:
        metad_line = (f"{label}OPES_METAD ARG={arg} PACE={pace} BARRIER={height} "
                      f"SIGMA={sigma} TEMP={temperature_str} "
                      f"STATE_WFILE=STATE STATE_WSTRIDE={pace}")
        sum_hills_input = (f'python3 {fes_cmd} --state STATE --outfile fes.dat'
                           f'{fes_grid} --bin {grid_bin} --kt {kt_str}')
    else:
        metad_grid = (f" GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}"
                      if grid_min is not None else '')
        metad_line = (f"{label}METAD ARG={arg} PACE={pace} HEIGHT={height} "
                      f"SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} "
                      f"FILE=HILLS{metad_grid}")
        sum_hills_input = (f'plumed sum_hills --hills HILLS --outfile fes.dat'
                           f'{fes_grid} --bin {grid_bin} --kt {kt_str}')
    return metad_line, sum_hills_input


def estimate_path_lambda(pdb_path: str) -> float:
    """
    Estimate the optimal LAMBDA parameter for PLUMED's PATH collective variable
    based on the mean squared displacement (MSD) between consecutive frames in a
    path (trajectory) provided as a PDB file.

    This function loads a multi-frame PDB file, aligns all frames to the first frame,
    computes the MSD between each pair of consecutive frames, and suggests a LAMBDA
    value for use in PLUMED path-based collective variables. It also prints a summary
    of the path statistics and warnings if the path is unevenly spaced or if the
    recommended LAMBDA is unusually high.

    Parameters
    ----------
    pdb_path : str
        Path to the multi-frame PDB file representing the path.

    Returns
    -------
    float
        The recommended LAMBDA value for PLUMED.

    Prints
    ------
    - Number of frames in the path
    - Average and maximum MSD between frames (in nm^2)
    - Recommended LAMBDA value for PLUMED
    - Warnings if frames are unevenly spaced or if LAMBDA is very high

    Notes
    -----
    - The function assumes the input PDB contains multiple frames (e.g., a NEB path).
    - The MSD is multiplied by 100 to convert from nm^2 to Å^2 before use in the LAMBDA calculation.
    - A typical path should have 15 to 30 frames for best results.
    """
    traj = md.load(pdb_path)
    traj.superpose(traj[0])

    msds = [
        np.mean(np.sum((traj.xyz[i] - traj.xyz[i + 1]) ** 2, axis=1)) * 100
        for i in range(len(traj) - 1)
    ]
    avg_msd = np.mean(msds)
    max_msd = np.max(msds)
    ideal_lambda = 2.3 / avg_msd

    print("--- Path Analysis ---", flush=True)
    print(f"Number of frames: {len(traj)} (aim for 15 to 30)", flush=True)
    print(f"Average MSD between frames: {avg_msd:.6f} nm^2", flush=True)
    print(f"Maximum MSD between frames: {max_msd:.6f} nm^2", flush=True)
    print(f"Recommended LAMBDA for PLUMED: {ideal_lambda:.2f}", flush=True)

    if max_msd > 2 * avg_msd:
        print("WARNING: Your path frames are unevenly spaced.", flush=True)
        print("Consider interpolating your path for better stability.", flush=True)

    if ideal_lambda > 500.0:
        print("WARNING: The recommended LAMBDA is very high", flush=True)

    return ideal_lambda


def _switching_value(r, r_0, nn=6, mm=None):
    """
    Evaluate PLUMED's default rational switching function.

    This is the same function ``COORDINATION`` applies to every pair distance,
    ``s(r) = (1 - (r/r_0)^nn) / (1 - (r/r_0)^mm)``, so it can be used to work
    out what a coordination-based CV is worth for a given geometry without
    running PLUMED.

    Parameters
    ----------
    r : float
        Distance between the two atoms, in the same units as *r_0*.
    r_0 : float
        The ``R_0`` parameter of the switching function.
    nn : int, optional
        Numerator exponent. Default is 6, as in PLUMED.
    mm : int or None, optional
        Denominator exponent. If None, ``2 * nn`` is used, again as in PLUMED.

    Returns
    -------
    float
        The value of the switching function, between 0 and 1.
    """
    mm = 2 * nn if mm is None else mm
    x = (r / r_0) ** nn
    y = (r / r_0) ** mm
    if np.isclose(y, 1.0):
        # r == r_0 makes both halves vanish; the limit there is nn / mm
        return nn / mm
    return (1.0 - x) / (1.0 - y)


def plumed_input_steered(cv_block,
                         cv_start,
                         cv_stop,
                         steps,
                         cv_name='cv',
                         kappa=2000.0,
                         stride=100,
                         steps_equil=0,
                         steps_relax=0,
                         colvar_file='COLVAR_SMD',
                         extra_lines=None):
    """
    Build a PLUMED input that drags a collective variable from one value to
    another with a moving harmonic restraint (steered MD).

    The restraint centre is moved linearly from *cv_start* to *cv_stop* over
    *steps* MD steps, optionally after holding at the starting value for
    *steps_equil* steps and before holding at the final value for
    *steps_relax* steps. The trajectory this produces is a first guess at the
    reaction path, which :func:`openmmnqe.path.path_from_steered_md` turns
    into a reference for ``PATHMSD``.

    Parameters
    ----------
    cv_block : str
        PLUMED lines defining the CV to steer, ending with an action labelled
        *cv_name*.
    cv_start : float
        Value of the CV the restraint starts at, normally the value of the
        reactant.
    cv_stop : float
        Value of the CV the restraint finishes at, normally the value of the
        product.
    steps : int
        Number of MD steps spent pulling from *cv_start* to *cv_stop*. Pulling
        slowly costs more time but leaves a path that is closer to the free
        energy valley.
    cv_name : str, optional
        Label of the CV defined in *cv_block*. Must be a plain label rather
        than a component such as ``path.sss``, because it is used to name the
        restraint's output components. Default is ``'cv'``.
    kappa : float, optional
        Spring constant of the moving restraint in kJ/mol per CV unit squared.
        Default is 2000.0. Too soft and the system lags behind the restraint,
        too stiff and the pulling heats it.
    stride : int, optional
        How often the CV is written to *colvar_file*, in steps. Default is 100.
        Match this to the reporter interval of the MD run so that every
        trajectory frame has a CV value.
    steps_equil : int, optional
        Steps held at *cv_start* before pulling starts. Default is 0.
    steps_relax : int, optional
        Steps held at *cv_stop* after pulling finishes. Default is 0.
    colvar_file : str, optional
        File the CV, the restraint centre and the work are written to.
        Default is ``'COLVAR_SMD'``.
    extra_lines : str or None, optional
        Further PLUMED lines (walls, restraints, extra prints) inserted after
        the CV definition. Default is None.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    n_steps : int
        Total number of MD steps the pulling schedule covers, i.e.
        ``steps_equil + steps + steps_relax``. Pass this to the MD run so the
        simulation does not stop mid-pull.
    """
    # Each milestone is a (step, restraint centre) pair; PLUMED interpolates
    # the centre linearly in between them.
    milestones = [(0, cv_start)]
    step = 0
    if steps_equil > 0:
        step += steps_equil
        milestones.append((step, cv_start))
    step += steps
    milestones.append((step, cv_stop))
    if steps_relax > 0:
        step += steps_relax
        milestones.append((step, cv_stop))

    schedule = " ".join(f"STEP{i}={at_step} AT{i}={at:.4f} KAPPA{i}={kappa}"
                        for i, (at_step, at) in enumerate(milestones))

    plumed_input = f"""
# Collective variable
{cv_block.strip()}
{extra_lines.strip() if extra_lines else ''}
# Steered MD: pull the CV from {cv_start:.4f} to {cv_stop:.4f}
smd:        MOVINGRESTRAINT ARG={cv_name} {schedule}
PRINT       ARG={cv_name},smd.{cv_name}_cntr,smd.work STRIDE={stride} FILE={colvar_file}
        """
    return plumed_input, step


def plumed_input_steered_pt(modeller,
                            idx,
                            steps,
                            r_0=1.1,
                            wall=1.5,
                            angle_lim=130.0,
                            kappa=2000.0,
                            stride=100,
                            cv_start=None,
                            cv_stop=None,
                            steps_equil=0,
                            steps_relax=0,
                            colvar_file='COLVAR_SMD',
                            wall_kappa=500.0):
    """
    Build a steered MD input that pulls a proton across a hydrogen bond.

    The collective variable is the one :func:`plumed_input_1pt` biases, the
    difference between the donor-hydrogen and acceptor-hydrogen coordination
    numbers, and the same walls keep the donor and acceptor from drifting
    apart while the proton is dragged over. Unless they are given, the start
    and end values of the CV are read off the geometry in *modeller*: the
    current value for the start, and its mirror image for the end, which is
    where the proton sits once transferred.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the reactant geometry, used to size the switching
        function and to work out where the pull should start.
    idx : list of int
        Three 0-based atom indices, ordered donor, hydrogen, acceptor.
    steps : int
        Number of MD steps spent pulling the proton across.
    r_0 : float, optional
        Multiplier on the shorter of the two donor/acceptor-hydrogen distances
        that sets ``R_0`` of the switching function. Default is 1.1.
    wall : float, optional
        Multiplier on the donor-acceptor distance that sets the upper wall
        keeping the hydrogen bond intact. Default is 1.5.
    angle_lim : float, optional
        Lower wall on the donor-hydrogen-acceptor angle, in degrees.
        Default is 130.0.
    kappa : float, optional
        Spring constant of the moving restraint. Default is 2000.0.
    stride : int, optional
        How often the CV is written, in steps. Default is 100.
    cv_start, cv_stop : float or None, optional
        Explicit start and end values for the CV. If None, the start is
        computed from *modeller* and the end is its negative. Default is None.
    steps_equil, steps_relax : int, optional
        Steps held at the start and end values. Default is 0.
    colvar_file : str, optional
        File the CV is written to. Default is ``'COLVAR_SMD'``.
    wall_kappa : float, optional
        Spring constant of the distance and angle walls. Default is 500.0.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    n_steps : int
        Total number of MD steps the pulling schedule covers.
    """
    # Distances, in nm, of the donor-hydrogen, acceptor-hydrogen and
    # donor-acceptor pairs
    r_01 = distance_between_atoms(modeller, idx[0], idx[1]).value_in_unit(unit.nanometer)
    r_21 = distance_between_atoms(modeller, idx[2], idx[1]).value_in_unit(unit.nanometer)
    r_02 = distance_between_atoms(modeller, idx[0], idx[2]).value_in_unit(unit.nanometer)
    r_0 = np.round(min(r_01, r_21) * r_0, decimals=2)

    if cv_start is None:
        # What the CV is worth right now: bonded to the donor gives +1,
        # bonded to the acceptor gives -1
        cv_start = np.round(_switching_value(r_01, r_0) - _switching_value(r_21, r_0), decimals=2)
    if cv_stop is None:
        cv_stop = -cv_start

    wall = np.round(r_02 * wall, decimals=2)
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    idx = atom_indices_to_plumed(idx)

    cv_block = f"""c_d:        COORDINATION GROUPA={idx[0]} GROUPB={idx[1]} R_0={r_0}
c_a:        COORDINATION GROUPA={idx[2]} GROUPB={idx[1]} R_0={r_0}
pt_cv:      COMBINE ARG=c_d,c_a COEFFICIENTS=1,-1 PERIODIC=NO"""

    extra_lines = f"""
# Limits
dist_da:    DISTANCE ATOMS={idx[2]},{idx[0]}
dist_wall:  UPPER_WALLS ARG=dist_da AT={wall} KAPPA={wall_kappa}
ang_1:      ANGLE ATOMS={idx[2]},{idx[1]},{idx[0]}
ang_wall:   LOWER_WALLS ARG=ang_1 AT={angle_lim} KAPPA={wall_kappa}
"""

    return plumed_input_steered(cv_block,
                                cv_start,
                                cv_stop,
                                steps,
                                cv_name='pt_cv',
                                kappa=kappa,
                                stride=stride,
                                steps_equil=steps_equil,
                                steps_relax=steps_relax,
                                colvar_file=colvar_file,
                                extra_lines=extra_lines)


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
    """
    Build a PLUMED input that biases a single proton transfer with metadynamics.

    The collective variable is the same one :func:`plumed_input_steered_pt`
    drags: the difference between the donor-hydrogen and acceptor-hydrogen
    coordination numbers, running from +1 (bonded to the donor) to -1
    (bonded to the acceptor). Distance and angle walls keep the hydrogen
    bond intact while the proton moves. Use this to bias the transfer with
    metadynamics; use :func:`plumed_input_steered_pt` to drag it instead.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the reactant geometry, used to size the switching
        function and the walls.
    idx : list of int
        Three 0-based atom indices, ordered donor, hydrogen, acceptor.
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR/OPES scaling and to
        report ``--kt`` to the FES reconstruction command.
    r_0 : float, optional
        Multiplier on the shorter of the two donor/acceptor-hydrogen
        distances that sets ``R_0`` of the coordination switching function.
        Default is 1.1.
    wall : float, optional
        Multiplier on the donor-acceptor distance that sets the upper wall
        keeping the hydrogen bond intact. Default is 1.5.
    angle_lim : float, optional
        Lower wall on the donor-hydrogen-acceptor angle, in degrees.
        Default is 130.0.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width of the bias, in CV units. Default is 0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 20.0.
    grid_min, grid_max : float, optional
        Bounds of the bias/FES grid. Default is -1.1 and 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.
    kappa : float, optional
        Spring constant of the distance and angle walls. Default is 500.0.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*: ``plumed sum_hills``, or the bundled
        OPES ``FES_from_State.py`` when *f_opes* is True.
    """
    # PLUMED, driven from OpenMM, works in nm, so the distances are converted
    # here rather than carried as Quantities into the input string.
    r_01 = distance_between_atoms(modeller, idx[0], idx[1]).value_in_unit(unit.nanometer)
    r_21 = distance_between_atoms(modeller, idx[2], idx[1]).value_in_unit(unit.nanometer)
    r_02 = distance_between_atoms(modeller, idx[0], idx[2]).value_in_unit(unit.nanometer)
    r_0 = np.round(min(r_01, r_21) * r_0, decimals=2)

    wall = np.round(r_02 * wall, decimals=2)
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    idx = atom_indices_to_plumed(idx)

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    metad_line, sum_hills_input = _metad_and_sumhills(
        f_opes, 'pt_cv', pace, height, sigma, bias, temperature_str, kt_str,
        grid_bin, grid_min=grid_min, grid_max=grid_max)

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
    """
    Build a PLUMED input that biases two proton transfers as a single 1-D CV.

    Each proton transfer gets its own donor-hydrogen/acceptor-hydrogen
    coordination-difference CV, as in :func:`plumed_input_1pt`, and the two
    are averaged into one collective variable, ``pt_cv = 0.5 * cv_diff1 +
    0.5 * cv_diff2``, which is what the metadynamics bias acts on. Use this
    for a concerted double proton transfer where only the combined reaction
    coordinate matters; use :func:`plumed_input_2pt_2d` to resolve the two
    transfers on separate axes.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the reactant geometry, used to size the switching
        functions and the walls.
    idx1, idx2 : list of int
        Three 0-based atom indices each, ordered donor, hydrogen, acceptor,
        for the first and second proton transfer.
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR/OPES scaling and to
        report ``--kt`` to the FES reconstruction command.
    r_0 : float, optional
        Multiplier on the shorter donor/acceptor-hydrogen distance in each
        pair that sets ``R_0`` of its coordination switching function.
        Default is 1.1.
    wall : float, optional
        Multiplier on the larger of the two donor-acceptor distances that
        sets the upper wall keeping both hydrogen bonds intact. Default is
        1.5.
    angle_lim : float, optional
        Lower wall on each donor-hydrogen-acceptor angle, in degrees.
        Default is 130.0.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width of the bias, in CV units. Default is 0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 20.0.
    grid_min, grid_max : float, optional
        Bounds of the bias/FES grid. Default is -1.1 and 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.
    kappa : float, optional
        Spring constant of the distance and angle walls. Default is 500.0.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*.
    """
    # PLUMED, driven from OpenMM, works in nm, so the distances are converted
    # here rather than carried as Quantities into the input string.
    r1_01 = distance_between_atoms(modeller, idx1[0], idx1[1]).value_in_unit(unit.nanometer)
    r1_21 = distance_between_atoms(modeller, idx1[2], idx1[1]).value_in_unit(unit.nanometer)
    r1_02 = distance_between_atoms(modeller, idx1[0], idx1[2]).value_in_unit(unit.nanometer)
    r1_0 = np.round(min(r1_01, r1_21) * r_0, decimals=2)

    r2_01 = distance_between_atoms(modeller, idx2[0], idx2[1]).value_in_unit(unit.nanometer)
    r2_21 = distance_between_atoms(modeller, idx2[2], idx2[1]).value_in_unit(unit.nanometer)
    r2_02 = distance_between_atoms(modeller, idx2[0], idx2[2]).value_in_unit(unit.nanometer)
    r2_0 = np.round(min(r2_01, r2_21) * r_0, decimals=2)

    wall = np.round(max(r1_02, r2_02) * wall, decimals=2)
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    metad_line, sum_hills_input = _metad_and_sumhills(
        f_opes, 'pt_cv', pace, height, sigma, bias, temperature_str, kt_str,
        grid_bin, grid_min=grid_min, grid_max=grid_max)

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
    """
    Build a PLUMED input that biases two proton transfers on a 2-D surface.

    Each proton transfer gets its own donor-hydrogen/acceptor-hydrogen
    coordination-difference CV, as in :func:`plumed_input_1pt`, and the two
    are kept as separate axes (``ARG=cv_diff1,cv_diff2``) so the free-energy
    surface resolves how the transfers are correlated. Use
    :func:`plumed_input_2pt_1d` instead when only the combined coordinate is
    needed.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the reactant geometry, used to size the switching
        functions and the walls.
    idx1, idx2 : list of int
        Three 0-based atom indices each, ordered donor, hydrogen, acceptor,
        for the first and second proton transfer.
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR/OPES scaling and to
        report ``--kt`` to the FES reconstruction command.
    r_0 : float, optional
        Multiplier on the shorter donor/acceptor-hydrogen distance in each
        pair that sets ``R_0`` of its coordination switching function.
        Default is 1.1.
    wall : float, optional
        Multiplier on the larger of the two donor-acceptor distances that
        sets the upper wall keeping both hydrogen bonds intact. Default is
        1.5.
    angle_lim : float, optional
        Lower wall on each donor-hydrogen-acceptor angle, in degrees.
        Default is 130.0.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width of the bias along each axis, in CV units. Default is
        0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 20.0.
    grid_min, grid_max : float, optional
        Bounds of the bias/FES grid, applied to both axes. Default is -1.1
        and 1.1.
    grid_bin : int, optional
        Number of grid bins per axis. Default is 200.
    kappa : float, optional
        Spring constant of the distance and angle walls. Default is 500.0.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*.
    """
    # PLUMED, driven from OpenMM, works in nm, so the distances are converted
    # here rather than carried as Quantities into the input string.
    r1_01 = distance_between_atoms(modeller, idx1[0], idx1[1]).value_in_unit(unit.nanometer)
    r1_21 = distance_between_atoms(modeller, idx1[2], idx1[1]).value_in_unit(unit.nanometer)
    r1_02 = distance_between_atoms(modeller, idx1[0], idx1[2]).value_in_unit(unit.nanometer)
    r1_0 = np.round(min(r1_01, r1_21) * r_0, decimals=2)

    r2_01 = distance_between_atoms(modeller, idx2[0], idx2[1]).value_in_unit(unit.nanometer)
    r2_21 = distance_between_atoms(modeller, idx2[2], idx2[1]).value_in_unit(unit.nanometer)
    r2_02 = distance_between_atoms(modeller, idx2[0], idx2[2]).value_in_unit(unit.nanometer)
    r2_0 = np.round(min(r2_01, r2_21) * r_0, decimals=2)

    wall = np.round(max(r1_02, r2_02) * wall, decimals=2)
    angle_lim = np.round(np.deg2rad(angle_lim), decimals=2)

    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    metad_line, sum_hills_input = _metad_and_sumhills(
        f_opes, 'cv_diff1,cv_diff2', pace, height, f'{sigma},{sigma}', bias,
        temperature_str, kt_str, f'{grid_bin},{grid_bin}',
        grid_min=f'{grid_min},{grid_min}', grid_max=f'{grid_max},{grid_max}')

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
    """
    Build a PLUMED input biasing the asynchronicity of two proton transfers.

    Each proton transfer gets its own donor-hydrogen/acceptor-hydrogen
    coordination-difference CV, as in :func:`plumed_input_1pt`, but here the
    two are *subtracted* rather than averaged, ``pt_cv = cv_diff1 -
    cv_diff2``. This CV is near zero when the two transfers move together
    and grows when one runs ahead of the other, i.e. the wobble base pair's
    concerted-vs-stepwise double proton transfer coordinate. Unlike the
    ``1pt``/``2pt_*`` builders, *r_0* and *wall* here are absolute distances
    rather than multipliers on the current geometry, and the walls always
    use a fixed spring constant of 500 kJ/mol/nm^2.

    Parameters
    ----------
    idx1, idx2 : list of int
        Three 0-based atom indices each, ordered donor, hydrogen, acceptor,
        for the first and second proton transfer.
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR scaling and to
        report ``--kt`` to the FES reconstruction command.
    r_0 : float, optional
        ``R_0`` of each coordination switching function, in nm. Default is
        0.14.
    wall : float, optional
        Upper wall on each donor-acceptor distance, in nm. Default is 0.4.
    angle_lim : float, optional
        Lower wall on each donor-hydrogen-acceptor angle, in degrees.
        Default is 100.0.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``), in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width of the bias, in CV units. Default is 0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Default is 20.0.
    grid_min, grid_max : float, optional
        Bounds of the bias/FES grid. Default is -1.1 and 1.1.
    grid_bin : int, optional
        Number of grid bins. Default is 200.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command (``plumed sum_hills``) that reconstructs the
        free-energy surface from the bias written by *plumed_input*.
    """
    idx1 = atom_indices_to_plumed(idx1)
    idx2 = atom_indices_to_plumed(idx2)
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
    """
    Build a PLUMED input biasing a 5-term wobble base-pair hydrogen-bond CV.

    Ten atoms spanning the wobble base pair's hydrogen-bond network (N3, H3,
    O6, O4, N1, H1, O2, N2, matching PDB nucleobase atom names) are combined
    into five coordination-difference terms, ``z1`` through ``z5`` -- one
    per competing hydrogen bond -- which are summed into a single CV ``z``
    that metadynamics biases. This generalises the two-atom-pair CV of
    :func:`plumed_input_1pt` to the wobble pair's full hydrogen-bond
    network. Upper walls on the O6-O4 and N1-N3 distances keep the base
    pair from separating.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the reactant geometry, used to size the
        coordination switching functions.
    idx : sequence of int
        Eight 0-based atom indices, ordered N3, H3, O6, O4, N1, H1, O2, N2.
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR scaling and to
        report ``--kt`` to the FES reconstruction command.
    r_0 : float, optional
        Multiplier on each measured distance that sets the ``R_0`` of its
        coordination switching function. Default is 1.1.
    wall : float, optional
        Multiplier on the O6-O4 and N1-N3 distances that sets their upper
        walls. Default is 4.0.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``), in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width of the bias, in CV units. Default is 0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Default is 20.0.
    grid_min, grid_max : float, optional
        Unused; kept for signature parity with the other ``wob`` builders.
    grid_bin : int, optional
        Number of grid bins reported to the FES reconstruction command.
        Default is 200.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command (``plumed sum_hills``) that reconstructs the
        free-energy surface from the bias written by *plumed_input*.
    """
    idx_n3, idx_h3, idx_o6, idx_o4, idx_n1, idx_h1, idx_o2, idx_n2 = idx

    # PLUMED, driven from OpenMM, works in nm, so the distances are converted
    # here rather than carried as Quantities into the input string.
    def _r(a, b):
        d = distance_between_atoms(modeller, a, b).value_in_unit(unit.nanometer)
        return np.round(d * r_0, decimals=2)

    r_1 = _r(idx_n3, idx_h3)
    r_2 = _r(idx_o6, idx_h3)
    r_3 = _r(idx_o6, idx_h3)
    r_4 = _r(idx_o4, idx_h3)
    r_5 = _r(idx_n1, idx_h1)
    r_6 = _r(idx_n3, idx_h1)
    r_7 = _r(idx_n1, idx_o2)
    r_8 = _r(idx_n2, idx_o2)
    r_9 = _r(idx_n2, idx_o2)
    r_10 = _r(idx_n1, idx_n3)

    idx_n3 += 1
    idx_h3 += 1
    idx_o6 += 1
    idx_o4 += 1
    idx_n1 += 1
    idx_h1 += 1
    idx_o2 += 1
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
    """
    Build a PLUMED input biasing a wobble base pair's transfer and opening.

    The CV combines two terms: ``z1``, the O4-H3/O2-H3 distance difference
    (the proton-transfer coordinate), and ``z2``, the N2-O6-O4 angle (how
    open the wobble pair is). Walls bound the opening angle, the distance
    between the two bases' R-group atoms (keeps the bases from drifting
    apart) and the O6-O4/N2-O2 distances (keeps the pair from separating
    entirely).

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the reactant geometry, used to size the walls.
    idx_o4, idx_h3, idx_o2, idx_o6, idx_n2 : list of int
        Single-element lists each holding the 0-based index of the named
        atom (O4, H3, O2, O6, N2) in the wobble pair's hydrogen-bond
        network.
    idx_nr1, idx_nr2 : list of int
        Single-element lists holding the 0-based index of an R-group atom
        on each base, used to bound how far the bases can separate.
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR scaling and to
        report ``--kt`` to the FES reconstruction command.
    r_0 : float, optional
        Unused; kept for signature parity with the other ``wob`` builders.
    wall : float, optional
        Multiplier defining the R-group distance's lower/upper walls
        (``2 - wall`` and *wall* times the current distance) and the upper
        walls on the O6-O4 and N2-O2 distances. Default is 1.1.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``), in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width of the bias, in CV units. Default is 0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Default is 20.0.
    grid_min, grid_max : float, optional
        Unused; kept for signature parity with the other ``wob`` builders.
    grid_bin : int, optional
        Number of grid bins reported to the FES reconstruction command.
        Default is 200.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command (``plumed sum_hills``) that reconstructs the
        free-energy surface from the bias written by *plumed_input*.
    """
    # PLUMED, driven from OpenMM, works in nm, so the distances are converted
    # here rather than carried as Quantities into the input string.
    d_nr = distance_between_atoms(modeller, idx_nr1[0], idx_nr2[0]).value_in_unit(unit.nanometer)
    wall_u = np.round(d_nr * wall, decimals=2)
    wall_l = np.round(d_nr * (2.0 - wall), decimals=2)

    d_o6_o4 = distance_between_atoms(modeller, idx_o6[0], idx_o4[0]).value_in_unit(unit.nanometer)
    d_n2_o2 = distance_between_atoms(modeller, idx_n2[0], idx_o2[0]).value_in_unit(unit.nanometer)
    wall_1 = np.round(d_o6_o4 * wall, decimals=2)
    wall_2 = np.round(d_n2_o2 * wall, decimals=2)

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
                       kappa=2000.0,
                       f_opes=False):
    """
    Build a PLUMED input biasing a 5-term wobble base-pair CV with fixed walls.

    Like :func:`plumed_input_wob_2`, ten atoms spanning the wobble base
    pair's hydrogen-bond network (N3, H3, O6, O4, N1, H1, O2, N2, plus two
    R-group atoms NR1/NR2) are combined into five distance-difference terms
    summed into a single CV ``z`` that metadynamics biases, but here the
    sub-CVs use plain ``DISTANCE`` rather than ``COORDINATION``, and the
    hydrogen-bond and base-pair-opening distances are held with fixed
    (rather than geometry-scaled) upper/lower walls. A restraint on the
    inter-base rise (``dist.z``) and walls on an R-group angle further
    constrain the pair geometry.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller holding the reactant geometry, used to size the walls.
    idx : sequence of int
        Ten 0-based atom indices, ordered N3, H3, O6, O4, N1, H1, O2, N2,
        NR1, NR2 (NR1/NR2 being an R-group atom on each base).
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR/OPES scaling and to
        report ``--kt`` to the FES reconstruction command.
    r_0 : float, optional
        Unused; kept for signature parity with the other ``wob`` builders.
    wall : float, optional
        Multiplier defining the (currently unused) R-group distance walls
        ``rr_u``/``rr_l``. Default is 1.1.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width of the bias, in CV units. Default is 0.05.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 20.0.
    grid_bin : int, optional
        Number of grid bins reported to the FES reconstruction command.
        Default is 200.
    kappa : float, optional
        Spring constant used by every wall and the rise restraint. Default
        is 2000.0.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*: ``plumed sum_hills``, or the bundled
        OPES ``FES_from_State.py`` when *f_opes* is True.
    """
    idx = atom_indices_to_plumed(idx)
    # 0 1   2   3   4   5   6   7   8    9
    n3, h3, o6, o4, n1, h1, o2, n2, nr1, nr2 = idx
    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    metad_line, sum_hills_input = _metad_and_sumhills(
        f_opes, 'z', pace, height, sigma, bias, temperature_str, kt_str, grid_bin)

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

o6_o4: DISTANCE ATOMS={o6},{o4}
UPPER_WALLS ARG=o6_o4 AT=0.36  KAPPA={kappa}
LOWER_WALLS ARG=o6_o4 AT=0.25  KAPPA={kappa}
o6_n3: DISTANCE ATOMS={o6},{n3}
UPPER_WALLS ARG=o6_n3 AT=0.34  KAPPA={kappa}
LOWER_WALLS ARG=o6_n3 AT=0.26  KAPPA={kappa}

UPPER_WALLS ARG=n1_n3 AT=0.38  KAPPA={kappa}
LOWER_WALLS ARG=n1_n3 AT=0.29  KAPPA={kappa}

UPPER_WALLS ARG=n1_o2 AT=0.37  KAPPA={kappa}
LOWER_WALLS ARG=n1_o2 AT=0.29  KAPPA={kappa}

UPPER_WALLS ARG=n2_o2 AT=0.37  KAPPA={kappa}
LOWER_WALLS ARG=n2_o2 AT=0.30  KAPPA={kappa}

# rise restraint
base1: GROUP ATOMS={nr1},{o6},{n1},{n2}
base2: GROUP ATOMS={nr2},{o4},{n3},{o2}
com1: COM ATOMS=base1
com2: COM ATOMS=base2
dist: DISTANCE ATOMS=com1,com2 COMPONENTS
RESTRAINT ARG=dist.z AT=0.0 KAPPA={kappa}

# Constraints to prevent excessive opening of the base pair
nr_ang: ANGLE ATOMS={nr1},{o6},{nr2}
w_a1: LOWER_WALLS ARG=nr_ang AT={np.round(np.deg2rad(120.0), decimals=2)} KAPPA={kappa}
w_a2: UPPER_WALLS ARG=nr_ang AT={np.round(np.deg2rad(140.0), decimals=2)} KAPPA={kappa}

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
                          grid_min=0.0,
                          grid_max=26.0,
                          grid_bin=500,
                          kappa=500.0,
                          lambda_val=250.0,
                          neigh_size=8,
                          f_opes=False):
    """
    Build a PLUMED input that biases progress along a NEB-derived path.

    Uses PLUMED's ``PATHMSD`` collective variable against a reference path
    (``neb_path.pdb``, as produced by :func:`reactiontools.get_neb_path` /
    :func:`reactiontools.stitch_path`) to define the progress-along-path
    coordinate ``path.sss``, which metadynamics biases, and the
    distance-from-path coordinate ``path.zzz``, which is kept small by an
    upper wall so sampling stays near the path. ``FIT_TO_TEMPLATE`` aligns
    each frame to ``index_atoms.pdb`` before the path distances are
    computed, so both files must exist in the working directory the
    resulting script is run from.

    Parameters
    ----------
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR/OPES scaling and to
        report ``--kt`` to the FES reconstruction command.
    wall : float, optional
        Upper wall on ``path.zzz``, the mean-squared distance from the
        reference path. Default is 0.1.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in kJ/mol. Default is 15.0.
    sigma : float, optional
        Gaussian width of the bias along ``path.sss``, in nm. Default is
        0.1.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 5.0.
    grid_min, grid_max : float, optional
        Bounds of the ``path.sss`` bias/FES grid, in units of path node
        index. Default is 0.0 and 26.0.
    grid_bin : int, optional
        Number of grid bins. Default is 500.
    kappa : float, optional
        Spring constant of the ``path.zzz`` wall. Default is 500.0.
    lambda_val : float, optional
        ``LAMBDA`` parameter of ``PATHMSD``, controlling how sharply
        ``path.sss`` distinguishes neighbouring frames; see
        :func:`estimate_path_lambda`. Default is 250.0.
    neigh_size : int, optional
        ``NEIGH_SIZE`` of ``PATHMSD``, the number of reference frames
        considered when computing the path distance. Default is 8.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*: ``plumed sum_hills``, or the bundled
        OPES ``FES_from_State.py`` when *f_opes* is True.
    """
    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    metad_line, sum_hills_input = _metad_and_sumhills(
        f_opes, 'path.sss', pace, height, sigma, bias, temperature_str, kt_str,
        grid_bin, grid_min=grid_min, grid_max=grid_max, label='metad: ')

    plumed_input = f'''
FIT_TO_TEMPLATE REFERENCE=index_atoms.pdb TYPE=OPTIMAL
path: PATHMSD REFERENCE=neb_path.pdb LAMBDA={lambda_val} NEIGH_SIZE={neigh_size}
{metad_line}
path_limit: UPPER_WALLS ARG=path.zzz AT={wall} KAPPA={kappa}
PRINT ARG=path.sss,path.zzz,metad.bias STRIDE={pace} FILE=COLVAR
        '''
    return plumed_input, sum_hills_input


def plumed_input_neb_path_wob(idx,
                              temperature,
                              wall=0.1,
                              pace=500,
                              height=10.0,  # kJ/mol
                              sigma=0.1,  # nm
                              bias=5.0,
                              grid_min=0.0,
                              grid_max=26.0,
                              grid_bin=500,
                              kappa=500.0,
                              lambda_val=500.0,
                              neigh_size=8,
                              f_opes=False):
    """
    Build a PLUMED input biasing progress along a NEB path, with wobble-pair walls.

    Extends :func:`plumed_input_neb_path` with two extra restraints that
    keep a wobble base pair intact while ``path.sss`` is biased: a torsion
    wall bounding the CA1-NR1-CB1-NR2 dihedral, and a monitored (currently
    unrestrained) distance between the two bases' R-group atoms. As in
    :func:`plumed_input_neb_path`, ``neb_path.pdb`` and ``index_atoms.pdb``
    must exist in the working directory the resulting script is run from.

    Parameters
    ----------
    idx : sequence of int
        Six 0-based atom indices, ordered CA1, CA2, CB1, CB2, NR1, NR2: a
        backbone atom and an R-group atom on each base, used to define the
        dihedral wall and the monitored R-group distance.
    temperature : openmm.unit.Quantity
        Simulation temperature, used for the BIASFACTOR/OPES scaling and to
        report ``--kt`` to the FES reconstruction command.
    wall : float, optional
        Upper wall on ``path.zzz``, the mean-squared distance from the
        reference path. Default is 0.1.
    pace : int, optional
        ``PACE`` of the metadynamics bias, in steps. Default is 500.
    height : float, optional
        Gaussian height (``HEIGHT``) for standard METAD, or the ``BARRIER``
        for ``OPES_METAD``, in kJ/mol. Default is 10.0.
    sigma : float, optional
        Gaussian width of the bias along ``path.sss``, in nm. Default is
        0.1.
    bias : float, optional
        Well-tempered ``BIASFACTOR``. Ignored when *f_opes* is True.
        Default is 5.0.
    grid_min, grid_max : float, optional
        Bounds of the ``path.sss`` bias/FES grid, in units of path node
        index. Default is 0.0 and 26.0.
    grid_bin : int, optional
        Number of grid bins. Default is 500.
    kappa : float, optional
        Spring constant of the ``path.zzz`` and dihedral walls. Default is
        500.0.
    lambda_val : float, optional
        ``LAMBDA`` parameter of ``PATHMSD``; see :func:`estimate_path_lambda`.
        Default is 500.0.
    neigh_size : int, optional
        ``NEIGH_SIZE`` of ``PATHMSD``, the number of reference frames
        considered when computing the path distance. Default is 8.
    f_opes : bool, optional
        If True, bias with ``OPES_METAD`` instead of well-tempered
        ``METAD``. Default is False.

    Returns
    -------
    plumed_input : str
        The PLUMED input script.
    sum_hills_input : str
        Shell command that reconstructs the free-energy surface from the
        bias written by *plumed_input*: ``plumed sum_hills``, or the bundled
        OPES ``FES_from_State.py`` when *f_opes* is True.
    """
    idx = atom_indices_to_plumed(idx)
    ca1, _, cb1, _, nr1, nr2 = idx

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = temperature_to_kbt(temperature)

    metad_line, sum_hills_input = _metad_and_sumhills(
        f_opes, 'path.sss', pace, height, sigma, bias, temperature_str, kt_str,
        grid_bin, grid_min=grid_min, grid_max=grid_max, label='metad: ')

    plumed_input = f'''
FIT_TO_TEMPLATE REFERENCE=index_atoms.pdb TYPE=OPTIMAL
path: PATHMSD REFERENCE=neb_path.pdb LAMBDA={lambda_val} NEIGH_SIZE={neigh_size}
{metad_line}
path_limit: UPPER_WALLS ARG=path.zzz AT={wall} KAPPA={kappa}

# Constraint to bound the sliding of the bases
dih: TORSION ATOMS={ca1},{nr1},{cb1},{nr2}
w_a1: LOWER_WALLS ARG=dih AT=-1.0 KAPPA={kappa}
w_a2: UPPER_WALLS ARG=dih AT=1.0 KAPPA={kappa}

# w_a1: LOWER_WALLS ARG=dih AT={np.round(np.deg2rad(150.0), decimals=2)} KAPPA={kappa}
# w_a2: UPPER_WALLS ARG=dih AT={np.round(np.deg2rad(190.0), decimals=2)} KAPPA={kappa}

# Constraint to R-groups to keep bases together
d_rr: DISTANCE ATOMS={nr1},{nr2}
# w_d1: UPPER_WALLS ARG=d_rr AT=1.0 KAPPA={kappa}

PRINT ARG=path.sss,path.zzz,metad.bias,dih,d_rr STRIDE={pace} FILE=COLVAR
        '''
    return plumed_input, sum_hills_input
