"""Tests for estimating a path collective variable from steered MD.

Everything up to the last test needs numpy, mdtraj and OpenMM but no ML
potential and no GPU: the trajectory a steered run would produce is faked, so
the frame selection and the file it writes can be checked on their own.  The
workflow test at the end does the real thing on malonaldehyde.
"""
import os
import re
import shutil

import mdtraj as md
import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmml import MLPotential

import openmmnqe as nqe
import reactiontools as rt

MALONALDEHYDE = 'tests/data/pdb/malonaldehyde.pdb'

# Donor oxygen, the proton it shares, and the acceptor oxygen
DONOR, HYDROGEN, ACCEPTOR = 0, 8, 1


def write_colvar(path, cv, fields=('time', 'pt_cv', 'smd.pt_cv_cntr', 'smd.work')):
    """Write a COLVAR file holding *cv*, in the layout a steered run leaves."""
    with open(path, "w") as handle:
        handle.write("#! FIELDS " + " ".join(fields) + "\n")
        for i, value in enumerate(cv):
            handle.write(f"{i * 0.05:.4f} {value:.6f} {value:.6f} {0.1 * i:.6f}\n")


def fake_steered_traj(path, n_frames=60, noise=0.0, seed=0):
    """
    Fake the trajectory of a proton being dragged across malonaldehyde.

    The proton slides from the donor towards the acceptor at a constant rate,
    which makes the frame that belongs to any given point of the pull known in
    advance.  Returns the trajectory and the fraction transferred per frame.
    """
    reference = md.load(MALONALDEHYDE)
    xyz = np.repeat(reference.xyz, n_frames, axis=0)

    fraction = np.linspace(0.0, 1.0, n_frames)
    start = reference.xyz[0, HYDROGEN]
    finish = reference.xyz[0, ACCEPTOR] + (start - reference.xyz[0, DONOR])
    xyz[:, HYDROGEN] = start + (finish - start) * fraction[:, None]

    if noise:
        xyz += np.random.default_rng(seed).normal(scale=noise, size=xyz.shape)

    traj = md.Trajectory(xyz, reference.topology)
    traj.save_pdb(path)
    return traj, fraction


def test_select_frames_by_cv_spans_the_range_evenly():
    cv = np.linspace(-1.0, 1.0, 101)
    picks = nqe.select_frames_by_cv(cv, 11)

    assert list(picks) == list(range(0, 101, 10))


def test_select_frames_by_cv_honours_explicit_limits():
    cv = np.linspace(-1.0, 1.0, 101)
    picks = nqe.select_frames_by_cv(cv, 3, cv_start=-0.5, cv_stop=0.5)

    assert [round(cv[i], 2) for i in picks] == [-0.5, 0.0, 0.5]


def test_select_frames_by_cv_moves_forwards_through_a_noisy_pull():
    rng = np.random.default_rng(1)
    cv = np.linspace(-1.0, 1.0, 200) + rng.normal(scale=0.1, size=200)

    picks = nqe.select_frames_by_cv(cv, 15)

    assert np.all(np.diff(picks) > 0), "frames must be ordered along the path"
    assert len(picks) == 15


def test_select_frames_by_cv_needs_enough_frames():
    with pytest.raises(ValueError, match="Cannot pick"):
        nqe.select_frames_by_cv(np.linspace(0.0, 1.0, 5), 10)


def test_select_frames_by_msd_spaces_frames_by_displacement():
    # One atom that accelerates, so equal spacing in displacement is very
    # much not equal spacing in frame number
    n_frames = 100
    xyz = np.zeros((n_frames, 3, 3))
    xyz[:, 0, 0] = np.linspace(0.0, 1.0, n_frames) ** 2

    picks = nqe.select_frames_by_msd(xyz, 6)

    assert picks[0] == 0 and picks[-1] == n_frames - 1
    travelled = xyz[picks, 0, 0]
    spacing = np.diff(travelled)
    assert np.allclose(spacing, spacing[0], atol=0.02)


def test_cv_from_colvar_drops_the_row_written_at_step_zero(tmp_path):
    colvar = tmp_path / "COLVAR_SMD"
    cv = np.linspace(0.0, 1.0, 21)
    write_colvar(colvar, cv)

    assert np.allclose(nqe.cv_from_colvar(str(colvar), 20, cv_name='pt_cv'), cv[1:])


def test_cv_from_colvar_resamples_a_mismatched_stride(tmp_path):
    colvar = tmp_path / "COLVAR_SMD"
    write_colvar(colvar, np.linspace(0.0, 1.0, 51))

    cv = nqe.cv_from_colvar(str(colvar), 10, cv_name='pt_cv')

    assert cv.shape == (10,)
    assert np.isclose(cv[-1], 1.0)
    assert np.all(np.diff(cv) > 0)


def test_plumed_input_steered_schedules_the_pull():
    plumed_input, n_steps = nqe.plumed_input_steered("pt_cv: DISTANCE ATOMS=1,2",
                                                     1.0,
                                                     -1.0,
                                                     5_000,
                                                     cv_name='pt_cv',
                                                     steps_equil=1_000,
                                                     steps_relax=500,
                                                     stride=50)

    assert n_steps == 6_500
    assert "MOVINGRESTRAINT ARG=pt_cv" in plumed_input
    # Hold, pull, hold: four milestones, the last of them at the end value
    assert "STEP0=0 AT0=1.0000" in plumed_input
    assert "STEP1=1000 AT1=1.0000" in plumed_input
    assert "STEP2=6000 AT2=-1.0000" in plumed_input
    assert "STEP3=6500 AT3=-1.0000" in plumed_input
    assert "PRINT       ARG=pt_cv,smd.pt_cv_cntr,smd.work STRIDE=50 FILE=COLVAR_SMD" in plumed_input


def test_plumed_input_steered_without_holds_has_two_milestones():
    plumed_input, n_steps = nqe.plumed_input_steered("cv: DISTANCE ATOMS=1,2", 0.2, 0.5, 100)

    assert n_steps == 100
    assert "STEP0=0 AT0=0.2000" in plumed_input
    assert "STEP1=100 AT1=0.5000" in plumed_input
    assert "STEP2=" not in plumed_input


def test_plumed_input_steered_pt_reads_the_ends_off_the_geometry():
    pdb = app.PDBFile(MALONALDEHYDE)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    plumed_input, n_steps = nqe.plumed_input_steered_pt(modeller,
                                                        [DONOR, HYDROGEN, ACCEPTOR],
                                                        10_000)

    assert n_steps == 10_000
    # PLUMED counts atoms from one
    assert f"COORDINATION GROUPA={DONOR + 1} GROUPB={HYDROGEN + 1}" in plumed_input
    assert f"COORDINATION GROUPA={ACCEPTOR + 1} GROUPB={HYDROGEN + 1}" in plumed_input
    assert "UPPER_WALLS ARG=dist_da" in plumed_input

    # The proton starts on the donor, so the CV starts positive and is pulled
    # to the mirror image of wherever it started
    at_values = [float(value) for value in re.findall(r'\bAT\d+=(\S+)', plumed_input)]
    assert at_values[0] > 0.0
    assert at_values[-1] == pytest.approx(-at_values[0])


def test_path_from_steered_md_writes_a_pathmsd_reference(tmp_path):
    template = tmp_path / "index_atoms.pdb"
    shutil.copy(MALONALDEHYDE, template)
    traj_file = tmp_path / "smd_steps.pdb"
    _, fraction = fake_steered_traj(str(traj_file), n_frames=60, noise=0.001)
    write_colvar(tmp_path / "COLVAR_SMD", np.concatenate(([0.0], fraction)))

    output = tmp_path / "neb_path.pdb"
    lambda_val = nqe.path_from_steered_md(str(traj_file),
                                          template_pdb=str(template),
                                          output_file=str(output),
                                          colvar_file=str(tmp_path / "COLVAR_SMD"),
                                          cv_name='pt_cv',
                                          n_images=8)

    assert lambda_val > 0.0
    path = md.load(str(output))
    assert path.n_frames == 8
    assert path.n_atoms == md.load(MALONALDEHYDE).n_atoms
    assert os.path.exists(tmp_path / "neb_path.xyz")

    # The proton should walk from the donor to the acceptor along the path
    to_donor = md.compute_distances(path, [[HYDROGEN, DONOR]]).ravel()
    to_acceptor = md.compute_distances(path, [[HYDROGEN, ACCEPTOR]]).ravel()
    assert to_donor[0] < to_acceptor[0]
    assert to_donor[-1] > to_acceptor[-1]


def test_path_from_steered_md_without_a_colvar(tmp_path):
    template = tmp_path / "index_atoms.pdb"
    shutil.copy(MALONALDEHYDE, template)
    traj_file = tmp_path / "smd_steps.pdb"
    fake_steered_traj(str(traj_file), n_frames=40)

    output = tmp_path / "neb_path.pdb"
    nqe.path_from_steered_md(str(traj_file),
                             template_pdb=str(template),
                             output_file=str(output),
                             colvar_file=None,
                             n_images=6,
                             smooth=2)

    assert md.load(str(output)).n_frames == 6


def test_path_from_steered_md_checks_the_template_matches(tmp_path):
    template = tmp_path / "index_atoms.pdb"
    shutil.copy(MALONALDEHYDE, template)
    traj_file = tmp_path / "smd_steps.pdb"
    fake_steered_traj(str(traj_file), n_frames=20)

    with pytest.raises(ValueError, match="atoms but"):
        nqe.path_from_steered_md(str(traj_file),
                                 template_pdb=str(template),
                                 output_file=str(tmp_path / "neb_path.pdb"),
                                 colvar_file=None,
                                 atom_indices=[DONOR, HYDROGEN, ACCEPTOR],
                                 n_images=5)


def test_path_from_steered_md_wants_more_frames_than_images(tmp_path):
    template = tmp_path / "index_atoms.pdb"
    shutil.copy(MALONALDEHYDE, template)
    traj_file = tmp_path / "smd_steps.pdb"
    fake_steered_traj(str(traj_file), n_frames=5)

    with pytest.raises(ValueError, match="too few"):
        nqe.path_from_steered_md(str(traj_file),
                                 template_pdb=str(template),
                                 output_file=str(tmp_path / "neb_path.pdb"),
                                 colvar_file=None,
                                 n_images=15)


@pytest.mark.pipeline
def test_malonaldehyde_steered_path():
    """Pull the proton across, make a path out of it, then bias along it."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_smd = 20_000
    steps_prod = 50_000
    n_images = 10

    pdb = app.PDBFile(MALONALDEHYDE)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    forcefield = MLPotential('mace-off23-small')
    nqe.run_openmm_relaxation_simple(modeller, forcefield)

    nqe.pdb_remove_ter_index("minimized.pdb", "minimized.pdb")
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    path_atoms = [atom.index for atom in modeller.topology.atoms()]
    nqe.save_only_index_atoms(modeller, path_atoms, file_idx='index_atoms.pdb')

    # Drag the proton over to the far oxygen
    idx = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O1', 'LIG1:H5', 'LIG1:O2'])
    plumed_input, n_steps = nqe.plumed_input_steered_pt(modeller,
                                                        idx,
                                                        steps_smd,
                                                        steps_equil=2_000,
                                                        steps_relax=2_000,
                                                        stride=100)
    traj_file = nqe.run_openmm_steered(modeller,
                                       forcefield,
                                       plumed_input,
                                       n_steps,
                                       temperature=temperature,
                                       n_report=100)

    # ... and read the reaction path back out of the trajectory
    lambda_val = nqe.path_from_steered_md(traj_file,
                                          cv_name='pt_cv',
                                          n_images=n_images,
                                          smooth=2) * 0.5

    plumed_input, sum_hills_input = nqe.plumed_input_neb_path(temperature,
                                                              grid_min=0.0,
                                                              grid_max=float(n_images),
                                                              lambda_val=lambda_val,
                                                              neigh_size=n_images,
                                                              f_opes=True)
    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        temperature=temperature,
                        barostat_freq=None,
                        steps=steps_prod)

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)
    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('smd*')
    nqe.remove_file_pattern('prod*')
    nqe.remove_file('COLVAR')
    nqe.remove_file('COLVAR_SMD')
    nqe.remove_file('HILLS')
    nqe.remove_file('KERNELS')
    nqe.remove_file('STATE')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')
    nqe.remove_file('index_atoms.pdb')
    nqe.remove_file('neb_path.pdb')
    nqe.remove_file('neb_path.xyz')
