"""Build a malonaldehyde reference path and bias a simulation along it.

The frame selection, the CV builders and the file writing are all
reactiontools' now, and are unit-tested there against a faked trajectory. What
is left here is the thing only this package can check: that a real steered run
through OpenMM produces a trajectory those functions turn into a path PLUMED
will actually bias.
"""
from __future__ import annotations

import os

import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

import openmmnqe as nqe
import reactiontools as rt

MALONALDEHYDE = 'tests/data/pdb/malonaldehyde.pdb'


def main() -> None:
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

    rt.pdb_remove_ter_index("minimized.pdb", "minimized.pdb")
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    path_atoms = [atom.index for atom in modeller.topology.atoms()]
    nqe.save_only_index_atoms(modeller, path_atoms, file_idx='index_atoms.pdb')

    # Drag the proton over to the far oxygen
    idx = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O1', 'LIG1:H5', 'LIG1:O2'])
    plumed_input, n_steps = rt.plumed_input_steered_pt(modeller,
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
    lambda_val = rt.path_from_steered_md(traj_file,
                                         cv_name='pt_cv',
                                         n_images=n_images,
                                         smooth=2) * 0.5

    plumed_input, sum_hills_input = rt.plumed_input_neb_path(temperature,
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


if __name__ == "__main__":
    main()
