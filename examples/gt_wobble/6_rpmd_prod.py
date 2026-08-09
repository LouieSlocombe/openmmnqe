import os
import shutil

import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

import openmmnqe as nqe
import reactiontools as rt

if __name__ == "__main__":
    print(flush=True)
    n_beads = 4
    temperature = 300.0 * unit.kelvin
    steps_prod = 100_000
    input_pdb = 'G_T_wob.pdb'

    potential = MLPotential('mace-off23-small')

    forcefield_names = ("amber19-all.xml", "amber19/tip3pfb.xml")
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(forcefield_names, molecule)

    pdb = app.PDBFile("rpmd_ready_centroid.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]

    # n3, h3, o6, o4, n1, h1, o2, n2, nr1, nr2
    idx = ['AAB1:N2',
           'AAB1:H6',
           'AAA1:O1',
           'AAB1:O2',
           'AAA1:N3',
           'AAA1:H3',
           'AAB1:O1',
           'AAA1:N4',
           'AAA1:N1',
           'AAB1:N1']
    idx = nqe.atom_indices_from_vmd_picks(modeller, idx)

    plumed_input, sum_hills_input = nqe.plumed_input_wob_4(modeller,
                                                           idx,
                                                           temperature,
                                                           wall=1.0,
                                                           height=200.0,
                                                           bias=10.0,
                                                           f_opes=True)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)

    nqe.run_openmm_rpmd_prod(modeller,
                             forcefield,
                             plumed_script_path=plumed_script_path,
                             temperature=temperature,
                             barostat_freq=None,
                             steps=steps_prod,
                             potential=potential,
                             ml_idx=ml_atoms,
                             n_beads=n_beads,
                             checkpoint_file='rpmd_ready.chk'
                             )

    os.system(sum_hills_input)
    shutil.copy('fes.dat', 'rpmd_fes.dat')

    rt.plot_plumed_fes("rpmd_fes.dat", filename="rpmd_fes")
    plt.close()

    rt.plot_plumed_colvar("COLVAR", filename="rpmd_colvar")
    plt.close()
