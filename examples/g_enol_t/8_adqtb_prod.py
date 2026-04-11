import os

import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit
from ase.io import read, write
from mace.calculators.foundations_models import mace_mp

import openmmnqe as nqe

if __name__ == "__main__":
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 30_000
    input_pdb = 'G_enol_T.pdb'

    calc = mace_mp(model=os.path.join(os.environ['MACE_MODELS'], 'mace-mh-1.model'),
                   default_dtype="float32",
                   device="cuda",
                   head="omol",
                   dispersion=True,
                   dispersion_xc="pbe")

    forcefield_names = ("amber19-all.xml", "amber19/tip3pfb.xml")
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(forcefield_names, molecule)

    pdb = app.PDBFile("adqtb_ready.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['AAB1:O2', 'AAA1:H5', 'AAA1:O1'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['AAA1:N3', 'AAB1:H1', 'AAB1:N2'])
    plumed_input, sum_hills_input = nqe.plumed_input_2pt_1d(modeller,
                                                            idx1,
                                                            idx2,
                                                            temperature,
                                                            wall=1.0,
                                                            height=100.0,
                                                            bias=10.0,
                                                            f_opes=True)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)

    nqe.run_openmm_adqtb_prod(modeller,
                             forcefield,
                             plumed_script_path=plumed_script_path,
                             temperature=temperature,
                             barostat_freq=None,
                             steps=steps_prod,
                             calculator=calc,
                             ml_idx=ml_atoms,
                             )

    # Run PLUMED sum_hills to get FES
    os.system(sum_hills_input)

    nqe.plot_plumed_fes("fes.dat")
    plt.savefig("adqtb_fes.png")
    plt.savefig("adqtb_fes.pdf")
    plt.close()

    nqe.plot_plumed_colvar("COLVAR")
    plt.savefig("adqtb_colvar.png")
    plt.savefig("adqtb_colvar.pdf")
    plt.close()
