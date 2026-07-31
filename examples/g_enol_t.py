import os

import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit
from mace.calculators.foundations_models import mace_mp

import openmmnqe as nqe

if __name__ == "__main__":
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 30_000
    # input_pdb = os.path.join(os.path.dirname(nqe.openmm_nqe_dir), 'tests/data/pdb/G_enol_T.pdb')
    input_pdb ='G_enol_T.pdb'

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

    # Prepare the box
    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     calculator=calc,
                                     ml_idx=ml_atoms)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
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

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_heating(modeller,
    #                        forcefield,
    #                        calculator=calc,
    #                        ml_idx=ml_atoms,
    #                        target_temp=temperature)
    #
    # pdb = app.PDBFile("equilibrate.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_npt(modeller,
    #                    forcefield,
    #                    calculator=calc,
    #                    ml_idx=ml_atoms,
    #                    temperature=temperature)
    #
    # pdb = app.PDBFile("npt_equilibrate.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        temperature=temperature,
                        barostat_freq=None,
                        steps=steps_prod,
                        calculator=calc,
                        ml_idx=ml_atoms)

    # Run PLUMED sum_hills to get FES
    os.system(sum_hills_input)

    nqe.plot_plumed_fes("fes.dat")
    plt.savefig("fes.png")
    plt.savefig("fes.pdf")
    plt.show()
    plt.close()

    nqe.plot_plumed_colvar("COLVAR")
    plt.savefig("colvar.png")
    plt.savefig("colvar.pdf")
    plt.show()
    plt.close()

    # nqe.remove_file_pattern('minimized*')
    # nqe.remove_file_pattern('equilibrate*')
    # nqe.remove_file_pattern('npt_equilibrate*')
    # # nqe.remove_file_pattern('prod*')
    # nqe.remove_file('COLVAR')
    # nqe.remove_file('HILLS')
    # nqe.remove_file('STATE')
    # nqe.remove_file('KERNELS')
    # nqe.remove_file('fes.dat')
    # nqe.remove_file('plumed.dat')
    # nqe.remove_file('index_atoms.pdb')
    # nqe.remove_file('neb_path.pdb')
    # nqe.remove_file('neb_path.xyz')
