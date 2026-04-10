import os

import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

import openmmnqe as nqe

if __name__ == "__main__":
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 50_000
    input_pdb = os.path.join(os.path.dirname(nqe.openmm_nqe_dir), 'tests/data/pdb/gc.pdb')
    ml_model = 'mace-off23-small'
    forcefield_names = ("amber14-all.xml", "amber14/tip3pfb.xml")
    potential = MLPotential(ml_model)  # mace-off23-large mace-off23-small
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
                                     potential=potential,
                                     ml_idx=ml_atoms)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.save_only_index_atoms(modeller, ml_atoms, file_idx='index_atoms.pdb')
    #
    # reactant = read('index_atoms.pdb')
    # # view(reactant)
    # product = nqe.swap_bonding_configuration(reactant, 17, 19, 11)
    # product = nqe.swap_bonding_configuration(product, 5, 7, 20)
    # # view(product)
    # calc = mace_off(model_name=ml_model, device='cuda')
    # product = nqe.optimise_geom(product, calc, fmax=0.01)
    # neb_path = nqe.quick_guess_path(reactant, product)
    # write("neb_path.xyz", neb_path)
    # view(neb_path)
    # nqe.convert_xyz_to_plumed_ref("neb_path.xyz", "index_atoms.pdb", "neb_path.pdb")
    # plumed_input, sum_hills_input = nqe.plumed_input_neb_path(temperature)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_heating(modeller,
    #                        forcefield,
    #                        potential=potential,
    #                        ml_idx=ml_atoms,
    #                        target_temp=temperature)
    #
    # pdb = app.PDBFile("equilibrate.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_npt(modeller,
    #                    forcefield,
    #                    potential=potential,
    #                    ml_idx=ml_atoms,
    #                    temperature=temperature)
    #
    # pdb = app.PDBFile("npt_equilibrate.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['GGG1:O1', 'CCC1:H2', 'CCC1:N1'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['CCC1:N2', 'GGG1:H3', 'GGG1:N3'])

    plumed_input, sum_hills_input = nqe.plumed_input_2pt_1d(modeller,
                                                            idx1,
                                                            idx2,
                                                            temperature,
                                                            wall=1.0,
                                                            height=40.0,
                                                            bias=10.0,
                                                            f_opes=True)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        temperature=temperature,
                        barostat_freq=None,
                        steps=steps_prod,
                        potential=potential,
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

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')
    # nqe.remove_file_pattern('prod*')
    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('STATE')
    nqe.remove_file('KERNELS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')
    nqe.remove_file('index_atoms.pdb')
    nqe.remove_file('neb_path.pdb')
    nqe.remove_file('neb_path.xyz')
