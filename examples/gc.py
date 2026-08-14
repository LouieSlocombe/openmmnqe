import logging
import os

import forcefill as ff
import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

import openmmnqe as nqe
import reactiontools as rt

if __name__ == "__main__":
    print(flush=True)
    # forcefill reports through logging where openmmnqe prints, so without this
    # its INFO lines -- which residues it is parameterising, the net charge it
    # assumed, where the intermediate files went -- go nowhere.
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    temperature = 300.0 * unit.kelvin
    steps_prod = 50_000
    input_pdb = os.path.join(os.path.dirname(nqe.openmm_nqe_dir), 'tests/data/pdb/gc.pdb')
    ml_model = 'mace-off23-small'
    forcefield_names = ("amber14-all.xml", "amber14/tip3pfb.xml")
    potential = MLPotential(ml_model)  # mace-off23-large mace-off23-small

    # The guanine and cytosine residues are not in any standard force field, so
    # forcefill parameterises them with GAFF2/AM1-BCC and writes an ffxml that
    # loads underneath the standard ones.  A charged ligand would need
    # net_charges={'GGG': -1}: a residue read out of a PDB carries no formal
    # charge, so forcefill assumes zero without saying so.
    result = ff.build_forcefield_xml(input_pdb,
                                     "ligands.xml",
                                     base_forcefield=forcefield_names,
                                     workdir="forcefill_work")
    print(f"Parameterised: {result.parameterized}, skipped: {result.skipped}", flush=True)

    # Built from the same file forcefill read, with nothing in between: the
    # templates describe those residues exactly as this file spells them, so an
    # edit here (adding hydrogens, deleting water) stops them matching.
    pdb_data = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    forcefield = app.ForceField(*forcefield_names, result.forcefield_xml)

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

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['GGG1:O1', 'CCC1:H2', 'CCC1:N1'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['CCC1:N2', 'GGG1:H3', 'GGG1:N3'])

    plumed_input, sum_hills_input = rt.plumed_input_2pt_1d(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", filename="fes", show=True)
    plt.close()

    rt.plot_plumed_colvar("COLVAR", filename="colvar", show=True)
    plt.close()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')
    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('STATE')
    nqe.remove_file('KERNELS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')
    nqe.remove_file('index_atoms.pdb')
    nqe.remove_file('neb_path.pdb')
    nqe.remove_file('neb_path.xyz')
    nqe.remove_file('ligands.xml')
    nqe.remove_directory('forcefill_work')
