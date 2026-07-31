import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

import openmmnqe as nqe

if __name__ == "__main__":
    print(flush=True)
    input_pdb = 'G_T_wob.pdb'

    potential = MLPotential('mace-off23-small')

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
                                     potential=potential,
                                     ml_idx=ml_atoms)
