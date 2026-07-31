import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

import openmmnqe as nqe

if __name__ == "__main__":
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    input_pdb = 'G_T_wob.pdb'

    potential = MLPotential('mace-off23-small')

    forcefield_names = ("amber19-all.xml", "amber19/tip3pfb.xml")
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(forcefield_names, molecule)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]
    nqe.run_openmm_npt(modeller,
                       forcefield,
                       potential=potential,
                       ml_idx=ml_atoms,
                       temperature=temperature)
