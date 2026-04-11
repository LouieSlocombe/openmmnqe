import os

import openmm.app as app
import openmm.unit as unit
from ase.io import read, write
from mace.calculators.foundations_models import mace_mp

import openmmnqe as nqe

if __name__ == "__main__":
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps = 10_000
    input_pdb = 'G_enol_T.pdb'

    calc = mace_mp(model=os.path.join(os.environ['MACE_MODELS'], 'mace-mh-1.model'),
                   default_dtype="float32",
                   device="cuda",
                   head="omol")

    forcefield_names = ("amber19-all.xml", "amber19/tip3pfb.xml")
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(forcefield_names, molecule)

    pdb = app.PDBFile("npt_equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]

    nqe.run_openmm_adqtb_eq(modeller,
                            forcefield,
                            steps=steps,
                            calculator=calc,
                            ml_idx=ml_atoms,
                            )
