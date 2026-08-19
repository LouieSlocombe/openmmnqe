"""Malonaldehyde intramolecular proton transfer, in explicit solvent.

The full pipeline in one script: forcefill parameterises the ligand, the
solute is treated with MACE and the water with the classical force field,
and a NEB guess of the transfer becomes the reference path a PATH collective
variable biases. The free-energy surface along that path is the output.
"""
import logging
import os

import forcefill as ff
import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit
from ase.io import read, write
from mace.calculators.foundations_models import mace_off
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
    steps_prod = 10_000
    input_pdb = os.path.join(os.path.dirname(nqe.openmm_nqe_dir), 'tests/data/pdb/malonaldehyde.pdb')
    ml_model = 'mace-off23-small'
    forcefield_names = ("amber14-all.xml", "amber14/tip3pfb.xml")
    potential = MLPotential(ml_model)  # mace-off23-large mace-off23-small

    # Malonaldehyde is not in any standard force field, so forcefill
    # parameterises it with GAFF2/AM1-BCC and writes an ffxml that loads
    # underneath the standard ones.  A charged ligand would need
    # net_charges={'LIG': -1}: a residue read out of a PDB carries no formal
    # charge, so forcefill assumes zero without saying so.
    result = ff.build_forcefield_xml(input_pdb,
                                     "ligands.xml",
                                     base_forcefield=forcefield_names,
                                     workdir="forcefill_work")
    # A skipped residue would only fail later, at createSystem, and forcefield_xml
    # is None when the base files already matched everything -- ForceField() will
    # not take a None.
    if result.skipped:
        raise RuntimeError(f"forcefill skipped residues: {result.skipped}")
    extra = [] if result.forcefield_xml is None else [result.forcefield_xml]
    print(f"Parameterised: {result.parameterized}", flush=True)

    # Built from the same file forcefill read, with nothing in between: the
    # templates describe those residues exactly as this file spells them, so an
    # edit here (adding hydrogens, deleting water) stops them matching.
    pdb_data = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    forcefield = app.ForceField(*forcefield_names, *extra)

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     potential=potential,
                                     ml_idx=ml_atoms)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    nqe.save_only_index_atoms(modeller, ml_atoms, file_idx='index_atoms.pdb')

    reactant = read('index_atoms.pdb')
    product = rt.swap_bonding_configuration(reactant, 0, 8, 1)
    calc = mace_off(model_name=ml_model, device='cuda')
    product = rt.optimise_geom(product, calc, fmax=0.01)
    neb_path = rt.quick_guess_path(reactant, product)
    write("neb_path.xyz", neb_path)
    rt.convert_xyz_to_plumed_ref("neb_path.xyz", "index_atoms.pdb", "neb_path.pdb")

    plumed_input, sum_hills_input = rt.plumed_input_neb_path(temperature)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller,
                           forcefield,
                           potential=potential,
                           ml_idx=ml_atoms,
                           target_temp=temperature)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller,
                       forcefield,
                       potential=potential,
                       ml_idx=ml_atoms,
                       temperature=temperature)

    pdb = app.PDBFile("npt_equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

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
    nqe.remove_file_pattern('prod*')
    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')
    nqe.remove_file('index_atoms.pdb')
    nqe.remove_file('neb_path.pdb')
    nqe.remove_file('neb_path.xyz')
    nqe.remove_file('ligands.xml')
    nqe.remove_directory('forcefill_work')
