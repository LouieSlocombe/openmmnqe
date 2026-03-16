import os

import matplotlib.pyplot as plt
import numpy as np
import openmm.app as app
import openmm.unit as unit
from ase.io import read
from openmmml import MLPotential

import openmmnqe as nqe


def test_run_openmm_relaxation():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    # nqe.center_in_box(modeller)

    nqe.run_openmm_relaxation(modeller, forcefield)
    nqe.remove_file('minimized.pdb')


def test_run_openmm_heating():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    nqe.run_openmm_heating(modeller, forcefield)
    nqe.remove_file_pattern('equilibrate*')


def test_run_openmm_heating_deuterate():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    nqe.run_openmm_heating(modeller, forcefield, deuterate=True)
    nqe.remove_file_pattern('equilibrate*')


def test_run_openmm_npt():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    nqe.run_openmm_npt(modeller, forcefield)
    nqe.remove_file_pattern('npt_equilibrate*')


def test_eq_workflow():
    print(flush=True)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    nqe.run_openmm_relaxation(modeller, forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield)

    nqe.remove_file('minimized.pdb')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')


def test_eq_workflow_mixed():
    print(flush=True)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    nqe.run_openmm_relaxation(modeller, forcefield, potential=potential, ml_idx=ml_atoms)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield, potential=potential, ml_idx=ml_atoms)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield, potential=potential, ml_idx=ml_atoms)

    nqe.remove_file('minimized.pdb')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')


def test_eq_workflow_plumed_dihedral():
    print(flush=True)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['ALA1:C', 'ALA2:N', 'ALA2:CA', 'ALA2:C'])
    idx_str = ",".join([str(i + 1) for i in idx])

    plumed_input = f"""
phi: TORSION ATOMS={idx_str}
# Units: default length=nm, energy=kJ/mol, time=ps (depends on your MD engine interface)
metad: METAD ARG=phi PACE=500 HEIGHT=4.0 SIGMA=0.35 BIASFACTOR=12 TEMP=300 GRID_MIN=-pi GRID_MAX=pi GRID_BIN=300 FILE=HILLS
PRINT STRIDE=200 ARG=phi,metad.bias FILE=COLVAR
"""
    # Write PLUMED script to a temporary file
    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
    # Minimise the system first
    nqe.run_openmm_relaxation_simple(modeller, forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield)

    pdb = app.PDBFile("npt_equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        steps=50_000)

    # Run PLUMED sum_hills to get FES
    os.system(f'plumed sum_hills --hills HILLS --outfile fes.dat --min -pi --max pi --bin 300 --kt 2.494')
    # Plot FES
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    n_beads = 4
    pdb = app.PDBFile("npt_equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_rpmd_equilibration(modeller,
                                      forcefield,
                                      n_beads=n_beads,
                                      n_report=100,
                                      n_1=1_000,
                                      n_2=1_000)

    pdb = app.PDBFile("rpmd_ready_centroid.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_rpmd_prod(modeller,
                             forcefield,
                             n_beads=n_beads,
                             steps=50_000,
                             plumed_script_path=plumed_script_path,
                             checkpoint_file='rpmd_ready.chk')

    # Run PLUMED sum_hills to get FES
    os.system(f'plumed sum_hills --hills HILLS --outfile fes.dat --min -pi --max pi --bin 300 --kt 2.494')
    # Plot FES
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file_pattern('rpmd_ready*')
    nqe.remove_file_pattern('rpmd_prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')

    nqe.remove_file('bck.0.COLVAR')
    nqe.remove_file('bck.0.HILLS')
    nqe.remove_file('bck.0.fes.dat')


def test_eq_workflow_plumed_dihedral_opes():
    print(flush=True)
    n_steps = 100_000
    temperature = 300.0 * unit.kelvin
    kbt = nqe.temperature_to_kbt(temperature)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['ALA1:C', 'ALA2:N', 'ALA2:CA', 'ALA2:C'])
    idx_str = ",".join([str(i + 1) for i in idx])

    plumed_input = f"""
phi: TORSION ATOMS={idx_str}
metad: OPES_METAD ARG=phi PACE=500 BARRIER=4.0 SIGMA=0.2 TEMP={temperature.value_in_unit(unit.kelvin)} STATE_WFILE=STATE STATE_WSTRIDE=500
PRINT STRIDE=200 ARG=phi,metad.bias FILE=COLVAR
"""

    # Write PLUMED script to a temporary file
    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
    # Minimise the system first
    nqe.run_openmm_relaxation_simple(modeller, forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield)

    pdb = app.PDBFile("npt_equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        steps=n_steps)

    # Run PLUMED sum_hills to get FES
    fes_cmd = os.path.join(nqe.openmm_nqe_dir, "opes", "FES_from_State.py")
    os.system(f'python3 {fes_cmd} --state STATE --min 0 --max 4.0 --bin 100 --kt {kbt}')
    # Plot FES
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('KERNELS')
    nqe.remove_file_pattern('*STATE')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_malonaldehyde_pt():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    pdb = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    forcefield = MLPotential('mace-omol-0-extra-large')  # mace-off23-large mace-off23-small
    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O2', 'LIG1:H5', 'LIG1:O1'])
    plumed_input, sum_hills_input = nqe.plumed_input_1pt(modeller,
                                                         idx,
                                                         temperature,
                                                         height=8.0,
                                                         bias=5.0)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
    forcefield = MLPotential('mace-off23-small')
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        temperature=temperature,
                        barostat_freq=None,
                        steps=steps_prod)

    # Run PLUMED sum_hills to get FES
    os.system(sum_hills_input)
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_malonaldehyde_pt_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 40_000

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

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

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O2', 'LIG1:H5', 'LIG1:O1'])
    plumed_input, sum_hills_input = nqe.plumed_input_1pt(modeller,
                                                         idx,
                                                         temperature,
                                                         height=8.0,
                                                         bias=5.0)

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
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_malonaldehyde_pt_quantum_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000
    n_beads = 2

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

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

    nqe.run_openmm_rpmd_equilibration(modeller,
                                      forcefield,
                                      n_beads=n_beads,
                                      n_report=1,
                                      n_1=100,
                                      n_2=100)

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O2', 'LIG1:H5', 'LIG1:O1'])
    plumed_input, sum_hills_input = nqe.plumed_input_1pt(modeller,
                                                         idx,
                                                         temperature,
                                                         height=8.0,
                                                         bias=5.0)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)

    nqe.run_openmm_rpmd_prod(modeller,
                             forcefield,
                             n_beads=n_beads,
                             plumed_script_path=plumed_script_path,
                             temperature=temperature,
                             barostat_freq=None,
                             steps=steps_prod,
                             potential=potential,
                             ml_idx=ml_atoms)

    # Run PLUMED sum_hills to get FES
    os.system(sum_hills_input)
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('rpmd_ready*')
    nqe.remove_file_pattern('rpmd_prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_malonaldehyde_pt_adqtb_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')

    rm_ions = ['Na+',
               'Cl-',
               'NA']
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb, rm_ions=rm_ions)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

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

    nqe.run_openmm_adqtb_eq(modeller,
                            forcefield,
                            n_report=1,
                            steps=100)

    pdb = app.PDBFile("adqtb_ready.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O2', 'LIG1:H5', 'LIG1:O1'])
    plumed_input, sum_hills_input = nqe.plumed_input_1pt(modeller,
                                                         idx,
                                                         temperature,
                                                         height=8.0,
                                                         bias=5.0)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)

    nqe.run_openmm_adqtb_prod(modeller,
                              forcefield,
                              plumed_script_path=plumed_script_path,
                              temperature=temperature,
                              steps=steps_prod,
                              potential=potential,
                              ml_idx=ml_atoms)

    # Run PLUMED sum_hills to get FES
    os.system(sum_hills_input)
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    # nqe.remove_file_pattern('adqtb_ready*')
    # nqe.remove_file_pattern('adqtb_prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_malonaldehyde_pt_solvated_full():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 50_000

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    rm_ions = ['Na+',
               'Cl-',
               'NA']
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb, rm_ions=rm_ions)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

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
    nqe.run_openmm_heating(modeller,
                           forcefield,
                           potential=potential,
                           ml_idx=ml_atoms)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller,
                       forcefield,
                       potential=potential,
                       ml_idx=ml_atoms)

    pdb = app.PDBFile("npt_equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O2', 'LIG1:H5', 'LIG1:O1'])
    plumed_input, sum_hills_input = nqe.plumed_input_1pt(modeller,
                                                         idx,
                                                         temperature,
                                                         height=8.0,
                                                         bias=5.0)

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
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_fad_pt_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 20_000

    input_pdb = 'tests/data/pdb/fad.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]
    print(f"ML atoms: {ml_atoms}", flush=True)

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     potential=potential,
                                     ml_idx=ml_atoms)
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['FAD1:O1', 'LIG1:H2', 'LIG1:O2'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O1', 'FAD1:H2', 'FAD1:O2'])

    plumed_input, sum_hills_input = nqe.plumed_input_2pt_1d(modeller,
                                                            idx1,
                                                            idx2,
                                                            temperature,
                                                            wall=1.0,
                                                            height=20.0,
                                                            bias=10.0)

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
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_gc_pt_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 40_000

    input_pdb = 'tests/data/pdb/gc.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]
    print(f"ML atoms: {ml_atoms}", flush=True)

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     potential=potential,
                                     ml_idx=ml_atoms)
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['GGG1:O1', 'CCC1:H2', 'CCC1:N1'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['CCC1:N2', 'GGG1:H3', 'GGG1:N3'])

    plumed_input, sum_hills_input = nqe.plumed_input_2pt_1d(modeller,
                                                            idx1,
                                                            idx2,
                                                            temperature,
                                                            wall=1.0,
                                                            height=20.0,
                                                            bias=10.0)

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
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_gc_pt_quantum_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 60_000
    n_beads = 4

    input_pdb = 'tests/data/pdb/gc.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]
    print(f"ML atoms: {ml_atoms}", flush=True)

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     potential=potential,
                                     ml_idx=ml_atoms)
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    nqe.run_openmm_rpmd_equilibration(modeller,
                                      forcefield,
                                      n_beads=n_beads,
                                      n_report=1,
                                      n_1=100,
                                      n_2=100)

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['GGG1:O1', 'CCC1:H2', 'CCC1:N1'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['CCC1:N2', 'GGG1:H3', 'GGG1:N3'])

    plumed_input, sum_hills_input = nqe.plumed_input_2pt_1d(modeller,
                                                            idx1,
                                                            idx2,
                                                            temperature,
                                                            wall=1.0,
                                                            height=20.0,
                                                            bias=10.0)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)

    nqe.run_openmm_rpmd_prod(modeller,
                             forcefield,
                             n_beads=n_beads,
                             plumed_script_path=plumed_script_path,
                             temperature=temperature,
                             barostat_freq=None,
                             steps=steps_prod,
                             potential=potential,
                             ml_idx=ml_atoms)

    # Run PLUMED sum_hills to get FES
    os.system(sum_hills_input)
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('rpmd_ready*')
    nqe.remove_file_pattern('rpmd_prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_g_enol_t_pt_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000

    input_pdb = 'tests/data/pdb/G_enol_T.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]
    print(f"ML atoms: {ml_atoms}", flush=True)

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     potential=potential,
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
                                                            height=10.0,
                                                            bias=10.0)

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
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_gt_wob_pt_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000

    input_pdb = 'tests/data/pdb/G_T_wob.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()] + [atom.index for atom in chains[1].atoms()]
    print(f"ML atoms: {ml_atoms}", flush=True)

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     potential=potential,
                                     ml_idx=ml_atoms)
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

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
                                                           height=100.0,
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
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    # nqe.remove_file_pattern('minimized*')
    # nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def convert_xyz_to_plumed_ref(xyz_file, template_pdb, output_file):
    with open(template_pdb, 'r') as f:
        template_atoms = [line for line in f if line.startswith("HETATM")]

    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    frames = []
    for i in range(0, len(lines), num_atoms + 2):
        frame_coords = lines[i + 2: i + num_atoms + 2]
        frames.append([l.split()[1:] for l in frame_coords])

    with open(output_file, 'w') as f:
        f.write("REMARK TYPE=MULTI-ST-PDB\n")
        f.write("REMARK ARG=path.s,path.z\n")
        for i, frame in enumerate(frames):
            f.write(f"REMARK NUMBER={i + 1}\n")
            f.write(f"REMARK STEP={i}\n")
            for j, coords in enumerate(frame):
                t_line = template_atoms[j]
                new_line = (t_line[:30] +
                            f"{float(coords[0]):8.3f}{float(coords[1]):8.3f}{float(coords[2]):8.3f}" +
                            t_line[54:])
                f.write(new_line)
            f.write("ENDMDL\n")


def calculate_plumed_lambda(xyz_path, units_to_nm=True):
    """
    Calculates the optimal LAMBDA for PLUMED PATHMSD.
    Logic: lambda = 2.3 / average_MSD_between_frames

    Parameters:
    xyz_path (str): Path to your .xyz NEB trajectory.
    units_to_nm (bool): Convert ASE Angstroms to nm (standard for PLUMED).
    """
    # Load all frames from the xyz file
    frames = read(xyz_path, index=':')
    num_frames = len(frames)

    if num_frames < 2:
        raise ValueError("The path file must contain at least 2 frames.")

    msd_values = []

    for i in range(num_frames - 1):
        # Get positions as numpy arrays
        pos1 = frames[i].get_positions()
        pos2 = frames[i + 1].get_positions()

        # Convert to nm if necessary
        if units_to_nm:
            pos1 /= 10.0
            pos2 /= 10.0

        # Calculate MSD between frame i and frame i+1
        # Squared displacement per atom, then averaged over all atoms
        sq_diff = np.sum((pos2 - pos1) ** 2, axis=1)
        msd = np.mean(sq_diff)
        msd_values.append(msd)

    avg_msd = np.mean(msd_values)

    # The heuristic for a smooth path is lambda = 2.3 / avg_msd
    optimal_lambda = 2.3 / avg_msd

    print(f"--- Path Analysis ---")
    print(f"Total frames: {num_frames}")
    print(f"Average MSD:  {avg_msd:.6f} {'nm^2' if units_to_nm else 'A^2'}")
    print(f"Suggested LAMBDA: {optimal_lambda:.2f}")

    return optimal_lambda


def test_malonaldehyde_pathmsd():
    from ase.io import read, write
    from mace.calculators.foundations_models import mace_off
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 20_000
    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule)

    # padding = 1.5
    # box_shape = 'cube'
    # modeller.addSolvent(forcefield,
    #                     padding=padding * unit.nanometer,
    #                     boxShape=box_shape)
    # nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield,
                                     potential=potential,
                                     ml_idx=ml_atoms)

    reactant = read("minimized.pdb")
    # view(reactant)

    product = nqe.swap_bonding_configuration(reactant, 0, 8, 1)

    calc = mace_off(model_name='mace-off23-small', device='cuda')
    product = nqe.optimise_geom(product, calc, fmax=0.1)
    # view(product)

    neb_path = nqe.quick_guess_path(reactant, product)
    # view(neb_path)

    write("neb_path.xyz", neb_path)
    convert_xyz_to_plumed_ref("neb_path.xyz", "minimized.pdb", "neb_path.pdb")
    lam = calculate_plumed_lambda("neb_path.xyz")

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    temperature_str = temperature.value_in_unit(unit.kelvin)
    kt_str = nqe.temperature_to_kbt(temperature)

    plumed_input = f'''
FIT_TO_TEMPLATE REFERENCE=neb_path.pdb TYPE=OPTIMAL
path: PATHMSD REFERENCE=neb_path.pdb LAMBDA=250.0 NEIGH_SIZE=8 
metad: METAD ARG=path.sss PACE=500 HEIGHT=8.0 SIGMA=0.1 BIASFACTOR=5 TEMP={temperature_str} FILE=HILLS CALC_RCT GRID_MIN=1.0 GRID_MAX=25.0 GRID_BIN=500
path_limit: UPPER_WALLS ARG=path.zzz AT=0.05 KAPPA=500.0
PRINT ARG=path.sss,path.zzz,metad.bias STRIDE=500 FILE=COLVAR
    '''
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --kt {kt_str}'
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
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()
    #
    # nqe.remove_file_pattern('minimized*')
    # nqe.remove_file_pattern('prod*')
    #
    # nqe.remove_file('COLVAR')
    # nqe.remove_file('HILLS')
    # nqe.remove_file('fes.dat')
    # nqe.remove_file('plumed.dat')
