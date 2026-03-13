import os

import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit
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


def test_malonaldehyde_pt():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    pdb = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    forcefield = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small
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
    potential = MLPotential('mace-omat-0-medium')  # mace-off23-large mace-off23-small

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

    # nqe.remove_file_pattern('minimized*')
    # nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_eq_workflow_plumed_pt():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000

    pdb = app.PDBFile("tests/data/pdb/gt_wob_solv_clean.pdb")
    forcefield = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    nqe.center_in_box(modeller)

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['DGN1:O6', 'DTN1:H3', 'DTN1:N3'])
    plumed_input, sum_hills_input = nqe.plumed_input_1pt(modeller, idx, temperature)

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['DGN1:O6', 'DTN1:H3', 'DTN1:N3'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['DTN1:O2', 'DGN1:H1', 'DGN1:N1'])
    plumed_input, sum_hills_input = nqe.plumed_input_2pt_1d(modeller, idx1, idx2, temperature)

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['DGN1:O6', 'DTN1:H3', 'DTN1:N3'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['DTN1:O2', 'DGN1:H1', 'DGN1:N1'])
    plumed_input, sum_hills_input = nqe.plumed_input_2pt_2d(modeller, idx1, idx2, temperature)

    # idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['DGN1:O6', 'DTN1:H3', 'DTN1:N3'])
    # idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['DGN1:O6', 'DTN1:H3', 'DTN1:O4'])
    # plumed_input, sum_hills_input = nqe.plumed_input_wob_1(idx1, idx2, temperature)

    # idx_n3,
    # idx_h3,
    # idx_o6,
    # idx_o4,
    # idx_n1,
    # idx_h1,
    # idx_o2,
    # idx_n2,
    # DGN1 DTN1
    picks = ['DTN1:N3',
             'DTN1:H3',
             'DGN1:O6',
             'DTN1:O4',
             'DGN1:N1',
             'DGN1:H1',
             'DTN1:O2',
             'DGN1:N2',
             ]

    idx = nqe.atom_indices_from_vmd_picks(modeller, picks)

    (plumed_input,
     sum_hills_input) = nqe.plumed_input_wob_2(modeller,
                                               idx,
                                               temperature)

    # Write PLUMED script to a temporary file
    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)

    # Minimise the system first
    nqe.run_openmm_relaxation_simple(modeller,
                                     forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        temperature=temperature,
                        steps=steps_prod)

    # Run PLUMED sum_hills to get FES
    os.system(sum_hills_input)
    # Plot FES
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    # n_beads = 4
    # pdb = app.PDBFile("minimized.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_rpmd_equilibration(modeller,
    #                                   forcefield,
    #                                   n_beads=n_beads,
    #                                   n_report=10,
    #                                   n_1=100,
    #                                   n_2=100)
    #
    # pdb = app.PDBFile("rpmd_ready_centroid.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_rpmd_prod(modeller,
    #                          forcefield,
    #                          n_beads=n_beads,
    #                          steps=steps_prod,
    #                          plumed_script_path=plumed_script_path,
    #                          checkpoint_file='rpmd_ready.chk')
    #
    # # Run PLUMED sum_hills to get FES
    # os.system(sum_hills_input)
    # # Plot FES
    # nqe.plot_plumed_fes("fes.dat")
    # plt.show()
    # nqe.plot_plumed_colvar("COLVAR")
    # plt.show()

    nqe.remove_file_pattern('minimized*')
    # nqe.remove_file_pattern('prod*')
    nqe.remove_file_pattern('rpmd_ready*')
    nqe.remove_file_pattern('rpmd_prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')

    nqe.remove_file('bck.0.COLVAR')
    nqe.remove_file('bck.0.HILLS')
    nqe.remove_file('bck.0.fes.dat')
