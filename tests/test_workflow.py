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

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_relaxation(modeller, forcefield, platform_name='CUDA')
    nqe.remove_file('minimized.pdb')


def test_run_openmm_heating():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_heating(modeller, forcefield)
    nqe.remove_file_pattern('equilibrate*')


def test_run_openmm_heating_deuterate():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_heating(modeller, forcefield, deuterate=True)
    nqe.remove_file_pattern('equilibrate*')


def test_run_openmm_npt():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_npt(modeller, forcefield)
    nqe.remove_file_pattern('npt_equilibrated*')


def test_eq_workflow():
    print(flush=True)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')

    nqe.run_openmm_relaxation(modeller, forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield)

    nqe.remove_file('minimized.pdb')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrated*')


def test_eq_workflow_mixed():
    print(flush=True)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    padding = 2.0
    box_shape = 'dodecahedron'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

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
    nqe.remove_file_pattern('npt_equilibrated*')


def test_eq_workflow_plumed_dihedral():
    print(flush=True)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    modeller.addSolvent(forcefield,
                        padding=1.0 * unit.nanometer,
                        boxShape='cube')
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
    nqe.run_openmm_relaxation_simple(modeller, forcefield, platform_name='CUDA')

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield, platform_name='CUDA')

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield, platform_name='CUDA')

    pdb = app.PDBFile("npt_equilibrated.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        platform_name='CUDA',
                        steps=50_000)

    # Run PLUMED sum_hills to get FES
    os.system(f'plumed sum_hills --hills HILLS --outfile fes.dat --min -pi --max pi --bin 300 --kt 2.494')
    # Plot FES
    nqe.plot_plumed_fes("fes.dat")
    plt.show()

    nqe.plot_plumed_colvar("COLVAR")
    plt.show()

    n_beads = 4
    pdb = app.PDBFile("npt_equilibrated.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_rpmd_equilibration(modeller,
                                      forcefield,
                                      platform_name='CUDA',
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
                             platform_name='CUDA',
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
    nqe.remove_file_pattern('npt_equilibrated*')
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
                                     forcefield,
                                     platform_name='CUDA')

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
                        platform_name='CUDA',
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
                        platform_name='CUDA',
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
                                      platform_name='CUDA',
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
                             platform_name='CUDA',
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

    pdb = app.PDBFile("npt_equilibrated.pdb")
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
                        platform_name='CUDA',
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
    nqe.remove_file_pattern('npt_equilibrated*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def test_fad_pt_solvated():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 40_000

    input_pdb = 'tests/data/pdb/fad.pdb'
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

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:O2', 'LIG1:H2', 'FAD1:O1'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['FAD1:O2', 'FAD1:H2', 'LIG1:O1'])
    plumed_input, sum_hills_input = nqe.plumed_input_2pt_1d(modeller,
                                                            idx1,
                                                            idx2,
                                                            temperature,
                                                            wall=1.0,
                                                            height=12.0,
                                                            bias=5.0)

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)

    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        platform_name='CUDA',
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


def test_eq_workflow_plumed_pt():
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000

    pdb = app.PDBFile("tests/data/pdb/gt_wob_solv_clean.pdb")
    forcefield = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

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
                                     forcefield,
                                     platform_name='CUDA')

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        platform_name='CUDA',
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
    #                                   platform_name='CUDA',
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
    #                          platform_name='CUDA',
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
