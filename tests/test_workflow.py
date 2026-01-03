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
    os.remove('minimized.pdb')


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
    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')


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
    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')


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
    os.remove('npt_equilibrated.chk')
    os.remove('npt_equilibrated.log')
    os.remove('npt_equilibrated.pdb')
    os.remove('npt_equilibrated_steps.pdb')


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

    os.remove('minimized.pdb')

    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')

    os.remove('npt_equilibrated.chk')
    os.remove('npt_equilibrated.log')
    os.remove('npt_equilibrated.pdb')
    os.remove('npt_equilibrated_steps.pdb')


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

    os.remove('minimized.pdb')

    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')

    os.remove('npt_equilibrated.chk')
    os.remove('npt_equilibrated.log')
    os.remove('npt_equilibrated.pdb')
    os.remove('npt_equilibrated_steps.pdb')


def test_eq_workflow_plumed_dihedral():
    print(flush=True)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    padding = 1.0
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    idx = nqe.atom_indices_from_vmd_picks(modeller, ['ALA1:C', 'ALA2:N', 'ALA2:CA', 'ALA2:C'])
    idx_str = ",".join([str(i - 1) for i in idx])

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

    os.remove('rpmd_ready.chk')
    os.remove('rpmd_ready.log')
    os.remove('rpmd_ready_centroid.pdb')
    for i in range(n_beads):
        os.remove(f'rpmd_ready_bead_{i}.pdb')

    os.remove('rpmd_prod.pdb')
    os.remove('rpmd_prod.chk')
    os.remove('rpmd_prod.log')
    os.remove('rpmd_prod_centroid.pdb')
    for i in range(n_beads):
        os.remove(f'rpmd_prod_bead_{i}.pdb')

    os.remove('minimized.chk')
    os.remove('minimized.log')
    os.remove('minimized.pdb')
    os.remove('minimized_steps.pdb')

    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')

    os.remove('npt_equilibrated.chk')
    os.remove('npt_equilibrated.log')
    os.remove('npt_equilibrated.pdb')
    os.remove('npt_equilibrated_steps.pdb')

    os.remove('prod.chk')
    os.remove('prod.log')
    os.remove('prod.pdb')
    os.remove('prod_steps.pdb')

    os.remove('COLVAR')
    os.remove('HILLS')
    os.remove('fes.dat')
    os.remove('plumed.dat')

    os.remove('bck.0.COLVAR')
    os.remove('bck.0.HILLS')
    os.remove('bck.0.fes.dat')


def test_eq_workflow_plumed_pt():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/gt_wob_solv_clean.pdb")
    forcefield = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # A..H-D: 'DGN1:O6', 'DTN1:H3', 'DTN1:N3'
    # A..H-D: 'DTN1:O2', 'DGN1:H1', 'DGN1:N1'
    picks = ['DTN1:H3', 'DTN1:N3']
    idx = nqe.atom_indices_from_vmd_picks(modeller, picks)
    idx_str = ",".join([str(i - 1) for i in idx])

    plumed_input = f"""
    r: DISTANCE ATOMS={idx_str}
    metad: METAD ARG=r PACE=500 HEIGHT=10.0 SIGMA=0.35 BIASFACTOR=12 TEMP=300 FILE=HILLS
    PRINT STRIDE=200 ARG=r,metad.bias FILE=COLVAR
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
    #
    # n_beads = 4
    # pdb = app.PDBFile("npt_equilibrated.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_rpmd_equilibration(modeller,
    #                                   forcefield,
    #                                   platform_name='CUDA',
    #                                   n_beads=n_beads,
    #                                   n_report=100,
    #                                   n_1=1_000,
    #                                   n_2=1_000)
    #
    # pdb = app.PDBFile("rpmd_ready_centroid.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_rpmd_prod(modeller,
    #                          forcefield,
    #                          n_beads=n_beads,
    #                          steps=50_000,
    #                          plumed_script_path=plumed_script_path,
    #                          platform_name='CUDA',
    #                          checkpoint_file='rpmd_ready.chk')
    #
    # # Run PLUMED sum_hills to get FES
    # os.system(f'plumed sum_hills --hills HILLS --outfile fes.dat --min -pi --max pi --bin 300 --kt 2.494')
    # # Plot FES
    # nqe.plot_plumed_fes("fes.dat")
    # plt.show()
    #
    # os.remove('rpmd_ready.chk')
    # os.remove('rpmd_ready.log')
    # os.remove('rpmd_ready_centroid.pdb')
    # for i in range(n_beads):
    #     os.remove(f'rpmd_ready_bead_{i}.pdb')
    #
    # os.remove('rpmd_prod.pdb')
    # os.remove('rpmd_prod.chk')
    # os.remove('rpmd_prod.log')
    # os.remove('rpmd_prod_centroid.pdb')
    # for i in range(n_beads):
    #     os.remove(f'rpmd_prod_bead_{i}.pdb')
    #
    # os.remove('minimized.chk')
    # os.remove('minimized.log')
    # os.remove('minimized.pdb')
    # os.remove('minimized_steps.pdb')
    #
    # os.remove('equilibrate.chk')
    # os.remove('equilibrate.log')
    # os.remove('equilibrate.pdb')
    # os.remove('equilibrate_steps.pdb')
    #
    # os.remove('npt_equilibrated.chk')
    # os.remove('npt_equilibrated.log')
    # os.remove('npt_equilibrated.pdb')
    # os.remove('npt_equilibrated_steps.pdb')
    #
    # os.remove('prod.chk')
    # os.remove('prod.log')
    # os.remove('prod.pdb')
    # os.remove('prod_steps.pdb')
    #
    # os.remove('COLVAR')
    # os.remove('HILLS')
    # os.remove('fes.dat')
    # os.remove('plumed.dat')
    #
    # os.remove('bck.0.COLVAR')
    # os.remove('bck.0.HILLS')
    # os.remove('bck.0.fes.dat')
