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


def plumed_input_1pt(idx,
                     temperature,
                     r_0=0.14,  # nm
                     wall=0.37,  # nm
                     pace=500,
                     height=10.0,  # kJ/mol
                     sigma=0.05,  # nm
                     bias=10.0,
                     grid_min=-1.2,
                     grid_max=1.2,
                     grid_bin=200):
    idx = nqe.atom_indices_to_plumed(idx)
    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    c_d: COORDINATION GROUPA={idx[0]} GROUPB={idx[1]} R_0={r_0}
    c_a: COORDINATION GROUPA={idx[2]} GROUPB={idx[1]} R_0={r_0}
    pt_cv: COMBINE ARG=c_d,c_a COEFFICIENTS=1,-1 PERIODIC=NO
    
    dist_da: DISTANCE ATOMS={idx[2]},{idx[0]}
    uwall: UPPER_WALLS ARG=dist_da AT={wall} KAPPA=3000
    
    metad: METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}
    PRINT ARG=c_d,c_a,pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input


def plumed_input_2pt_1d(idx1,
                        idx2,
                        temperature,
                        r_0=0.14,  # nm
                        wall=0.37,  # nm
                        pace=500,
                        height=10.0,  # kJ/mol
                        sigma=0.05,  # nm
                        bias=10.0,
                        grid_min=-1.2,
                        grid_max=1.2,
                        grid_bin=200):

    idx1 = nqe.atom_indices_to_plumed(idx1)
    idx2 = nqe.atom_indices_to_plumed(idx2)

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    c_d1: COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r_0}
    c_a1: COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r_0}
    cv_diff1: COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1,-1 PERIODIC=NO
    
    dist_da_1: DISTANCE ATOMS={idx1[2]},{idx1[0]}
    u_wall_1: UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA=3000
    
    c_d2: COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r_0}
    c_a2: COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r_0}
    cv_diff2: COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1,-1 PERIODIC=NO
    
    dist_da_2: DISTANCE ATOMS={idx2[2]},{idx2[0]}
    u_wall_2: UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA=3000
    
    pt_cv: COMBINE ARG=cv_diff1,cv_diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO
    
    metad: METAD ARG=pt_cv PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}
    PRINT ARG=pt_cv,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min} --max {grid_max} --bin {grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input


def plumed_input_2pt_2d(idx1,
                        idx2,
                        temperature,
                        r_0=0.14,  # nm
                        wall=0.37,  # nm
                        pace=500,
                        height=20.0,  # kJ/mol
                        sigma=0.05,  # nm
                        bias=10.0,
                        grid_min=-1.2,
                        grid_max=1.2,
                        grid_bin=200):
    idx1 = nqe.atom_indices_to_plumed(idx1)
    idx2 = nqe.atom_indices_to_plumed(idx2)

    temperature_str = str(temperature.value_in_unit(unit.kelvin))
    kt = unit.MOLAR_GAS_CONSTANT_R * temperature
    kt_str = kt.value_in_unit(unit.kilojoule_per_mole)
    plumed_input = f"""
    c_d1: COORDINATION GROUPA={idx1[0]} GROUPB={idx1[1]} R_0={r_0}
    c_a1: COORDINATION GROUPA={idx1[2]} GROUPB={idx1[1]} R_0={r_0}
    cv_diff1: COMBINE ARG=c_d1,c_a1 COEFFICIENTS=1,-1 PERIODIC=NO

    dist_da_1: DISTANCE ATOMS={idx1[2]},{idx1[0]}
    u_wall_1: UPPER_WALLS ARG=dist_da_1 AT={wall} KAPPA=3000

    c_d2: COORDINATION GROUPA={idx2[0]} GROUPB={idx2[1]} R_0={r_0}
    c_a2: COORDINATION GROUPA={idx2[2]} GROUPB={idx2[1]} R_0={r_0}
    cv_diff2: COMBINE ARG=c_d2,c_a2 COEFFICIENTS=1,-1 PERIODIC=NO

    dist_da_2: DISTANCE ATOMS={idx2[2]},{idx2[0]}
    u_wall_2: UPPER_WALLS ARG=dist_da_2 AT={wall} KAPPA=3000

    metad: METAD ARG=cv_diff1,cv_diff2 PACE={pace} HEIGHT={height} SIGMA={sigma},{sigma} BIASFACTOR={bias} TEMP={temperature_str} FILE=HILLS GRID_MIN={grid_min},{grid_min} GRID_MAX={grid_max},{grid_max} GRID_BIN={grid_bin},{grid_bin}
    PRINT ARG=cv_diff1,cv_diff2,metad.bias STRIDE={pace} FILE=COLVAR
        """
    sum_hills_input = f'plumed sum_hills --hills HILLS --outfile fes.dat --min {grid_min},{grid_min} --max {grid_max},{grid_max} --bin {grid_bin},{grid_bin} --kt {kt_str}'
    return plumed_input, sum_hills_input

def test_eq_workflow_plumed_pt():
    print(flush=True)
    temperature = 300.0 * unit.kelvin

    pdb = app.PDBFile("tests/data/pdb/gt_wob_solv_clean.pdb")
    forcefield = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # A..H-D: 'DGN1:O6', 'DTN1:H3', 'DTN1:N3'
    # A..H-D: 'DTN1:O2', 'DGN1:H1', 'DGN1:N1'
    idx = nqe.atom_indices_from_vmd_picks(modeller, ['DGN1:O6', 'DTN1:H3', 'DTN1:N3'])
    plumed_input, sum_hills_input = plumed_input_1pt(idx, temperature)

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['DGN1:O6', 'DTN1:H3', 'DTN1:N3'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['DTN1:O2', 'DGN1:H1', 'DGN1:N1'])
    plumed_input, sum_hills_input = plumed_input_2pt_1d(idx1, idx2, temperature)

    idx1 = nqe.atom_indices_from_vmd_picks(modeller, ['DGN1:O6', 'DTN1:H3', 'DTN1:N3'])
    idx2 = nqe.atom_indices_from_vmd_picks(modeller, ['DTN1:O2', 'DGN1:H1', 'DGN1:N1'])
    plumed_input, sum_hills_input = plumed_input_2pt_2d(idx1, idx2, temperature)

    # Write PLUMED script to a temporary file
    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)

    # Minimise the system first
    nqe.run_openmm_relaxation_simple(modeller, forcefield, platform_name='CUDA')

    pdb = app.PDBFile("minimized.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_heating(modeller, forcefield, platform_name='CUDA')
    #
    # pdb = app.PDBFile("equilibrate.pdb")
    # modeller = app.Modeller(pdb.topology, pdb.positions)
    # nqe.run_openmm_npt(modeller, forcefield, platform_name='CUDA')
    #
    # pdb = app.PDBFile("npt_equilibrated.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_prod(modeller,
                        forcefield,
                        plumed_script_path=plumed_script_path,
                        platform_name='CUDA',
                        temperature=temperature,
                        steps=50_000)

    # Run PLUMED sum_hills to get FES
    os.system(sum_hills_input)
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
