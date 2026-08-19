"""Classical, ML/MM, enhanced-sampling, RPMD, and adQTB workflows."""

import argparse
import logging
import os

import forcefill as ff
import openmm.app as app
import openmm.unit as unit
from ase.io import read, write
from mace.calculators.foundations_models import mace_off
from openmmml import MLPotential

import openmmnqe as nqe
import reactiontools as rt

BASE_FORCEFIELD = ("amber14-all.xml", "amber14/tip3pfb.xml")


def ligand_forcefield(input_pdb, base_forcefield=BASE_FORCEFIELD):
    """
    Parameterise ligands and return the matching modeller and force field.

    The Modeller is built from the same file forcefill read, with nothing in
    between, because the generated templates describe those residues exactly
    as that file spells them.

    Parameters
    ----------
    input_pdb : str
        Structure to parameterise. It must already be repaired: forcefill
        decides what needs parameters by asking what the base force field
        cannot match.
    base_forcefield : tuple of str, optional
        Standard force field files the generated ffxml loads underneath.
        Default is ``BASE_FORCEFIELD``.

    Returns
    -------
    modeller : openmm.app.Modeller
        Modeller holding *input_pdb* exactly as forcefill saw it.
    forcefield : openmm.app.ForceField
        The base files plus the generated ligand ffxml.

    Raises
    ------
    RuntimeError
        If forcefill skipped any residue, which would leave the system
        without parameters for it.
    """
    result = ff.build_forcefield_xml(
        input_pdb,
        "ligands.xml",
        base_forcefield=base_forcefield,
        workdir="forcefill_work",
    )
    if result.skipped:
        raise RuntimeError(f"forcefill skipped residues: {result.skipped}")
    extra = [] if result.forcefield_xml is None else [result.forcefield_xml]
    pdb = app.PDBFile(input_pdb)
    return (
        app.Modeller(pdb.topology, pdb.positions),
        app.ForceField(*base_forcefield, *extra),
    )


def run_openmm_relaxation():
    """Minimise a solvated peptide with the staged, restrained relaxation."""
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

    nqe.run_openmm_relaxation(modeller, forcefield)
    nqe.remove_file('minimized.pdb')


def run_openmm_heating():
    """Heat a solvated peptide to its target temperature under restraints."""
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


def run_openmm_heating_deuterate():
    """Heat the same system with every hydrogen replaced by deuterium."""
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


def run_openmm_npt():
    """Equilibrate a solvated peptide at constant pressure."""
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


def run_eq_workflow():
    """Run the full classical equilibration: relax, heat, then NPT."""
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


def run_eq_workflow_mixed():
    """Run the same equilibration with MACE on the solute and MM on the water."""
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


def run_eq_workflow_plumed_dihedral():
    """Bias a peptide dihedral with metadynamics, classically and then with RPMD."""
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
    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
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

    os.system('plumed sum_hills --hills HILLS --outfile fes.dat --min -pi --max pi --bin 300 --kt 2.494')
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    n_beads = 4
    pdb = app.PDBFile("npt_equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_rpmd_equilibration(modeller,
                                      forcefield,
                                      n_beads=n_beads,
                                      n_report=100,
                                      n_1=1_000,
                                      n_2=1_000)

    pdb = app.PDBFile("rpmd_ready_final.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_rpmd_prod(modeller,
                             forcefield,
                             n_beads=n_beads,
                             steps=50_000,
                             plumed_script_path=plumed_script_path,
                             checkpoint_file='rpmd_ready.chk')

    os.system('plumed sum_hills --hills HILLS --outfile fes.dat --min -pi --max pi --bin 300 --kt 2.494')
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

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


def run_eq_workflow_plumed_dihedral_opes():
    """Bias the same dihedral with OPES instead of metadynamics."""
    print(flush=True)
    n_steps = 100_000
    temperature = 300.0 * unit.kelvin
    kbt = rt.thermal_energy(temperature)

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

    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
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

    rt.run_opes_fes(state="STATE", grid_min=0, grid_max=4.0, grid_bin=100, kt=kbt)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('KERNELS')
    nqe.remove_file_pattern('*STATE')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_malonaldehyde_pt():
    """Malonaldehyde proton transfer in vacuum, biased on the single transfer coordinate."""
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
    plumed_input, sum_hills_input = rt.plumed_input_1pt(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_malonaldehyde_pt_solvated():
    """The same transfer with explicit solvent and an ML/MM split."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 40_000

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small
    modeller, forcefield = ligand_forcefield(input_pdb)

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
    plumed_input, sum_hills_input = rt.plumed_input_1pt(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_malonaldehyde_pt_quantum_solvated():
    """The solvated transfer again with RPMD, so the proton itself is quantised."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000
    n_beads = 2

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    modeller, forcefield = ligand_forcefield(input_pdb)

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
    plumed_input, sum_hills_input = rt.plumed_input_1pt(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('rpmd_ready*')
    nqe.remove_file_pattern('rpmd_prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_malonaldehyde_pt_adqtb_solvated():
    """The solvated transfer with adQTB supplying the nuclear quantum effects instead."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 10_000

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')

    modeller, forcefield = ligand_forcefield(input_pdb)

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
    plumed_input, sum_hills_input = rt.plumed_input_1pt(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_malonaldehyde_pt_solvated_full():
    """The solvated transfer after a full relax-heat-NPT equilibration."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 50_000

    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    modeller, forcefield = ligand_forcefield(input_pdb)

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
    plumed_input, sum_hills_input = rt.plumed_input_1pt(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_fad_pt_solvated():
    """Formic acid dimer double proton transfer, on a one-dimensional two-proton CV."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 20_000

    input_pdb = 'tests/data/pdb/fad.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    modeller, forcefield = ligand_forcefield(input_pdb)

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

    plumed_input, sum_hills_input = rt.plumed_input_2pt_1d(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_gc_pt_solvated():
    """Guanine-cytosine double proton transfer, on the same two-proton CV."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 40_000

    input_pdb = 'tests/data/pdb/gc.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    modeller, forcefield = ligand_forcefield(input_pdb)

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

    plumed_input, sum_hills_input = rt.plumed_input_2pt_1d(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_gc_pt_quantum_solvated():
    """The guanine-cytosine transfer again with RPMD on four beads."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 60_000
    n_beads = 4

    input_pdb = 'tests/data/pdb/gc.pdb'
    potential = MLPotential('mace-off23-small')  # mace-off23-large mace-off23-small

    modeller, forcefield = ligand_forcefield(input_pdb)

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

    plumed_input, sum_hills_input = rt.plumed_input_2pt_1d(modeller,
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

    os.system(sum_hills_input)
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('rpmd_ready*')
    nqe.remove_file_pattern('rpmd_prod*')

    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')


def run_malonaldehyde_pathmsd():
    """Bias malonaldehyde along a NEB reference path with PATHMSD rather than a bond CV."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    steps_prod = 100_000
    n_images = 10
    input_pdb = 'tests/data/pdb/malonaldehyde.pdb'
    ml_model = 'mace-off23-small'
    potential = MLPotential(ml_model)  # mace-off23-large mace-off23-small
    modeller, forcefield = ligand_forcefield(input_pdb)
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
    rt.pdb_remove_ter_index("minimized.pdb", "minimized.pdb")
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    nqe.save_only_index_atoms(modeller, ml_atoms, file_idx='index_atoms.pdb')

    reactant = read('index_atoms.pdb')

    product = rt.swap_bonding_configuration(reactant, 0, 8, 1)
    calc = mace_off(model_name=ml_model, device='cuda')
    product = rt.optimise_geom(product, calc, fmax=0.01)
    neb_path = rt.quick_guess_path(reactant, product, n_images=n_images)
    write("neb_path.xyz", neb_path)
    rt.convert_xyz_to_plumed_ref("neb_path.xyz", "index_atoms.pdb", "neb_path.pdb")
    lambda_val = rt.estimate_path_lambda("neb_path.pdb") * 0.5

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['LIG1:H5'])
    rt.strip_hydrogens_keep_indices("neb_path.pdb", "neb_path_tmp.pdb", keep=idx)
    rt.strip_hydrogens_keep_indices("index_atoms.pdb", "index_atoms_tmp.pdb", keep=idx)
    # Written to _tmp names above since strip_hydrogens_keep_indices can't write in place.
    os.rename("neb_path_tmp.pdb", "neb_path.pdb")
    os.rename("index_atoms_tmp.pdb", "index_atoms.pdb")

    rt.pdb_remove_ter_index("minimized.pdb", "minimized.pdb")
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    plumed_input, sum_hills_input = rt.plumed_input_neb_path(temperature,
                                                             grid_min=0.0,
                                                             grid_max=6.0,
                                                             lambda_val=lambda_val,
                                                             neigh_size=n_images,
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
    rt.plot_plumed_fes("fes.dat", show=True)

    rt.plot_plumed_colvar("COLVAR", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file('COLVAR')
    nqe.remove_file('HILLS')
    nqe.remove_file('KERNELS')
    nqe.remove_file('STATE')
    nqe.remove_file('fes.dat')
    nqe.remove_file('plumed.dat')
    nqe.remove_file('index_atoms.pdb')
    nqe.remove_file('neb_path.pdb')
    nqe.remove_file('neb_path.xyz')


EXAMPLES = {
    name.removeprefix("run_"): function
    for name, function in list(globals().items())
    if name.startswith("run_") and callable(function)
}


def main():
    """Run whichever workflow is named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    EXAMPLES[args.example]()


if __name__ == "__main__":
    main()
