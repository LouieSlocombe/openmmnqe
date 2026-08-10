from sys import stdout

import numpy as np
import openmm.app as app
import openmm.unit as unit
from matplotlib import pyplot as plt
from openmm import openmm
from openmmml import MLPotential
import pytest

import openmmnqe as nqe

pytestmark = pytest.mark.pipeline

device = "CUDA"


def test_openmm_rpmd():
    print(flush=True)
    n_steps = 1_000
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    simulation.step(n_steps)


def test_openmm_rpmd_solvated():
    print(flush=True)
    n_steps = 200
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.0 * unit.nanometer,
                        boxShape='dodecahedron')

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=0.5 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    simulation.step(n_steps)


def test_openmm_rpmd_ml():
    print(flush=True)
    n_steps = 200
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = potential.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    simulation.step(n_steps)


def test_openmm_rpmd_mixed():
    print(flush=True)
    n_steps = 200
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 4
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    padding = 1.5
    box_shape = 'dodecahedron'

    pdb = app.PDBFile(in_pdb)
    potential = MLPotential('mace-off23-small')
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    has_box = modeller.topology.getUnitCellDimensions() is not None
    mm_system = forcefield.createSystem(modeller.topology,
                                        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
                                        nonbondedCutoff=1.0 * unit.nanometer,
                                        constraints=None,
                                        rigidWater=False,
                                        removeCMMotion=True)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms", flush=True)
    print(f"Number of ML atoms: {len(ml_atoms)}", flush=True)
    print(f"Number of MM atoms: {n_atoms - len(ml_atoms)}", flush=True)

    system = potential.createMixedSystem(modeller.topology,
                                         mm_system,
                                         ml_atoms)

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)

    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    simulation.step(n_steps)


def test_rpmd_quantum_spread_reporter():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    n_beads = 32
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond
    integrator = openmm.RPMDIntegrator(n_beads,
                                       temperature,
                                       friction,
                                       dt)

    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    for i in range(n_beads):
        integrator.setPositions(i, modeller.positions)
    nqe.init_beads(modeller, simulation, n_beads)

    atoms_to_watch = [0, 1]
    atom_names = ["Atom0", "Atom1"]

    simulation.reporters.append(nqe.RPMDQuantumSpreadReporter(
        file="quantum_spread.txt",
        reportInterval=1,
        atom_indices=atoms_to_watch,
        names=atom_names
    ))

    print("Running RPMD with Quantum Spread reporting...")
    simulation.step(500)
    print("Done. Check 'quantum_spread.txt'.")
    data = np.loadtxt("quantum_spread.txt", skiprows=1, delimiter='\t')

    plt.plot(data[:, 0], data[:, 1], label=atom_names[0])
    plt.plot(data[:, 0], data[:, 2], label=atom_names[1])
    plt.xlabel('Step')
    plt.ylabel('Quantum Rg (nm)')
    plt.legend()
    plt.show()

    nqe.remove_file("quantum_spread.txt")


def test_rpmd_bead_reporter():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    n_beads = 4
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond
    integrator = openmm.RPMDIntegrator(n_beads,
                                       temperature,
                                       friction,
                                       dt)

    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    nqe.init_beads(modeller, simulation, n_beads)

    simulation.reporters.append(nqe.RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name="out",
        reportInterval=10,
        num_beads=n_beads,
    ))

    simulation.step(100)
    for i in range(n_beads):
        nqe.remove_file(f'out_bead_{i}.pdb')


def test_rpmd_centroid_reporter():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    n_beads = 32
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond
    integrator = openmm.RPMDIntegrator(n_beads,
                                       temperature,
                                       friction,
                                       dt)

    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    nqe.init_beads(modeller, simulation, n_beads)

    simulation.reporters.append(nqe.RPMDCentroidReporter(
        topology=modeller.topology,
        file_name="centroid.pdb",
        reportInterval=10,
        num_beads=n_beads,
    ))

    simulation.step(100)
    nqe.remove_file('centroid.pdb')


def test_openmm_adqtb():
    print(flush=True)
    n_steps = 1_000
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms.")

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    integrator = openmm.QTBIntegrator(temperature, friction, dt)
    integrator.setSegmentLength(0.5 * unit.picosecond)

    nqe.set_adqtb_particle_types_by_element(integrator,
                                            topology=modeller.topology,
                                            system=system)

    integrator.setDefaultAdaptationRate(0.5)

    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    simulation.step(n_steps)


def test_run_openmm_rpmd_equilibration():
    print(flush=True)
    n_beads = 2
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_rpmd_equilibration(modeller,
                                      forcefield,
                                      n_beads=n_beads,
                                      n_report=1,
                                      platform_name=device,
                                      n_1=10,
                                      n_2=10)
    nqe.remove_file_pattern('rpmd_ready*')


def test_run_openmm_rpmd_prod():
    print(flush=True)
    n_beads = 2
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_rpmd_equilibration(modeller,
                                      forcefield,
                                      n_beads=n_beads,
                                      n_report=1,
                                      platform_name=device,
                                      n_1=10,
                                      n_2=10)

    nqe.run_openmm_rpmd_prod(modeller,
                             forcefield,
                             n_beads=n_beads,
                             platform_name=device,
                             steps=100)
    nqe.remove_file_pattern('rpmd_ready*')
    nqe.remove_file_pattern('rpmd_prod*')


def test_run_openmm_rpmd_contracted():
    print(flush=True)
    n_beads = 8
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_rpmd_equilibration(modeller,
                                      forcefield,
                                      n_beads=n_beads,
                                      platform_name=device,
                                      n_1=100,
                                      n_2=100)

    nqe.run_openmm_rpmd_contracted(modeller,
                                   forcefield,
                                   n_beads=n_beads,
                                   steps=100,
                                   n_report=1,
                                   platform_name=device)

    nqe.remove_file_pattern('rpmd_ready*')
    nqe.remove_file_pattern('rpmd_prod_contracted*')


def test_run_openmm_adqtb_eq():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_adqtb_eq(modeller,
                            forcefield,
                            platform_name=device,
                            n_report=1,
                            steps=100)

    nqe.remove_file_pattern('adqtb_ready*')


def test_run_openmm_adqtb_prod():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')

    nqe.run_openmm_adqtb_eq(modeller,
                            forcefield,
                            platform_name=device,
                            n_report=1,
                            steps=100)

    nqe.run_openmm_adqtb_prod(modeller,
                              forcefield,
                              platform_name=device,
                              n_report=1,
                              steps=100)

    nqe.remove_file_pattern('adqtb_ready*')
    nqe.remove_file_pattern('adqtb_prod*')
