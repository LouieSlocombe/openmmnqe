"""RPMD, adQTB, and quantum-reporter simulation examples."""

import argparse
from sys import stdout

import openmm.app as app
import openmm.unit as unit
from openmm import openmm
from openmmml import MLPotential

import openmmnqe as nqe

device = "CUDA"


def run_openmm_rpmd():
    """Run RPMD on a peptide in vacuum, driving the integrator by hand."""
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
        hydrogenMass=None,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      time=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    nqe.step_rpmd(simulation, n_steps)


def run_openmm_rpmd_solvated():
    """Run RPMD on the same peptide with explicit solvent around it."""
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
        hydrogenMass=None,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      time=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    nqe.step_rpmd(simulation, n_steps)


def run_openmm_rpmd_ml():
    """Run RPMD with MACE, rather than a classical force field, as the potential."""
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
        hydrogenMass=None,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      time=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    nqe.step_rpmd(simulation, n_steps)


def run_openmm_rpmd_mixed():
    """Run RPMD with MACE on the solute and the classical force field on the water."""
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
                                        removeCMMotion=True,
                                        hydrogenMass=None)

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
                                                      time=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    nqe.step_rpmd(simulation, n_steps)


def run_rpmd_quantum_spread_reporter():
    """Log how far two atoms' beads spread, then plot the result."""
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
        hydrogenMass=None,
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

    atoms_to_watch = [0, 1]
    atom_names = ["Atom0", "Atom1"]
    distance_pair = (0, 1)
    distance_name = "Atom0-Atom1"

    simulation.reporters.append(nqe.RPMDQuantumSpreadReporter(
        file="quantum_expansion.tsv",
        reportInterval=1,
        atom_indices=atoms_to_watch,
        names=atom_names,
        metric="mean",
        distance_pairs=[distance_pair],
        distance_names=[distance_name],
    ))

    print("Running RPMD with bead-expansion reporting...")
    nqe.step_rpmd(simulation, 500)
    plot_file = "quantum_expansion_vs_distance.png"
    nqe.plot_rpmd_atom_expansion(
        "quantum_expansion.tsv",
        distance_columns=f"Distance_{distance_name}(nm)",
        length_unit="angstrom",
        filename=plot_file,
        show=True,
    )
    print(f"Done. Wrote '{plot_file}'.")

    nqe.remove_file("quantum_expansion.tsv")


def run_rpmd_bead_reporter():
    """Write each bead's own trajectory to its own PDB file."""
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
        hydrogenMass=None,
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

    nqe.step_rpmd(simulation, 100)
    for i in range(n_beads):
        nqe.remove_file(f'out_bead_{i}.pdb')


def run_rpmd_centroid_reporter():
    """Write the bead-averaged centroid trajectory to a single PDB file."""
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
        hydrogenMass=None,
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

    nqe.step_rpmd(simulation, 100)
    nqe.remove_file('centroid.pdb')


def run_openmm_adqtb():
    """Run adQTB on a peptide, assigning particle types by element by hand."""
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
        hydrogenMass=None,
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


def run_openmm_rpmd_equilibration():
    """Equilibrate an RPMD system through the driver, leaving a bead-aware restart."""
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


def run_openmm_rpmd_prod():
    """Equilibrate, then run RPMD production from the restart it left behind."""
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


def run_openmm_rpmd_contracted():
    """Run contracted RPMD, evaluating the slow forces on fewer beads than the fast ones."""
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


def run_openmm_adqtb_eq():
    """Equilibrate an adQTB system through the driver, adapting its friction as it goes."""
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


def run_openmm_adqtb_prod():
    """Equilibrate, then run adQTB production on a solvated peptide."""
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


EXAMPLES = {
    name.removeprefix("run_"): function
    for name, function in list(globals().items())
    if name.startswith("run_") and callable(function)
}


def main():
    """Run whichever example is named on the command line."""
    global device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("--platform", default=device)
    args = parser.parse_args()

    device = args.platform
    EXAMPLES[args.example]()


if __name__ == "__main__":
    main()
