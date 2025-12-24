import os
from sys import stdout

import numpy as np
import openmm.app as app
import openmm.unit as unit
from matplotlib import pyplot as plt
from openmm import openmm
from openmmml import MLPotential

import openmmnqe as nqe


def test_openmm_rpmd():
    # Simple run parameters
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
    platform = openmm.Platform.getPlatformByName("CUDA")
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
    # Simple run parameters
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

    # Solvate
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
    platform = openmm.Platform.getPlatformByName("CUDA")
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
    # Simple run parameters
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
    platform = openmm.Platform.getPlatformByName("CUDA")
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
    # Simple run parameters
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

    platform = openmm.Platform.getPlatformByName("CUDA")
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
    integrator = openmm.RPMDIntegrator(n_beads,
                                       300 * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       0.5 * unit.femtosecond)

    simulation = app.Simulation(modeller.topology, system, integrator)
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

    os.remove("quantum_spread.txt")


def test_rpmd_bead_reporter():
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
    integrator = openmm.RPMDIntegrator(n_beads,
                                       300 * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       0.5 * unit.femtosecond)

    simulation = app.Simulation(modeller.topology, system, integrator)

    nqe.init_beads(modeller, simulation, n_beads)

    simulation.reporters.append(nqe.RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name="out",
        reportInterval=10,
        num_beads=n_beads,
    ))

    simulation.step(100)
    for i in range(n_beads):
        os.remove(f'out_bead_{i}.pdb')


def test_rpmd_centroid_reporter():
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
    integrator = openmm.RPMDIntegrator(n_beads,
                                       300 * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       0.5 * unit.femtosecond)

    simulation = app.Simulation(modeller.topology, system, integrator)

    nqe.init_beads(modeller, simulation, n_beads)

    simulation.reporters.append(nqe.RPMDCentroidReporter(
        topology=modeller.topology,
        file_name="centroid.pdb",
        reportInterval=10,
        num_beads=n_beads,
    ))

    simulation.step(100)
    os.remove('centroid.pdb')


def test_run_openmm_rpmd_equilibration():
    print(flush=True)
    n_beads = 4
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_rpmd_equilibration(modeller, forcefield, n_beads=n_beads, n_2=1_000)

    os.remove('rpmd_ready.chk')
    os.remove('rpmd_ready.log')
    os.remove('rpmd_ready_centroid.pdb')
    for i in range(n_beads):
        os.remove(f'rpmd_ready_{i}.pdb')


from typing import Dict, Optional, Sequence, Any


def set_adqtb_particle_types_by_element(
        integrator: Any,
        *,
        topology: Optional[Any] = None,
        system: Optional[Any] = None,
        particle_elements: Optional[Sequence[Any]] = None,
        start_type: int = 0,
        unknown_symbol: str = "X",
) -> Dict[str, int]:
    """
    Assign OpenMM adQTB (QTBIntegrator) particle types so that all particles with the same
    chemical element share the same integer type.

    IMPORTANT: call this BEFORE creating a Context/Simulation; particle types are fixed at
    Context creation time for QTBIntegrator.

    Parameters
    ----------
    integrator
        An OpenMM QTBIntegrator (adQTB). Must provide setParticleType(index, type).
    topology
        openmm.app.Topology used to infer element per particle (atom.element), if
        particle_elements is not provided.
    system
        openmm.System (optional) used to sanity-check that the number of particles matches.
    particle_elements
        Optional explicit per-particle element spec (length == system.getNumParticles()).
        Each entry can be an openmm.app.element.Element, a symbol string like "C", or None.
        Use this if your System has extra particles not present in the Topology (e.g., Drudes).
    start_type
        First type id to use (default 0).
    unknown_symbol
        Symbol to use when an element is missing/None (all such particles share a type).

    Returns
    -------
    element_to_type
        Mapping from element symbol (e.g., "H", "C") to assigned integer type id.
    """

    if not hasattr(integrator, "setParticleType"):
        raise TypeError(
            "integrator must support setParticleType(index, type); expected an OpenMM QTBIntegrator."
        )

    def _sym_and_Z(el: Any) -> tuple[str, int]:
        # OpenMM Element has .symbol and .atomic_number; allow strings too.
        if el is None:
            return unknown_symbol, 10 ** 9
        sym = getattr(el, "symbol", None)
        if sym is None:
            sym = str(el)
        Z = getattr(el, "atomic_number", 10 ** 9)
        try:
            Z = int(Z)
        except Exception:
            Z = 10 ** 9
        return sym, Z

    # Infer per-particle elements
    if particle_elements is None:
        if topology is None:
            raise ValueError("Provide topology or particle_elements.")
        atoms = list(topology.atoms())
        particle_elements = [a.element for a in atoms]

    n = len(particle_elements)

    # Optional sanity check against the System particle count
    if system is not None:
        n_sys = system.getNumParticles()
        if n_sys != n:
            raise ValueError(
                f"Element list has length {n}, but System has {n_sys} particles. "
                "If your System includes extra particles (Drudes/virtual sites/etc.), "
                "pass particle_elements with one entry per System particle."
            )

    # Build a stable symbol -> type_id mapping (sorted by atomic number, then symbol)
    symbol_to_Z: dict[str, int] = {}
    for el in particle_elements:
        sym, Z = _sym_and_Z(el)
        symbol_to_Z.setdefault(sym, Z)

    ordered_symbols = sorted(symbol_to_Z.items(), key=lambda kv: (kv[1], kv[0]))
    element_to_type = {sym: start_type + i for i, (sym, _) in enumerate(ordered_symbols)}

    # Assign types to particles
    for idx, el in enumerate(particle_elements):
        sym, _ = _sym_and_Z(el)
        integrator.setParticleType(idx, int(element_to_type[sym]))

    return element_to_type


def test_openmm_adqtb():
    # Simple run parameters
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

    set_adqtb_particle_types_by_element(integrator, topology=modeller.topology, system=system)

    integrator.setDefaultAdaptationRate(0.5)

    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    simulation.step(n_steps)
