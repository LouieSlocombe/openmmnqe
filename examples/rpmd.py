"""RPMD, adQTB, and quantum-reporter simulation examples."""

from __future__ import annotations

import argparse
from sys import stdout

import openmm.app as app
import openmm.unit as unit
from openmm import openmm
from openmmml import MLPotential

import openmmnqe as nqe

device = "CUDA"


def run_openmm_rpmd() -> None:
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


def run_openmm_rpmd_solvated() -> None:
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


def run_openmm_rpmd_ml() -> None:
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


def run_openmm_rpmd_mixed() -> None:
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


def run_rpmd_quantum_spread_reporter() -> None:
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


def run_rpmd_bead_reporter() -> None:
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


def run_rpmd_centroid_reporter() -> None:
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


def run_rpmd_thermodynamic_reporter() -> None:
    """Log ring-polymer energy estimators, then average and plot the log."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        # The centroid-virial estimator needs unconstrained forces, so the
        # beads run fully flexible rather than with rigid bonds or water.
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

    simulation.reporters.append(nqe.RPMDThermodynamicReporter(
        file="thermo.log",
        reportInterval=10,
    ))

    nqe.step_rpmd(simulation, 500)

    # One reading taken directly, without going through the log.
    values = nqe.rpmd_thermodynamics(simulation)
    print(f"quantum energy now: {values['energy_quantum']}", flush=True)

    averages = nqe.rpmd_thermodynamic_averages("thermo.log", discard=0.2)
    for column in ("KE_cv(kJ/mol)", "PE_mean(kJ/mol)", "E_quantum(kJ/mol)",
                   "T_ring(K)", "T_centroid(K)"):
        mean, error = averages[column]
        print(f"{column:>20s}  {mean:12.3f} +/- {error:.3f}", flush=True)

    nqe.plot_rpmd_thermodynamics(
        "thermo.log",
        energy_columns=["KE_cv(kJ/mol)", "PE_mean(kJ/mol)", "E_quantum(kJ/mol)"],
        filename="rpmd-thermodynamics.png",
    )

    nqe.remove_file('thermo.log')
    nqe.remove_file('rpmd-thermodynamics.png')


def _flexible_peptide_rpmd(n_beads: int = 32,
                           substituted_mass: unit.Quantity | None = None,
                           ) -> tuple[app.Simulation, list[int]]:
    """
    Build a flexible peptide RPMD simulation and pick out two hydrogens.

    With *substituted_mass* those two carry that mass instead of hydrogen's,
    set before the Context exists so the beads are seeded at the right mass.
    """
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        # The centroid-virial estimator needs unconstrained forces, so the
        # beads run fully flexible rather than with rigid bonds or water.
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
        hydrogenMass=None,
    )

    atoms = [
        atom.index
        for atom in modeller.topology.atoms()
        if atom.element is app.element.hydrogen
    ][:2]
    if substituted_mass is not None:
        for atom in atoms:
            system.setParticleMass(atom, substituted_mass)

    integrator = openmm.RPMDIntegrator(n_beads,
                                       300.0 * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       0.5 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName(device)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    nqe.init_beads(modeller, simulation, n_beads)
    return simulation, atoms


def run_rpmd_kinetic_decomposition() -> None:
    """Log per-atom quantum kinetic energies, then average and plot them."""
    print(flush=True)
    simulation, atoms = _flexible_peptide_rpmd()

    simulation.reporters.append(nqe.RPMDKineticDecompositionReporter(
        file="kinetic.log",
        reportInterval=10,
        atom_indices=atoms,
        names=["H1", "H2"],
    ))

    nqe.step_rpmd(simulation, 500)

    # One reading taken directly, without going through the log. A classical
    # atom would sit at 3kT/2, which is 3.74 kJ/mol at 300 K.
    now = nqe.rpmd_kinetic_decomposition(simulation, atoms)
    for index, kinetic in now.items():
        print(f"atom {index}: {kinetic}", flush=True)

    averages = nqe.rpmd_kinetic_decomposition_averages(
        "kinetic.log", discard=0.2,
    )
    for name in ("H1", "H2"):
        mean, error = averages[f"Kcv_{name}(kJ/mol)"]
        print(f"{name:>4s}  {mean:8.3f} +/- {error:.3f} kJ/mol", flush=True)

    nqe.plot_rpmd_kinetic_decomposition(
        "kinetic.log",
        temperature=300.0 * unit.kelvin,
        filename="rpmd-kinetic.png",
    )

    nqe.remove_file('kinetic.log')
    nqe.remove_file('rpmd-kinetic.png')


def run_rpmd_isotope_free_energy() -> None:
    """Integrate over mass for an H-to-D substitution free energy."""
    print(flush=True)
    temperature = 300.0 * unit.kelvin

    # Which masses to run at. Not the physical isotope masses: these are the
    # Gauss-Legendre nodes of the integral over ln(m) between them.
    plan = nqe.rpmd_mass_integration_nodes(
        app.element.hydrogen.mass,
        app.element.deuterium.mass,
    )
    print(f"run at: {plan.masses.round(3)} Da", flush=True)

    means, errors = [], []
    for node, mass in enumerate(plan.masses):
        # The substituted atoms carry the node mass, on the same potential
        # energy surface: an isotope substitution changes nothing else.
        simulation, atoms = _flexible_peptide_rpmd(
            substituted_mass=mass * unit.dalton,
        )

        log = f"node_{node}_kinetic.log"
        simulation.reporters.append(nqe.RPMDKineticDecompositionReporter(
            file=log,
            reportInterval=10,
            atom_indices=atoms,
            names=["H1", "H2"],
        ))
        nqe.step_rpmd(simulation, 500)

        # Both substituted atoms go into one integrand, so n_substituted is 2.
        averages = nqe.rpmd_kinetic_decomposition_averages(log, discard=0.2)
        node_means = [averages[f"Kcv_{name}(kJ/mol)"] for name in ("H1", "H2")]
        means.append(sum(mean for mean, _ in node_means))
        errors.append(sum(error ** 2 for _, error in node_means) ** 0.5)
        nqe.remove_file(log)

    result = nqe.rpmd_isotope_free_energy(
        plan,
        means,
        kinetic_stderr=errors,
        temperature=temperature,
        n_substituted=2,
    )
    print(
        f"dF(H->D) = {result.free_energy:.3f} "
        f"+/- {result.free_energy_stderr:.3f} kJ/mol",
        flush=True,
    )
    print(f"excess over classical: {result.free_energy_excess:.3f} kJ/mol",
          flush=True)

    # Two sites' substitution free energies give their fractionation ratio.
    ln_alpha, error = nqe.rpmd_fractionation_factor(
        result, result, temperature=temperature,
    )
    print(f"ln(alpha) against itself: {ln_alpha:.3f} +/- {error:.3f}",
          flush=True)


def run_openmm_adqtb() -> None:
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


def run_openmm_rpmd_equilibration() -> None:
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


def run_openmm_rpmd_prod() -> None:
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


def run_openmm_rpmd_contracted() -> None:
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


def run_openmm_adqtb_eq() -> None:
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


def run_openmm_adqtb_prod() -> None:
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


def run_adqtb_verification() -> None:
    """Equilibrate an adQTB run, then show that its friction has converged."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # particle_types="element" is the default, so every element adapts one
    # spectrum and <prefix>_friction.log carries one block of columns each.
    nqe.run_openmm_adqtb_eq(modeller,
                            forcefield,
                            segment_length=0.5 * unit.picosecond,
                            time_step=0.5 * unit.femtosecond,
                            platform_name=device,
                            n_report=10_000,
                            steps=200_000,
                            output_prefix='adqtb_ready')

    log = 'adqtb_ready_friction.log'
    for label, verdict in nqe.adqtb_convergence(log).items():
        state = "converged" if verdict.converged else "STILL ADAPTING"
        print(f"{label}: {state}, drift ratio {verdict.drift_ratio:.2f}, "
              f"max |gamma_r - 1| {verdict.max_deviation:.3f}, "
              f"{verdict.clamped_fraction:.1%} of bins clamped at zero")

    nqe.plot_adqtb_friction_spectra(log, filename='adqtb_friction.png')
    nqe.plot_adqtb_fdt_residual(log, filename='adqtb_residual.png')


def run_rpmd_binary_trajectory() -> None:
    """Write the bead and centroid trajectories as XTC rather than PDB."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # One trajectory per bead is the most expensive output the package has, so
    # this is where a binary format pays most: 16 beads of PDB text is an
    # order of magnitude more than 16 of XTC. The atom subset keeps the
    # solute and drops the water, which is the larger saving of the two.
    solute = [atom.index for atom in modeller.topology.atoms()
              if atom.residue.name not in ("HOH", "WAT")]
    trajectory = nqe.TrajectoryOptions("xtc", atom_indices=solute)

    nqe.run_openmm_rpmd_equilibration(modeller, forcefield, n_beads=16,
                                      output_prefix="rpmd_ready",
                                      trajectory=trajectory,
                                      platform_name=device)
    nqe.run_openmm_rpmd_prod(modeller, forcefield, n_beads=16,
                             steps=50_000,
                             n_report=1_000,
                             output_prefix="rpmd_prod",
                             trajectory=trajectory,
                             platform_name=device)

    # XTC carries no topology, so every stage that writes one also writes
    # <prefix>_topology.pdb -- matching the subset, and written before the run
    # rather than after it, so a crashed run still leaves a readable pair.
    print("centroid:  rpmd_prod_centroid.xtc + rpmd_prod_topology.pdb")
    print("bead 0:    rpmd_prod_bead_0.xtc")


EXAMPLES = {
    name.removeprefix("run_"): function
    for name, function in list(globals().items())
    if name.startswith("run_") and callable(function)
}


def main() -> None:
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
