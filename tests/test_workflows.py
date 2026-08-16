"""Fast orchestration tests for the public simulation-stage drivers."""

from types import SimpleNamespace

import openmm.unit as unit
import pytest
from openmm import Vec3

import openmmnqe.openmm as nqe_openmm


class _Topology:
    def __init__(self):
        self._atoms = [
            SimpleNamespace(index=0, name="CA"),
            SimpleNamespace(index=1, name="SIDE"),
        ]

    def atoms(self):
        return iter(self._atoms)

    def getNumAtoms(self):
        return len(self._atoms)


class _System:
    def __init__(self):
        self.forces = []

    def addForce(self, force):
        self.forces.append(force)
        return len(self.forces) - 1

    def getForces(self):
        return list(self.forces)


class _ExternalForce:
    def __init__(self, expression):
        self.expression = expression
        self.global_parameters = []
        self.per_particle_parameters = []
        self.particles = []

    def addGlobalParameter(self, name, value):
        self.global_parameters.append((name, value))

    def addPerParticleParameter(self, name):
        self.per_particle_parameters.append(name)

    def addParticle(self, index, parameters):
        self.particles.append((index, parameters))


class _Integrator:
    def __init__(self, kind, args):
        self.kind = kind
        self.args = args
        self.temperatures = []
        self.step_sizes = []
        self.segment_lengths = []
        self.adaptation_rates = []

    def setTemperature(self, temperature):
        self.temperatures.append(temperature)

    def setStepSize(self, step_size):
        self.step_sizes.append(step_size)

    def setSegmentLength(self, segment_length):
        self.segment_lengths.append(segment_length)

    def setDefaultAdaptationRate(self, rate):
        self.adaptation_rates.append(rate)


class _Context:
    def __init__(self):
        self.positions = []
        self.velocity_temperatures = []
        self.parameters = []

    def setPositions(self, positions):
        self.positions.append(positions)

    def setVelocitiesToTemperature(self, temperature):
        self.velocity_temperatures.append(temperature)

    def setParameter(self, name, value):
        self.parameters.append((name, value))


class _Simulation:
    def __init__(self, topology, system, integrator, platform):
        self.topology = topology
        self.system = system
        self.integrator = integrator
        self.platform = platform
        self.context = _Context()
        self.reporters = []
        self.steps = []
        self.minimizations = []

    def step(self, steps):
        self.steps.append(steps)

    def minimizeEnergy(self, **kwargs):
        self.minimizations.append(kwargs)


@pytest.fixture
def workflow_runtime(monkeypatch):
    """Replace costly OpenMM objects with recorders while retaining driver logic."""
    calls = SimpleNamespace(
        builds=[],
        deuterations=[],
        plumed=[],
        standard_reporters=[],
        rpmd_reporters=[],
        saved=[],
        checkpoints=[],
        bead_initializations=[],
        integrators=[],
        simulations=[],
        barostats=[],
    )
    topology = _Topology()
    modeller = SimpleNamespace(
        topology=topology,
        positions=[Vec3(0.0, 0.0, 0.0), Vec3(0.1, 0.0, 0.0)]
        * unit.nanometer,
    )
    system = _System()
    platform = object()

    def build_system(*args):
        calls.builds.append(args)
        return system, platform

    def make_integrator(kind):
        def factory(*args):
            integrator = _Integrator(kind, args)
            calls.integrators.append(integrator)
            return integrator

        return factory

    def make_barostat(kind):
        def factory(*args):
            barostat = SimpleNamespace(kind=kind, args=args)
            calls.barostats.append(barostat)
            return barostat

        return factory

    def make_simulation(*args):
        simulation = _Simulation(*args)
        calls.simulations.append(simulation)
        return simulation

    monkeypatch.setattr(nqe_openmm, "_build_system", build_system)
    monkeypatch.setattr(
        nqe_openmm,
        "_maybe_deuterate",
        lambda *args: calls.deuterations.append(args),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "_load_plumed",
        lambda *args: calls.plumed.append(args),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "_add_standard_reporters",
        lambda *args, **kwargs: calls.standard_reporters.append((args, kwargs)),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "_add_rpmd_reporters",
        lambda *args, **kwargs: calls.rpmd_reporters.append((args, kwargs)),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "_save_final_state",
        lambda *args, **kwargs: calls.saved.append((args, kwargs)),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "_load_checkpoint",
        lambda *args, **kwargs: calls.checkpoints.append((args, kwargs)),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "init_beads",
        lambda *args: calls.bead_initializations.append(args),
    )
    monkeypatch.setattr(nqe_openmm.app, "Simulation", make_simulation)
    monkeypatch.setattr(nqe_openmm.openmm, "CustomExternalForce", _ExternalForce)
    monkeypatch.setattr(
        nqe_openmm.openmm,
        "LangevinMiddleIntegrator",
        make_integrator("langevin"),
    )
    monkeypatch.setattr(
        nqe_openmm.openmm,
        "RPMDIntegrator",
        make_integrator("rpmd"),
    )
    monkeypatch.setattr(
        nqe_openmm.openmm,
        "QTBIntegrator",
        make_integrator("qtb"),
    )
    monkeypatch.setattr(
        nqe_openmm.openmm,
        "MonteCarloBarostat",
        make_barostat("classical"),
    )
    monkeypatch.setattr(
        nqe_openmm.openmm,
        "RPMDMonteCarloBarostat",
        make_barostat("rpmd"),
    )

    return SimpleNamespace(
        calls=calls,
        modeller=modeller,
        system=system,
        platform=platform,
    )


def _temperature_values(temperatures):
    return [temperature.value_in_unit(unit.kelvin) for temperature in temperatures]


def test_restrained_relaxation_runs_all_stages_and_saves_without_checkpoint(
    workflow_runtime,
):
    runtime = workflow_runtime

    nqe_openmm.run_openmm_relaxation(
        runtime.modeller,
        forcefield=object(),
        output_prefix="relaxed",
        n_1=3,
        n_2=5,
        n_3=7,
        ks_1=90.0,
        ks_2=9.0,
        ks_3=0.5,
    )

    simulation = runtime.calls.simulations[0]
    restraint = runtime.system.forces[0]
    spring_unit = unit.kilojoules_per_mole / unit.nanometer**2

    assert [entry["maxIterations"] for entry in simulation.minimizations] == [
        3,
        5,
        7,
    ]
    assert [value.value_in_unit(spring_unit) for _, value in simulation.context.parameters] == [
        90.0,
        9.0,
        0.5,
    ]
    assert [index for index, _ in restraint.particles] == [0]
    assert runtime.calls.saved == [
        ((simulation, "relaxed"), {"save_checkpoint": False})
    ]


@pytest.mark.parametrize(
    ("target", "increment", "expected_temperatures", "expected_steps"),
    [
        (125.0, 50.0, [50.0, 100.0, 125.0], [3, 3, 3, 7]),
        (25.0, 50.0, [25.0], [3, 7]),
    ],
)
def test_heating_reaches_target_exactly_and_initializes_velocities_once(
    workflow_runtime,
    target,
    increment,
    expected_temperatures,
    expected_steps,
):
    runtime = workflow_runtime

    nqe_openmm.run_openmm_heating(
        runtime.modeller,
        forcefield=object(),
        output_prefix="heated",
        target_temp=target * unit.kelvin,
        temp_step=increment * unit.kelvin,
        n_report=11,
        steps_per_stage=3,
        steps_final=7,
    )

    simulation = runtime.calls.simulations[0]
    integrator = runtime.calls.integrators[0]

    assert _temperature_values(integrator.temperatures) == expected_temperatures
    assert _temperature_values(simulation.context.velocity_temperatures) == [
        expected_temperatures[0]
    ]
    assert simulation.steps == expected_steps
    assert runtime.calls.standard_reporters == [
        ((simulation, "heated", 11), {"pdb_steps": True})
    ]
    assert runtime.calls.saved == [((simulation, "heated"), {})]


def test_npt_runs_restrained_and_unrestrained_phases_without_optional_barostat(
    workflow_runtime,
):
    runtime = workflow_runtime

    nqe_openmm.run_openmm_npt(
        runtime.modeller,
        forcefield=object(),
        output_prefix="npt",
        barostat_freq=None,
        n_report=13,
        n_1=2,
        n_2=4,
    )

    simulation = runtime.calls.simulations[0]

    assert runtime.calls.barostats == []
    assert simulation.steps == [2, 4]
    assert simulation.context.parameters == [("k", 0.0)]
    assert len(runtime.system.forces) == 1
    assert runtime.calls.standard_reporters == [
        (
            (simulation, "npt", 13),
            {"pdb_steps": True, "stdout_volume": True},
        )
    ]
    assert runtime.calls.saved == [((simulation, "npt"), {})]


def test_classical_production_wires_optional_features_and_periodic_checkpoint(
    workflow_runtime,
):
    runtime = workflow_runtime

    nqe_openmm.run_openmm_prod(
        runtime.modeller,
        forcefield=object(),
        plumed_script_path="bias.dat",
        barostat_freq=17,
        n_report=7,
        steps=23,
        output_prefix="prod",
        deuterate=True,
        deuterate_option="protein",
    )

    simulation = runtime.calls.simulations[0]

    assert runtime.calls.deuterations == [
        (runtime.modeller, runtime.system, True, "protein")
    ]
    assert runtime.calls.barostats[0].kind == "classical"
    assert runtime.system.forces == [runtime.calls.barostats[0]]
    assert runtime.calls.plumed == [(runtime.system, "bias.dat")]
    assert simulation.context.positions == [runtime.modeller.positions]
    assert _temperature_values(simulation.context.velocity_temperatures) == [300.0]
    assert simulation.steps == [23]
    assert runtime.calls.standard_reporters == [
        (
            (simulation, "prod", 7),
            {"pdb_steps": True, "checkpoint_interval": 70},
        )
    ]
    assert runtime.calls.saved == [((simulation, "prod"), {})]


@pytest.mark.parametrize(
    ("plumed_input", "expected_path", "expected_contents"),
    [
        ("DISTANCE ATOMS=1,2\nPRINT ARG=*", "pull_plumed.dat", "DISTANCE ATOMS=1,2\nPRINT ARG=*"),
        ("existing.dat", "existing.dat", None),
    ],
)
def test_steered_md_accepts_inline_or_file_plumed_input(
    monkeypatch,
    plumed_input,
    expected_path,
    expected_contents,
):
    delegated = []
    monkeypatch.setattr(
        nqe_openmm,
        "run_openmm_prod",
        lambda *args, **kwargs: delegated.append((args, kwargs)),
    )
    modeller = object()
    forcefield = object()

    trajectory = nqe_openmm.run_openmm_steered(
        modeller,
        forcefield,
        plumed_input,
        steps=40,
        output_prefix="pull",
        n_report=4,
    )

    args, kwargs = delegated[0]
    assert args == (modeller, forcefield)
    assert kwargs["plumed_script_path"] == expected_path
    assert kwargs["steps"] == 40
    assert kwargs["n_report"] == 4
    assert kwargs["barostat_freq"] is None
    assert trajectory == "pull_steps.pdb"
    if expected_contents is not None:
        with open(expected_path) as handle:
            assert handle.read() == expected_contents


def test_rpmd_equilibration_expands_beads_then_restores_full_timestep(
    workflow_runtime,
):
    runtime = workflow_runtime

    nqe_openmm.run_openmm_rpmd_equilibration(
        runtime.modeller,
        forcefield=object(),
        output_prefix="rpmd_ready",
        n_beads=8,
        timestep=0.6 * unit.femtoseconds,
        n_report=9,
        n_1=2,
        n_2=5,
        atoms_to_watch=[1],
    )

    simulation = runtime.calls.simulations[0]
    integrator = runtime.calls.integrators[0]

    assert [size.value_in_unit(unit.femtoseconds) for size in integrator.step_sizes] == [
        0.3,
        0.6,
    ]
    assert simulation.steps == [2, 5]
    assert simulation.context.positions == []
    assert simulation.context.velocity_temperatures == []
    assert runtime.calls.bead_initializations == [
        (runtime.modeller, simulation, 8)
    ]
    assert runtime.calls.rpmd_reporters == [
        ((simulation, runtime.modeller.topology, "rpmd_ready", 9, 8, [1]), {})
    ]
    assert runtime.calls.saved == [
        (
            (simulation, "rpmd_ready"),
            {"pdb_suffix": "_final.pdb", "n_beads": 8},
        )
    ]


def test_rpmd_production_loads_checkpoint_and_saves_centroid(
    workflow_runtime,
):
    runtime = workflow_runtime

    nqe_openmm.run_openmm_rpmd_prod(
        runtime.modeller,
        forcefield=object(),
        checkpoint_file="ready.chk",
        output_prefix="rpmd_prod",
        n_beads=6,
        barostat_freq=None,
        n_report=5,
        steps=12,
    )

    simulation = runtime.calls.simulations[0]

    assert runtime.calls.barostats == []
    assert runtime.calls.checkpoints == [
        ((simulation, "ready.chk"), {"n_beads": 6})
    ]
    assert simulation.context.positions == []
    assert simulation.steps == [12]
    assert runtime.calls.rpmd_reporters == [
        ((simulation, runtime.modeller.topology, "rpmd_prod", 5, 6, None), {})
    ]
    assert runtime.calls.saved == [
        (
            (simulation, "rpmd_prod"),
            {"pdb_suffix": "_final.pdb", "n_beads": 6},
        )
    ]


def test_contracted_rpmd_assigns_force_groups_and_default_contractions(
    monkeypatch,
    workflow_runtime,
):
    runtime = workflow_runtime

    class GroupedForce:
        def __init__(self):
            self.force_groups = []

        def setForceGroup(self, group):
            self.force_groups.append(group)

    class NonbondedForce(GroupedForce):
        def __init__(self):
            super().__init__()
            self.reciprocal_groups = []

        def setReciprocalSpaceForceGroup(self, group):
            self.reciprocal_groups.append(group)

    class BondedForce(GroupedForce):
        pass

    class OtherForce(GroupedForce):
        pass

    monkeypatch.setattr(nqe_openmm.openmm, "NonbondedForce", NonbondedForce)
    monkeypatch.setattr(nqe_openmm.openmm, "HarmonicBondForce", BondedForce)
    monkeypatch.setattr(nqe_openmm.openmm, "HarmonicAngleForce", BondedForce)
    monkeypatch.setattr(nqe_openmm.openmm, "PeriodicTorsionForce", BondedForce)
    monkeypatch.setattr(nqe_openmm.openmm, "RBTorsionForce", BondedForce)
    monkeypatch.setattr(nqe_openmm.openmm, "CMAPTorsionForce", BondedForce)
    nonbonded = NonbondedForce()
    bonded = BondedForce()
    other = OtherForce()
    runtime.system.forces = [nonbonded, bonded, other]

    nqe_openmm.run_openmm_rpmd_contracted(
        runtime.modeller,
        forcefield=object(),
        checkpoint_file="ready.chk",
        output_prefix="contracted",
        barostat_freq=None,
        steps=4,
    )

    simulation = runtime.calls.simulations[0]
    integrator = runtime.calls.integrators[0]

    assert nonbonded.force_groups == [1]
    assert nonbonded.reciprocal_groups == [2]
    assert bonded.force_groups == [0]
    assert other.force_groups == [0]
    assert integrator.args[-1] == {1: 8, 2: 1}
    assert runtime.calls.checkpoints == [
        ((simulation, "ready.chk"), {"n_beads": 32})
    ]
    assert simulation.steps == [4]
    assert runtime.calls.saved == [
        (
            (simulation, "contracted"),
            {"pdb_suffix": "_final.pdb", "n_beads": 32},
        )
    ]


def test_adqtb_equilibration_configures_adaptation_and_checkpoint_reporting(
    workflow_runtime,
):
    runtime = workflow_runtime

    nqe_openmm.run_openmm_adqtb_eq(
        runtime.modeller,
        forcefield=object(),
        segment_length=0.8 * unit.picosecond,
        adaptation_rate=0.25,
        n_report=6,
        steps=14,
        output_prefix="adqtb",
    )

    simulation = runtime.calls.simulations[0]
    integrator = runtime.calls.integrators[0]

    assert [value.value_in_unit(unit.picosecond) for value in integrator.segment_lengths] == [
        0.8
    ]
    assert integrator.adaptation_rates == [0.25]
    assert simulation.context.positions == [runtime.modeller.positions]
    assert _temperature_values(simulation.context.velocity_temperatures) == [300.0]
    assert simulation.steps == [14]
    assert runtime.calls.standard_reporters == [
        (
            (simulation, "adqtb", 6),
            {"pdb_steps": True, "checkpoint_interval": 60},
        )
    ]
    assert runtime.calls.saved == [((simulation, "adqtb"), {})]
