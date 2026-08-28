"""Fast orchestration tests for the public simulation-stage drivers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3

import openmmnqe.openmm as nqe_openmm


class _Topology:
    def __init__(self) -> None:
        self._atoms = [
            SimpleNamespace(
                index=0, name="CA", element=app.Element.getBySymbol("C"),
            ),
            SimpleNamespace(
                index=1, name="SIDE", element=app.Element.getBySymbol("H"),
            ),
        ]

    def atoms(self) -> Iterator[SimpleNamespace]:
        return iter(self._atoms)

    def getNumAtoms(self) -> int:
        return len(self._atoms)


class _System:
    def __init__(self) -> None:
        self.forces = []
        self.periodic = True
        self.masses = [12.011, 1.008]

    def getNumParticles(self) -> int:
        return len(self.masses)

    def getParticleMass(self, index: int) -> unit.Quantity:
        return self.masses[index] * unit.dalton

    def addForce(self, force: Any) -> int:
        self.forces.append(force)
        return len(self.forces) - 1

    def getForces(self) -> list[Any]:
        return list(self.forces)

    def usesPeriodicBoundaryConditions(self) -> bool:
        return self.periodic


class _ExternalForce:
    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.global_parameters = []
        self.per_particle_parameters = []
        self.particles = []

    def addGlobalParameter(self, name: str, value: Any) -> None:
        self.global_parameters.append((name, value))

    def addPerParticleParameter(self, name: str) -> None:
        self.per_particle_parameters.append(name)

    def addParticle(self, index: int, parameters: list[Any]) -> None:
        self.particles.append((index, parameters))


class _Barostat:
    def __init__(self, kind: str, args: tuple[Any, ...]) -> None:
        self.kind = kind
        self.args = args
        self._group = 0
        self.random_seeds = []

    def setRandomNumberSeed(self, seed: int) -> None:
        self.random_seeds.append(seed)

    def setForceGroup(self, group: int) -> None:
        self._group = group

    def getForceGroup(self) -> int:
        return self._group


class _Integrator:
    def __init__(self, kind: str, args: tuple[Any, ...]) -> None:
        self.kind = kind
        self.args = args
        self.temperatures = []
        self.step_sizes = []
        self.segment_lengths = []
        self.adaptation_rates = []
        self.random_seeds = []
        self.particle_types: dict[int, int] = {}

    def setParticleType(self, particle: int, type_index: int) -> None:
        self.particle_types[particle] = type_index

    def getParticleTypes(self) -> dict[int, int]:
        return dict(self.particle_types)

    def getStepSize(self) -> unit.Quantity:
        if self.step_sizes:
            return self.step_sizes[-1]
        return self.args[2]

    def getSegmentLength(self) -> unit.Quantity:
        return self.segment_lengths[-1]

    def setTemperature(self, temperature: unit.Quantity) -> None:
        self.temperatures.append(temperature)

    def setStepSize(self, step_size: unit.Quantity) -> None:
        self.step_sizes.append(step_size)

    def setSegmentLength(self, segment_length: unit.Quantity) -> None:
        self.segment_lengths.append(segment_length)

    def setDefaultAdaptationRate(self, rate: float) -> None:
        self.adaptation_rates.append(rate)

    def setRandomNumberSeed(self, seed: int) -> None:
        self.random_seeds.append(seed)


class _Context:
    def __init__(self) -> None:
        self.positions = []
        self.velocity_temperatures = []
        self.velocity_seeds = []
        self.parameters = []

    def setPositions(self, positions: Any) -> None:
        self.positions.append(positions)

    def setVelocitiesToTemperature(self, temperature: unit.Quantity,
                                   seed: int | None = None) -> None:
        self.velocity_temperatures.append(temperature)
        self.velocity_seeds.append(seed)

    def setParameter(self, name: str, value: Any) -> None:
        self.parameters.append((name, value))


class _Simulation:
    def __init__(self, topology: Any, system: Any, integrator: Any, platform: Any) -> None:
        self.topology = topology
        self.system = system
        self.integrator = integrator
        self.platform = platform
        self.context = _Context()
        self.reporters = []
        self.steps = []
        self.minimizations = []

    def step(self, steps: int) -> None:
        self.steps.append(steps)

    def minimizeEnergy(self, **kwargs: Any) -> None:
        self.minimizations.append(kwargs)


@pytest.fixture
def workflow_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace costly OpenMM objects with recorders while retaining driver logic.

    Returns a namespace of ``calls``, ``modeller``, ``system`` and
    ``platform``; ``calls`` collects what each patched seam was handed, so a
    test can assert on the driver's decisions without running any dynamics.
    """
    calls = SimpleNamespace(
        builds=[],
        deuterations=[],
        plumed=[],
        standard_reporters=[],
        adqtb_reporters=[],
        rpmd_progress_reporters=[],
        rpmd_reporters=[],
        rpmd_steps=[],
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

    def build_system(*args: Any) -> tuple[Any, Any]:
        calls.builds.append(args)
        return system, platform

    def make_integrator(kind: str) -> Callable[..., _Integrator]:
        def factory(*args: Any) -> _Integrator:
            integrator = _Integrator(kind, args)
            calls.integrators.append(integrator)
            return integrator

        return factory

    def make_barostat(kind: str) -> Callable[..., _Barostat]:
        def factory(*args: Any) -> _Barostat:
            barostat = _Barostat(kind, args)
            calls.barostats.append(barostat)
            return barostat

        return factory

    def make_simulation(*args: Any) -> _Simulation:
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
        "_add_adqtb_reporters",
        lambda *args, **kwargs: calls.adqtb_reporters.append((args, kwargs)),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "_add_rpmd_reporters",
        lambda *args, **kwargs: calls.rpmd_reporters.append((args, kwargs)),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "_add_rpmd_progress_reporters",
        lambda *args, **kwargs: calls.rpmd_progress_reporters.append(
            (args, kwargs)
        ),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "step_rpmd",
        lambda simulation, steps: calls.rpmd_steps.append(
            (simulation, steps)
        ),
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
        lambda *args, **kwargs: calls.bead_initializations.append(
            (args, kwargs)
        ),
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


def _temperature_values(temperatures: Sequence[unit.Quantity]) -> list[float]:
    """Strip units from a sequence of temperatures, giving plain kelvin floats."""
    return [temperature.value_in_unit(unit.kelvin) for temperature in temperatures]


def test_restrained_relaxation_runs_all_stages_and_saves_without_checkpoint(
    workflow_runtime: SimpleNamespace,
) -> None:
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
    workflow_runtime: SimpleNamespace,
    target: float,
    increment: float,
    expected_temperatures: list[float],
    expected_steps: list[int],
) -> None:
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


@pytest.mark.parametrize(
    "increment",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_heating_rejects_nonpositive_or_nonfinite_increment_before_building(
    workflow_runtime: SimpleNamespace,
    increment: float,
) -> None:
    runtime = workflow_runtime

    with pytest.raises(ValueError, match="temp_step must be finite and positive"):
        nqe_openmm.run_openmm_heating(
            runtime.modeller,
            forcefield=object(),
            temp_step=increment * unit.kelvin,
        )

    assert runtime.calls.builds == []


@pytest.mark.parametrize(
    "target",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_heating_rejects_nonpositive_or_nonfinite_target_before_building(
    workflow_runtime: SimpleNamespace,
    target: float,
) -> None:
    runtime = workflow_runtime

    with pytest.raises(ValueError, match="target_temp must be finite and positive"):
        nqe_openmm.run_openmm_heating(
            runtime.modeller,
            forcefield=object(),
            target_temp=target * unit.kelvin,
        )

    assert runtime.calls.builds == []


def test_npt_runs_restrained_and_unrestrained_phases_without_optional_barostat(
    workflow_runtime: SimpleNamespace,
) -> None:
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
    workflow_runtime: SimpleNamespace,
) -> None:
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
        ("DISTANCE ATOMS=1,2", "pull_plumed.dat", "DISTANCE ATOMS=1,2"),
        (
            "DISTANCE ATOMS=1,2\nPRINT ARG=* FILE=outputs/colvar",
            "pull_plumed.dat",
            "DISTANCE ATOMS=1,2\nPRINT ARG=* FILE=outputs/colvar",
        ),
        (
            "MATHEVAL ARG=x FUNC=x/2 PERIODIC=NO",
            "pull_plumed.dat",
            "MATHEVAL ARG=x FUNC=x/2 PERIODIC=NO",
        ),
        ("existing.dat", "existing.dat", None),
        ("inputs/PLUMED", "inputs/PLUMED", None),
        ("missing dir/plumed.dat", "missing dir/plumed.dat", None),
        (Path("PLUMED"), "PLUMED", None),
    ],
)
def test_steered_md_accepts_inline_or_file_plumed_input(
    monkeypatch: pytest.MonkeyPatch,
    plumed_input: str | Path,
    expected_path: str,
    expected_contents: str | None,
) -> None:
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


def test_rpmd_reporter_finalizer_closes_outputs_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Reporter:
        def __init__(self, close_error: Exception | None = None) -> None:
            self.close_calls = 0
            self.close_error = close_error

        def close(self) -> None:
            self.close_calls += 1
            if self.close_error is not None:
                raise self.close_error

    monkeypatch.setattr(nqe_openmm, "RPMDQuantumSpreadReporter", Reporter)
    monkeypatch.setattr(nqe_openmm, "RPMDCentroidReporter", Reporter)
    monkeypatch.setattr(nqe_openmm, "RPMDBeadReporter", Reporter)

    successful = Reporter()
    with nqe_openmm._finalize_reporters(
        SimpleNamespace(reporters=[successful]),
    ):
        pass
    assert successful.close_calls == 1

    close_failure = Reporter(OSError("close failed"))
    still_closed = Reporter()
    with pytest.raises(OSError, match="close failed"):
        with nqe_openmm._finalize_reporters(
            SimpleNamespace(reporters=[close_failure, still_closed]),
        ):
            pass
    assert close_failure.close_calls == 1
    assert still_closed.close_calls == 1

    failing = Reporter(OSError("close failed"))
    with pytest.raises(RuntimeError, match="simulation failed"):
        with nqe_openmm._finalize_reporters(
            SimpleNamespace(reporters=[failing]),
        ):
            raise RuntimeError("simulation failed")
    assert failing.close_calls == 1


_BAROSTAT_STAGES = [
    nqe_openmm.run_openmm_npt,
    nqe_openmm.run_openmm_prod,
    nqe_openmm.run_openmm_rpmd_contracted,
    nqe_openmm.run_openmm_rpmd_prod,
    nqe_openmm.run_openmm_adqtb_prod,
]


@pytest.mark.parametrize("stage", _BAROSTAT_STAGES)
def test_barostat_stages_reject_nonperiodic_systems_before_context_creation(
    workflow_runtime: SimpleNamespace,
    stage: Callable[..., None],
) -> None:
    runtime = workflow_runtime
    runtime.system.periodic = False

    with pytest.raises(ValueError, match="barostat requires a periodic System"):
        stage(runtime.modeller, forcefield=object())

    assert runtime.calls.barostats == []
    assert runtime.calls.simulations == []


@pytest.mark.parametrize("stage", _BAROSTAT_STAGES)
@pytest.mark.parametrize(
    ("frequency", "error", "message"),
    [
        (True, TypeError, "barostat_freq must be an integer"),
        (1.5, TypeError, "barostat_freq must be an integer"),
        (0, ValueError, "barostat_freq must be a positive integer"),
        (-1, ValueError, "barostat_freq must be a positive integer"),
    ],
)
def test_barostat_stages_validate_frequency_before_context_creation(
    workflow_runtime: SimpleNamespace,
    stage: Callable[..., None],
    frequency: Any,
    error: type[Exception],
    message: str,
) -> None:
    runtime = workflow_runtime

    with pytest.raises(error, match=message):
        stage(
            runtime.modeller,
            forcefield=object(),
            barostat_freq=frequency,
        )

    assert runtime.calls.barostats == []
    assert runtime.calls.simulations == []


def test_rpmd_equilibration_expands_beads_then_restores_full_timestep(
    workflow_runtime: SimpleNamespace,
) -> None:
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
        scale_factor=0.75,
        seed=1234,
        expansion_metric="mean",
        distance_pairs_to_watch=[(0, 1)],
    )

    simulation = runtime.calls.simulations[0]
    integrator = runtime.calls.integrators[0]

    assert [size.value_in_unit(unit.femtoseconds) for size in integrator.step_sizes] == [
        0.3,
        0.6,
    ]
    assert simulation.steps == []
    assert runtime.calls.rpmd_steps == [(simulation, 2), (simulation, 5)]
    assert simulation.context.positions == []
    assert simulation.context.velocity_temperatures == []
    initialization_seed, thermostat_seed = nqe_openmm._derive_seeds(
        1234, "initialization", "thermostat"
    )
    assert integrator.random_seeds == [thermostat_seed]
    assert runtime.calls.bead_initializations == [(
        (runtime.modeller, simulation, 8),
        {"scale_factor": 0.75, "seed": initialization_seed},
    )]
    assert runtime.calls.rpmd_reporters == [
        (
            (simulation, runtime.modeller.topology, "rpmd_ready", 9, 8, [1]),
            {"expansion_metric": "mean", "distance_pairs": [(0, 1)]},
        )
    ]
    assert runtime.calls.rpmd_progress_reporters == [
        ((simulation, "rpmd_ready", 9), {})
    ]
    assert runtime.calls.saved == [
        (
            (simulation, "rpmd_ready"),
            {"pdb_suffix": "_final.pdb", "n_beads": 8},
        )
    ]


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_derive_seeds_rejects_invalid_values(seed: Any) -> None:
    with pytest.raises(ValueError, match="seed must be"):
        nqe_openmm._derive_seeds(seed, "thermostat")


def test_derive_seeds_rejects_unregistered_streams() -> None:
    with pytest.raises(KeyError, match="unknown seed stream"):
        nqe_openmm._derive_seeds(1, "entropy")


def test_derive_seeds_leaves_every_stream_unseeded_without_a_master_seed() -> None:
    assert nqe_openmm._derive_seeds(None, "thermostat", "barostat") == (
        None,
        None,
    )


def test_derive_seeds_gives_each_stream_its_own_reproducible_value() -> None:
    streams = ("initialization", "thermostat", "velocities", "barostat")

    first = nqe_openmm._derive_seeds(7, *streams)

    assert first == nqe_openmm._derive_seeds(7, *streams)
    assert len(set(first)) == len(streams)
    assert first != nqe_openmm._derive_seeds(8, *streams)


def test_derive_seeds_ignores_which_other_streams_were_asked_for() -> None:
    # A stage drawing two streams and one drawing four have to agree on what
    # a shared stream means, or a single master seed could not reproduce a
    # whole workflow rather than one stage of it.
    alone = nqe_openmm._derive_seeds(99, "thermostat")
    _, alongside, _ = nqe_openmm._derive_seeds(
        99, "velocities", "thermostat", "barostat"
    )

    assert alone == (alongside,)


@pytest.mark.parametrize("seed", [0, 1, 2**31, 2**64 + 3])
def test_derive_seeds_keeps_openmm_streams_in_its_positive_range(
    seed: int,
) -> None:
    # Zero is what OpenMM reads as "pick one for me", which is the seed=None
    # path, so no explicit master seed may ever derive it.
    derived = nqe_openmm._derive_seeds(
        seed, "thermostat", "velocities", "barostat"
    )

    assert all(1 <= value <= 2_147_483_646 for value in derived)


# Every fixture-drivable stage, and the random streams it draws.
# run_openmm_steered is absent because it runs no dynamics of its own; the
# delegation test below covers it instead.
_SEEDED_STAGES = [
    pytest.param(
        nqe_openmm.run_openmm_relaxation, ("thermostat",), id="relaxation"
    ),
    pytest.param(
        nqe_openmm.run_openmm_relaxation_simple,
        ("thermostat",),
        id="relaxation_simple",
    ),
    pytest.param(
        nqe_openmm.run_openmm_heating,
        ("thermostat", "velocities"),
        id="heating",
    ),
    pytest.param(
        nqe_openmm.run_openmm_npt,
        ("thermostat", "velocities", "barostat"),
        id="npt",
    ),
    pytest.param(
        nqe_openmm.run_openmm_prod,
        ("thermostat", "velocities", "barostat"),
        id="prod",
    ),
    pytest.param(
        nqe_openmm.run_openmm_rpmd_equilibration,
        ("thermostat",),
        id="rpmd_equilibration",
    ),
    pytest.param(
        nqe_openmm.run_openmm_rpmd_contracted,
        ("thermostat", "barostat"),
        id="rpmd_contracted",
    ),
    pytest.param(
        nqe_openmm.run_openmm_rpmd_prod,
        ("thermostat", "barostat"),
        id="rpmd_prod",
    ),
    pytest.param(
        nqe_openmm.run_openmm_adqtb_eq,
        ("thermostat", "velocities"),
        id="adqtb_eq",
    ),
    pytest.param(
        nqe_openmm.run_openmm_adqtb_prod,
        ("thermostat", "barostat"),
        id="adqtb_prod",
    ),
]


@pytest.mark.parametrize(("stage", "streams"), _SEEDED_STAGES)
def test_every_stage_fixes_each_random_stream_it_draws(
    workflow_runtime: SimpleNamespace,
    stage: Callable[..., Any],
    streams: tuple[str, ...],
) -> None:
    runtime = workflow_runtime

    stage(runtime.modeller, forcefield=object(), seed=4321)

    expected = dict(
        zip(streams, nqe_openmm._derive_seeds(4321, *streams), strict=True)
    )
    context = runtime.calls.simulations[0].context
    barostat_seeds = [
        seed
        for barostat in runtime.calls.barostats
        for seed in barostat.random_seeds
    ]

    assert runtime.calls.integrators[0].random_seeds == [expected["thermostat"]]
    assert context.velocity_seeds == (
        [expected["velocities"]] if "velocities" in expected else []
    )
    assert barostat_seeds == (
        [expected["barostat"]] if "barostat" in expected else []
    )


@pytest.mark.parametrize(("stage", "streams"), _SEEDED_STAGES)
def test_no_stage_fixes_a_random_stream_without_a_master_seed(
    workflow_runtime: SimpleNamespace,
    stage: Callable[..., Any],
    streams: tuple[str, ...],
) -> None:
    runtime = workflow_runtime

    stage(runtime.modeller, forcefield=object())

    context = runtime.calls.simulations[0].context

    assert runtime.calls.integrators[0].random_seeds == []
    assert all(
        barostat.random_seeds == [] for barostat in runtime.calls.barostats
    )
    # No seed argument at all, so OpenMM falls back to its entropy source.
    assert set(context.velocity_seeds) <= {None}


def test_one_master_seed_still_gives_each_consumer_its_own_numbers(
    workflow_runtime: SimpleNamespace,
) -> None:
    # A single seed must not reach the thermostat, the velocity draw and the
    # barostat as the same number: correlated streams are not three
    # independent sources of noise.
    runtime = workflow_runtime

    nqe_openmm.run_openmm_prod(runtime.modeller, forcefield=object(), seed=5)

    context = runtime.calls.simulations[0].context
    drawn = (
        runtime.calls.integrators[0].random_seeds
        + runtime.calls.barostats[0].random_seeds
        + context.velocity_seeds
    )

    assert len(set(drawn)) == 3


def test_steered_md_hands_its_seed_to_the_production_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated = []
    monkeypatch.setattr(
        nqe_openmm,
        "run_openmm_prod",
        lambda *args, **kwargs: delegated.append(kwargs),
    )

    nqe_openmm.run_openmm_steered(
        object(), object(), Path("PLUMED"), steps=4, seed=17
    )

    assert delegated[0]["seed"] == 17


def test_rpmd_production_loads_checkpoint_and_saves_centroid(
    workflow_runtime: SimpleNamespace,
) -> None:
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
    assert simulation.steps == []
    assert runtime.calls.rpmd_steps == [(simulation, 12)]
    assert runtime.calls.rpmd_reporters == [
        (
            (simulation, runtime.modeller.topology, "rpmd_prod", 5, 6, None),
            {"expansion_metric": "rms", "distance_pairs": None},
        )
    ]
    assert runtime.calls.rpmd_progress_reporters == [
        ((simulation, "rpmd_prod", 5), {})
    ]
    assert runtime.calls.saved == [
        (
            (simulation, "rpmd_prod"),
            {"pdb_suffix": "_final.pdb", "n_beads": 6},
        )
    ]


def test_contracted_rpmd_assigns_force_groups_and_default_contractions(
    monkeypatch: pytest.MonkeyPatch,
    workflow_runtime: SimpleNamespace,
) -> None:
    runtime = workflow_runtime

    class GroupedForce:
        def __init__(self, group: int=0) -> None:
            self.force_groups = []
            self._group = group

        def setForceGroup(self, group: int) -> None:
            self.force_groups.append(group)
            self._group = group

        def getForceGroup(self) -> int:
            return self._group

    class NonbondedForce(GroupedForce):
        def __init__(self, group: int=0) -> None:
            super().__init__(group)
            self.reciprocal_groups = []

        def setReciprocalSpaceForceGroup(self, group: int) -> None:
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
    other = OtherForce(group=31)
    runtime.system.forces = [nonbonded, bonded, other]

    nqe_openmm.run_openmm_rpmd_contracted(
        runtime.modeller,
        forcefield=object(),
        checkpoint_file="ready.chk",
        output_prefix="contracted",
        barostat_freq=None,
        steps=4,
        atoms_to_watch=[1],
        expansion_metric="mean",
        distance_pairs_to_watch=[(0, 1)],
    )

    simulation = runtime.calls.simulations[0]
    integrator = runtime.calls.integrators[0]

    assert nonbonded.force_groups == [1]
    assert nonbonded.reciprocal_groups == [2]
    assert bonded.force_groups == [0]
    assert other.force_groups == [], "an unrecognised force keeps its group"
    assert other.getForceGroup() == 31
    assert integrator.args[-1] == {1: 8, 2: 1}
    assert runtime.calls.checkpoints == [
        ((simulation, "ready.chk"), {"n_beads": 32})
    ]
    assert simulation.steps == []
    assert runtime.calls.rpmd_steps == [(simulation, 4)]
    assert runtime.calls.rpmd_reporters == [
        (
            (simulation, runtime.modeller.topology, "contracted", 1_000, 32, [1]),
            {"expansion_metric": "mean", "distance_pairs": [(0, 1)]},
        )
    ]
    assert runtime.calls.rpmd_progress_reporters == [
        ((simulation, "contracted", 1_000), {})
    ]
    assert runtime.calls.saved == [
        (
            (simulation, "contracted"),
            {"pdb_suffix": "_final.pdb", "n_beads": 32},
        )
    ]


@pytest.mark.parametrize(
    ("n_beads", "direct_space_copies"),
    [(1, 1), (2, 2), (7, 7), (8, 8), (9, 8)],
)
def test_contracted_rpmd_default_contractions_fit_the_bead_count(
    workflow_runtime: SimpleNamespace,
    n_beads: int,
    direct_space_copies: int,
) -> None:
    runtime = workflow_runtime

    nqe_openmm.run_openmm_rpmd_contracted(
        runtime.modeller,
        forcefield=object(),
        checkpoint_file="ready.chk",
        n_beads=n_beads,
        barostat_freq=None,
        steps=0,
    )

    assert runtime.calls.integrators[0].args[-1] == {
        1: direct_space_copies,
        2: 1,
    }


@pytest.mark.parametrize(
    ("contractions", "error", "message"),
    [
        ([(1, 1)], TypeError, "contractions must be a mapping"),
        ({True: 1}, TypeError, "contraction force group must be an integer"),
        ({-1: 1}, ValueError, "contraction force group must be a non-negative integer"),
        ({32: 1}, ValueError, "contraction force group must be between 0 and 31"),
        ({1: True}, TypeError, r"contractions\[1\] must be an integer"),
        ({1: 1.5}, TypeError, r"contractions\[1\] must be an integer"),
        ({1: 0}, ValueError, r"contractions\[1\] must be a positive integer"),
        ({1: 9}, ValueError, r"contractions\[1\]=9 cannot exceed n_beads=8"),
    ],
)
def test_contracted_rpmd_validates_explicit_contractions_before_building(
    workflow_runtime: SimpleNamespace,
    contractions: Any,
    error: type[Exception],
    message: str,
) -> None:
    runtime = workflow_runtime

    with pytest.raises(error, match=message):
        nqe_openmm.run_openmm_rpmd_contracted(
            runtime.modeller,
            forcefield=object(),
            n_beads=8,
            contractions=contractions,
            barostat_freq=None,
        )

    assert runtime.calls.builds == []


@pytest.mark.parametrize(
    "stage",
    [
        nqe_openmm.run_openmm_rpmd_contracted,
        nqe_openmm.run_openmm_rpmd_prod,
        nqe_openmm.run_openmm_adqtb_prod,
    ],
)
def test_nqe_production_stages_warn_but_run_default_barostat_on_python_force(
    workflow_runtime: SimpleNamespace,
    stage: Callable[..., None],
) -> None:
    runtime = workflow_runtime
    python_force = nqe_openmm.openmm.PythonForce(lambda *args: 0.0)
    runtime.system.forces = [python_force]

    with pytest.warns(UserWarning, match="PythonForce"):
        stage(runtime.modeller, forcefield=object())

    barostat = runtime.calls.barostats[0]
    assert runtime.system.forces == [python_force, barostat]


def test_adqtb_equilibration_configures_adaptation_and_checkpoint_reporting(
    workflow_runtime: SimpleNamespace,
) -> None:
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
    assert runtime.calls.standard_reporters == []
    assert runtime.calls.adqtb_reporters == [
        (
            (simulation, "adqtb", 6),
            {
                "segment_steps": 800,
                "type_names": {0: "H", 1: "C"},
                "friction_log": True,
                "checkpoint_interval": 60,
            },
        )
    ]
    assert integrator.particle_types == {0: 1, 1: 0}
    assert runtime.calls.saved == [((simulation, "adqtb"), {})]
