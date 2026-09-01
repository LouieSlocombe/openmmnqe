"""Focused unit tests for simulation setup and geometry helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm
from scipy import constants

import openmmnqe as nqe
import openmmnqe.tools as nqe_tools


class _State:
    def __init__(self, positions: Sequence[Any], box_vectors: Any) -> None:
        self._positions = np.asarray(positions, dtype=float) * unit.nanometer
        self._box_vectors = np.asarray(box_vectors, dtype=float) * unit.nanometer

    def getPositions(self, asNumpy: bool=False) -> unit.Quantity:
        return self._positions

    def getPeriodicBoxVectors(self, asNumpy: bool=False) -> unit.Quantity:
        return self._box_vectors


class _Integrator:
    def __init__(self, bead_positions: Sequence[Any]=(), box_vectors: Any=None,
                 temperature: unit.Quantity=300.0 * unit.kelvin) -> None:
        box_vectors = np.eye(3) if box_vectors is None else box_vectors
        self._states = [
            _State(positions, box_vectors)
            for positions in bead_positions
        ]
        self.positions = {}
        self.velocities = {}
        self.particle_types = {}
        self.temperature = temperature

    def getState(self, bead: int, **kwargs: Any) -> _State:
        return self._states[bead]

    def setPositions(self, bead: int, positions: Any) -> None:
        self.positions[bead] = positions

    def setVelocities(self, bead: int, velocities: Any) -> None:
        self.velocities[bead] = velocities

    def getTemperature(self) -> unit.Quantity:
        return self.temperature

    def setParticleType(self, particle: int, particle_type: int) -> None:
        self.particle_types[particle] = particle_type


class _System:
    def __init__(self, periodic: bool=False, masses: Sequence[float]=()) -> None:
        self._periodic = periodic
        self._masses = [mass * unit.dalton for mass in masses]

    def usesPeriodicBoundaryConditions(self) -> bool:
        return self._periodic

    def getNumParticles(self) -> int:
        return len(self._masses)

    def getParticleMass(self, index: int) -> unit.Quantity:
        return self._masses[index]


def _single_atom_modeller(position: tuple[float, float, float]=(0.0, 0.0, 0.0)) -> app.Modeller:
    topology = app.Topology()
    residue = topology.addResidue("LIG", topology.addChain("A"), id="1")
    topology.addAtom("H1", app.Element.getBySymbol("H"), residue)
    return app.Modeller(topology, [Vec3(*position)] * unit.nanometer)


def _multi_component_modeller_and_system() -> tuple[app.Modeller, openmm.System]:
    topology = app.Topology()
    chain = topology.addChain("A")
    positions = []
    system = openmm.System()
    for residue_name in ("ALA", "HOH", "DA", "RA", "LIG"):
        residue = topology.addResidue(residue_name, chain)
        for atom_name, symbol in (("H", "H"), ("C", "C")):
            element = app.Element.getBySymbol(symbol)
            topology.addAtom(atom_name, element, residue)
            system.addParticle(element.mass)
            positions.append(Vec3(0.0, 0.0, 0.0))
    return app.Modeller(topology, positions * unit.nanometer), system


def test_zero_velocities_returns_unit_bearing_vectors() -> None:
    velocities = nqe.zero_velocities(3)

    assert unit.is_quantity(velocities)
    assert velocities.unit.is_compatible(unit.nanometer / unit.picosecond)
    assert np.array(velocities.value_in_unit(unit.nanometer / unit.picosecond)).shape == (3, 3)
    assert np.allclose(
        velocities.value_in_unit(unit.nanometer / unit.picosecond),
        0.0,
    )


def test_write_multimodel_pdb_delegates_model_index(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        nqe_tools.app.PDBFile,
        "writeModel",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    topology = object()
    positions = object()
    handle = object()

    nqe.write_multimodel_pdb(topology, positions, handle, model_index=4)

    assert calls == [
        ((topology, positions, handle), {"modelIndex": 4})
    ]


def test_thermal_de_broglie_wavelength_accepts_numbers_and_quantities() -> None:
    bare = nqe.get_thermal_de_broglie_wavelength(1.0, 300.0)
    quantified = nqe.get_thermal_de_broglie_wavelength(
        1.0 * unit.dalton,
        300.0 * unit.kelvin,
    )
    heavier = nqe.get_thermal_de_broglie_wavelength(4.0, 300.0)

    assert unit.is_quantity(bare)
    assert bare.value_in_unit(unit.nanometer) == pytest.approx(
        quantified.value_in_unit(unit.nanometer)
    )
    assert heavier.value_in_unit(unit.nanometer) == pytest.approx(
        bare.value_in_unit(unit.nanometer) / 2.0
    )


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan, np.inf, [1.0]])
def test_thermal_de_broglie_wavelength_rejects_invalid_mass(mass: Any) -> None:
    with pytest.raises(ValueError, match="mass"):
        nqe.get_thermal_de_broglie_wavelength(mass, 300.0)


@pytest.mark.parametrize(
    "temperature",
    [0.0, -1.0, np.nan, np.inf, [300.0]],
)
def test_thermal_de_broglie_wavelength_rejects_invalid_temperature(
    temperature: Any,
) -> None:
    with pytest.raises(ValueError, match="temperature"):
        nqe.get_thermal_de_broglie_wavelength(1.0, temperature)


def test_thermal_de_broglie_wavelength_rejects_incompatible_units() -> None:
    with pytest.raises(ValueError, match="mass"):
        nqe.get_thermal_de_broglie_wavelength(
            1.0 * unit.nanometer,
            300.0 * unit.kelvin,
        )
    with pytest.raises(ValueError, match="temperature"):
        nqe.get_thermal_de_broglie_wavelength(
            1.0 * unit.dalton,
            300.0 * unit.nanometer,
        )


def test_init_beads_is_deterministic_and_sets_independent_thermal_velocities() -> None:
    modeller = _single_atom_modeller((1.0, 2.0, 3.0))
    first = _Integrator()
    second = _Integrator()
    first_simulation = SimpleNamespace(
        integrator=first,
        system=_System(masses=[1.0]),
    )
    second_simulation = SimpleNamespace(
        integrator=second,
        system=_System(masses=[1.0]),
    )

    nqe.init_beads(
        modeller,
        first_simulation,
        2,
        scale_factor=0.1,
        seed=11,
    )
    nqe.init_beads(
        modeller,
        second_simulation,
        2,
        scale_factor=0.1,
        seed=11,
    )

    assert set(first.positions) == {0, 1}
    for bead in (0, 1):
        first_nm = first.positions[bead].value_in_unit(unit.nanometer)
        second_nm = second.positions[bead].value_in_unit(unit.nanometer)
        assert np.allclose(first_nm, second_nm)
        assert not np.allclose(first_nm, [[1.0, 2.0, 3.0]])
        assert np.allclose(
            first.velocities[bead].value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            second.velocities[bead].value_in_unit(
                unit.nanometer / unit.picosecond
            ),
        )
        assert not np.allclose(
            first.velocities[bead].value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            0.0,
        )
    assert not np.allclose(
        first.velocities[0].value_in_unit(unit.nanometer / unit.picosecond),
        first.velocities[1].value_in_unit(unit.nanometer / unit.picosecond),
    )
    bead_positions = np.asarray([
        first.positions[bead].value_in_unit(unit.nanometer)
        for bead in (0, 1)
    ])
    assert np.allclose(bead_positions.mean(axis=0), [[1.0, 2.0, 3.0]])


def test_init_beads_velocity_scale_tracks_mass_and_skips_massless_particles() -> None:
    masses = [1.0, 4.0, 0.0]
    modeller = SimpleNamespace(
        positions=[openmm.Vec3(0.0, 0.0, 0.0) for _ in masses]
        * unit.nanometer,
    )
    integrator = _Integrator()
    simulation = SimpleNamespace(
        integrator=integrator,
        system=_System(masses=masses),
    )

    n_beads = 2_000
    temperature = 300.0 * unit.kelvin
    nqe.init_beads(
        modeller,
        simulation,
        n_beads=n_beads,
        scale_factor=0.0,
        temperature=temperature,
        seed=123,
    )
    velocities = np.asarray([
        integrator.velocities[bead].value_in_unit(
            unit.nanometer / unit.picosecond
        )
        for bead in range(n_beads)
    ])

    light_std = velocities[:, 0].std()
    heavy_std = velocities[:, 1].std()
    expected_light_std = np.sqrt(
        (
            unit.MOLAR_GAS_CONSTANT_R
            * n_beads
            * temperature
            / (masses[0] * unit.dalton)
        ).value_in_unit((unit.nanometer / unit.picosecond) ** 2)
    )
    assert light_std == pytest.approx(expected_light_std, rel=0.03)
    assert light_std / heavy_std == pytest.approx(2.0, rel=0.05)
    assert np.allclose(velocities[:, 2], 0.0)


def test_init_beads_sets_velocities_on_real_rpmd_copies() -> None:
    modeller = _single_atom_modeller()
    system = openmm.System()
    system.addParticle(1.0 * unit.dalton)
    integrator = openmm.RPMDIntegrator(
        2,
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.1 * unit.femtosecond,
    )
    simulation = app.Simulation(
        modeller.topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )

    nqe.init_beads(modeller, simulation, n_beads=2, seed=7)
    velocities = [
        integrator.getState(bead, getVelocities=True)
        .getVelocities(asNumpy=True)
        .value_in_unit(unit.nanometer / unit.picosecond)
        for bead in range(2)
    ]

    assert not np.allclose(velocities[0], 0.0)
    assert not np.allclose(velocities[1], 0.0)
    assert not np.allclose(velocities[0], velocities[1])


def test_step_rpmd_advances_context_count_and_schedules_reporters() -> None:
    class _RecordingReporter:
        def __init__(self) -> None:
            self.steps = []

        def describeNextReport(self, simulation: Any) -> tuple[int, bool, bool, bool, bool]:
            steps = 2 - simulation.currentStep % 2
            return (steps, False, False, False, False)

        def report(self, simulation: Any, state: Any) -> None:
            self.steps.append(simulation.currentStep)

    modeller = _single_atom_modeller()
    system = openmm.System()
    system.addParticle(1.0 * unit.dalton)
    timestep = 0.1 * unit.femtosecond
    integrator = openmm.RPMDIntegrator(
        2,
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        timestep,
    )
    simulation = app.Simulation(
        modeller.topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    nqe.init_beads(modeller, simulation, n_beads=2, seed=7)
    reporter = _RecordingReporter()
    simulation.reporters.append(reporter)

    nqe.step_rpmd(simulation, 5)

    assert simulation.currentStep == 5
    assert simulation.context.getTime().value_in_unit(
        unit.femtosecond
    ) == pytest.approx(0.5)
    assert reporter.steps == [2, 4]


def test_step_rpmd_does_not_double_count_native_context_updates() -> None:
    class _CountingContext:
        def __init__(self) -> None:
            self.step_count = 4

        def getStepCount(self) -> int:
            return self.step_count

        def setStepCount(self, count: int) -> None:
            self.step_count = count

    class _NativeCountingIntegrator:
        def __init__(self, context: Any) -> None:
            self.context = context
            self.calls = []

        def getNumCopies(self) -> int:
            return 2

        def step(self, count: int) -> None:
            self.calls.append(count)
            self.context.setStepCount(self.context.getStepCount() + count)

    class _CountingSimulation:
        def __init__(self) -> None:
            self.context = _CountingContext()
            self.integrator = _NativeCountingIntegrator(self.context)

        @property
        def currentStep(self) -> int:
            return self.context.getStepCount()

        @currentStep.setter
        def currentStep(self, count: int) -> None:
            self.context.setStepCount(count)

        def step(self, count: int) -> None:
            self.integrator.step(count)

    simulation = _CountingSimulation()
    native_step = simulation.integrator.step

    nqe.step_rpmd(simulation, 3)

    assert simulation.currentStep == 7
    assert simulation.integrator.calls == [3]
    assert simulation.integrator.step == native_step


@pytest.mark.parametrize(
    ("steps", "error"),
    [(True, TypeError), (1.5, TypeError), (-1, ValueError)],
)
def test_step_rpmd_rejects_invalid_step_counts(steps: Any, error: type[Exception]) -> None:
    with pytest.raises(error, match="steps must be a non-negative integer"):
        nqe.step_rpmd(SimpleNamespace(), steps)


def test_step_rpmd_restores_native_step_after_failure() -> None:
    class _FailingIntegrator:
        def getNumCopies(self) -> int:
            return 2

        def step(self, count: int) -> None:
            raise RuntimeError(f"failed after request for {count} steps")

    class _FailingSimulation:
        currentStep = 0

        def __init__(self) -> None:
            self.integrator = _FailingIntegrator()

        def step(self, count: int) -> None:
            self.integrator.step(count)

    simulation = _FailingSimulation()
    native_step = simulation.integrator.step

    with pytest.raises(RuntimeError, match="failed after request"):
        nqe.step_rpmd(simulation, 3)

    assert simulation.integrator.step == native_step


def test_init_beads_uses_thermal_position_scaling_and_preserves_centroid() -> None:
    integrator = _Integrator()
    simulation = SimpleNamespace(
        system=_System(masses=[1.0, 16.0, 0.0]),
        integrator=integrator,
    )
    positions = np.asarray([
        [1.0, 2.0, 3.0],
        [-2.0, -1.0, 0.5],
        [0.4, 0.5, 0.6],
    ])
    modeller = SimpleNamespace(positions=positions * unit.nanometer)
    n_beads = 200

    nqe.init_beads(
        modeller,
        simulation,
        n_beads=n_beads,
        temperature=300.0 * unit.kelvin,
        scale_factor=1.0,
        seed=7,
    )
    bead_positions = np.asarray([
        integrator.positions[bead].value_in_unit(unit.nanometer)
        for bead in range(n_beads)
    ])
    displacements = bead_positions - positions[np.newaxis]
    velocities = [
        integrator.velocities[bead].value_in_unit(
            unit.nanometer / unit.picosecond
        )
        for bead in range(2)
    ]

    assert displacements[:, 0].std() / displacements[:, 1].std() == (
        pytest.approx(4.0, rel=0.15)
    )
    assert np.allclose(bead_positions.mean(axis=0), positions)
    assert np.allclose(displacements[:, 2], 0.0)
    assert not np.allclose(velocities[0], 0.0)
    assert not np.allclose(velocities[1], 0.0)
    assert not np.allclose(velocities[0], velocities[1])
    assert np.allclose(np.asarray(velocities)[:, 2], 0.0)


@pytest.mark.parametrize("n_beads", [1, 5, 8])
def test_init_beads_samples_free_ring_polymer_normal_modes(n_beads: int) -> None:
    n_atoms = 3_000
    mass_amu = 1.0
    temperature_k = 300.0
    integrator = _Integrator(temperature=temperature_k * unit.kelvin)
    simulation = SimpleNamespace(
        system=_System(masses=np.full(n_atoms, mass_amu)),
        integrator=integrator,
    )
    modeller = SimpleNamespace(
        positions=np.zeros((n_atoms, 3)) * unit.nanometer,
    )

    nqe.init_beads(modeller, simulation, n_beads=n_beads, seed=1234)
    displacements_nm = np.asarray([
        integrator.positions[bead].value_in_unit(unit.nanometer)
        for bead in range(n_beads)
    ])
    modes_nm = (
        np.fft.rfft(displacements_nm, axis=0) / np.sqrt(n_beads)
    )

    omega_p = n_beads * constants.k * temperature_k / constants.hbar
    mass_kg = mass_amu * constants.atomic_mass
    for mode in range(1, n_beads // 2 + 1):
        omega_k = 2.0 * omega_p * np.sin(np.pi * mode / n_beads)
        expected_variance_nm2 = (
            n_beads * constants.k * temperature_k
            / (mass_kg * omega_k**2)
            * 1.0e18
        )
        assert np.mean(np.abs(modes_nm[mode]) ** 2) == pytest.approx(
            expected_variance_nm2,
            rel=0.05,
        )

    link_displacements_m = (
        np.roll(displacements_nm, -1, axis=0) - displacements_nm
    ) * 1.0e-9
    spring_energy = (
        0.5
        * mass_kg
        * omega_p**2
        * np.sum(link_displacements_m**2, axis=0)
    )
    assert np.mean(
        spring_energy / (n_beads * constants.k * temperature_k)
    ) == pytest.approx((n_beads - 1) / 2.0, rel=0.03)
    assert np.allclose(displacements_nm.mean(axis=0), 0.0, atol=1.0e-15)


@pytest.mark.parametrize("n_beads", [0, -1, 2.0, True])
def test_init_beads_rejects_invalid_bead_counts(n_beads: Any) -> None:
    simulation = SimpleNamespace(
        system=_System(masses=[1.0]),
        integrator=_Integrator(),
    )

    with pytest.raises(ValueError, match="n_beads must be a positive integer"):
        nqe.init_beads(_single_atom_modeller(), simulation, n_beads=n_beads)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_init_beads_rejects_invalid_seeds(seed: Any) -> None:
    simulation = SimpleNamespace(
        system=_System(masses=[1.0]),
        integrator=_Integrator(),
    )

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        nqe.init_beads(
            _single_atom_modeller(),
            simulation,
            n_beads=2,
            seed=seed,
        )


@pytest.mark.parametrize("invalid_coordinate", [np.nan, np.inf, -np.inf])
def test_init_beads_rejects_nonfinite_positions(invalid_coordinate: float) -> None:
    modeller = SimpleNamespace(
        positions=np.asarray([[invalid_coordinate, 0.0, 0.0]])
        * unit.nanometer,
    )
    simulation = SimpleNamespace(
        system=_System(masses=[1.0]),
        integrator=_Integrator(),
    )

    with pytest.raises(ValueError, match="positions must be finite"):
        nqe.init_beads(modeller, simulation, n_beads=2)


def test_init_beads_rejects_temperature_inconsistent_with_integrator() -> None:
    modeller = _single_atom_modeller()
    simulation = SimpleNamespace(
        system=_System(masses=[1.0]),
        integrator=_Integrator(temperature=300.0 * unit.kelvin),
    )

    with pytest.raises(ValueError, match="must match the RPMDIntegrator"):
        nqe.init_beads(
            modeller,
            simulation,
            n_beads=2,
            temperature=310.0 * unit.kelvin,
        )


def _bonded_chain_topology(sizes: Sequence[int]) -> app.Topology:
    """A Topology of bonded chains, one residue each, with no System behind it."""
    topology = app.Topology()
    chain = topology.addChain()
    for size in sizes:
        residue = topology.addResidue("MOL", chain)
        previous = None
        for _ in range(size):
            atom = topology.addAtom("C", app.element.carbon, residue)
            if previous is not None:
                topology.addBond(previous, atom)
            previous = atom
    return topology


def _bonded_chain_system(topology: app.Topology, box: np.ndarray,
                         bonded: bool) -> tuple[openmm.System, np.ndarray]:
    """
    Build a System for *topology*, optionally without any bonded terms.

    With *bonded* False the System carries the bonds nowhere, which is what
    ``MLPotential.createMixedSystem`` leaves behind for a molecule modelled
    entirely by the ML potential: it deletes every bonded term inside the ML
    region, and the ``openmm.PythonForce`` that replaces them reports no
    pairs. OpenMM then treats each of those atoms as its own molecule.
    """
    system = openmm.System()
    force = openmm.HarmonicBondForce()
    for _ in topology.atoms():
        system.addParticle(12.011 * unit.dalton)
    if bonded:
        for bond in topology.bonds():
            force.addBond(
                bond.atom1.index,
                bond.atom2.index,
                0.15 * unit.nanometer,
                1000.0 * unit.kilojoule_per_mole / unit.nanometer**2,
            )
    system.addForce(force)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    for _ in topology.atoms():
        nonbonded.addParticle(
            0.0, 0.3 * unit.nanometer, 0.1 * unit.kilojoule_per_mole
        )
    system.addForce(nonbonded)
    system.setDefaultPeriodicBoxVectors(
        *[Vec3(*row) * unit.nanometer for row in box]
    )
    topology.setPeriodicBoxVectors([Vec3(*row) * unit.nanometer for row in box])
    return system, box


@pytest.mark.parametrize("box", [
    np.array([[2.5, 0.0, 0.0], [0.0, 2.7, 0.0], [0.0, 0.0, 2.4]]),
    np.array([[2.5, 0.0, 0.0], [0.4, 2.6, 0.0], [0.3, 0.35, 2.4]]),
], ids=["orthorhombic", "triclinic"])
def test_wrap_molecules_reproduces_openmm_where_openmm_is_right(
    box: np.ndarray,
) -> None:
    # Where the System does carry the bonds, OpenMM's own molecule list is
    # complete and enforcePeriodicBox is correct. Matching it exactly there
    # is what makes the Topology-driven replacement a drop-in.
    topology = _bonded_chain_topology([3] * 40)
    system, box = _bonded_chain_system(topology, box, bonded=True)
    generator = np.random.default_rng(1)
    positions = np.concatenate([
        base + np.array([[0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.24, 0.0, 0.0]])
        for base in generator.uniform(-1.0, 3.0, (40, 3))[:, np.newaxis, :]
    ])

    integrator = openmm.RPMDIntegrator(
        2, 300 * unit.kelvin, 1.0 / unit.picosecond, 0.0002 * unit.picoseconds
    )
    simulation = app.Simulation(
        topology, system, integrator,
        openmm.Platform.getPlatform("Reference"),
    )
    simulation.context.setPositions(positions * unit.nanometer)
    for bead in range(2):
        integrator.setPositions(bead, positions * unit.nanometer)

    raw = integrator.getState(
        copy=0, getPositions=True, enforcePeriodicBox=False
    ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    expected = integrator.getState(
        copy=0, getPositions=True, enforcePeriodicBox=True
    ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    wrapped = nqe_tools._wrap_molecules(
        raw, box, nqe_tools._topology_molecule_tree(topology)
    )
    assert wrapped == pytest.approx(expected, abs=1e-6)


def test_wrap_molecules_keeps_a_molecule_the_system_has_no_bonds_for() -> None:
    # The mixed ML/MM case: the Topology knows the molecule, the System does
    # not, and OpenMM therefore wraps each atom on its own.
    box = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    topology = _bonded_chain_topology([4])
    system, box = _bonded_chain_system(topology, box, bonded=False)
    # A chain straddling x = 0, so wrapping has something to get wrong.
    positions = np.array([
        [-0.10, 1.0, 1.0], [0.05, 1.0, 1.0], [0.20, 1.0, 1.0], [0.35, 1.0, 1.0],
    ])

    integrator = openmm.RPMDIntegrator(
        2, 300 * unit.kelvin, 1.0 / unit.picosecond, 0.0002 * unit.picoseconds
    )
    simulation = app.Simulation(
        topology, system, integrator,
        openmm.Platform.getPlatform("Reference"),
    )
    simulation.context.setPositions(positions * unit.nanometer)
    for bead in range(2):
        integrator.setPositions(bead, positions * unit.nanometer)

    openmm_wrapped = integrator.getState(
        copy=0, getPositions=True, enforcePeriodicBox=True
    ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    ours = nqe_tools._wrap_molecules(
        positions, box, nqe_tools._topology_molecule_tree(topology)
    )

    def span(values: np.ndarray) -> float:
        return float(np.ptp(values[:, 0]))

    # OpenMM throws the first atom a whole box length away from the rest.
    assert span(openmm_wrapped) > 1.5
    assert span(ours) == pytest.approx(0.45)
    # Every bond survives, and the molecule's centre lands inside the box.
    assert np.linalg.norm(np.diff(ours, axis=0), axis=1) == pytest.approx(0.15)
    assert 0.0 <= ours.mean(axis=0)[0] < 2.0


def _straddling_chain_simulation() -> tuple[app.Simulation, np.ndarray]:
    """A 4-atom chain across x = 0 whose bonds the System does not carry."""
    box = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    topology = _bonded_chain_topology([4])
    system, box = _bonded_chain_system(topology, box, bonded=False)
    positions = np.array([
        [-0.10, 1.0, 1.0], [0.05, 1.0, 1.0], [0.20, 1.0, 1.0], [0.35, 1.0, 1.0],
    ])
    integrator = openmm.RPMDIntegrator(
        2, 300 * unit.kelvin, 1.0 / unit.picosecond, 0.0002 * unit.picoseconds
    )
    simulation = app.Simulation(
        topology, system, integrator,
        openmm.Platform.getPlatform("Reference"),
    )
    simulation.context.setPositions(positions * unit.nanometer)
    for bead in range(2):
        integrator.setPositions(bead, positions * unit.nanometer)
    return simulation, positions


def test_centroid_positions_wraps_by_topology_molecules() -> None:
    simulation, positions = _straddling_chain_simulation()

    centroid = np.asarray(
        nqe.centroid_positions(simulation, n_atoms=4, n_beads=2).value_in_unit(
            unit.nanometer
        )
    )

    # Every bond survives, and the molecule's centre lands inside the box.
    assert np.linalg.norm(
        np.diff(centroid, axis=0), axis=1
    ) == pytest.approx(0.15)
    assert 0.0 <= centroid.mean(axis=0)[0] < 2.0


def test_centroid_positions_can_be_told_not_to_wrap() -> None:
    simulation, positions = _straddling_chain_simulation()

    centroid = np.asarray(
        nqe.centroid_positions(
            simulation, n_atoms=4, n_beads=2, enforce_periodic_box=False
        ).value_in_unit(unit.nanometer)
    )

    assert centroid == pytest.approx(positions)


def test_molecule_tree_groups_atoms_by_topology_bonds() -> None:
    tree = nqe_tools._topology_molecule_tree(_bonded_chain_topology([3, 2, 1]))

    assert tree.n_molecules == 3
    assert tree.molecule.tolist() == [0, 0, 0, 1, 1, 2]
    # Roots carry no parent; every other atom is reached from a bonded one.
    assert tree.parent.tolist() == [-1, 0, 1, -1, 3, -1]
    # Levels hold the atoms at each bond distance, so a whole level can be
    # unwrapped at once after its parents are settled.
    assert [level.tolist() for level in tree.levels] == [[1, 4], [2]]


def test_centroid_positions_unwraps_beads_across_box_boundary() -> None:
    simulation = SimpleNamespace(
        integrator=_Integrator(
            bead_positions=[[[0.1, 0.0, 0.0]], [[1.9, 0.0, 0.0]]],
            box_vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        ),
        system=_System(periodic=True),
    )

    centroid = nqe.centroid_positions(simulation, n_atoms=1, n_beads=2)

    assert np.allclose(
        centroid.value_in_unit(unit.nanometer)[0],
        [0.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_centroid_positions_handles_triclinic_box() -> None:
    box = [[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.2, 0.3, 2.0]]
    simulation = SimpleNamespace(
        integrator=_Integrator(
            bead_positions=[[[0.1, 0.1, 0.1]], [[2.0, 2.2, 1.9]]],
            box_vectors=box,
        ),
        system=_System(periodic=True),
    )

    centroid = nqe.centroid_positions(simulation, n_atoms=1, n_beads=2)

    assert np.allclose(
        centroid.value_in_unit(unit.nanometer)[0],
        [-0.3, 0.0, 0.0],
        atol=1e-12,
    )


def test_centroid_positions_does_not_wrap_nonperiodic_systems() -> None:
    simulation = SimpleNamespace(
        integrator=_Integrator(
            bead_positions=[[[0.1, 0.0, 0.0]], [[1.9, 0.0, 0.0]]],
            box_vectors=np.eye(3) * 2.0,
        ),
        system=_System(periodic=False),
    )

    centroid = nqe.centroid_positions(simulation, n_atoms=1, n_beads=2)

    assert np.allclose(
        centroid.value_in_unit(unit.nanometer)[0],
        [1.0, 0.0, 0.0],
    )


def test_count_dna_charge_accounts_for_unphosphorylated_5_prime_termini() -> None:
    topology = app.Topology()
    first_strand = topology.addChain()
    for residue_name in ("DA5", "DC", "DG3"):
        topology.addResidue(residue_name, first_strand)

    second_strand = topology.addChain()
    for residue_name in ("DT5", "DA3"):
        topology.addResidue(residue_name, second_strand)

    other = topology.addChain()
    for residue_name in ("DT", "ALA", "RA"):
        topology.addResidue(residue_name, other)

    assert nqe.count_dna_and_estimate_charge(topology) == -4


def test_count_dna_charge_uses_phosphates_after_pdb_name_canonicalization(
    tmp_path: Path,
) -> None:
    topology = app.Topology()
    chain = topology.addChain("A")
    five_prime = topology.addResidue("DA5", chain, id="1")
    topology.addAtom("C1'", app.Element.getBySymbol("C"), five_prime)
    three_prime = topology.addResidue("DA3", chain, id="2")
    topology.addAtom("P", app.Element.getBySymbol("P"), three_prime)
    positions = [Vec3(0, 0, 0), Vec3(0.1, 0, 0)] * unit.nanometer
    pdb_path = tmp_path / "dna.pdb"
    with pdb_path.open("w") as handle:
        app.PDBFile.writeFile(topology, positions, handle)

    loaded = app.PDBFile(str(pdb_path)).topology

    assert [residue.name for residue in loaded.residues()] == ["DA", "DA"]
    assert nqe.count_dna_and_estimate_charge(loaded) == -1


@pytest.mark.parametrize(
    ("option", "target_resname", "expected_hydrogens"),
    [
        ("all", None, {0, 2, 4, 6, 8}),
        ("protein", None, {0}),
        ("water", None, {2}),
        ("dna", None, {4}),
        ("rna", None, {6}),
        ("nucleic", None, {4, 6}),
        ("ligand", "LIG", {8}),
    ],
)
def test_deuterate_system_selects_requested_component(
    option: str,
    target_resname: str | None,
    expected_hydrogens: set[int],
) -> None:
    modeller, system = _multi_component_modeller_and_system()

    nqe.deuterate_system(
        modeller,
        system,
        option=option,
        target_resname=target_resname,
    )
    deuterium = app.element.deuterium.mass.value_in_unit(unit.dalton)
    changed = {
        index
        for index in range(system.getNumParticles())
        if system.getParticleMass(index).value_in_unit(unit.dalton)
        == pytest.approx(deuterium)
    }

    assert changed == expected_hydrogens


def test_deuterate_system_validates_options() -> None:
    modeller, system = _multi_component_modeller_and_system()

    with pytest.raises(ValueError, match="target_resname"):
        nqe.deuterate_system(modeller, system, option="ligand")
    with pytest.raises(ValueError, match="Option must be"):
        nqe.deuterate_system(modeller, system, option="invalid")


def test_deuterate_system_warns_when_target_is_absent(capsys: pytest.CaptureFixture[str]) -> None:
    modeller, system = _multi_component_modeller_and_system()

    nqe.deuterate_system(modeller, system, option="ligand", target_resname="NOPE")

    assert "No ligand named 'NOPE'" in capsys.readouterr().out


def test_get_atoms_in_residue_supports_global_and_chain_indexes(data_dir: Path) -> None:
    source = data_dir / "pdb" / "gc.pdb"

    assert nqe.get_atoms_in_residue(source, 0) == list(range(16))
    assert nqe.get_atoms_in_residue(source, 0, chain_id="B") == list(range(16, 29))


def test_get_atoms_in_residue_reports_missing_chain_and_index(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = data_dir / "pdb" / "gc.pdb"

    assert nqe.get_atoms_in_residue(source, 0, chain_id="Z") is None
    assert "not found" in capsys.readouterr().out
    assert nqe.get_atoms_in_residue(source, -1) is None
    assert "out of bounds" in capsys.readouterr().out


def test_set_adqtb_particle_types_orders_elements_by_atomic_number() -> None:
    integrator = _Integrator()
    elements = ["O", "H", "C", "H", None]

    mapping = nqe.set_adqtb_particle_types_by_element(
        integrator,
        particle_elements=elements,
        start_type=3,
    )

    assert mapping == {"H": 3, "C": 4, "O": 5, "X": 6}
    assert integrator.particle_types == {0: 5, 1: 3, 2: 4, 3: 3, 4: 6}


def test_set_adqtb_particle_types_reads_topology_and_checks_system_size() -> None:
    modeller, _ = _multi_component_modeller_and_system()
    integrator = _Integrator()

    with pytest.raises(ValueError, match="System has 1 particles"):
        nqe.set_adqtb_particle_types_by_element(
            integrator,
            topology=modeller.topology,
            system=_System(masses=[1.0]),
        )


def test_set_adqtb_particle_types_validates_required_interfaces() -> None:
    with pytest.raises(TypeError, match="setParticleType"):
        nqe.set_adqtb_particle_types_by_element(object(), particle_elements=["H"])
    with pytest.raises(ValueError, match="topology or particle_elements"):
        nqe.set_adqtb_particle_types_by_element(_Integrator())


@pytest.mark.parametrize("start_type", [True, 1.5, "3"])
def test_set_adqtb_particle_types_rejects_noninteger_start_type(
    start_type: Any,
) -> None:
    integrator = _Integrator()

    with pytest.raises(TypeError, match="start_type must be an integer"):
        nqe.set_adqtb_particle_types_by_element(
            integrator,
            particle_elements=["H"],
            start_type=start_type,
        )

    assert integrator.particle_types == {}


def test_set_adqtb_particle_types_rejects_negative_start_type() -> None:
    integrator = _Integrator()

    with pytest.raises(ValueError, match="start_type must be a non-negative"):
        nqe.set_adqtb_particle_types_by_element(
            integrator,
            particle_elements=["H"],
            start_type=-1,
        )

    assert integrator.particle_types == {}


def _ambiguous_modeller() -> app.Modeller:
    topology = app.Topology()
    positions = []
    for chain_id in ("A", "B"):
        residue = topology.addResidue("ALA", topology.addChain(chain_id), id="12")
        topology.addAtom("CA", app.Element.getBySymbol("C"), residue)
        positions.append(Vec3(0.0, 0.0, 0.0))
    inserted = topology.addResidue(
        "HIE", topology.addChain("C"), id="258", insertionCode="A"
    )
    topology.addAtom("CD2", app.Element.getBySymbol("C"), inserted)
    positions.append(Vec3(0.0, 0.0, 0.0))
    alphanumeric = topology.addResidue(
        "CH4",
        topology.addChain("D"),
        id="1",
    )
    topology.addAtom("C1", app.Element.getBySymbol("C"), alphanumeric)
    positions.append(Vec3(0.0, 0.0, 0.0))
    return app.Modeller(topology, positions * unit.nanometer)


def test_atom_indices_from_vmd_picks_handles_chains_modes_and_insertions() -> None:
    modeller = _ambiguous_modeller()

    assert nqe.atom_indices_from_vmd_picks(
        modeller, ["ALA12:CA"], chain_id="B"
    ) == [1]
    assert nqe.atom_indices_from_vmd_picks(
        modeller, ["ALA12:CA"], match_mode="first"
    ) == [0]
    assert nqe.atom_indices_from_vmd_picks(
        modeller, ["ALA12:CA"], match_mode="all"
    ) == [[0, 1]]
    assert nqe.atom_indices_from_vmd_picks(modeller, ["HIE258A:CD2"]) == [2]
    assert nqe.atom_indices_from_vmd_picks(modeller, ["ALA 12:CA"], chain_id="A") == [0]
    assert nqe.atom_indices_from_vmd_picks(modeller, ["HIE258 A:CD2"]) == [2]
    assert nqe.atom_indices_from_vmd_picks(modeller, ["CH41:C1"]) == [3]


def test_atom_indices_from_vmd_picks_treats_a_blank_insertion_code_as_none() -> None:
    # PDB marks "no insertion code" with a blank column, and OpenMM's PDB
    # reader hands that over as ' ' rather than '' -- truthy, so a naive
    # `or` keeps the space and no plain pick ever matches a file-read
    # topology.
    topology = app.Topology()
    residue = topology.addResidue(
        "LIG", topology.addChain("A"), id="1", insertionCode=" "
    )
    topology.addAtom("O2", app.Element.getBySymbol("O"), residue)
    modeller = app.Modeller(topology, [Vec3(0.0, 0.0, 0.0)] * unit.nanometer)

    assert nqe.atom_indices_from_vmd_picks(modeller, ["LIG1:O2"]) == [0]


@pytest.mark.parametrize(
    ("pick", "kwargs", "message"),
    [
        ("bad", {}, "malformed"),
        ("ALA99:CA", {}, "No atom matches"),
        ("ALA12:CA", {}, "matched 2 atoms"),
        ("ALA12:CA", {"match_mode": "bad"}, "Unknown match_mode"),
    ],
)
def test_atom_indices_from_vmd_picks_rejects_invalid_or_ambiguous_picks(
    pick: str,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        nqe.atom_indices_from_vmd_picks(_ambiguous_modeller(), [pick], **kwargs)


def test_distance_between_atoms_preserves_units() -> None:
    modeller = SimpleNamespace(
        positions=[Vec3(0.0, 0.0, 0.0), Vec3(0.3, 0.4, 0.0)]
        * unit.nanometer
    )

    distance = nqe.distance_between_atoms(modeller, 0, 1)

    assert unit.is_quantity(distance)
    assert distance.value_in_unit(unit.nanometer) == pytest.approx(0.5)
    assert distance.value_in_unit(unit.angstrom) == pytest.approx(5.0)


def test_distance_between_atoms_requires_positions() -> None:
    with pytest.raises(ValueError, match="positions is None"):
        nqe.distance_between_atoms(SimpleNamespace(positions=None), 0, 1)


def test_angle_between_atoms_returns_radians_or_degrees() -> None:
    modeller = SimpleNamespace(
        positions=[
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
        ]
        * unit.nanometer
    )

    assert nqe.angle_between_atoms(modeller, 0, 1, 2) == pytest.approx(np.pi / 2)
    assert nqe.angle_between_atoms(
        modeller, 0, 1, 2, degrees=True
    ) == pytest.approx(90.0)


def test_angle_between_atoms_rejects_zero_length_vector() -> None:
    modeller = SimpleNamespace(
        positions=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        ]
        * unit.nanometer
    )

    with pytest.raises(ValueError, match="zero length"):
        nqe.angle_between_atoms(modeller, 0, 1, 2)


def test_check_platform_preserves_explicit_choice() -> None:
    assert nqe.check_platform("Reference") == "Reference"


@pytest.mark.parametrize(
    ("available", "expected"),
    [(["Reference", "CPU"], "CPU"), (["Reference", "CPU", "CUDA"], "CUDA")],
)
def test_check_platform_prefers_cuda(monkeypatch: pytest.MonkeyPatch, available: list[str], expected: str) -> None:
    class FakePlatform:
        @staticmethod
        def getNumPlatforms() -> int:
            return len(available)

        @staticmethod
        def getPlatform(index: int) -> SimpleNamespace:
            return SimpleNamespace(getName=lambda: available[index])

    monkeypatch.setattr(nqe_tools.openmm, "Platform", FakePlatform)

    assert nqe.check_platform() == expected
