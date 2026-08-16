"""Focused unit tests for simulation setup and geometry helpers."""

from types import SimpleNamespace

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm
from scipy import constants

import openmmnqe as nqe
import openmmnqe.tools as nqe_tools


class _State:
    def __init__(self, positions, box_vectors):
        self._positions = np.asarray(positions, dtype=float) * unit.nanometer
        self._box_vectors = np.asarray(box_vectors, dtype=float) * unit.nanometer

    def getPositions(self, asNumpy=False):
        return self._positions

    def getPeriodicBoxVectors(self, asNumpy=False):
        return self._box_vectors


class _Integrator:
    def __init__(self, bead_positions=(), box_vectors=None,
                 temperature=300.0 * unit.kelvin):
        box_vectors = np.eye(3) if box_vectors is None else box_vectors
        self._states = [
            _State(positions, box_vectors)
            for positions in bead_positions
        ]
        self.positions = {}
        self.velocities = {}
        self.particle_types = {}
        self.temperature = temperature

    def getState(self, bead, **kwargs):
        return self._states[bead]

    def setPositions(self, bead, positions):
        self.positions[bead] = positions

    def setVelocities(self, bead, velocities):
        self.velocities[bead] = velocities

    def getTemperature(self):
        return self.temperature

    def setParticleType(self, particle, particle_type):
        self.particle_types[particle] = particle_type


class _System:
    def __init__(self, periodic=False, masses=()):
        self._periodic = periodic
        self._masses = [mass * unit.dalton for mass in masses]

    def usesPeriodicBoundaryConditions(self):
        return self._periodic

    def getNumParticles(self):
        return len(self._masses)

    def getParticleMass(self, index):
        return self._masses[index]


def _single_atom_modeller(position=(0.0, 0.0, 0.0)):
    topology = app.Topology()
    residue = topology.addResidue("LIG", topology.addChain("A"), id="1")
    topology.addAtom("H1", app.Element.getBySymbol("H"), residue)
    return app.Modeller(topology, [Vec3(*position)] * unit.nanometer)


def _multi_component_modeller_and_system():
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


def test_zero_velocities_returns_unit_bearing_vectors():
    velocities = nqe.zero_velocities(3)

    assert unit.is_quantity(velocities)
    assert velocities.unit.is_compatible(unit.nanometer / unit.picosecond)
    assert np.array(velocities.value_in_unit(unit.nanometer / unit.picosecond)).shape == (3, 3)
    assert np.allclose(
        velocities.value_in_unit(unit.nanometer / unit.picosecond),
        0.0,
    )


def test_write_multimodel_pdb_delegates_model_index(monkeypatch):
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


def test_thermal_de_broglie_wavelength_accepts_numbers_and_quantities():
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


def test_init_beads_is_deterministic_and_sets_independent_thermal_velocities():
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


def test_init_beads_velocity_scale_tracks_mass_and_skips_massless_particles():
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


def test_init_beads_sets_velocities_on_real_rpmd_copies():
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


def test_step_rpmd_advances_context_count_and_schedules_reporters():
    class _RecordingReporter:
        def __init__(self):
            self.steps = []

        def describeNextReport(self, simulation):
            steps = 2 - simulation.currentStep % 2
            return (steps, False, False, False, False)

        def report(self, simulation, state):
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


def test_step_rpmd_does_not_double_count_native_context_updates():
    class _CountingContext:
        def __init__(self):
            self.step_count = 4

        def getStepCount(self):
            return self.step_count

        def setStepCount(self, count):
            self.step_count = count

    class _NativeCountingIntegrator:
        def __init__(self, context):
            self.context = context
            self.calls = []

        def getNumCopies(self):
            return 2

        def step(self, count):
            self.calls.append(count)
            self.context.setStepCount(self.context.getStepCount() + count)

    class _CountingSimulation:
        def __init__(self):
            self.context = _CountingContext()
            self.integrator = _NativeCountingIntegrator(self.context)

        @property
        def currentStep(self):
            return self.context.getStepCount()

        @currentStep.setter
        def currentStep(self, count):
            self.context.setStepCount(count)

        def step(self, count):
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
def test_step_rpmd_rejects_invalid_step_counts(steps, error):
    with pytest.raises(error, match="steps must be a non-negative integer"):
        nqe.step_rpmd(SimpleNamespace(), steps)


def test_step_rpmd_restores_native_step_after_failure():
    class _FailingIntegrator:
        def getNumCopies(self):
            return 2

        def step(self, count):
            raise RuntimeError(f"failed after request for {count} steps")

    class _FailingSimulation:
        currentStep = 0

        def __init__(self):
            self.integrator = _FailingIntegrator()

        def step(self, count):
            self.integrator.step(count)

    simulation = _FailingSimulation()
    native_step = simulation.integrator.step

    with pytest.raises(RuntimeError, match="failed after request"):
        nqe.step_rpmd(simulation, 3)

    assert simulation.integrator.step == native_step


def test_init_beads_uses_thermal_position_scaling_and_preserves_centroid():
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
def test_init_beads_samples_free_ring_polymer_normal_modes(n_beads):
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
def test_init_beads_rejects_invalid_bead_counts(n_beads):
    simulation = SimpleNamespace(
        system=_System(masses=[1.0]),
        integrator=_Integrator(),
    )

    with pytest.raises(ValueError, match="n_beads must be a positive integer"):
        nqe.init_beads(_single_atom_modeller(), simulation, n_beads=n_beads)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_init_beads_rejects_invalid_seeds(seed):
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
def test_init_beads_rejects_nonfinite_positions(invalid_coordinate):
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


def test_init_beads_rejects_temperature_inconsistent_with_integrator():
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


def test_centroid_positions_unwraps_beads_across_box_boundary():
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


def test_centroid_positions_handles_triclinic_box():
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


def test_centroid_positions_does_not_wrap_nonperiodic_systems():
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


def test_count_dna_charge_recognises_internal_and_terminal_names():
    topology = app.Topology()
    chain = topology.addChain()
    for residue_name in ("DA", "DC5", "DG3", "DT", "ALA", "RA"):
        topology.addResidue(residue_name, chain)

    assert nqe.count_dna_and_estimate_charge(topology) == -4


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
    option,
    target_resname,
    expected_hydrogens,
):
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


def test_deuterate_system_validates_options():
    modeller, system = _multi_component_modeller_and_system()

    with pytest.raises(ValueError, match="target_resname"):
        nqe.deuterate_system(modeller, system, option="ligand")
    with pytest.raises(ValueError, match="Option must be"):
        nqe.deuterate_system(modeller, system, option="invalid")


def test_deuterate_system_warns_when_target_is_absent(capsys):
    modeller, system = _multi_component_modeller_and_system()

    nqe.deuterate_system(modeller, system, option="ligand", target_resname="NOPE")

    assert "No ligand named 'NOPE'" in capsys.readouterr().out


def test_get_atoms_in_residue_supports_global_and_chain_indexes(data_dir):
    source = data_dir / "pdb" / "gc.pdb"

    assert nqe.get_atoms_in_residue(source, 0) == list(range(16))
    assert nqe.get_atoms_in_residue(source, 0, chain_id="B") == list(range(16, 29))


def test_get_atoms_in_residue_reports_missing_chain_and_index(data_dir, capsys):
    source = data_dir / "pdb" / "gc.pdb"

    assert nqe.get_atoms_in_residue(source, 0, chain_id="Z") is None
    assert "not found" in capsys.readouterr().out
    assert nqe.get_atoms_in_residue(source, -1) is None
    assert "out of bounds" in capsys.readouterr().out


def test_set_adqtb_particle_types_orders_elements_by_atomic_number():
    integrator = _Integrator()
    elements = ["O", "H", "C", "H", None]

    mapping = nqe.set_adqtb_particle_types_by_element(
        integrator,
        particle_elements=elements,
        start_type=3,
    )

    assert mapping == {"H": 3, "C": 4, "O": 5, "X": 6}
    assert integrator.particle_types == {0: 5, 1: 3, 2: 4, 3: 3, 4: 6}


def test_set_adqtb_particle_types_reads_topology_and_checks_system_size():
    modeller, _ = _multi_component_modeller_and_system()
    integrator = _Integrator()

    with pytest.raises(ValueError, match="System has 1 particles"):
        nqe.set_adqtb_particle_types_by_element(
            integrator,
            topology=modeller.topology,
            system=_System(masses=[1.0]),
        )


def test_set_adqtb_particle_types_validates_required_interfaces():
    with pytest.raises(TypeError, match="setParticleType"):
        nqe.set_adqtb_particle_types_by_element(object(), particle_elements=["H"])
    with pytest.raises(ValueError, match="topology or particle_elements"):
        nqe.set_adqtb_particle_types_by_element(_Integrator())


def _ambiguous_modeller():
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
    return app.Modeller(topology, positions * unit.nanometer)


def test_atom_indices_from_vmd_picks_handles_chains_modes_and_insertions():
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
    pick,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        nqe.atom_indices_from_vmd_picks(_ambiguous_modeller(), [pick], **kwargs)


def test_distance_between_atoms_preserves_units():
    modeller = SimpleNamespace(
        positions=[Vec3(0.0, 0.0, 0.0), Vec3(0.3, 0.4, 0.0)]
        * unit.nanometer
    )

    distance = nqe.distance_between_atoms(modeller, 0, 1)

    assert unit.is_quantity(distance)
    assert distance.value_in_unit(unit.nanometer) == pytest.approx(0.5)
    assert distance.value_in_unit(unit.angstrom) == pytest.approx(5.0)


def test_distance_between_atoms_requires_positions():
    with pytest.raises(ValueError, match="positions is None"):
        nqe.distance_between_atoms(SimpleNamespace(positions=None), 0, 1)


def test_angle_between_atoms_returns_radians_or_degrees():
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


def test_angle_between_atoms_rejects_zero_length_vector():
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


def test_check_platform_preserves_explicit_choice():
    assert nqe.check_platform("Reference") == "Reference"


@pytest.mark.parametrize(
    ("available", "expected"),
    [(["Reference", "CPU"], "CPU"), (["Reference", "CPU", "CUDA"], "CUDA")],
)
def test_check_platform_prefers_cuda(monkeypatch, available, expected):
    class FakePlatform:
        @staticmethod
        def getNumPlatforms():
            return len(available)

        @staticmethod
        def getPlatform(index):
            return SimpleNamespace(getName=lambda: available[index])

    monkeypatch.setattr(nqe_tools.openmm, "Platform", FakePlatform)

    assert nqe.check_platform() == expected
