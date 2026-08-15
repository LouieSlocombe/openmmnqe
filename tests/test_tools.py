"""Focused unit tests for simulation setup and geometry helpers."""

from types import SimpleNamespace

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm

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
    def __init__(self, bead_positions=(), box_vectors=None):
        box_vectors = np.eye(3) if box_vectors is None else box_vectors
        self._states = [
            _State(positions, box_vectors)
            for positions in bead_positions
        ]
        self.positions = {}
        self.velocities = {}
        self.particle_types = {}

    def getState(self, bead, **kwargs):
        return self._states[bead]

    def setPositions(self, bead, positions):
        self.positions[bead] = positions

    def setVelocities(self, bead, velocities):
        self.velocities[bead] = velocities

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


def test_init_beads_is_deterministic_and_sets_zero_velocities():
    modeller = _single_atom_modeller((1.0, 2.0, 3.0))
    first = _Integrator()
    second = _Integrator()

    nqe.init_beads(modeller, SimpleNamespace(integrator=first), 2, perturb=0.01)
    nqe.init_beads(modeller, SimpleNamespace(integrator=second), 2, perturb=0.01)

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
            0.0,
        )


def test_init_beads_scaled_spreads_light_atoms_further():
    integrator = _Integrator()
    temperatures = []
    simulation = SimpleNamespace(
        system=_System(masses=[1.0, 16.0]),
        integrator=integrator,
        context=SimpleNamespace(
            setVelocitiesToTemperature=lambda temperature: temperatures.append(temperature)
        ),
    )
    positions = np.zeros((2, 3))

    nqe.init_beads_scaled(
        simulation,
        positions,
        n_beads=1,
        temperature=300.0 * unit.kelvin,
        scale_factor=1.0,
    )
    displacement = np.abs(
        integrator.positions[0].value_in_unit(unit.nanometer)
    )

    assert displacement[0].mean() > displacement[1].mean()
    assert temperatures == [300.0 * unit.kelvin]


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
