"""Regression tests for the ML/MM configuration matrix in ``_build_system``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from ase import Atoms
from ase import units as ase_units
from ase.calculators.lj import LennardJones
from openmm import openmm
from openmmml import MLPotential

import openmmnqe.openmm as nqe_openmm
from openmmnqe.openmm import PreparedSystem, _build_system

TOLUENE = Path(__file__).resolve().parent / "data" / "pdb" / "toluene.pdb"


def _toluene_modeller() -> app.Modeller:
    pdb = app.PDBFile(str(TOLUENE))
    return app.Modeller(pdb.topology, pdb.positions)


class _Topology:
    def __init__(self, periodic: bool=False, n_atoms: int=4) -> None:
        self.periodic = periodic
        self.n_atoms = n_atoms

    def getUnitCellDimensions(self) -> object | None:
        return object() if self.periodic else None

    def getNumAtoms(self) -> int:
        return self.n_atoms


class _ForceField:
    def __init__(self) -> None:
        self.calls = []

    def createSystem(self, topology: Any, **kwargs: Any) -> str:
        self.calls.append((topology, kwargs))
        return "mm-system"


class _Potential:
    def __init__(self) -> None:
        self.calls = []

    def createMixedSystem(self, topology: Any, mm_system: Any, ml_idx: list[int], **kwargs: Any) -> str:
        self.calls.append((topology, mm_system, ml_idx, kwargs))
        return "mixed-system"


@pytest.fixture
def fake_platform(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    requested = []

    class Platform:
        @staticmethod
        def getPlatformByName(name: str) -> str:
            requested.append(name)
            return f"platform:{name}"

    monkeypatch.setattr(nqe_openmm.openmm, "Platform", Platform)
    monkeypatch.setattr(
        nqe_openmm,
        "check_platform",
        lambda name=None: "CPU" if name is None else name,
    )
    return requested


def test_calculator_with_plain_forcefield_needs_ml_idx() -> None:
    modeller = _toluene_modeller()
    forcefield = app.ForceField("amber14-all.xml")

    with pytest.raises(ValueError, match="ml_idx"):
        _build_system(
            modeller,
            forcefield,
            "CPU",
            potential=None,
            ml_idx=None,
            calculator=LennardJones(),
        )


def test_potential_needs_ml_idx() -> None:
    modeller = _toluene_modeller()
    forcefield = app.ForceField("amber14-all.xml")

    with pytest.raises(ValueError, match="ml_idx"):
        _build_system(
            modeller,
            forcefield,
            "CPU",
            potential=MLPotential("ase"),
            ml_idx=None,
            calculator=LennardJones(),
        )


@pytest.mark.parametrize(
    ("potential", "calculator", "ml_idx", "message"),
    [
        (None, None, [0], "without an ML potential"),
        (object(), None, [], "ml_idx is empty"),
        (None, object(), [], "ml_idx is empty"),
    ],
)
def test_invalid_ml_region_configurations_raise(potential: Any, calculator: Any, ml_idx: list[int] | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build_system(
            SimpleNamespace(topology=_Topology()),
            _ForceField(),
            "CPU",
            potential=potential,
            ml_idx=ml_idx,
            calculator=calculator,
        )


@pytest.mark.parametrize(
    ("ml_idx", "error", "message"),
    [
        ([True], TypeError, r"ml_idx\[0\] must be an integer"),
        ([1.0], TypeError, r"ml_idx\[0\] must be an integer"),
        ([-1], ValueError, r"ml_idx\[0\] must be a non-negative integer"),
        ([4], ValueError, r"ml_idx\[0\]=4 is outside the topology"),
        ([1, 1], ValueError, "duplicate atom index 1"),
    ],
)
def test_mixed_system_rejects_invalid_ml_indices(
    ml_idx: list[Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _build_system(
            SimpleNamespace(topology=_Topology()),
            _ForceField(),
            "CPU",
            potential=_Potential(),
            ml_idx=ml_idx,
            calculator=None,
        )


def test_mixed_system_normalizes_numpy_integer_indices(
    fake_platform: list[str],
) -> None:
    topology = _Topology()
    potential = _Potential()

    _build_system(
        SimpleNamespace(topology=topology),
        _ForceField(),
        "CPU",
        potential=potential,
        ml_idx=np.array([1, 3], dtype=np.int64),
        calculator=None,
    )

    normalized = potential.calls[0][2]
    assert normalized == [1, 3]
    assert all(type(index) is int for index in normalized)


def test_pure_ml_forcefield_with_calculator_builds_system() -> None:
    modeller = _toluene_modeller()

    system, platform = _build_system(
        modeller,
        MLPotential("ase"),
        "CPU",
        potential=None,
        ml_idx=None,
        calculator=LennardJones(),
    )

    assert system.getNumParticles() == modeller.topology.getNumAtoms()
    assert platform.getName() == "CPU"


@pytest.mark.parametrize(
    ("periodic", "expected_method"),
    [(False, app.CutoffNonPeriodic), (True, app.PME)],
)
def test_pure_mm_build_uses_boundary_appropriate_nonbonded_method(
    fake_platform: list[str],
    periodic: bool,
    expected_method: Any,
) -> None:
    topology = _Topology(periodic=periodic)
    forcefield = _ForceField()

    system, platform = _build_system(
        SimpleNamespace(topology=topology),
        forcefield,
        None,
        potential=None,
        ml_idx=None,
        calculator=None,
    )

    assert system == "mm-system"
    assert platform == "platform:CPU"
    assert fake_platform == ["CPU"]
    assert forcefield.calls[0][1]["nonbondedMethod"] == expected_method
    assert forcefield.calls[0][1]["constraints"] is None
    assert forcefield.calls[0][1]["rigidWater"] is False
    assert forcefield.calls[0][1]["hydrogenMass"] is None


def test_explicit_potential_builds_mixed_system_and_forces_cuda(fake_platform: list[str]) -> None:
    topology = _Topology()
    forcefield = _ForceField()
    potential = _Potential()
    calculator = object()

    system, platform = _build_system(
        SimpleNamespace(topology=topology),
        forcefield,
        "CPU",
        potential=potential,
        ml_idx=[1, 3],
        calculator=calculator,
    )

    assert system == "mixed-system"
    assert platform == "platform:CUDA"
    assert fake_platform == ["CUDA"]
    assert forcefield.calls[0][1].get("calculator") is None
    assert forcefield.calls[0][1]["hydrogenMass"] is None
    _, mm_system, ml_idx, kwargs = potential.calls[0]
    assert mm_system == "mm-system"
    assert ml_idx == [1, 3]
    assert kwargs["calculator"] is calculator
    assert kwargs["hydrogenMass"] is None


def _one_particle_openmm_system(n_particles: int=1) -> openmm.System:
    system = openmm.System()
    for _ in range(n_particles):
        system.addParticle(39.9 * unit.dalton)
    return system


def test_prepared_system_returns_held_system_and_ignores_kwargs(one_particle_system: tuple[app.Modeller, Any]) -> None:
    modeller, _ = one_particle_system
    system = _one_particle_openmm_system()
    prepared = PreparedSystem(system)

    built = prepared.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        rigidWater=False,
        calculator=object(),
    )

    assert built is system
    assert prepared.system is system


def test_prepared_system_rejects_particle_count_mismatch(one_particle_system: tuple[app.Modeller, Any]) -> None:
    modeller, _ = one_particle_system
    prepared = PreparedSystem(_one_particle_openmm_system(n_particles=2))

    with pytest.raises(
        ValueError,
        match="holds 2 particles but the stage topology has 1 atoms",
    ):
        prepared.createSystem(modeller.topology)


def test_prepared_system_requires_openmm_system() -> None:
    with pytest.raises(TypeError, match="openmm.System"):
        PreparedSystem("not-a-system")


def test_prepared_system_flows_through_build_system(fake_platform: list[str], one_particle_system: tuple[app.Modeller, Any]) -> None:
    modeller, _ = one_particle_system
    system = _one_particle_openmm_system()

    built, platform = _build_system(
        modeller,
        PreparedSystem(system),
        None,
        potential=None,
        ml_idx=None,
        calculator=None,
    )

    assert built is system
    assert platform == "platform:CPU"


def test_prepared_system_with_calculator_still_needs_ml_idx(one_particle_system: tuple[app.Modeller, Any]) -> None:
    modeller, _ = one_particle_system

    with pytest.raises(ValueError, match="ml_idx"):
        _build_system(
            modeller,
            PreparedSystem(_one_particle_openmm_system()),
            "CPU",
            potential=None,
            ml_idx=None,
            calculator=object(),
        )


def test_bare_calculator_uses_ase_potential_fallback(monkeypatch: pytest.MonkeyPatch, fake_platform: list[str]) -> None:
    topology = _Topology()
    forcefield = _ForceField()
    fallback = _Potential()
    constructed = []

    def make_potential(name: str) -> _Potential:
        constructed.append(name)
        return fallback

    monkeypatch.setattr(nqe_openmm, "MLPotential", make_potential)
    calculator = object()

    system, _ = _build_system(
        SimpleNamespace(topology=topology),
        forcefield,
        "Reference",
        potential=None,
        ml_idx=[0],
        calculator=calculator,
    )

    assert system == "mixed-system"
    assert constructed == ["ase"]
    assert fallback.calls[0][2] == [0]
    assert fallback.calls[0][3]["calculator"] is calculator


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("explicit_potential", [False, True])
def test_ase_mixed_system_preserves_energies_and_sparse_subset_forces(
    monkeypatch: pytest.MonkeyPatch,
    periodic: bool,
    explicit_potential: bool,
) -> None:
    """Exercise OpenMM-ML's actual embedding and PythonForce.setParticles path."""
    # Building a mixed system normally selects CUDA. Reference lets this
    # dependency integration check also run on CI machines without a GPU.
    monkeypatch.setattr(nqe_openmm, "check_platform", lambda name: "Reference")
    topology = app.Topology()
    chain = topology.addChain()
    mm_system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    for index, charge in enumerate([0.25, 0.0, -0.25]):
        residue = topology.addResidue("AR", chain)
        topology.addAtom(f"Ar{index}", app.element.argon, residue)
        mm_system.addParticle(app.element.argon.mass)
        nonbonded.addParticle(charge, 0.2, 0.0)
    nonbonded.setCutoffDistance(0.9 * unit.nanometer)
    if periodic:
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
        box = np.eye(3) * 3.0 * unit.nanometer
        topology.setPeriodicBoxVectors(box)
        mm_system.setDefaultPeriodicBoxVectors(*box)
    else:
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    mm_system.addForce(nonbonded)
    positions = np.array([[0.2, 0.2, 0.2], [1.7, 1.3, 1.3], [0.55, 0.2, 0.2]])
    modeller = app.Modeller(topology, positions * unit.nanometer)
    ml_idx = [2, 0]  # Non-contiguous and deliberately out of topology order.
    calculator = LennardJones(sigma=2.8, epsilon=1.0)

    system, platform = _build_system(
        modeller,
        PreparedSystem(mm_system),
        "Reference",
        potential=MLPotential("ase") if explicit_potential else None,
        ml_idx=ml_idx,
        calculator=calculator,
    )

    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions(modeller.positions)
    state = context.getState(getEnergy=True, getForces=True)

    # The pre-1.8 mechanical convention removes the direct ML pair and
    # retains MM periodic-image electrostatics. Compare with that independently
    # constructed MM reference plus ASE's own energy and scattered forces.
    reference_system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
    reference_nonbonded = reference_system.getForce(0)
    reference_nonbonded.addException(2, 0, 0.0, 1.0, 0.0, True)
    reference_context = openmm.Context(
        reference_system, openmm.VerletIntegrator(0.001), platform,
    )
    reference_context.setPositions(modeller.positions)
    reference_state = reference_context.getState(getEnergy=True, getForces=True)
    atoms = Atoms("Ar2", positions=positions[ml_idx] * 10.0, pbc=periodic)
    if periodic:
        atoms.set_cell(np.eye(3) * 30.0)
    atoms.calc = LennardJones(sigma=2.8, epsilon=1.0)
    ev_to_kj_mol = 1.0 / (ase_units.kJ / ase_units.mol)
    expected_energy = reference_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    expected_energy += atoms.get_potential_energy() * ev_to_kj_mol
    expected_forces = reference_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer,
    )
    expected_forces[ml_idx] += atoms.get_forces() * 10.0 * ev_to_kj_mol

    assert state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        expected_energy, abs=1e-5,
    )
    np.testing.assert_allclose(
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
        expected_forces,
        rtol=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("long_range", [False, True])
def test_periodic_mixed_system_leaves_known_model_range_to_openmmml(
    fake_platform: list[str],
    long_range: bool,
) -> None:
    potential = _Potential()
    potential._impl = SimpleNamespace(getMLLongRange=lambda: long_range)

    _build_system(
        SimpleNamespace(topology=_Topology(periodic=True)),
        _ForceField(),
        "CPU",
        potential=potential,
        ml_idx=[0],
        calculator=None,
    )

    # OpenMM-ML rejects mlLongRange whenever its model declares a known value.
    assert "mlLongRange" not in potential.calls[0][3]


def test_mixed_system_rejects_added_link_atoms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nqe_openmm, "check_platform", lambda name: "Reference")
    modeller = _toluene_modeller()
    mm_system = openmm.System()
    for atom in modeller.topology.atoms():
        mm_system.addParticle(atom.element.mass)

    with pytest.raises(ValueError, match="Select complete molecules"):
        _build_system(
            modeller,
            PreparedSystem(mm_system),
            "Reference",
            potential=MLPotential("ase"),
            ml_idx=[0],
            calculator=LennardJones(),
        )
