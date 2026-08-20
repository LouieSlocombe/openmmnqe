"""Regression tests for the ML/MM configuration matrix in ``_build_system``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ase.calculators.lj import LennardJones
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import openmm
from openmmml import MLPotential

from openmmnqe.openmm import PreparedSystem, _build_system
import openmmnqe.openmm as nqe_openmm

TOLUENE = Path(__file__).resolve().parent / "data" / "pdb" / "toluene.pdb"


def _toluene_modeller() -> app.Modeller:
    pdb = app.PDBFile(str(TOLUENE))
    return app.Modeller(pdb.topology, pdb.positions)


class _Topology:
    def __init__(self, periodic: bool=False) -> None:
        self.periodic = periodic

    def getUnitCellDimensions(self) -> object | None:
        return object() if self.periodic else None


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
