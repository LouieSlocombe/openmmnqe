"""Regression tests for the ML/MM configuration matrix in ``_build_system``."""

from pathlib import Path
from types import SimpleNamespace

from ase.calculators.lj import LennardJones
import openmm.app as app
import pytest
from openmmml import MLPotential

from openmmnqe.openmm import _build_system
import openmmnqe.openmm as nqe_openmm

TOLUENE = Path(__file__).resolve().parent / "data" / "pdb" / "toluene.pdb"


def _toluene_modeller():
    pdb = app.PDBFile(str(TOLUENE))
    return app.Modeller(pdb.topology, pdb.positions)


class _Topology:
    def __init__(self, periodic=False):
        self.periodic = periodic

    def getUnitCellDimensions(self):
        return object() if self.periodic else None


class _ForceField:
    def __init__(self):
        self.calls = []

    def createSystem(self, topology, **kwargs):
        self.calls.append((topology, kwargs))
        return "mm-system"


class _Potential:
    def __init__(self):
        self.calls = []

    def createMixedSystem(self, topology, mm_system, ml_idx, **kwargs):
        self.calls.append((topology, mm_system, ml_idx, kwargs))
        return "mixed-system"


@pytest.fixture
def fake_platform(monkeypatch):
    requested = []

    class Platform:
        @staticmethod
        def getPlatformByName(name):
            requested.append(name)
            return f"platform:{name}"

    monkeypatch.setattr(nqe_openmm.openmm, "Platform", Platform)
    monkeypatch.setattr(
        nqe_openmm,
        "check_platform",
        lambda name=None: "CPU" if name is None else name,
    )
    return requested


def test_calculator_with_plain_forcefield_needs_ml_idx():
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


def test_potential_needs_ml_idx():
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
def test_invalid_ml_region_configurations_raise(potential, calculator, ml_idx, message):
    with pytest.raises(ValueError, match=message):
        _build_system(
            SimpleNamespace(topology=_Topology()),
            _ForceField(),
            "CPU",
            potential=potential,
            ml_idx=ml_idx,
            calculator=calculator,
        )


def test_pure_ml_forcefield_with_calculator_builds_system():
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
    fake_platform,
    periodic,
    expected_method,
):
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


def test_explicit_potential_builds_mixed_system_and_forces_cuda(fake_platform):
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


def test_bare_calculator_uses_ase_potential_fallback(monkeypatch, fake_platform):
    topology = _Topology()
    forcefield = _ForceField()
    fallback = _Potential()
    constructed = []

    def make_potential(name):
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
