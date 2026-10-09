"""Check shared example setup without running research-scale simulations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import openmm.unit as unit
import pytest

import openmmnqe._setup as setup


def _example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("generated", [None, Path("ligands.xml")])
def test_parameterized_structure_preserves_the_input_and_template_order(
    monkeypatch, generated,
):
    topology, positions = object(), object()
    modeller, forcefield = object(), object()
    calls = []

    def read_pdb(path):
        calls.append(("pdb", path))
        return SimpleNamespace(topology=topology, positions=positions)

    def make_modeller(top, pos):
        calls.append(("modeller", top, pos))
        return modeller

    def make_forcefield(*names):
        calls.append(("forcefield", names))
        return forcefield

    monkeypatch.setattr(setup, "app", SimpleNamespace(
        PDBFile=read_pdb, Modeller=make_modeller, ForceField=make_forcefield,
    ))
    base = ("protein.xml", "water.xml")

    assert setup.load_parameterized_structure(Path("repaired.pdb"), base, generated) == (
        modeller, forcefield,
    )
    assert calls == [
        ("pdb", "repaired.pdb"),
        ("modeller", topology, positions),
        ("forcefield", base + (() if generated is None else ("ligands.xml",))),
    ]


@pytest.mark.parametrize("name", ["workflows", "rates"])
@pytest.mark.parametrize("skipped", [(), ("LIG",)])
def test_ligand_examples_preserve_parameterization_and_skip_policy(
    monkeypatch, name, skipped,
):
    example = _example(name)
    calls = []

    def parameterize(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(skipped=skipped, forcefield_xml="generated.xml")

    loaded = []
    pair = (object(), object())

    def load(*args):
        loaded.append(args)
        return pair

    monkeypatch.setattr(example.ff, "build_forcefield_xml", parameterize)
    monkeypatch.setattr(example, "load_parameterized_structure", load)
    if skipped:
        error = RuntimeError if name == "workflows" else AssertionError
        with pytest.raises(error, match="forcefill skipped residues"):
            example.ligand_forcefield("input.pdb")
        assert loaded == []
    else:
        assert example.ligand_forcefield("input.pdb") == pair
        assert loaded == [("input.pdb", example.BASE_FORCEFIELD, "generated.xml")]
    kwargs = {"base_forcefield": example.BASE_FORCEFIELD}
    if name == "workflows":
        kwargs["workdir"] = "forcefill_work"
    assert calls == [(("input.pdb", "ligands.xml"), kwargs)]


def test_standard_peptide_preparation_keeps_water_and_box_choices(monkeypatch):
    example = _example("workflows")
    calls = []
    topology, positions, forcefield = object(), object(), object()

    class Modeller:
        def __init__(self, top, pos):
            assert (top, pos) == (topology, positions)

        def deleteWater(self):
            calls.append("delete_water")

        def addHydrogens(self):
            calls.append("add_hydrogens")

        def addSolvent(self, ff, *, padding, boxShape):
            assert ff is forcefield
            calls.append(("solvate", padding.value_in_unit(unit.nanometer), boxShape))

    def read_pdb(path):
        calls.append(("pdb", path))
        return SimpleNamespace(topology=topology, positions=positions)

    def make_forcefield(*names):
        calls.append(("forcefield", names))
        return forcefield

    monkeypatch.setattr(example, "app", SimpleNamespace(
        PDBFile=read_pdb, Modeller=Modeller, ForceField=make_forcefield,
    ))
    modeller, result = example._solvated_peptide()

    assert isinstance(modeller, Modeller)
    assert result is forcefield
    assert calls == [
        ("pdb", "tests/data/pdb/input.pdb"),
        ("forcefield", ("amber14-all.xml", "amber14/tip3p.xml")),
        "delete_water", "add_hydrogens", ("solvate", 1.5, "cube"),
    ]


@pytest.mark.parametrize(
    ("entrypoint", "beads", "reporter", "interval", "steps"),
    [
        ("run_rpmd_quantum_spread_reporter", 32, "RPMDQuantumSpreadReporter", 1, 500),
        ("run_rpmd_bead_reporter", 4, "RPMDBeadReporter", 10, 100),
        ("run_rpmd_centroid_reporter", 32, "RPMDCentroidReporter", 10, 100),
        ("run_rpmd_thermodynamic_reporter", 32, "RPMDThermodynamicReporter", 10, 500),
    ],
)
def test_reporter_examples_keep_their_beads_observables_and_cadence(
    monkeypatch, entrypoint, beads, reporter, interval, steps,
):
    example = _example("rpmd")
    simulation = SimpleNamespace(topology=object(), reporters=[])
    calls = []

    def prepare(*, n_beads):
        calls.append(("prepare", n_beads))
        return simulation, [0, 1]

    monkeypatch.setattr(example, "_flexible_peptide_rpmd", prepare)
    monkeypatch.setattr(example.nqe, reporter, lambda **kwargs: kwargs)
    monkeypatch.setattr(example.nqe, "step_rpmd", lambda sim, count: calls.append((sim, count)))
    for name in ("remove_file", "plot_rpmd_atom_expansion", "plot_rpmd_thermodynamics"):
        monkeypatch.setattr(example.nqe, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(example.nqe, "rpmd_thermodynamics", lambda sim: {"energy_quantum": 1.0})
    monkeypatch.setattr(example.nqe, "rpmd_thermodynamic_averages", lambda *args, **kwargs: {
        column: (1.0, 0.1) for column in (
            "KE_cv(kJ/mol)", "PE_mean(kJ/mol)", "E_quantum(kJ/mol)", "T_ring(K)", "T_centroid(K)",
        )
    })

    getattr(example, entrypoint)()

    assert calls == [("prepare", beads), (simulation, steps)]
    assert len(simulation.reporters) == 1
    options = simulation.reporters[0]
    assert options["reportInterval"] == interval
    if reporter in ("RPMDBeadReporter", "RPMDCentroidReporter"):
        assert options["num_beads"] == beads
        assert options["topology"] is simulation.topology
    elif reporter == "RPMDQuantumSpreadReporter":
        assert options["metric"] == "mean"
        assert options["atom_indices"] == [0, 1]
        assert options["distance_pairs"] == [(0, 1)]
