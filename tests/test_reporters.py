"""Unit tests for RPMD reporter calculations and output protocols."""

from types import SimpleNamespace

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3

from openmmnqe.reporters import (
    RPMDBeadReporter,
    RPMDCentroidReporter,
    RPMDQuantumSpreadReporter,
    _calculate_quantum_spread,
    track_rpmd_atom_expansion,
)
import openmmnqe.reporters as reporters


class _State:
    def __init__(self, positions):
        self._positions = np.asarray(positions, dtype=float) * unit.nanometer

    def getPositions(self, asNumpy=False):
        return self._positions


class _Integrator:
    def __init__(self, bead_positions):
        self._states = [_State(positions) for positions in bead_positions]
        self.calls = []

    def getNumCopies(self):
        return len(self._states)

    def getState(self, copy=None, **kwargs):
        self.calls.append((copy, kwargs))
        return self._states[copy]


def _topology():
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    return topology


def test_calculate_quantum_spread_returns_per_atom_rms_radius():
    integrator = _Integrator(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        ]
    )

    spread = _calculate_quantum_spread(integrator)
    selected = _calculate_quantum_spread(integrator, atom_indices=[1])

    assert unit.is_quantity(spread)
    assert np.allclose(spread.value_in_unit(unit.nanometer), [1.0, 2.0])
    assert np.allclose(selected.value_in_unit(unit.nanometer), [2.0])
    assert all(call[1] == {"getPositions": True} for call in integrator.calls)


def test_quantum_spread_reporter_writes_header_and_values(tmp_path):
    output = tmp_path / "spread.tsv"
    reporter = RPMDQuantumSpreadReporter(
        output,
        reportInterval=5,
        atom_indices=[0, 1],
        names=["H", "O"],
    )
    simulation = SimpleNamespace(
        currentStep=7,
        integrator=_Integrator(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            ]
        ),
    )

    assert reporter.describeNextReport(simulation) == (3, False, False, False, False)
    reporter.report(simulation, state=None)
    reporter._out.close()

    assert output.read_text().splitlines() == [
        "Step\tRg_H(nm)\tRg_O(nm)",
        "7\t1.000000\t2.000000",
    ]


def test_track_rpmd_atom_expansion_attaches_single_atom_reporter(tmp_path):
    output = tmp_path / "atom_expansion.tsv"
    simulation = SimpleNamespace(
        currentStep=12,
        reporters=[],
        integrator=_Integrator(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[8.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        ),
    )

    reporter = track_rpmd_atom_expansion(
        simulation,
        atom_index=1,
        file=output,
        report_interval=5,
        name="target",
    )

    assert simulation.reporters == [reporter]
    assert reporter.describeNextReport(simulation) == (3, False, False, False, False)

    reporter.report(simulation, state=None)
    reporter._out.close()
    assert output.read_text().splitlines() == [
        "Step\tRg_target(nm)",
        "12\t1.000000",
    ]


@pytest.mark.parametrize("atom_index", [-1, 1.5, True])
def test_track_rpmd_atom_expansion_rejects_invalid_atom_index(
    tmp_path, atom_index,
):
    simulation = SimpleNamespace(reporters=[])
    error = ValueError if atom_index == -1 else TypeError

    with pytest.raises(error, match="atom_index"):
        track_rpmd_atom_expansion(
            simulation,
            atom_index=atom_index,
            file=tmp_path / "spread.tsv",
            report_interval=1,
        )

    assert simulation.reporters == []
    assert not (tmp_path / "spread.tsv").exists()


@pytest.mark.parametrize(
    "factory",
    [
        lambda path: RPMDQuantumSpreadReporter(path, 0, [0]),
        lambda path: RPMDBeadReporter(path, 0, 1, _topology()),
        lambda path: RPMDBeadReporter(path, 1, 0, _topology()),
        lambda path: RPMDCentroidReporter(path, 0, 1, _topology()),
        lambda path: RPMDCentroidReporter(path, 1, 0, _topology()),
    ],
)
def test_reporters_reject_nonpositive_intervals_or_bead_counts(tmp_path, factory):
    with pytest.raises(ValueError, match="must be positive"):
        factory(tmp_path / "output")


def test_quantum_spread_reporter_validates_names(tmp_path):
    with pytest.raises(ValueError, match="one entry per atom"):
        RPMDQuantumSpreadReporter(
            tmp_path / "spread.tsv",
            reportInterval=1,
            atom_indices=[0, 1],
            names=["only-one"],
        )


def test_bead_reporter_writes_one_model_per_bead(tmp_path):
    base = tmp_path / "beads"
    reporter = RPMDBeadReporter(
        file_base_name=str(base),
        reportInterval=4,
        num_beads=2,
        topology=_topology(),
    )
    integrator = _Integrator([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    simulation = SimpleNamespace(currentStep=5, integrator=integrator)

    assert reporter.describeNextReport(simulation) == (3, False, False, False, False)
    reporter.report(simulation, state=None)
    reporter.__del__()

    for bead in (0, 1):
        contents = (tmp_path / f"beads_bead_{bead}.pdb").read_text()
        assert contents.count("MODEL") == 1
        assert contents.rstrip().endswith("END")
    assert integrator.calls == [
        (0, {"getPositions": True, "enforcePeriodicBox": True}),
        (1, {"getPositions": True, "enforcePeriodicBox": True}),
    ]


def test_centroid_reporter_delegates_centroid_calculation(monkeypatch, tmp_path):
    output = tmp_path / "centroid.pdb"
    calls = []
    monkeypatch.setattr(
        reporters,
        "centroid_positions",
        lambda simulation, n_atoms, n_beads: (
            calls.append((simulation, n_atoms, n_beads))
            or [Vec3(0.5, 0.0, 0.0)] * unit.nanometer
        ),
    )
    reporter = RPMDCentroidReporter(
        file_name=output,
        reportInterval=10,
        num_beads=2,
        topology=_topology(),
    )
    simulation = SimpleNamespace(currentStep=10)

    reporter.report(simulation, state=None)
    reporter.__del__()

    assert calls == [(simulation, 1, 2)]
    assert output.read_text().count("MODEL") == 1
