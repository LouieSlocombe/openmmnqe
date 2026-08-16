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
    _calculate_bead_expansion,
    _calculate_quantum_spread,
    _read_expansion_log,
    plot_rpmd_atom_expansion,
    track_rpmd_atom_expansion,
)
import openmmnqe.reporters as reporters


class _State:
    def __init__(self, positions, box_vectors=None):
        self._positions = np.asarray(positions, dtype=float) * unit.nanometer
        if box_vectors is None:
            box_vectors = np.eye(3)
        self._box_vectors = (
            np.asarray(box_vectors, dtype=float) * unit.nanometer
        )

    def getPositions(self, asNumpy=False):
        return self._positions

    def getPeriodicBoxVectors(self, asNumpy=False):
        return self._box_vectors


class _Integrator:
    def __init__(self, bead_positions, box_vectors=None):
        self._states = [
            _State(positions, box_vectors) for positions in bead_positions
        ]
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


def _two_atom_topology():
    topology = app.Topology()
    residue = topology.addResidue("LIG", topology.addChain())
    topology.addAtom("H", app.Element.getBySymbol("H"), residue)
    topology.addAtom("O", app.Element.getBySymbol("O"), residue)
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


def test_calculate_bead_expansion_is_mean_radius_not_rms_radius():
    integrator = _Integrator(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ]
    )

    expansion = _calculate_bead_expansion(integrator)
    rms = _calculate_quantum_spread(integrator)

    assert expansion.value_in_unit(unit.nanometer) == pytest.approx([4 / 3])
    assert rms.value_in_unit(unit.nanometer) == pytest.approx([np.sqrt(2)])


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


def test_expansion_reporter_writes_aligned_centroid_distances(tmp_path):
    output = tmp_path / "expansion.tsv"
    reporter = RPMDQuantumSpreadReporter(
        output,
        reportInterval=5,
        atom_indices=[0],
        names=["H"],
        metric="mean",
        distance_pairs=[(0, 1)],
        distance_names=["H-O"],
    )
    simulation = SimpleNamespace(
        currentStep=15,
        topology=_two_atom_topology(),
        system=SimpleNamespace(usesPeriodicBoundaryConditions=lambda: False),
        integrator=_Integrator(
            [
                [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ]
        ),
    )

    reporter.report(simulation, state=None)
    reporter._out.close()

    assert output.read_text().splitlines() == [
        "Step\tExpansion_H(nm)\tDistance_H-O(nm)",
        "15\t1.333333\t4.000000",
    ]


@pytest.mark.parametrize(("metric", "prefix"), [("mean", "Expansion"), ("rms", "Rg")])
def test_expansion_and_centroid_distance_use_periodic_minimum_images(
    tmp_path, metric, prefix,
):
    output = tmp_path / "periodic.tsv"
    reporter = RPMDQuantumSpreadReporter(
        output,
        reportInterval=1,
        atom_indices=[0],
        metric=metric,
        distance_pairs=[(0, 1)],
    )
    simulation = SimpleNamespace(
        currentStep=1,
        topology=_two_atom_topology(),
        system=SimpleNamespace(usesPeriodicBoundaryConditions=lambda: True),
        integrator=_Integrator(
            [
                [[0.95, 0.0, 0.0], [0.05, 0.0, 0.0]],
                [[0.05, 0.0, 0.0], [0.15, 0.0, 0.0]],
            ],
            box_vectors=np.eye(3),
        ),
    )

    reporter.report(simulation, state=None)
    reporter._out.close()

    assert output.read_text().splitlines() == [
        f"Step\t{prefix}_Atom0(nm)\tDistance_Atom0-Atom1(nm)",
        "1\t0.050000\t0.100000",
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


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"metric": "median"}, ValueError, "metric"),
        ({"distance_pairs": [(0,)]}, ValueError, "exactly two"),
        ({"distance_pairs": [(0, 1.5)]}, TypeError, "must be integers"),
        ({"distance_pairs": [(-1, 1)]}, ValueError, "non-negative"),
        (
            {"distance_pairs": [(0, 1)], "distance_names": []},
            ValueError,
            "one entry per distance pair",
        ),
    ],
)
def test_expansion_reporter_validates_metric_and_distances(
    tmp_path, kwargs, error, message,
):
    output = tmp_path / "spread.tsv"

    with pytest.raises(error, match=message):
        RPMDQuantumSpreadReporter(output, 1, [0], **kwargs)

    assert not output.exists()


@pytest.mark.parametrize(
    ("atom_indices", "error", "message"),
    [
        ([], ValueError, "must not be empty"),
        ([True], TypeError, "must be integers"),
        ([1.5], TypeError, "must be integers"),
        ([-1], ValueError, "non-negative"),
    ],
)
def test_expansion_reporter_validates_atom_indices_before_opening(
    tmp_path, atom_indices, error, message,
):
    output = tmp_path / "spread.tsv"

    with pytest.raises(error, match=message):
        RPMDQuantumSpreadReporter(output, 1, atom_indices)

    assert not output.exists()


def test_expansion_reporter_rejects_duplicate_output_columns(tmp_path):
    output = tmp_path / "spread.tsv"

    with pytest.raises(ValueError, match="must be unique"):
        RPMDQuantumSpreadReporter(output, 1, [0, 1], names=["H", "H"])

    assert not output.exists()


def test_track_expansion_validates_indices_against_topology(tmp_path):
    output = tmp_path / "spread.tsv"
    simulation = SimpleNamespace(
        topology=SimpleNamespace(getNumAtoms=lambda: 1),
        reporters=[],
    )

    with pytest.raises(ValueError, match="outside topology"):
        track_rpmd_atom_expansion(
            simulation,
            atom_index=0,
            file=output,
            report_interval=1,
            distance_pairs=[(0, 1)],
        )

    assert not output.exists()


def test_direct_reporter_gives_clear_error_for_late_topology_mismatch(tmp_path):
    reporter = RPMDQuantumSpreadReporter(tmp_path / "spread.tsv", 1, [1])
    simulation = SimpleNamespace(
        currentStep=1,
        topology=SimpleNamespace(getNumAtoms=lambda: 1),
        integrator=_Integrator([[[0.0, 0.0, 0.0]]]),
    )

    with pytest.raises(ValueError, match="outside topology"):
        reporter.report(simulation, state=None)
    reporter._out.close()


def _write_plot_log(path):
    path.write_text(
        "Step\tExpansion_H(nm)\tDistance_D-H(nm)\tDistance_A-H(nm)\n"
        "0\t0.010\t0.100\t0.300\n"
        "1\t0.020\t0.200\t0.200\n"
        "2\t0.015\t0.300\t0.100\n"
    )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("Time\tExpansion_H(nm)\n0\t0.1\n", "start with a Step"),
        (
            "Step\tExpansion_H(nm)\tExpansion_H(nm)\n0\t0.1\t0.2\n",
            "duplicate column",
        ),
        ("Step\tExpansion_H(nm)\n0\tbad\n", "could not parse"),
        ("Step\tExpansion_H(nm)\n", "no data rows"),
        ("Step\tExpansion_H(nm)\n0\t0.1\t0.2\n", "do not match"),
    ],
)
def test_read_expansion_log_rejects_malformed_input(
    tmp_path, contents, message,
):
    log = tmp_path / "bad.tsv"
    log.write_text(contents)

    with pytest.raises(ValueError, match=message):
        _read_expansion_log(log)


def test_plot_rpmd_atom_expansion_against_distance(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    log = tmp_path / "expansion.tsv"
    image = tmp_path / "distance.png"
    _write_plot_log(log)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    figure, axes = plot_rpmd_atom_expansion(
        log,
        expansion_columns=["Expansion_H(nm)"],
        distance_columns="Distance_D-H(nm)",
        length_unit="angstrom",
        filename=image,
        show=True,
    )

    assert len(axes) == 1
    offsets = np.asarray(axes[0].collections[0].get_offsets())
    assert offsets[:, 0] == pytest.approx([1.0, 2.0, 3.0])
    assert offsets[:, 1] == pytest.approx([0.1, 0.2, 0.15])
    assert "Bead expansion" in axes[0].get_ylabel()
    assert image.stat().st_size > 0
    assert shown == [True]
    plt.close(figure)


def test_plot_rpmd_atom_expansion_along_path_progress(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    log = tmp_path / "expansion.tsv"
    _write_plot_log(log)

    figure, axes = plot_rpmd_atom_expansion(
        log,
        path_progress=[0.0, 0.5, 1.0],
        length_unit="angstrom",
    )

    assert len(axes) == 2
    assert axes[0].lines[0].get_xdata() == pytest.approx([0.0, 0.5, 1.0])
    assert len(axes[1].lines) == 2
    assert axes[1].get_xlabel() == "Path progress (unitless)"
    plt.close(figure)


def test_plot_rpmd_atom_expansion_averages_and_sorts_path_samples(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    log = tmp_path / "expansion.tsv"
    _write_plot_log(log)

    figure, axes = plot_rpmd_atom_expansion(
        log,
        path_progress=[1.0, 0.0, 0.0],
        progress_bins=2,
        length_unit="angstrom",
    )

    assert axes[0].lines[0].get_xdata() == pytest.approx([0.0, 1.0])
    assert axes[0].lines[0].get_ydata() == pytest.approx([0.175, 0.1])
    plt.close(figure)


def test_plot_rpmd_atom_expansion_handles_constant_binned_progress(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    log = tmp_path / "expansion.tsv"
    _write_plot_log(log)

    figure, axes = plot_rpmd_atom_expansion(
        log,
        path_progress=[0.5, 0.5, 0.5],
        progress_bins=4,
    )

    assert axes[0].lines[0].get_xdata() == pytest.approx([0.5])
    assert axes[0].lines[0].get_ydata() == pytest.approx([0.015])
    plt.close(figure)


def test_plot_rpmd_atom_expansion_supports_progress_without_distances(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    log = tmp_path / "expansion.tsv"
    log.write_text(
        "Step\tExpansion_H(nm)\n"
        "0\t0.010\n"
        "1\t0.020\n"
    )

    figure, axes = plot_rpmd_atom_expansion(
        log,
        path_progress=[0.0, 1.0],
    )

    assert len(axes) == 1
    assert axes[0].get_xlabel() == "Path progress (unitless)"
    plt.close(figure)


def test_plot_rpmd_atom_expansion_validates_coordinate_selection(tmp_path):
    pytest.importorskip("matplotlib")
    log = tmp_path / "expansion.tsv"
    _write_plot_log(log)

    with pytest.raises(ValueError, match="exactly one distance"):
        plot_rpmd_atom_expansion(log)
    with pytest.raises(ValueError, match="one value per"):
        plot_rpmd_atom_expansion(log, path_progress=[0.0, 1.0])
    with pytest.raises(ValueError, match="unknown expansion"):
        plot_rpmd_atom_expansion(
            log,
            expansion_columns="Expansion_missing(nm)",
            distance_columns="Distance_D-H(nm)",
        )
    with pytest.raises(ValueError, match="unknown distance"):
        plot_rpmd_atom_expansion(
            log,
            distance_columns="Distance_missing(nm)",
        )
    with pytest.raises(ValueError, match="length_unit"):
        plot_rpmd_atom_expansion(
            log,
            distance_columns="Distance_D-H(nm)",
            length_unit="bohr",
        )
    with pytest.raises(ValueError, match="positive integer"):
        plot_rpmd_atom_expansion(
            log,
            path_progress=[0.0, 0.5, 1.0],
            progress_bins=0,
        )

    no_expansion = tmp_path / "no_expansion.tsv"
    no_expansion.write_text(
        "Step\tDistance_D-H(nm)\n"
        "0\t0.1\n"
    )
    with pytest.raises(ValueError, match="no expansion columns"):
        plot_rpmd_atom_expansion(no_expansion, path_progress=[0.0])


@pytest.mark.parametrize(
    ("contents", "path_progress", "message"),
    [
        (
            "Step\tExpansion_H(nm)\tDistance_D-H(nm)\n"
            "0\tnan\t0.1\n",
            None,
            "log values must be finite",
        ),
        (
            "Step\tExpansion_H(nm)\n0\t0.1\n1\t0.2\n",
            [0.0, np.inf],
            "progress values must be finite",
        ),
    ],
)
def test_plot_rpmd_atom_expansion_rejects_nonfinite_values(
    tmp_path, contents, path_progress, message,
):
    pytest.importorskip("matplotlib")
    log = tmp_path / "nonfinite.tsv"
    log.write_text(contents)

    kwargs = {"path_progress": path_progress}
    if path_progress is None:
        kwargs["distance_columns"] = "Distance_D-H(nm)"
    with pytest.raises(ValueError, match=message):
        plot_rpmd_atom_expansion(log, **kwargs)


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
