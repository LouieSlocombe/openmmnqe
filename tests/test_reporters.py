"""Unit tests for RPMD reporter calculations and output protocols."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm
from scipy import constants

import openmmnqe.reporters as reporters
from openmmnqe.reporters import (
    RPMDBeadReporter,
    RPMDCentroidReporter,
    RPMDQuantumSpreadReporter,
    RPMDThermodynamicReporter,
    _calculate_bead_expansion,
    _calculate_quantum_spread,
    _read_expansion_log,
    _thermodynamic_degrees_of_freedom,
    plot_rpmd_atom_expansion,
    plot_rpmd_thermodynamics,
    rpmd_energy_conservation,
    rpmd_thermodynamic_averages,
    rpmd_thermodynamics,
    track_rpmd_atom_expansion,
)


class _State:
    def __init__(self, positions: Sequence[Any], box_vectors: Any=None) -> None:
        self._positions = np.asarray(positions, dtype=float) * unit.nanometer
        if box_vectors is None:
            box_vectors = np.eye(3)
        self._box_vectors = (
            np.asarray(box_vectors, dtype=float) * unit.nanometer
        )

    def getPositions(self, asNumpy: bool=False) -> unit.Quantity:
        return self._positions

    def getPeriodicBoxVectors(self, asNumpy: bool=False) -> unit.Quantity:
        return self._box_vectors


class _Integrator:
    def __init__(self, bead_positions: Sequence[Any], box_vectors: Any=None) -> None:
        self._states = [
            _State(positions, box_vectors) for positions in bead_positions
        ]
        self.calls = []

    def getNumCopies(self) -> int:
        return len(self._states)

    def getState(self, copy: int | None=None, **kwargs: Any) -> _State:
        self.calls.append((copy, kwargs))
        return self._states[copy]


def _topology() -> app.Topology:
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    return topology


def _two_atom_topology() -> app.Topology:
    topology = app.Topology()
    residue = topology.addResidue("LIG", topology.addChain())
    topology.addAtom("H", app.Element.getBySymbol("H"), residue)
    topology.addAtom("O", app.Element.getBySymbol("O"), residue)
    return topology


def test_calculate_quantum_spread_returns_per_atom_rms_radius() -> None:
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


def test_calculate_bead_expansion_is_mean_radius_not_rms_radius() -> None:
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


def test_quantum_spread_reporter_writes_header_and_values(tmp_path: Path) -> None:
    output = tmp_path / "spread.tsv"
    simulation = SimpleNamespace(
        currentStep=7,
        integrator=_Integrator(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            ]
        ),
    )

    with RPMDQuantumSpreadReporter(
        output,
        reportInterval=5,
        atom_indices=[0, 1],
        names=["H", "O"],
    ) as reporter:
        assert reporter.describeNextReport(simulation) == (
            3,
            False,
            False,
            False,
            False,
        )
        reporter.report(simulation, state=None)

    reporter.close()
    reporter.__del__()
    assert reporter._out.closed

    assert output.read_text().splitlines() == [
        "Step\tRg_H(nm)\tRg_O(nm)",
        "7\t1.000000\t2.000000",
    ]


def test_expansion_reporter_writes_aligned_centroid_distances(tmp_path: Path) -> None:
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
    reporter.close()

    assert output.read_text().splitlines() == [
        "Step\tExpansion_H(nm)\tDistance_H-O(nm)",
        "15\t1.333333\t4.000000",
    ]


@pytest.mark.parametrize(("metric", "prefix"), [("mean", "Expansion"), ("rms", "Rg")])
def test_expansion_and_centroid_distance_use_periodic_minimum_images(
    tmp_path: Path, metric: str, prefix: str,
) -> None:
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
    reporter.close()

    assert output.read_text().splitlines() == [
        f"Step\t{prefix}_Atom0(nm)\tDistance_Atom0-Atom1(nm)",
        "1\t0.050000\t0.100000",
    ]


def test_track_rpmd_atom_expansion_attaches_single_atom_reporter(tmp_path: Path) -> None:
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
    reporter.close()
    assert output.read_text().splitlines() == [
        "Step\tRg_target(nm)",
        "12\t1.000000",
    ]


@pytest.mark.parametrize("atom_index", [-1, 1.5, True])
def test_track_rpmd_atom_expansion_rejects_invalid_atom_index(
    tmp_path: Path, atom_index: Any,
) -> None:
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
def test_reporters_reject_nonpositive_intervals_or_bead_counts(tmp_path: Path, factory: Callable[[Path], Any]) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        factory(tmp_path / "output")


@pytest.mark.parametrize("invalid", [True, 1.5])
@pytest.mark.parametrize(
    "factory",
    [
        lambda path, value: RPMDQuantumSpreadReporter(path, value, [0]),
        lambda path, value: RPMDBeadReporter(path, value, 1, _topology()),
        lambda path, value: RPMDBeadReporter(path, 1, value, _topology()),
        lambda path, value: RPMDCentroidReporter(path, value, 1, _topology()),
        lambda path, value: RPMDCentroidReporter(path, 1, value, _topology()),
    ],
)
def test_reporters_reject_noninteger_intervals_or_bead_counts(
    tmp_path: Path,
    factory: Callable[[Path, Any], Any],
    invalid: Any,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        factory(tmp_path / "output", invalid)

    assert not list(tmp_path.iterdir())


def test_quantum_spread_reporter_validates_names(tmp_path: Path) -> None:
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
    tmp_path: Path, kwargs: dict[str, Any], error: type[Exception], message: str,
) -> None:
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
    tmp_path: Path, atom_indices: list[Any], error: type[Exception], message: str,
) -> None:
    output = tmp_path / "spread.tsv"

    with pytest.raises(error, match=message):
        RPMDQuantumSpreadReporter(output, 1, atom_indices)

    assert not output.exists()


def test_expansion_reporter_rejects_duplicate_output_columns(tmp_path: Path) -> None:
    output = tmp_path / "spread.tsv"

    with pytest.raises(ValueError, match="must be unique"):
        RPMDQuantumSpreadReporter(output, 1, [0, 1], names=["H", "H"])

    assert not output.exists()


def test_track_expansion_validates_indices_against_topology(tmp_path: Path) -> None:
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


def test_direct_reporter_gives_clear_error_for_late_topology_mismatch(tmp_path: Path) -> None:
    reporter = RPMDQuantumSpreadReporter(tmp_path / "spread.tsv", 1, [1])
    simulation = SimpleNamespace(
        currentStep=1,
        topology=SimpleNamespace(getNumAtoms=lambda: 1),
        integrator=_Integrator([[[0.0, 0.0, 0.0]]]),
    )

    with pytest.raises(ValueError, match="outside topology"):
        reporter.report(simulation, state=None)
    reporter.close()


def _write_plot_log(path: Path) -> None:
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
    tmp_path: Path, contents: str, message: str,
) -> None:
    log = tmp_path / "bad.tsv"
    log.write_text(contents)

    with pytest.raises(ValueError, match=message):
        _read_expansion_log(log)


def test_plot_rpmd_atom_expansion_against_distance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_plot_rpmd_atom_expansion_along_path_progress(tmp_path: Path) -> None:
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


def test_plot_rpmd_atom_expansion_averages_and_sorts_path_samples(tmp_path: Path) -> None:
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


def test_plot_rpmd_atom_expansion_handles_constant_binned_progress(tmp_path: Path) -> None:
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


def test_plot_rpmd_atom_expansion_supports_progress_without_distances(tmp_path: Path) -> None:
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


def test_plot_rpmd_atom_expansion_validates_coordinate_selection(tmp_path: Path) -> None:
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
    tmp_path: Path, contents: str, path_progress: list[float] | None, message: str,
) -> None:
    pytest.importorskip("matplotlib")
    log = tmp_path / "nonfinite.tsv"
    log.write_text(contents)

    kwargs = {"path_progress": path_progress}
    if path_progress is None:
        kwargs["distance_columns"] = "Distance_D-H(nm)"
    with pytest.raises(ValueError, match=message):
        plot_rpmd_atom_expansion(log, **kwargs)


def test_bead_reporter_writes_consecutive_models_and_one_footer(
    tmp_path: Path,
) -> None:
    base = tmp_path / "beads"
    integrator = _Integrator([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    simulation = SimpleNamespace(currentStep=5, integrator=integrator)

    with RPMDBeadReporter(
        file_base_name=str(base),
        reportInterval=4,
        num_beads=2,
        topology=_topology(),
    ) as reporter:
        assert reporter.describeNextReport(simulation) == (
            3,
            False,
            False,
            False,
            False,
        )
        reporter.report(simulation, state=None)
        reporter.report(simulation, state=None)

    reporter.close()
    reporter.__del__()

    for bead in (0, 1):
        contents = (tmp_path / f"beads_bead_{bead}.pdb").read_text()
        assert [
            line for line in contents.splitlines() if line.startswith("MODEL")
        ] == ["MODEL        1", "MODEL        2"]
        assert contents.splitlines().count("END") == 1
    assert integrator.calls == [
        (0, {"getPositions": True, "enforcePeriodicBox": True}),
        (1, {"getPositions": True, "enforcePeriodicBox": True}),
        (0, {"getPositions": True, "enforcePeriodicBox": True}),
        (1, {"getPositions": True, "enforcePeriodicBox": True}),
    ]


def test_centroid_reporter_writes_consecutive_models_and_one_footer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    simulation = SimpleNamespace(currentStep=10)

    with RPMDCentroidReporter(
        file_name=output,
        reportInterval=10,
        num_beads=2,
        topology=_topology(),
    ) as reporter:
        reporter.report(simulation, state=None)
        reporter.report(simulation, state=None)

    reporter.close()
    reporter.__del__()

    assert calls == [(simulation, 1, 2), (simulation, 1, 2)]
    contents = output.read_text()
    assert [
        line for line in contents.splitlines() if line.startswith("MODEL")
    ] == ["MODEL        1", "MODEL        2"]
    assert contents.splitlines().count("END") == 1


# ---------------------------------------------------------------------------
# Thermodynamic estimators
# ---------------------------------------------------------------------------

_BOLTZMANN = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin
)

# hbar in the package's MD units, kJ*ps/mol.
_HBAR = constants.hbar * constants.Avogadro * 1.0e-3 * 1.0e12


class _ThermoState:
    """A bead state serving every field the thermodynamic reader asks for."""

    def __init__(self, positions: Any, velocities: Any, forces: Any,
                 potential: float, kinetic: float, time: float = 1.5) -> None:
        self._positions = np.asarray(positions, dtype=float) * unit.nanometer
        self._velocities = (
            np.asarray(velocities, dtype=float)
            * unit.nanometer
            / unit.picosecond
        )
        self._forces = (
            np.asarray(forces, dtype=float)
            * unit.kilojoule_per_mole
            / unit.nanometer
        )
        self._potential = potential * unit.kilojoule_per_mole
        self._kinetic = kinetic * unit.kilojoule_per_mole
        self._time = time * unit.picosecond

    def getPositions(self, asNumpy: bool=False) -> unit.Quantity:
        return self._positions

    def getVelocities(self, asNumpy: bool=False) -> unit.Quantity:
        return self._velocities

    def getForces(self, asNumpy: bool=False) -> unit.Quantity:
        return self._forces

    def getPotentialEnergy(self) -> unit.Quantity:
        return self._potential

    def getKineticEnergy(self) -> unit.Quantity:
        return self._kinetic

    def getTime(self) -> unit.Quantity:
        return self._time


class _ThermoIntegrator:
    """Minimal stand-in for an ``RPMDIntegrator`` holding fixed bead states."""

    def __init__(self, states: Sequence[_ThermoState], total_energy: float,
                 temperature: float = 300.0) -> None:
        self._states = list(states)
        self._total_energy = total_energy * unit.kilojoule_per_mole
        self._temperature = temperature * unit.kelvin
        self.calls: list[dict[str, Any]] = []

    def getNumCopies(self) -> int:
        return len(self._states)

    def getState(self, copy: int | None=None, **kwargs: Any) -> _ThermoState:
        self.calls.append(kwargs)
        assert copy is not None
        return self._states[copy]

    def getTotalEnergy(self) -> unit.Quantity:
        return self._total_energy

    def getTemperature(self) -> unit.Quantity:
        return self._temperature


def _one_particle_system(mass: float = 1.0) -> openmm.System:
    """Build a bare one-particle System with no forces or constraints."""
    system = openmm.System()
    system.addParticle(mass * unit.dalton)
    return system


def _hand_built_simulation() -> SimpleNamespace:
    """
    Two beads of one particle with hand-chosen positions, forces and energies.

    Centroid is at x = 0.1 nm, so the bead displacements are -0.1 and +0.1 nm
    and the virial sum is (-0.1)(1.0) + (0.1)(-3.0) = -0.4 kJ/mol.
    """
    states = [
        _ThermoState(
            positions=[[0.0, 0.0, 0.0]],
            velocities=[[1.0, 0.0, 0.0]],
            forces=[[1.0, 0.0, 0.0]],
            potential=1.0,
            kinetic=5.0,
        ),
        _ThermoState(
            positions=[[0.2, 0.0, 0.0]],
            velocities=[[3.0, 0.0, 0.0]],
            forces=[[-3.0, 0.0, 0.0]],
            potential=3.0,
            kinetic=7.0,
        ),
    ]
    return SimpleNamespace(
        integrator=_ThermoIntegrator(states, total_energy=100.0),
        system=_one_particle_system(),
        currentStep=40,
    )


def test_thermodynamic_degrees_of_freedom_follows_state_data_reporter_rule() -> None:
    system = openmm.System()
    for _ in range(3):
        system.addParticle(1.0 * unit.dalton)
    assert _thermodynamic_degrees_of_freedom(system) == 9

    system.addParticle(0.0 * unit.dalton)
    assert _thermodynamic_degrees_of_freedom(system) == 9

    system.addConstraint(0, 1, 0.1 * unit.nanometer)
    assert _thermodynamic_degrees_of_freedom(system) == 8

    system.addForce(openmm.CMMotionRemover())
    assert _thermodynamic_degrees_of_freedom(system) == 5


def test_thermodynamic_degrees_of_freedom_rejects_a_massless_system() -> None:
    system = openmm.System()
    system.addParticle(0.0 * unit.dalton)

    with pytest.raises(ValueError, match="no positive degrees of freedom"):
        _thermodynamic_degrees_of_freedom(system)


def test_rpmd_thermodynamics_matches_hand_computed_estimators() -> None:
    simulation = _hand_built_simulation()

    values = rpmd_thermodynamics(simulation)
    kilojoules = {
        key: values[key].value_in_unit(unit.kilojoule_per_mole)
        for key in (
            "kinetic_centroid_virial",
            "potential_mean",
            "energy_quantum",
            "potential_sd",
            "energy_ring",
            "energy_spring",
        )
    }
    kt = _BOLTZMANN * 300.0

    # 0.5 * dof * kT - virial / (2 * P), with virial = -0.4 and P = 2.
    expected_kinetic = 0.5 * 3 * kt + 0.1
    assert kilojoules["kinetic_centroid_virial"] == pytest.approx(expected_kinetic)
    assert kilojoules["potential_mean"] == pytest.approx(2.0)
    assert kilojoules["potential_sd"] == pytest.approx(1.0)
    assert kilojoules["energy_quantum"] == pytest.approx(expected_kinetic + 2.0)
    assert kilojoules["energy_ring"] == pytest.approx(100.0)
    # 100 - (1 + 3) - (5 + 7)
    assert kilojoules["energy_spring"] == pytest.approx(84.0)

    # 2 * sum(KE) / (dof * P**2 * k_B)
    assert values["temperature_ring"].value_in_unit(unit.kelvin) == pytest.approx(
        2.0 * 12.0 / (3 * 4 * _BOLTZMANN)
    )
    # Centroid velocity is 2 nm/ps on a 1 Da particle, so KE_centroid = 2 kJ/mol.
    assert values["temperature_centroid"].value_in_unit(unit.kelvin) == pytest.approx(
        2.0 * 2.0 / (3 * _BOLTZMANN)
    )
    assert values["time"].value_in_unit(unit.picosecond) == pytest.approx(1.5)


def test_rpmd_thermodynamics_reads_every_bead_once_without_wrapping() -> None:
    simulation = _hand_built_simulation()

    rpmd_thermodynamics(simulation)

    assert len(simulation.integrator.calls) == 2
    assert all(
        call == {
            "getPositions": True,
            "getVelocities": True,
            "getForces": True,
            "getEnergy": True,
            "enforcePeriodicBox": False,
        }
        for call in simulation.integrator.calls
    )


def test_rpmd_thermodynamics_honours_temperature_and_dof_overrides() -> None:
    simulation = _hand_built_simulation()

    default = rpmd_thermodynamics(simulation)
    overridden = rpmd_thermodynamics(
        simulation,
        temperature=600 * unit.kelvin,
        degrees_of_freedom=6,
    )

    # Only the free-particle term moves: the virial contribution is unchanged.
    difference = (
        overridden["kinetic_centroid_virial"] - default["kinetic_centroid_virial"]
    ).value_in_unit(unit.kilojoule_per_mole)
    assert difference == pytest.approx(0.5 * (6 * 600.0 - 3 * 300.0) * _BOLTZMANN)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
def test_rpmd_thermodynamics_rejects_unphysical_temperatures(temperature: float) -> None:
    simulation = _hand_built_simulation()

    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        rpmd_thermodynamics(simulation, temperature=temperature)


@pytest.mark.parametrize("degrees_of_freedom", [2.0, True, np.bool_(False)])
def test_rpmd_thermodynamics_rejects_non_integer_degrees_of_freedom(
    degrees_of_freedom: Any,
) -> None:
    simulation = _hand_built_simulation()

    with pytest.raises(TypeError, match="degrees_of_freedom must be an integer"):
        rpmd_thermodynamics(simulation, degrees_of_freedom=degrees_of_freedom)


@pytest.mark.parametrize("degrees_of_freedom", [0, -1])
def test_rpmd_thermodynamics_rejects_non_positive_degrees_of_freedom(
    degrees_of_freedom: int,
) -> None:
    simulation = _hand_built_simulation()

    with pytest.raises(ValueError, match="degrees_of_freedom must be a positive"):
        rpmd_thermodynamics(simulation, degrees_of_freedom=degrees_of_freedom)


def _constrained_simulation() -> SimpleNamespace:
    """Two constrained particles, whose virial estimator is not trustworthy."""
    states = [
        _ThermoState(
            positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            velocities=[[0.0, 0.0, 0.0]] * 2,
            forces=[[0.0, 0.0, 0.0]] * 2,
            potential=1.0,
            kinetic=2.0,
        ),
        _ThermoState(
            positions=[[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
            velocities=[[0.0, 0.0, 0.0]] * 2,
            forces=[[0.0, 0.0, 0.0]] * 2,
            potential=1.0,
            kinetic=2.0,
        ),
    ]
    system = openmm.System()
    system.addParticle(1.0 * unit.dalton)
    system.addParticle(1.0 * unit.dalton)
    system.addConstraint(0, 1, 0.1 * unit.nanometer)
    return SimpleNamespace(
        integrator=_ThermoIntegrator(states, total_energy=10.0),
        system=system,
        currentStep=0,
    )


def _rpmd_simulation(*, n_beads: int, mass: float,
                     force_constant: float | None,
                     temperature: float = 300.0,
                     friction: float = 5.0) -> app.Simulation:
    """Build a seeded one-particle RPMD simulation on the Reference platform."""
    system = openmm.System()
    system.addParticle(mass * unit.dalton)
    if force_constant is not None:
        force = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
        force.addGlobalParameter("k", force_constant)
        force.addParticle(0, [])
        system.addForce(force)

    integrator = openmm.RPMDIntegrator(
        n_beads,
        temperature * unit.kelvin,
        friction / unit.picosecond,
        0.0005 * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(1234)
    simulation = app.Simulation(
        _topology(),
        system,
        integrator,
        openmm.Platform.getPlatform("Reference"),
    )
    generator = np.random.default_rng(5)
    for bead in range(n_beads):
        integrator.setPositions(
            bead,
            generator.normal(0.0, 0.01, (1, 3)) * unit.nanometer,
        )
    return simulation


def test_centroid_virial_estimator_is_exact_for_a_free_particle() -> None:
    # With no potential every force vanishes, so the virial term is identically
    # zero and the estimator collapses to the classical free-particle value.
    simulation = _rpmd_simulation(n_beads=6, mass=39.9, force_constant=None)
    simulation.integrator.step(100)

    values = rpmd_thermodynamics(simulation)

    assert values["kinetic_centroid_virial"].value_in_unit(
        unit.kilojoule_per_mole
    ) == pytest.approx(1.5 * _BOLTZMANN * 300.0, rel=1e-12)


def test_reported_spring_energy_matches_openmm_ring_polymer_convention() -> None:
    # Guards the one convention this module leans on: that getTotalEnergy()
    # is the bead energies plus springs of frequency omega_P = P k_B T / hbar.
    n_beads, mass, temperature = 8, 1.008, 300.0
    simulation = _rpmd_simulation(
        n_beads=n_beads,
        mass=mass,
        force_constant=1000.0,
        temperature=temperature,
    )
    simulation.integrator.step(200)

    values = rpmd_thermodynamics(simulation)

    positions = np.asarray([
        simulation.integrator.getState(
            copy=bead,
            getPositions=True,
            enforcePeriodicBox=False,
        ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        for bead in range(n_beads)
    ])
    omega_p = n_beads * _BOLTZMANN * temperature / _HBAR
    displacements = np.roll(positions, -1, axis=0) - positions
    expected = 0.5 * mass * omega_p ** 2 * np.sum(displacements ** 2)

    assert values["energy_spring"].value_in_unit(
        unit.kilojoule_per_mole
    ) == pytest.approx(expected, rel=1e-5)


def test_thermodynamic_estimators_reproduce_the_exact_harmonic_ring_polymer() -> None:
    n_beads, mass, force_constant, temperature = 8, 1.008, 1000.0, 300.0
    # Friction near the well frequency keeps the centroid mode from ringing,
    # which is what limits how fast the mean potential energy converges.
    simulation = _rpmd_simulation(
        n_beads=n_beads,
        mass=mass,
        force_constant=force_constant,
        temperature=temperature,
        friction=30.0,
    )
    simulation.integrator.step(4_000)

    kinetic = []
    potential = []
    ring_temperature = []
    for _ in range(300):
        simulation.integrator.step(20)
        values = rpmd_thermodynamics(simulation)
        kinetic.append(
            values["kinetic_centroid_virial"].value_in_unit(unit.kilojoule_per_mole)
        )
        potential.append(
            values["potential_mean"].value_in_unit(unit.kilojoule_per_mole)
        )
        ring_temperature.append(
            values["temperature_ring"].value_in_unit(unit.kelvin)
        )

    # A P-bead harmonic ring polymer separates into normal modes of frequency
    # sqrt(omega_k**2 + omega**2), each holding P/(2 beta) of energy, which
    # gives closed forms for both estimators.  They are equal by the virial
    # theorem, and tend to the exact quantum result as P grows.
    omega = np.sqrt(force_constant / mass)
    omega_p = n_beads * _BOLTZMANN * temperature / _HBAR
    omega_k = 2.0 * omega_p * np.sin(np.pi * np.arange(n_beads) / n_beads)
    beta = 1.0 / (_BOLTZMANN * temperature)
    exact = 3.0 * (omega ** 2 / (2.0 * beta)) * np.sum(
        1.0 / (omega_k ** 2 + omega ** 2)
    )

    # Above the classical 3kT/2: the zero-point energy is the whole point.
    assert exact > 1.5 * _BOLTZMANN * temperature
    assert np.mean(kinetic) == pytest.approx(exact, rel=0.02)
    assert np.mean(ring_temperature) == pytest.approx(temperature, rel=0.05)

    # The mean bead potential energy is the same quantity in expectation, but
    # it converges far more slowly because the centroid mode dominates its
    # fluctuations -- which is precisely the noise the centroid-virial
    # estimator replaces with an analytic term.  A few picoseconds buys it
    # only about ten percent, so this asserts the scale, not the value; the
    # exact arithmetic is pinned by the hand-built test above.
    assert np.mean(potential) == pytest.approx(exact, rel=0.20)
    assert np.std(kinetic) < 0.1 * np.std(potential)


def test_thermodynamic_reporter_writes_header_and_values(tmp_path: Path) -> None:
    simulation = _hand_built_simulation()
    output = tmp_path / "thermo.log"

    with RPMDThermodynamicReporter(output, 10) as reporter:
        assert reporter.describeNextReport(simulation) == (
            10,
            False,
            False,
            False,
            False,
        )
        reporter.report(simulation, None)

    lines = output.read_text().splitlines()
    assert lines[0].split("\t") == [
        "Step",
        *(column for _, column in reporters._THERMO_COLUMNS),
    ]
    row = lines[1].split("\t")
    assert row[0] == "40"

    expected = rpmd_thermodynamics(simulation)
    for written, (key, _) in zip(row[1:], reporters._THERMO_COLUMNS, strict=True):
        assert float(written) == pytest.approx(
            expected[key].value_in_unit(reporters._THERMO_UNITS[key]),
            abs=1e-6,
        )


def test_thermodynamic_reporter_warns_once_about_constraints(tmp_path: Path) -> None:
    simulation = _constrained_simulation()

    with RPMDThermodynamicReporter(tmp_path / "thermo.log", 10) as reporter:
        with pytest.warns(UserWarning, match="biased by constraints"):
            reporter.report(simulation, None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reporter.report(simulation, None)


def test_thermodynamic_reporter_stays_quiet_without_constraints(tmp_path: Path) -> None:
    simulation = _hand_built_simulation()

    with RPMDThermodynamicReporter(tmp_path / "thermo.log", 10) as reporter:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reporter.report(simulation, None)


@pytest.mark.parametrize("report_interval", [2.0, True, np.bool_(False)])
def test_thermodynamic_reporter_rejects_non_integer_intervals(
    tmp_path: Path,
    report_interval: Any,
) -> None:
    with pytest.raises(TypeError, match="reportInterval must be an integer"):
        RPMDThermodynamicReporter(tmp_path / "thermo.log", report_interval)


@pytest.mark.parametrize("report_interval", [0, -1])
def test_thermodynamic_reporter_rejects_non_positive_intervals(
    tmp_path: Path,
    report_interval: int,
) -> None:
    with pytest.raises(ValueError, match="reportInterval must be a positive"):
        RPMDThermodynamicReporter(tmp_path / "thermo.log", report_interval)


def test_thermodynamic_reporter_validates_overrides_up_front(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        RPMDThermodynamicReporter(tmp_path / "thermo.log", 10, temperature=0.0)
    with pytest.raises(ValueError, match="degrees_of_freedom must be a positive"):
        RPMDThermodynamicReporter(
            tmp_path / "thermo.log",
            10,
            degrees_of_freedom=0,
        )


def test_thermodynamic_reporter_close_is_repeatable(tmp_path: Path) -> None:
    reporter = RPMDThermodynamicReporter(tmp_path / "thermo.log", 10)

    reporter.close()
    reporter.close()

    assert (tmp_path / "thermo.log").exists()


def _write_thermodynamic_log(path: Path, rows: int) -> Path:
    """Write a log whose every column ramps linearly with the row index."""
    header = "Step\t" + "\t".join(
        column for _, column in reporters._THERMO_COLUMNS
    )
    lines = [header]
    for index in range(rows):
        values = "\t".join(
            f"{float(index):.6f}" for _ in reporters._THERMO_COLUMNS
        )
        lines.append(f"{index}\t{values}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_thermodynamic_averages_use_block_standard_errors(tmp_path: Path) -> None:
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=20)

    averages = rpmd_thermodynamic_averages(log, blocks=5)

    assert "Step" not in averages
    assert set(averages) == {
        column for _, column in reporters._THERMO_COLUMNS
    }
    mean, error = averages["E_quantum(kJ/mol)"]
    assert mean == pytest.approx(9.5)
    # Blocks of four rows give means 1.5, 5.5, 9.5, 13.5, 17.5.
    block_means = np.array([1.5, 5.5, 9.5, 13.5, 17.5])
    assert error == pytest.approx(block_means.std(ddof=1) / np.sqrt(5))


def test_thermodynamic_averages_discard_drops_leading_rows(tmp_path: Path) -> None:
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=20)

    averages = rpmd_thermodynamic_averages(log, discard=0.5, blocks=5)

    mean, error = averages["KE_cv(kJ/mol)"]
    assert mean == pytest.approx(14.5)
    block_means = np.array([10.5, 12.5, 14.5, 16.5, 18.5])
    assert error == pytest.approx(block_means.std(ddof=1) / np.sqrt(5))


def test_thermodynamic_averages_drop_the_leading_remainder(tmp_path: Path) -> None:
    # 22 rows into 5 blocks keeps the last 20, so the tail is what is averaged.
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=22)

    mean, _ = rpmd_thermodynamic_averages(log, blocks=5)["T_ring(K)"]

    assert mean == pytest.approx(np.arange(2, 22).mean())


@pytest.mark.parametrize("discard", [-0.1, 1.0, 1.5, float("nan"), "half", True])
def test_thermodynamic_averages_reject_invalid_discards(tmp_path: Path,
                                                        discard: Any) -> None:
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=20)

    with pytest.raises(ValueError, match=r"discard must be a number in \[0, 1\)"):
        rpmd_thermodynamic_averages(log, discard=discard)


@pytest.mark.parametrize("blocks", [2.0, True, np.bool_(False)])
def test_thermodynamic_averages_reject_non_integer_blocks(tmp_path: Path,
                                                          blocks: Any) -> None:
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=20)

    with pytest.raises(TypeError, match="blocks must be an integer"):
        rpmd_thermodynamic_averages(log, blocks=blocks)


@pytest.mark.parametrize("blocks", [0, 1])
def test_thermodynamic_averages_need_at_least_two_blocks(tmp_path: Path,
                                                         blocks: int) -> None:
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=20)

    with pytest.raises(ValueError, match="greater than or equal to 2"):
        rpmd_thermodynamic_averages(log, blocks=blocks)


def test_thermodynamic_averages_need_enough_rows_for_the_blocks(tmp_path: Path) -> None:
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=3)

    with pytest.raises(ValueError, match="too few for 5 blocks"):
        rpmd_thermodynamic_averages(log, blocks=5)


def test_thermodynamic_log_reader_rejects_a_foreign_header(tmp_path: Path) -> None:
    log = tmp_path / "thermo.log"
    log.write_text("Frame\tKE_cv(kJ/mol)\n0\t1.0\n")

    with pytest.raises(ValueError, match="thermodynamic log must start with a Step"):
        rpmd_thermodynamic_averages(log)


def test_plot_rpmd_thermodynamics_stacks_energy_and_temperature(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=8)
    output = tmp_path / "thermo.png"

    figure, axes = plot_rpmd_thermodynamics(log, filename=output)

    assert len(axes) == 2
    energy_axis, temperature_axis = axes
    assert energy_axis.get_ylabel() == "Energy (kJ/mol)"
    assert temperature_axis.get_ylabel() == "Temperature (K)"
    assert temperature_axis.get_xlabel() == "Time (ps)"
    # Legend labels drop the trailing unit that the column name carries.
    labels = [line.get_label() for line in energy_axis.get_lines()]
    assert "E_quantum" in labels
    assert output.exists()
    matplotlib.pyplot.close(figure)


def test_plot_rpmd_thermodynamics_selects_columns_and_units(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=8)

    figure, axes = plot_rpmd_thermodynamics(
        log,
        energy_columns="KE_cv(kJ/mol)",
        temperature_columns=[],
        x_axis="step",
        energy_unit="kilocalorie_per_mole",
    )

    assert len(axes) == 1
    assert axes[0].get_xlabel() == "Step"
    assert axes[0].get_ylabel() == "Energy (kcal/mol)"
    plotted = axes[0].get_lines()[0].get_ydata()
    expected = np.arange(8) * (1.0 * unit.kilojoule_per_mole).value_in_unit(
        unit.kilocalorie_per_mole
    )
    assert np.allclose(plotted, expected)
    matplotlib.pyplot.close(figure)


def test_plot_rpmd_thermodynamics_rejects_bad_options(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    log = _write_thermodynamic_log(tmp_path / "thermo.log", rows=8)

    with pytest.raises(ValueError, match="energy_unit must be one of"):
        plot_rpmd_thermodynamics(log, energy_unit="furlongs")
    with pytest.raises(ValueError, match="x_axis must be one of"):
        plot_rpmd_thermodynamics(log, x_axis="wallclock")
    with pytest.raises(ValueError, match="unknown energy column"):
        plot_rpmd_thermodynamics(log, energy_columns="E_missing(kJ/mol)")


def test_plot_rpmd_thermodynamics_needs_an_energy_column(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    log = tmp_path / "thermo.log"
    log.write_text("Step\tT_ring(K)\n0\t300.0\n1\t301.0\n")

    with pytest.raises(ValueError, match="no energy columns"):
        plot_rpmd_thermodynamics(log)


def test_plot_rpmd_thermodynamics_rejects_non_finite_values(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    log = tmp_path / "thermo.log"
    log.write_text("Step\tE_quantum(kJ/mol)\n0\t1.0\n1\tnan\n")

    with pytest.raises(ValueError, match="must be finite"):
        plot_rpmd_thermodynamics(log)


def test_thermodynamic_reporter_uses_its_overrides_when_reporting(
    tmp_path: Path,
) -> None:
    simulation = _hand_built_simulation()
    output = tmp_path / "thermo.log"

    with RPMDThermodynamicReporter(
        output,
        10,
        temperature=600 * unit.kelvin,
        degrees_of_freedom=6,
    ) as reporter:
        reporter.report(simulation, None)

    written = float(output.read_text().splitlines()[1].split("\t")[2])
    expected = rpmd_thermodynamics(
        simulation,
        temperature=600 * unit.kelvin,
        degrees_of_freedom=6,
    )["kinetic_centroid_virial"].value_in_unit(unit.kilojoule_per_mole)

    assert written == pytest.approx(expected, abs=1e-6)
    # Distinguishable from the integrator's own 300 K and counted dof of 3.
    assert written != pytest.approx(
        rpmd_thermodynamics(simulation)["kinetic_centroid_virial"].value_in_unit(
            unit.kilojoule_per_mole
        )
    )


# ---------------------------------------------------------------------------
# Energy conservation


def _write_energy_log(path: Path, steps: Sequence[int], times: Sequence[float],
                      energies: Sequence[float]) -> Path:
    """Write a thermodynamic log whose only meaningful column is E_ring."""
    header = "Step\t" + "\t".join(
        column for _, column in reporters._THERMO_COLUMNS
    )
    ring_position = [
        column for _, column in reporters._THERMO_COLUMNS
    ].index("E_ring(kJ/mol)")
    lines = [header]
    for step, time, energy in zip(steps, times, energies, strict=True):
        values = [0.0] * len(reporters._THERMO_COLUMNS)
        values[0] = time
        values[ring_position] = energy
        lines.append(
            f"{step}\t" + "\t".join(f"{value:.10g}" for value in values)
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def test_energy_conservation_flags_a_linear_ramp(tmp_path: Path) -> None:
    steps = [index * 10 for index in range(10)]
    times = [step * 1.0e-4 for step in steps]
    energies = [5.0 + 2000.0 * time for time in times]
    log = _write_energy_log(tmp_path / "thermo.log", steps, times, energies)

    verdict = rpmd_energy_conservation(log, temperature=300.0)

    assert verdict.drift_rate == pytest.approx(2000.0)
    assert verdict.drift_per_step == pytest.approx(0.2)
    assert verdict.total_drift == pytest.approx(2000.0 * 0.009)
    # A perfect ramp leaves only rounding-noise residuals, so the ratio is
    # astronomically large and the verdict an unambiguous failure.
    assert verdict.fluctuation == pytest.approx(0.0, abs=1.0e-9)
    assert verdict.drift_ratio > 1.0e6
    assert not verdict.conserved
    kbt = reporters._BOLTZMANN_KJ_PER_MOL_K * 300.0
    assert verdict.drift_per_ps_over_kbt == pytest.approx(2000.0 / kbt)


def test_energy_conservation_accepts_driftless_oscillation(tmp_path: Path) -> None:
    # The +0.5, -0.5, -0.5, +0.5 pattern is orthogonal to a linear trend, so
    # the fitted slope is exactly zero and the RMS residual is exactly 0.5.
    steps = list(range(12))
    times = [step * 1.0e-3 for step in steps]
    pattern = [0.5, -0.5, -0.5, 0.5] * 3
    energies = [10.0 + offset for offset in pattern]
    log = _write_energy_log(tmp_path / "thermo.log", steps, times, energies)

    verdict = rpmd_energy_conservation(log, temperature=300.0)

    assert verdict.drift_rate == pytest.approx(0.0, abs=1.0e-9)
    assert verdict.fluctuation == pytest.approx(0.5)
    assert verdict.drift_ratio == pytest.approx(0.0, abs=1.0e-9)
    assert verdict.conserved


def test_energy_conservation_discard_drops_the_settling_period(tmp_path: Path) -> None:
    steps = list(range(20))
    times = [step * 1.0e-3 for step in steps]
    # First half a steep ramp, second half the driftless pattern.
    energies = [50.0 - 4.0 * index for index in range(10)]
    energies += [10.0 + offset for offset in ([0.5, -0.5, -0.5, 0.5] * 3)[:10]]
    log = _write_energy_log(tmp_path / "thermo.log", steps, times, energies)

    assert not rpmd_energy_conservation(log, temperature=300.0).conserved
    assert rpmd_energy_conservation(
        log, temperature=300.0, discard=0.5,
    ).conserved


def test_energy_conservation_input_validation(tmp_path: Path) -> None:
    steps = [0, 1, 2, 3]
    times = [0.0, 0.001, 0.002, 0.003]
    log = _write_energy_log(
        tmp_path / "thermo.log", steps, times, [1.0, 1.0, 1.0, 1.0],
    )

    with pytest.raises(ValueError, match="temperature"):
        rpmd_energy_conservation(log, temperature=0.0)
    with pytest.raises(ValueError, match="discard"):
        rpmd_energy_conservation(log, temperature=300.0, discard=1.5)
    with pytest.raises(ValueError, match="tolerance"):
        rpmd_energy_conservation(log, temperature=300.0, tolerance=0.0)
    with pytest.raises(ValueError, match="at least 3"):
        rpmd_energy_conservation(log, temperature=300.0, discard=0.6)

    stalled = _write_energy_log(
        tmp_path / "stalled.log", [0, 0, 0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
    )
    with pytest.raises(ValueError, match="advance in time"):
        rpmd_energy_conservation(stalled, temperature=300.0)

    headerless = tmp_path / "short.log"
    headerless.write_text("Step\tTime(ps)\n0\t0.0\n1\t0.1\n2\t0.2\n")
    with pytest.raises(ValueError, match="lacks column"):
        rpmd_energy_conservation(headerless, temperature=300.0)


# ---------------------------------------------------------------------------
# Velocity recording and correlation functions


def _one_particle_nve_simulation(n_beads: int) -> app.Simulation:
    """One argon in a harmonic well under thermostat-off RPMD."""
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    system = openmm.System()
    system.addParticle(39.9 * unit.dalton)
    force = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    force.addGlobalParameter(
        "k", 100.0 * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    force.addParticle(0, [])
    system.addForce(force)
    integrator = openmm.RPMDIntegrator(
        n_beads,
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtoseconds,
    )
    integrator.setApplyThermostat(False)
    simulation = app.Simulation(
        topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    for bead in range(n_beads):
        integrator.setPositions(
            bead, np.array([[0.1, 0.0, 0.0]]) * unit.nanometer,
        )
        integrator.setVelocities(
            bead, np.zeros((1, 3)) * unit.nanometer / unit.picosecond,
        )
    return simulation


def test_velocity_reporter_writes_centroid_frames_on_close(tmp_path: Path) -> None:
    from openmmnqe import step_rpmd

    simulation = _one_particle_nve_simulation(2)
    output = tmp_path / "velocities.npz"

    with reporters.RPMDVelocityReporter(output, 2) as reporter:
        simulation.reporters.append(reporter)
        step_rpmd(simulation, 6)

    with np.load(output) as archive:
        times = archive["times_ps"]
        velocities = archive["velocities_nm_per_ps"]
        assert np.array_equal(archive["atom_indices"], [0])
        assert archive["masses_dalton"] == pytest.approx([39.9])
    assert velocities.shape == (3, 1, 3)
    assert np.isfinite(velocities).all()
    assert np.diff(times) == pytest.approx([0.004, 0.004])


def test_velocity_reporter_validates_its_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reportInterval"):
        reporters.RPMDVelocityReporter(tmp_path / "v.npz", 0)
    with pytest.raises(ValueError, match="must not be empty"):
        reporters.RPMDVelocityReporter(tmp_path / "v.npz", 2, atom_indices=[])
    with pytest.raises(ValueError, match="duplicate"):
        reporters.RPMDVelocityReporter(
            tmp_path / "v.npz", 2, atom_indices=[0, 0],
        )

    simulation = _one_particle_nve_simulation(2)
    out_of_range = reporters.RPMDVelocityReporter(
        tmp_path / "v.npz", 2, atom_indices=[5],
    )
    with pytest.raises(ValueError, match="outside the System"):
        out_of_range.report(simulation, None)

    not_rpmd = SimpleNamespace(
        integrator=SimpleNamespace(), system=simulation.system,
    )
    fresh = reporters.RPMDVelocityReporter(tmp_path / "v2.npz", 2)
    with pytest.raises(TypeError, match="RPMDIntegrator"):
        fresh.report(not_rpmd, None)

    # A reporter that never recorded a frame writes nothing on close.
    silent = reporters.RPMDVelocityReporter(tmp_path / "silent.npz", 2)
    silent.close()
    silent.close()
    assert not (tmp_path / "silent.npz").exists()


def _write_cosine_archive(path: Path, *, n_frames: int, dt: float,
                          angular_frequency: float, mass: float) -> None:
    """Archive holding one atom whose x velocity is a pure cosine."""
    times = np.arange(n_frames) * dt
    velocities = np.zeros((n_frames, 1, 3))
    velocities[:, 0, 0] = np.cos(angular_frequency * times)
    np.savez(
        path,
        times_ps=times,
        velocities_nm_per_ps=velocities,
        atom_indices=np.array([0]),
        masses_dalton=np.array([mass]),
    )


def test_velocity_autocorrelation_of_a_cosine_is_a_cosine(tmp_path: Path) -> None:
    omega, dt, n_frames, mass = 314.159, 0.001, 2000, 12.0
    archive = tmp_path / "cosine.npz"
    _write_cosine_archive(
        archive, n_frames=n_frames, dt=dt, angular_frequency=omega, mass=mass,
    )

    times, vacf = reporters.rpmd_velocity_autocorrelation(
        archive, max_time=0.2,
    )

    assert len(times) == 201
    signal = np.cos(omega * np.arange(n_frames) * dt)
    assert vacf[0] == pytest.approx(mass * np.mean(signal**2))
    # The unbiased estimator of a pure cosine is (mass/2) cos(omega t) up to
    # the truncation cross-term, which shrinks with the record length.
    assert vacf == pytest.approx(
        0.5 * mass * np.cos(omega * times), abs=0.02 * mass,
    )

    _, unweighted = reporters.rpmd_velocity_autocorrelation(
        archive, max_time=0.2, mass_weighted=False,
    )
    assert unweighted == pytest.approx(vacf / mass)


def test_vibrational_spectrum_peaks_at_the_cosine_frequency(tmp_path: Path) -> None:
    omega, dt, n_frames = 314.159, 0.001, 2000
    archive = tmp_path / "cosine.npz"
    _write_cosine_archive(
        archive, n_frames=n_frames, dt=dt, angular_frequency=omega, mass=1.0,
    )

    frequencies, intensities = reporters.rpmd_vibrational_spectrum(archive)

    speed_of_light_cm_per_ps = 1.0e2 * unit.SPEED_OF_LIGHT_C.value_in_unit(
        unit.meter / unit.picosecond
    )
    expected = omega / (2.0 * np.pi) / speed_of_light_cm_per_ps
    peak = frequencies[np.argmax(intensities)]
    grid_spacing = frequencies[1] - frequencies[0]
    assert abs(peak - expected) <= grid_spacing

    with pytest.raises(ValueError, match="window"):
        reporters.rpmd_vibrational_spectrum(archive, window="hamming")


def test_vibrational_spectrum_finds_a_real_harmonic_frequency(tmp_path: Path) -> None:
    # A classical (one-bead) thermostat-off particle released off-centre in
    # the harmonic well oscillates at exactly omega = sqrt(k/m), so the
    # whole chain -- reporter, archive, correlation, transform -- must put
    # the spectral peak there.
    from openmmnqe import step_rpmd

    simulation = _one_particle_nve_simulation(1)
    output = tmp_path / "well.npz"
    reporter = reporters.RPMDVelocityReporter(output, 10)
    simulation.reporters.append(reporter)
    step_rpmd(simulation, 20_000)
    reporter.close()

    frequencies, intensities = reporters.rpmd_vibrational_spectrum(output)

    speed_of_light_cm_per_ps = 1.0e2 * unit.SPEED_OF_LIGHT_C.value_in_unit(
        unit.meter / unit.picosecond
    )
    omega = np.sqrt(100.0 / 39.9)  # rad/ps, since 1 kJ/mol = 1 Da nm^2/ps^2
    expected = omega / (2.0 * np.pi) / speed_of_light_cm_per_ps
    peak = frequencies[np.argmax(intensities)]
    assert peak == pytest.approx(expected, abs=1.0)


def test_velocity_archive_validation(tmp_path: Path) -> None:
    good = tmp_path / "good.npz"
    _write_cosine_archive(
        good, n_frames=10, dt=0.001, angular_frequency=1.0, mass=1.0,
    )
    with pytest.raises(ValueError, match="max_time"):
        reporters.rpmd_velocity_autocorrelation(good, max_time=0.0)

    missing = tmp_path / "missing.npz"
    np.savez(missing, times_ps=np.arange(3.0))
    with pytest.raises(ValueError, match="lacks field"):
        reporters.rpmd_velocity_autocorrelation(missing)

    short = tmp_path / "short.npz"
    np.savez(
        short,
        times_ps=np.array([0.0]),
        velocities_nm_per_ps=np.zeros((1, 1, 3)),
        atom_indices=np.array([0]),
        masses_dalton=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="at least two frames"):
        reporters.rpmd_velocity_autocorrelation(short)

    mismatched = tmp_path / "mismatched.npz"
    np.savez(
        mismatched,
        times_ps=np.arange(3.0) * 0.001,
        velocities_nm_per_ps=np.zeros((3, 2, 3)),
        atom_indices=np.array([0]),
        masses_dalton=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="shapes disagree"):
        reporters.rpmd_velocity_autocorrelation(mismatched)

    uneven = tmp_path / "uneven.npz"
    np.savez(
        uneven,
        times_ps=np.array([0.0, 0.001, 0.005]),
        velocities_nm_per_ps=np.zeros((3, 1, 3)),
        atom_indices=np.array([0]),
        masses_dalton=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        reporters.rpmd_velocity_autocorrelation(uneven)
