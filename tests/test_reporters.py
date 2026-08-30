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
    RPMDKineticDecompositionReporter,
    RPMDQuantumSpreadReporter,
    RPMDThermodynamicReporter,
    _calculate_bead_expansion,
    _calculate_quantum_spread,
    _read_expansion_log,
    _thermodynamic_degrees_of_freedom,
    plot_rpmd_atom_expansion,
    plot_rpmd_kinetic_decomposition,
    plot_rpmd_thermodynamics,
    rpmd_energy_conservation,
    rpmd_kinetic_decomposition,
    rpmd_kinetic_decomposition_averages,
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


def _virtual_site_simulation() -> SimpleNamespace:
    """
    Two massive particles plus the average site they carry, over two beads.

    Particle 2 is a ``TwoParticleAverageSite(0, 1, 0.25, 0.75)``, so its bead
    displacement is the same weighted average of its parents'. With the site's
    own force redistributed onto those parents in the 0.25/0.75 ratio, the
    site row's virial contribution is identically equal to the parents' --
    which is exactly why summing every row would double-count it.

    Displacements about the centroid are (-0.1, -0.3, -0.25) nm for bead 0 and
    the negatives of those for bead 1, so the virial over the massive rows is
    (0.1 + 0.9) + (0.2 + 1.8) = 3.0 kJ/mol, and over every row 6.0 kJ/mol.
    """
    states = [
        _ThermoState(
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.75, 0.0, 0.0]],
            velocities=np.zeros((3, 3)),
            forces=[[-1.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
            potential=1.0,
            kinetic=5.0,
        ),
        _ThermoState(
            positions=[[0.2, 0.0, 0.0], [1.6, 0.0, 0.0], [1.25, 0.0, 0.0]],
            velocities=np.zeros((3, 3)),
            forces=[[2.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            potential=3.0,
            kinetic=7.0,
        ),
    ]
    system = openmm.System()
    system.addParticle(1.0 * unit.dalton)
    system.addParticle(1.0 * unit.dalton)
    system.addParticle(0.0 * unit.dalton)
    system.setVirtualSite(2, openmm.TwoParticleAverageSite(0, 1, 0.25, 0.75))
    return SimpleNamespace(
        integrator=_ThermoIntegrator(states, total_energy=100.0),
        system=system,
        currentStep=40,
    )


def test_virtual_site_forces_are_not_double_counted_in_the_virial() -> None:
    simulation = _virtual_site_simulation()

    values = rpmd_thermodynamics(simulation)
    kinetic = values["kinetic_centroid_virial"].value_in_unit(
        unit.kilojoule_per_mole
    )

    # dof counts the two massive particles only, and the virial over those
    # rows is 3.0 kJ/mol shared between two beads.
    kt = _BOLTZMANN * 300.0
    assert _thermodynamic_degrees_of_freedom(simulation.system) == 6
    assert kinetic == pytest.approx(0.5 * 6 * kt - 0.75)

    # Summing every row instead would land 0.75 kJ/mol lower, because the
    # site's contribution exactly repeats its parents'.
    assert kinetic != pytest.approx(0.5 * 6 * kt - 1.5)


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
# Per-atom kinetic decomposition


def _three_particle_reduced_dof_simulation() -> SimpleNamespace:
    """
    Three particles whose dof count is reduced by a constraint and a CMM.

    All three have mass, so ``3 * n_massive`` is 9 while
    ``_thermodynamic_degrees_of_freedom`` returns ``9 - 1 - 3 = 5``. The
    per-atom virials are -0.2, -0.8 and -2.1 kJ/mol, summing to -3.1.
    """
    states = [
        _ThermoState(
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            velocities=np.zeros((3, 3)),
            forces=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            potential=1.0,
            kinetic=2.0,
        ),
        _ThermoState(
            positions=[[0.2, 0.0, 0.0], [1.4, 0.0, 0.0], [2.6, 0.0, 0.0]],
            velocities=np.zeros((3, 3)),
            forces=[[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
            potential=1.0,
            kinetic=2.0,
        ),
    ]
    system = openmm.System()
    for _ in range(3):
        system.addParticle(1.0 * unit.dalton)
    system.addConstraint(0, 1, 0.1 * unit.nanometer)
    system.addForce(openmm.CMMotionRemover())
    return SimpleNamespace(
        integrator=_ThermoIntegrator(states, total_energy=10.0),
        system=system,
        currentStep=0,
    )


def test_kinetic_decomposition_matches_the_hand_computed_virial() -> None:
    simulation = _hand_built_simulation()

    kinetic = rpmd_kinetic_decomposition(simulation)
    kt = _BOLTZMANN * 300.0

    # One particle, virial -0.4 shared between two beads.
    assert set(kinetic) == {0}
    assert kinetic[0].value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        1.5 * kt + 0.1
    )
    # dof is 3 here, so the decomposition reproduces the system estimator.
    system_total = rpmd_thermodynamics(simulation)["kinetic_centroid_virial"]
    assert kinetic[0].value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        system_total.value_in_unit(unit.kilojoule_per_mole)
    )


def test_kinetic_decomposition_sums_to_the_system_estimator_with_a_dof_offset() -> None:
    simulation = _three_particle_reduced_dof_simulation()
    kt = _BOLTZMANN * 300.0

    kinetic = rpmd_kinetic_decomposition(simulation)
    assert set(kinetic) == {0, 1, 2}
    for index, virial in zip((0, 1, 2), (-0.2, -0.8, -2.1), strict=True):
        assert kinetic[index].value_in_unit(
            unit.kilojoule_per_mole
        ) == pytest.approx(1.5 * kt - 0.25 * virial)

    total = sum(
        value.value_in_unit(unit.kilojoule_per_mole)
        for value in kinetic.values()
    )
    system_total = rpmd_thermodynamics(simulation)[
        "kinetic_centroid_virial"
    ].value_in_unit(unit.kilojoule_per_mole)

    # Three degrees of freedom per atom against the System's five: one
    # constraint and a CMMotionRemover account for the whole difference.
    assert _thermodynamic_degrees_of_freedom(simulation.system) == 5
    assert total - system_total == pytest.approx(
        (9 - 5) * kt / 2
    )


def test_kinetic_decomposition_reads_every_bead_once_without_wrapping() -> None:
    simulation = _hand_built_simulation()

    rpmd_kinetic_decomposition(simulation)

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


def test_kinetic_decomposition_omits_a_virtual_site() -> None:
    simulation = _virtual_site_simulation()

    kinetic = rpmd_kinetic_decomposition(simulation)

    assert set(kinetic) == {0, 1}
    kt = _BOLTZMANN * 300.0
    # Per-atom virials are (0.1 + 0.2) and (0.9 + 1.8) kJ/mol.
    assert kinetic[0].value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        1.5 * kt - 0.25 * 0.3
    )
    assert kinetic[1].value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        1.5 * kt - 0.25 * 2.7
    )
    # Those two are the whole of KE_cv, because dof is 3 * 2 here.
    total = sum(
        value.value_in_unit(unit.kilojoule_per_mole)
        for value in kinetic.values()
    )
    assert total == pytest.approx(
        rpmd_thermodynamics(simulation)[
            "kinetic_centroid_virial"
        ].value_in_unit(unit.kilojoule_per_mole)
    )


def test_kinetic_decomposition_rejects_a_massless_atom() -> None:
    simulation = _virtual_site_simulation()

    with pytest.raises(ValueError, match="atom 2 has zero mass"):
        rpmd_kinetic_decomposition(simulation, [0, 2])


def test_kinetic_decomposition_rejects_an_out_of_range_atom() -> None:
    simulation = _hand_built_simulation()

    with pytest.raises(ValueError, match="outside System with 1 particles"):
        rpmd_kinetic_decomposition(simulation, [0, 5])


def test_kinetic_decomposition_rejects_a_massless_system() -> None:
    simulation = _hand_built_simulation()
    simulation.system = openmm.System()
    simulation.system.addParticle(0.0 * unit.dalton)

    with pytest.raises(ValueError, match="no particles with mass"):
        rpmd_kinetic_decomposition(simulation)


@pytest.mark.parametrize("temperature", [0.0, -5.0, float("nan")])
def test_kinetic_decomposition_rejects_unphysical_temperatures(
    temperature: float,
) -> None:
    simulation = _hand_built_simulation()

    with pytest.raises(ValueError, match="temperature"):
        rpmd_kinetic_decomposition(simulation, temperature=temperature)


def _two_mass_rpmd_simulation(*, n_beads: int, masses: Sequence[float],
                              force_constant: float,
                              temperature: float = 300.0,
                              friction: float = 30.0) -> app.Simulation:
    """Two non-interacting particles of different mass in one harmonic well."""
    system = openmm.System()
    force = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    force.addGlobalParameter("k", force_constant)
    for index, mass in enumerate(masses):
        system.addParticle(mass * unit.dalton)
        force.addParticle(index, [])
    system.addForce(force)

    integrator = openmm.RPMDIntegrator(
        n_beads,
        temperature * unit.kelvin,
        friction / unit.picosecond,
        0.0005 * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(1234)
    simulation = app.Simulation(
        _two_atom_topology(),
        system,
        integrator,
        openmm.Platform.getPlatform("Reference"),
    )
    generator = np.random.default_rng(5)
    for bead in range(n_beads):
        integrator.setPositions(
            bead,
            generator.normal(0.0, 0.01, (len(masses), 3)) * unit.nanometer,
        )
    return simulation


def _exact_harmonic_kinetic(mass: float, force_constant: float, *,
                            n_beads: int, temperature: float) -> float:
    """Exact ring-polymer kinetic energy of one 3D harmonic oscillator."""
    omega = np.sqrt(force_constant / mass)
    omega_p = n_beads * _BOLTZMANN * temperature / _HBAR
    omega_k = 2.0 * omega_p * np.sin(np.pi * np.arange(n_beads) / n_beads)
    beta = 1.0 / (_BOLTZMANN * temperature)
    return float(
        3.0 * (omega ** 2 / (2.0 * beta))
        * np.sum(1.0 / (omega_k ** 2 + omega ** 2))
    )


def test_per_atom_estimator_separates_two_masses_in_one_harmonic_well() -> None:
    # hbar*omega/kT is about 4 for the light particle, so both are clearly
    # quantum and the two are clearly apart.
    n_beads, force_constant, temperature = 8, 25_000.0, 300.0
    masses = (1.008, 2.014)
    simulation = _two_mass_rpmd_simulation(
        n_beads=n_beads,
        masses=masses,
        force_constant=force_constant,
        temperature=temperature,
    )
    simulation.integrator.step(4_000)

    light: list[float] = []
    heavy: list[float] = []
    for _ in range(300):
        simulation.integrator.step(20)
        kinetic = rpmd_kinetic_decomposition(simulation)
        values = [
            kinetic[index].value_in_unit(unit.kilojoule_per_mole)
            for index in (0, 1)
        ]
        # The decomposition is an exact identity against the system estimator
        # here: no constraints, no CMMotionRemover, so dof is 3 * 2.
        assert sum(values) == pytest.approx(
            rpmd_thermodynamics(simulation)[
                "kinetic_centroid_virial"
            ].value_in_unit(unit.kilojoule_per_mole),
            rel=1e-9,
        )
        light.append(values[0])
        heavy.append(values[1])

    exact = [
        _exact_harmonic_kinetic(
            mass, force_constant, n_beads=n_beads, temperature=temperature,
        )
        for mass in masses
    ]
    assert np.mean(light) == pytest.approx(exact[0], rel=0.05)
    assert np.mean(heavy) == pytest.approx(exact[1], rel=0.05)
    # Each atom is nearer its own exact value than the other's, which the two
    # exact values being 28% apart makes a real discrimination.
    assert abs(np.mean(light) - exact[0]) < abs(np.mean(light) - exact[1])
    assert abs(np.mean(heavy) - exact[1]) < abs(np.mean(heavy) - exact[0])

    # The whole point: the lighter atom is the more quantum one, and both sit
    # well above the classical equipartition value -- here the proton carries
    # nearly twice its classical kinetic energy.
    kt = _BOLTZMANN * temperature
    assert np.mean(light) > np.mean(heavy) > 1.5 * kt
    assert np.mean(light) > 1.5 * (1.5 * kt)


def test_kinetic_reporter_writes_header_and_values(tmp_path: Path) -> None:
    simulation = _three_particle_reduced_dof_simulation()
    simulation.currentStep = 40
    output = tmp_path / "kinetic.log"

    with RPMDKineticDecompositionReporter(
        output, 10, [0, 2], names=["H1", "Donor"],
    ) as reporter:
        assert reporter.describeNextReport(simulation) == (
            10,
            False,
            False,
            False,
            False,
        )
        # This fixture is constrained, so the estimator warns before writing.
        with pytest.warns(UserWarning, match="biased by constraints"):
            reporter.report(simulation, None)

    lines = output.read_text().splitlines()
    assert lines[0].split("\t") == [
        "Step",
        "Time(ps)",
        "Kcv_H1(kJ/mol)",
        "Kcv_Donor(kJ/mol)",
    ]
    row = lines[1].split("\t")
    assert row[0] == "40"
    assert float(row[1]) == pytest.approx(1.5)

    expected = rpmd_kinetic_decomposition(simulation)
    for written, index in zip(row[2:], (0, 2), strict=True):
        assert float(written) == pytest.approx(
            expected[index].value_in_unit(unit.kilojoule_per_mole),
            abs=1e-6,
        )


def test_kinetic_reporter_defaults_column_names_to_atom_indices(
    tmp_path: Path,
) -> None:
    output = tmp_path / "kinetic.log"

    with RPMDKineticDecompositionReporter(output, 10, [0, 2]):
        pass

    assert output.read_text().splitlines()[0].split("\t")[2:] == [
        "Kcv_Atom0(kJ/mol)",
        "Kcv_Atom2(kJ/mol)",
    ]


def test_kinetic_reporter_validates_its_inputs(tmp_path: Path) -> None:
    output = tmp_path / "kinetic.log"

    with pytest.raises(ValueError, match="atom_indices must not be empty"):
        RPMDKineticDecompositionReporter(output, 10, [])
    with pytest.raises(TypeError, match="atom_indices must be integers"):
        RPMDKineticDecompositionReporter(output, 10, [0.5])
    with pytest.raises(ValueError, match="names must contain one entry"):
        RPMDKineticDecompositionReporter(output, 10, [0, 1], names=["only"])
    with pytest.raises(ValueError, match="column names must be unique"):
        RPMDKineticDecompositionReporter(
            output, 10, [0, 1], names=["same", "same"],
        )
    with pytest.raises(ValueError, match="temperature"):
        RPMDKineticDecompositionReporter(output, 10, [0], temperature=-1.0)
    assert not output.exists()


@pytest.mark.parametrize("interval", [2.0, True, np.bool_(False)])
def test_kinetic_reporter_rejects_non_integer_intervals(
    tmp_path: Path, interval: Any,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        RPMDKineticDecompositionReporter(tmp_path / "kinetic.log", interval, [0])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("interval", [0, -1])
def test_kinetic_reporter_rejects_non_positive_intervals(
    tmp_path: Path, interval: int,
) -> None:
    with pytest.raises(ValueError, match="must be a positive"):
        RPMDKineticDecompositionReporter(tmp_path / "kinetic.log", interval, [0])
    assert not list(tmp_path.iterdir())


def test_kinetic_reporter_warns_once_about_constraints(tmp_path: Path) -> None:
    simulation = _constrained_simulation()

    with RPMDKineticDecompositionReporter(
        tmp_path / "kinetic.log", 10, [0],
    ) as reporter:
        with pytest.warns(UserWarning, match="biased by constraints"):
            reporter.report(simulation, None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reporter.report(simulation, None)


def test_kinetic_reporter_rejects_a_massless_selection_at_first_report(
    tmp_path: Path,
) -> None:
    simulation = _virtual_site_simulation()

    with RPMDKineticDecompositionReporter(
        tmp_path / "kinetic.log", 10, [2],
    ) as reporter:
        with pytest.raises(ValueError, match="atom 2 has zero mass"):
            reporter.report(simulation, None)


def test_kinetic_reporter_follows_the_integrator_temperature(
    tmp_path: Path,
) -> None:
    simulation = _hand_built_simulation()
    output = tmp_path / "kinetic.log"

    with RPMDKineticDecompositionReporter(output, 10, [0]) as reporter:
        reporter.report(simulation, None)
        simulation.integrator._temperature = 600.0 * unit.kelvin
        reporter.report(simulation, None)

    rows = [
        line.split("\t") for line in output.read_text().splitlines()[1:]
    ]
    # Only the free-particle term moves; the virial contribution is fixed.
    assert float(rows[1][2]) - float(rows[0][2]) == pytest.approx(
        1.5 * _BOLTZMANN * 300.0, abs=1e-5
    )


def test_kinetic_reporter_uses_a_fixed_temperature_when_given(
    tmp_path: Path,
) -> None:
    simulation = _hand_built_simulation()
    output = tmp_path / "kinetic.log"

    with RPMDKineticDecompositionReporter(
        output, 10, [0], temperature=600 * unit.kelvin,
    ) as reporter:
        reporter.report(simulation, None)

    written = float(output.read_text().splitlines()[1].split("\t")[2])
    assert written == pytest.approx(1.5 * _BOLTZMANN * 600.0 + 0.1, abs=1e-6)


def test_kinetic_reporter_close_is_repeatable(tmp_path: Path) -> None:
    reporter = RPMDKineticDecompositionReporter(
        tmp_path / "kinetic.log", 10, [0],
    )

    reporter.close()
    reporter.close()


def _write_kinetic_log(path: Path, rows: int,
                       names: Sequence[str] = ("H1", "Donor")) -> Path:
    """Write a kinetic log whose every column ramps with the row index."""
    columns = ["Time(ps)", *(f"Kcv_{name}(kJ/mol)" for name in names)]
    lines = ["Step\t" + "\t".join(columns)]
    for index in range(rows):
        values = "\t".join(f"{float(index):.6f}" for _ in columns)
        lines.append(f"{index}\t{values}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_kinetic_averages_use_block_standard_errors(tmp_path: Path) -> None:
    log = _write_kinetic_log(tmp_path / "kinetic.log", rows=20)

    averages = rpmd_kinetic_decomposition_averages(log, blocks=5)

    assert set(averages) == {"Time(ps)", "Kcv_H1(kJ/mol)", "Kcv_Donor(kJ/mol)"}
    mean, error = averages["Kcv_H1(kJ/mol)"]
    assert mean == pytest.approx(9.5)
    block_means = np.array([1.5, 5.5, 9.5, 13.5, 17.5])
    assert error == pytest.approx(block_means.std(ddof=1) / np.sqrt(5))


def test_kinetic_averages_name_their_own_log_in_errors(tmp_path: Path) -> None:
    log = _write_kinetic_log(tmp_path / "kinetic.log", rows=3)

    with pytest.raises(
        ValueError, match="kinetic decomposition log has 3 rows",
    ):
        rpmd_kinetic_decomposition_averages(log, blocks=5)


def test_kinetic_columns_do_not_pollute_thermodynamic_column_selection(
    tmp_path: Path,
) -> None:
    log = _write_kinetic_log(tmp_path / "kinetic.log", rows=8)
    header = log.read_text().splitlines()[0].split("\t")

    # The Kcv_ prefix is chosen so a kinetic column can never be mistaken for
    # a thermodynamic energy trace.
    assert reporters._select_log_columns(
        header, None, ("KE_", "PE_", "E_"), "energy",
    ) == []
    assert reporters._select_log_columns(
        header, None, ("Kcv_",), "kinetic",
    ) == ["Kcv_H1(kJ/mol)", "Kcv_Donor(kJ/mol)"]

    thermo = _write_thermodynamic_log(tmp_path / "thermo.log", rows=8)
    assert set(rpmd_thermodynamic_averages(thermo)) == {
        column for _, column in reporters._THERMO_COLUMNS
    }


def test_plot_kinetic_decomposition_draws_a_classical_reference(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    log = _write_kinetic_log(tmp_path / "kinetic.log", rows=8)
    output = tmp_path / "kinetic.png"

    figure, axes = plot_rpmd_kinetic_decomposition(
        log, temperature=300.0, filename=output,
    )

    assert len(axes) == 1
    axis = axes[0]
    assert axis.get_ylabel() == "Kinetic energy (kJ/mol)"
    assert axis.get_xlabel() == "Time (ps)"
    labels = [line.get_label() for line in axis.get_lines()]
    assert labels[:2] == ["H1", "Donor"]
    assert "classical, 3kT/2" in labels[2]
    reference = axis.get_lines()[2].get_ydata()
    assert reference[0] == pytest.approx(1.5 * _BOLTZMANN * 300.0)
    assert output.exists()
    matplotlib.pyplot.close(figure)


def test_plot_kinetic_decomposition_selects_columns_and_units(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    log = _write_kinetic_log(tmp_path / "kinetic.log", rows=8)

    figure, axes = plot_rpmd_kinetic_decomposition(
        log,
        columns="Kcv_H1(kJ/mol)",
        x_axis="step",
        energy_unit="kilocalorie_per_mole",
    )

    assert axes[0].get_xlabel() == "Step"
    assert axes[0].get_ylabel() == "Kinetic energy (kcal/mol)"
    assert len(axes[0].get_lines()) == 1
    plotted = axes[0].get_lines()[0].get_ydata()
    expected = np.arange(8) * (1.0 * unit.kilojoule_per_mole).value_in_unit(
        unit.kilocalorie_per_mole
    )
    assert np.allclose(plotted, expected)
    matplotlib.pyplot.close(figure)


def test_plot_kinetic_decomposition_rejects_bad_options(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    log = _write_kinetic_log(tmp_path / "kinetic.log", rows=8)

    with pytest.raises(ValueError, match="energy_unit must be one of"):
        plot_rpmd_kinetic_decomposition(log, energy_unit="furlongs")
    with pytest.raises(ValueError, match="x_axis must be one of"):
        plot_rpmd_kinetic_decomposition(log, x_axis="wallclock")
    with pytest.raises(ValueError, match="unknown kinetic column"):
        plot_rpmd_kinetic_decomposition(log, columns="Kcv_missing(kJ/mol)")
    with pytest.raises(ValueError, match="temperature"):
        plot_rpmd_kinetic_decomposition(log, temperature=-1.0)


def test_plot_kinetic_decomposition_needs_a_kinetic_column(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    log = tmp_path / "kinetic.log"
    log.write_text("Step\tTime(ps)\n0\t0.0\n1\t0.1\n")

    with pytest.raises(ValueError, match="no Kcv_ columns"):
        plot_rpmd_kinetic_decomposition(log)


def test_plot_kinetic_decomposition_rejects_non_finite_values(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    log = tmp_path / "kinetic.log"
    log.write_text("Step\tTime(ps)\tKcv_H1(kJ/mol)\n0\t0.0\t1.0\n1\t0.1\tnan\n")

    with pytest.raises(ValueError, match="must be finite"):
        plot_rpmd_kinetic_decomposition(log)


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

    times, vacf = reporters.velocity_autocorrelation(
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

    _, unweighted = reporters.velocity_autocorrelation(
        archive, max_time=0.2, mass_weighted=False,
    )
    assert unweighted == pytest.approx(vacf / mass)


def test_vibrational_spectrum_peaks_at_the_cosine_frequency(tmp_path: Path) -> None:
    omega, dt, n_frames = 314.159, 0.001, 2000
    archive = tmp_path / "cosine.npz"
    _write_cosine_archive(
        archive, n_frames=n_frames, dt=dt, angular_frequency=omega, mass=1.0,
    )

    frequencies, intensities = reporters.vibrational_spectrum(archive)

    speed_of_light_cm_per_ps = 1.0e2 * unit.SPEED_OF_LIGHT_C.value_in_unit(
        unit.meter / unit.picosecond
    )
    expected = omega / (2.0 * np.pi) / speed_of_light_cm_per_ps
    peak = frequencies[np.argmax(intensities)]
    grid_spacing = frequencies[1] - frequencies[0]
    assert abs(peak - expected) <= grid_spacing

    with pytest.raises(ValueError, match="window"):
        reporters.vibrational_spectrum(archive, window="hamming")


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

    frequencies, intensities = reporters.vibrational_spectrum(output)

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
        reporters.velocity_autocorrelation(good, max_time=0.0)

    missing = tmp_path / "missing.npz"
    np.savez(missing, times_ps=np.arange(3.0))
    with pytest.raises(ValueError, match="lacks field"):
        reporters.velocity_autocorrelation(missing)

    short = tmp_path / "short.npz"
    np.savez(
        short,
        times_ps=np.array([0.0]),
        velocities_nm_per_ps=np.zeros((1, 1, 3)),
        atom_indices=np.array([0]),
        masses_dalton=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="at least two frames"):
        reporters.velocity_autocorrelation(short)

    mismatched = tmp_path / "mismatched.npz"
    np.savez(
        mismatched,
        times_ps=np.arange(3.0) * 0.001,
        velocities_nm_per_ps=np.zeros((3, 2, 3)),
        atom_indices=np.array([0]),
        masses_dalton=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="shapes disagree"):
        reporters.velocity_autocorrelation(mismatched)

    uneven = tmp_path / "uneven.npz"
    np.savez(
        uneven,
        times_ps=np.array([0.0, 0.001, 0.005]),
        velocities_nm_per_ps=np.zeros((3, 1, 3)),
        atom_indices=np.array([0]),
        masses_dalton=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        reporters.velocity_autocorrelation(uneven)


# ---------------------------------------------------------------------------
# Trajectory formats
# ---------------------------------------------------------------------------


def _periodic_two_atom_topology() -> app.Topology:
    topology = _two_atom_topology()
    topology.setPeriodicBoxVectors(
        [Vec3(3.0, 0.0, 0.0), Vec3(0.0, 3.0, 0.0), Vec3(0.0, 0.0, 3.0)]
        * unit.nanometer
    )
    return topology


_TWO_ATOM_POSITIONS = [Vec3(0.1, 0.0, 0.0), Vec3(0.2, 0.0, 0.0)] * unit.nanometer


@pytest.mark.parametrize("traj_format", ["pdb", "dcd", "xtc"])
def test_trajectory_writer_round_trips_every_format(
    tmp_path: Path, traj_format: str,
) -> None:
    pytest.importorskip("mdtraj")
    import mdtraj as md

    topology = _periodic_two_atom_topology()
    path = tmp_path / f"traj.{traj_format}"
    reference = tmp_path / "reference.pdb"
    with open(reference, "w") as handle:
        app.PDBFile.writeFile(topology, _TWO_ATOM_POSITIONS, handle)

    with reporters._TrajectoryWriter(
        path, topology, format=traj_format, reportInterval=5,
    ) as writer:
        assert writer.is_binary is (traj_format != "pdb")
        for _ in range(3):
            writer.write(
                writer.select(_TWO_ATOM_POSITIONS),
                step_size=0.002 * unit.picosecond,
                periodic_box_vectors=topology.getPeriodicBoxVectors(),
            )

    trajectory = md.load(str(path), top=str(reference))
    assert trajectory.n_frames == 3
    assert trajectory.n_atoms == 2
    # XTC quantizes to 1e-3 nm, so this is a tolerance rather than equality.
    assert np.abs(trajectory.xyz[0] - md.load(str(reference)).xyz[0]).max() < 2e-3


def test_trajectory_writer_pdb_writes_one_footer(tmp_path: Path) -> None:
    path = tmp_path / "traj.pdb"
    writer = reporters._TrajectoryWriter(path, _two_atom_topology())
    for _ in range(2):
        writer.write(_TWO_ATOM_POSITIONS, step_size=None)
    writer.close()
    writer.close()
    writer.__del__()

    contents = path.read_text().splitlines()
    assert [line for line in contents if line.startswith("MODEL")] == [
        "MODEL        1",
        "MODEL        2",
    ]
    assert contents.count("END") == 1


@pytest.mark.parametrize("traj_format", ["pdb", "dcd", "xtc"])
def test_trajectory_writer_fails_on_an_unwritable_path(
    tmp_path: Path, traj_format: str,
) -> None:
    with pytest.raises(OSError):
        reporters._TrajectoryWriter(
            tmp_path / "missing" / f"traj.{traj_format}",
            _two_atom_topology(),
            format=traj_format,
        )


def test_trajectory_writer_subset_keeps_residues_and_slices_positions(
    tmp_path: Path,
) -> None:
    topology = _periodic_two_atom_topology()
    writer = reporters._TrajectoryWriter(
        tmp_path / "subset.pdb", topology, atom_indices=[1],
    )

    assert writer.topology.getNumAtoms() == 1
    assert [residue.name for residue in writer.topology.residues()] == ["LIG"]
    assert [atom.name for atom in writer.topology.atoms()] == ["O"]
    assert writer.topology.getPeriodicBoxVectors() is not None

    selected = writer.select(_TWO_ATOM_POSITIONS)
    assert np.allclose(
        selected.value_in_unit(unit.nanometer), [[0.2, 0.0, 0.0]],
    )
    writer.close()


def test_trajectory_writer_select_is_a_no_op_without_a_subset(
    tmp_path: Path,
) -> None:
    writer = reporters._TrajectoryWriter(tmp_path / "all.pdb", _two_atom_topology())
    assert writer.select(_TWO_ATOM_POSITIONS) is _TWO_ATOM_POSITIONS
    writer.close()


def test_subset_topology_rejects_indices_outside_the_topology() -> None:
    with pytest.raises(ValueError, match="lie outside the topology"):
        reporters._subset_topology(_two_atom_topology(), [0, 5])


def test_integrator_formats_reject_h5_by_name_and_others_generically() -> None:
    with pytest.raises(ValueError, match="cannot be written from bead states"):
        reporters._require_integrator_format("h5")
    with pytest.raises(ValueError, match="unknown trajectory format"):
        reporters._require_integrator_format("mp4")


@pytest.mark.parametrize("traj_format", ["pdb", "dcd", "xtc"])
def test_bead_and_centroid_reporters_write_every_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, traj_format: str,
) -> None:
    pytest.importorskip("mdtraj")
    import mdtraj as md

    topology = _periodic_two_atom_topology()
    reference = tmp_path / "reference.pdb"
    with open(reference, "w") as handle:
        app.PDBFile.writeFile(topology, _TWO_ATOM_POSITIONS, handle)

    # A real State hands back Vec3 box vectors, which is what XTCFile reads;
    # the array-valued _State above is enough only for DCD.
    box = topology.getPeriodicBoxVectors()
    state = SimpleNamespace(
        getPositions=lambda asNumpy=False: _TWO_ATOM_POSITIONS,
        getPeriodicBoxVectors=lambda asNumpy=False: box,
    )
    integrator = SimpleNamespace(
        getNumCopies=lambda: 2,
        getState=lambda copy=None, **kwargs: state,
        getStepSize=lambda: 0.002 * unit.picosecond,
    )
    context = SimpleNamespace(getState=lambda **kwargs: state)
    simulation = SimpleNamespace(
        currentStep=0, integrator=integrator, context=context,
    )
    monkeypatch.setattr(
        reporters,
        "centroid_positions",
        lambda simulation, n_atoms, n_beads: _TWO_ATOM_POSITIONS,
    )

    with RPMDBeadReporter(
        file_base_name=str(tmp_path / "beads"),
        reportInterval=4,
        num_beads=2,
        topology=topology,
        format=traj_format,
    ) as beads, RPMDCentroidReporter(
        file_name=str(tmp_path / f"centroid.{traj_format}"),
        reportInterval=4,
        num_beads=2,
        topology=topology,
        format=traj_format,
    ) as centroid:
        for _ in range(2):
            beads.report(simulation, state=None)
            centroid.report(simulation, state=None)

    for path in (tmp_path / f"beads_bead_0.{traj_format}",
                 tmp_path / f"beads_bead_1.{traj_format}",
                 tmp_path / f"centroid.{traj_format}"):
        assert md.load(str(path), top=str(reference)).n_frames == 2


def test_centroid_reporter_asks_for_the_full_atom_count_under_a_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[tuple[Any, int, int]] = []
    monkeypatch.setattr(
        reporters,
        "centroid_positions",
        lambda simulation, n_atoms, n_beads: (
            calls.append((simulation, n_atoms, n_beads))
            or _TWO_ATOM_POSITIONS
        ),
    )
    simulation = SimpleNamespace(currentStep=0)

    with RPMDCentroidReporter(
        file_name=str(tmp_path / "centroid.pdb"),
        reportInterval=4,
        num_beads=3,
        topology=_two_atom_topology(),
        atom_indices=[0],
    ) as reporter:
        reporter.report(simulation, state=None)

    # Two, the whole System -- not the one atom the subset topology holds.
    assert calls == [(simulation, 2, 3)]


def test_bead_reporter_rejects_a_format_it_cannot_write(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be written from bead states"):
        RPMDBeadReporter(
            file_base_name=str(tmp_path / "beads"),
            reportInterval=1,
            num_beads=2,
            topology=_two_atom_topology(),
            format="h5",
        )


# ---------------------------------------------------------------------------
# The classical velocity archive
# ---------------------------------------------------------------------------


class _VelocityState:
    def __init__(self, velocities: Sequence[Any], time_ps: float) -> None:
        self._velocities = np.asarray(velocities, dtype=float) * (
            unit.nanometer / unit.picosecond
        )
        self._time = time_ps * unit.picosecond

    def getVelocities(self, asNumpy: bool=False) -> unit.Quantity:
        return self._velocities

    def getTime(self) -> unit.Quantity:
        return self._time


def _two_particle_system() -> openmm.System:
    system = openmm.System()
    system.addParticle(1.0 * unit.dalton)
    system.addParticle(16.0 * unit.dalton)
    return system


def test_velocity_archive_reporter_records_context_velocities(
    tmp_path: Path,
) -> None:
    output = tmp_path / "velocities.npz"
    simulation = SimpleNamespace(
        currentStep=4,
        integrator=SimpleNamespace(),
        system=_two_particle_system(),
    )

    with reporters.VelocityArchiveReporter(output, 2) as reporter:
        assert reporter.describeNextReport(simulation) == (
            2,
            False,
            True,
            False,
            False,
        )
        reporter.report(
            simulation, _VelocityState([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], 0.5),
        )
        reporter.report(
            simulation, _VelocityState([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], 1.0),
        )

    with np.load(output) as archive:
        assert sorted(archive.files) == [
            "atom_indices",
            "masses_dalton",
            "times_ps",
            "velocities_nm_per_ps",
        ]
        assert np.allclose(archive["times_ps"], [0.5, 1.0])
        assert archive["velocities_nm_per_ps"].shape == (2, 2, 3)
        assert np.allclose(archive["masses_dalton"], [1.0, 16.0])
        assert np.allclose(archive["atom_indices"], [0, 1])


def test_velocity_archive_reporter_keeps_only_selected_atoms(
    tmp_path: Path,
) -> None:
    output = tmp_path / "selected.npz"
    simulation = SimpleNamespace(
        currentStep=0,
        integrator=SimpleNamespace(),
        system=_two_particle_system(),
    )

    with reporters.VelocityArchiveReporter(output, 1, atom_indices=[1]) as reporter:
        reporter.report(
            simulation, _VelocityState([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], 0.0),
        )

    with np.load(output) as archive:
        assert np.allclose(archive["atom_indices"], [1])
        assert np.allclose(archive["masses_dalton"], [16.0])
        assert np.allclose(
            archive["velocities_nm_per_ps"], [[[0.0, 2.0, 0.0]]],
        )


def test_velocity_archive_reporter_refuses_an_rpmd_integrator(
    tmp_path: Path,
) -> None:
    simulation = SimpleNamespace(
        currentStep=0,
        integrator=SimpleNamespace(getNumCopies=lambda: 4),
        system=_two_particle_system(),
    )

    reporter = reporters.VelocityArchiveReporter(tmp_path / "v.npz", 1)
    with pytest.raises(TypeError, match="use RPMDVelocityReporter"):
        reporter.report(simulation, _VelocityState([[1.0, 0.0, 0.0]] * 2, 0.0))


def test_velocity_archive_reporter_writes_nothing_without_a_frame(
    tmp_path: Path,
) -> None:
    output = tmp_path / "silent.npz"
    reporters.VelocityArchiveReporter(output, 2).close()
    assert not output.exists()


def test_velocity_archives_share_one_schema(tmp_path: Path) -> None:
    """The classical and RPMD archives must be the same file, key for key."""
    system = _two_particle_system()
    classical = SimpleNamespace(
        currentStep=0, integrator=SimpleNamespace(), system=system,
    )
    bead_states = [
        _VelocityState([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], 0.5),
        _VelocityState([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], 0.5),
    ]
    integrator = SimpleNamespace(
        getNumCopies=lambda: 2,
        getState=lambda copy, **kwargs: bead_states[copy],
    )
    ring = SimpleNamespace(currentStep=0, integrator=integrator, system=system)

    with reporters.VelocityArchiveReporter(tmp_path / "a.npz", 1) as one:
        one.report(
            classical, _VelocityState([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], 0.5),
        )
    with reporters.RPMDVelocityReporter(tmp_path / "b.npz", 1) as other:
        other.report(ring, state=None)

    with np.load(tmp_path / "a.npz") as a, np.load(tmp_path / "b.npz") as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            assert a[key].shape == b[key].shape, key
            assert a[key].dtype == b[key].dtype, key
