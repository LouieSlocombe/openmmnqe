"""CPU regression tests for continuity between the two adQTB stages."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm

import openmmnqe as nqe
from openmmnqe import adqtb


def _adapted_friction_from_checkpoint(modeller: app.Modeller, forcefield: Any, checkpoint: Path) -> np.ndarray:
    system = forcefield.createSystem(modeller.topology)
    integrator = openmm.QTBIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        1.0 * unit.femtosecond,
    )
    integrator.setSegmentLength(0.5 * unit.picosecond)
    integrator.setDefaultAdaptationRate(0.5)
    simulation = app.Simulation(
        modeller.topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("CPU"),
    )
    simulation.loadCheckpoint(str(checkpoint))
    return np.asarray(integrator.getAdaptedFriction(0))


def test_adqtb_production_loads_equilibrated_friction_before_stepping(
    tmp_path: Path,
    one_particle_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = one_particle_system
    ready = tmp_path / "adqtb_ready"
    production = tmp_path / "adqtb_prod"

    nqe.run_openmm_adqtb_eq(
        modeller,
        forcefield,
        platform_name="CPU",
        n_report=1000,
        steps=1000,
        output_prefix=str(ready),
    )
    equilibrated = _adapted_friction_from_checkpoint(
        modeller, forcefield, ready.with_suffix(".chk")
    )
    assert not np.allclose(equilibrated, 1.0)
    assert ready.with_suffix(".pdb").is_file()
    assert ready.with_suffix(".chk").stat().st_size > 0

    nqe.run_openmm_adqtb_prod(
        modeller,
        forcefield,
        checkpoint_file=str(ready.with_suffix(".chk")),
        barostat_freq=None,
        platform_name="CPU",
        n_report=1000,
        steps=0,
        output_prefix=str(production),
    )
    continued = _adapted_friction_from_checkpoint(
        modeller, forcefield, production.with_suffix(".chk")
    )

    np.testing.assert_array_equal(continued, equilibrated)
    assert production.with_suffix(".pdb").is_file()
    assert production.with_suffix(".chk").stat().st_size > 0


def test_adqtb_production_rejects_missing_checkpoint(tmp_path: Path, one_particle_system: tuple[app.Modeller, Any]) -> None:
    modeller, forcefield = one_particle_system
    with pytest.raises(FileNotFoundError, match="equilibration stage"):
        nqe.run_openmm_adqtb_prod(
            modeller,
            forcefield,
            checkpoint_file=str(tmp_path / "missing.chk"),
            barostat_freq=None,
            platform_name="CPU",
            steps=0,
            output_prefix=str(tmp_path / "production"),
        )


class _State:
    """Minimal stand-in for the State a reporter is handed."""

    def __init__(self, time_ps: float) -> None:
        self._time = time_ps

    def getTime(self) -> unit.Quantity:
        return self._time * unit.picosecond


class _Integrator:
    """QTBIntegrator stand-in that returns scripted friction spectra."""

    def __init__(self, particle_types: dict[int, int] | None = None,
                 step_size: float = 0.001, segment_length: float = 0.005,
                 spectra: dict[int, list[float]] | None = None) -> None:
        self._particle_types = (
            {0: 0, 1: 0, 2: 1} if particle_types is None else particle_types
        )
        self._step_size = step_size
        self._segment_length = segment_length
        self._spectra = spectra
        self.reads: list[int] = []

    def getParticleTypes(self) -> dict[int, int]:
        return dict(self._particle_types)

    def getStepSize(self) -> unit.Quantity:
        return self._step_size * unit.picosecond

    def getSegmentLength(self) -> unit.Quantity:
        return self._segment_length * unit.picosecond

    def getAdaptedFriction(self, particle: int) -> tuple[float, ...]:
        self.reads.append(particle)
        if self._spectra is not None:
            return tuple(self._spectra[particle])
        n_freq = (3 * round(self._segment_length / self._step_size) + 1) // 2
        return tuple(1.0 + particle for _ in range(n_freq))


def _simulation(integrator: _Integrator, step: int) -> Any:
    """Return the smallest object a friction reporter can report on."""
    return SimpleNamespace(integrator=integrator, currentStep=step)


def _write_friction_log(path: Path, labels: Sequence[str],
                        spectra: Sequence[np.ndarray],
                        step_size: float = 0.001,
                        segment_steps: int = 5) -> None:
    """Write a friction log by hand, exactly as the reporter would."""
    n_freq = spectra[0].shape[1]
    columns = [
        f"Gamma_{label}_{index:04d}"
        for label in labels
        for index in range(n_freq)
    ]
    with open(path, "w") as handle:
        handle.write("Step\tTime(ps)\t" + "\t".join(columns) + "\n")
        for row in range(spectra[0].shape[0]):
            step = (row + 1) * segment_steps
            values = np.concatenate([block[row] for block in spectra])
            handle.write(
                f"{step}\t{step * step_size:.6f}"
                + "".join(f"\t{value:.6f}" for value in values)
                + "\n"
            )


# --------------------------------------------------------------------------
# QTBFrictionReporter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("interval", [True, np.bool_(False), 2.0, "5"])
def test_friction_reporter_rejects_non_integer_intervals(
    tmp_path: Path, interval: Any,
) -> None:
    log = tmp_path / "friction.log"
    with pytest.raises(TypeError, match="reportInterval must be an integer"):
        nqe.QTBFrictionReporter(log, interval, _Integrator())
    assert not log.exists()


def test_friction_reporter_rejects_intervals_across_a_segment_boundary(
    tmp_path: Path,
) -> None:
    log = tmp_path / "friction.log"
    with pytest.raises(ValueError, match="multiple of the segment length"):
        nqe.QTBFrictionReporter(log, 7, _Integrator())
    assert not log.exists()


def test_friction_reporter_rejects_an_untyped_integrator(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    with pytest.raises(ValueError, match="no adQTB particle types"):
        nqe.QTBFrictionReporter(log, 5, _Integrator(particle_types={}))
    assert not log.exists()


def test_friction_reporter_rejects_a_non_qtb_integrator(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    with pytest.raises(TypeError, match="getParticleTypes"):
        nqe.QTBFrictionReporter(log, 5, object())
    assert not log.exists()


def test_friction_reporter_rejects_a_fractional_segment(tmp_path: Path) -> None:
    integrator = _Integrator(step_size=0.001, segment_length=0.0055)
    with pytest.raises(ValueError, match="whole number of steps"):
        nqe.QTBFrictionReporter(tmp_path / "friction.log", 5, integrator)


@pytest.mark.parametrize(
    "type_names, expected",
    [
        (None, ["T0", "T1"]),
        ({0: "H", 1: "O"}, ["H", "O"]),
        ({0: "H"}, ["H", "T1"]),
    ],
)
def test_friction_reporter_header_names_every_type_and_bin(
    tmp_path: Path, type_names: dict[int, str] | None, expected: list[str],
) -> None:
    log = tmp_path / "friction.log"
    with nqe.QTBFrictionReporter(log, 5, _Integrator(), type_names):
        pass
    header = log.read_text().rstrip("\n").split("\t")

    assert header[:2] == ["Step", "Time(ps)"]
    # (3*5 + 1)//2 frequency bins per type, for two types
    assert len(header) == 2 + 2 * 8
    assert header[2] == f"Gamma_{expected[0]}_0000"
    assert header[9] == f"Gamma_{expected[0]}_0007"
    assert header[10] == f"Gamma_{expected[1]}_0000"


@pytest.mark.parametrize("names", [{0: "H", 1: "H"}, {0: "", 1: "O"},
                                   {0: "H\tO", 1: "O"}])
def test_friction_reporter_rejects_unusable_labels(
    tmp_path: Path, names: dict[int, str],
) -> None:
    log = tmp_path / "friction.log"
    with pytest.raises(ValueError):
        nqe.QTBFrictionReporter(log, 5, _Integrator(), names)
    assert not log.exists()


def test_friction_reporter_reports_on_the_segment_cadence(tmp_path: Path) -> None:
    with nqe.QTBFrictionReporter(tmp_path / "friction.log", 5,
                                 _Integrator()) as reporter:
        assert reporter.describeNextReport(_simulation(_Integrator(), 0)) == (
            5, False, False, False, False,
        )
        assert reporter.describeNextReport(_simulation(_Integrator(), 3))[0] == 2


def test_friction_reporter_writes_one_representative_per_type(
    tmp_path: Path,
) -> None:
    log = tmp_path / "friction.log"
    integrator = _Integrator()
    with nqe.QTBFrictionReporter(log, 5, integrator, {0: "H", 1: "O"}) as reporter:
        reporter.report(_simulation(integrator, 5), _State(0.005))
        reporter.report(_simulation(integrator, 10), _State(0.010))

    # Particles 0 and 1 share type 0, so only particle 0 is ever read.
    assert integrator.reads == [0, 2, 0, 2]
    rows = log.read_text().splitlines()[1:]
    first = rows[0].split("\t")
    assert first[0] == "5"
    assert first[1] == "0.005000"
    assert [float(value) for value in first[2:10]] == [1.0] * 8
    assert [float(value) for value in first[10:]] == [3.0] * 8
    assert rows[1].split("\t")[0] == "10"


def test_friction_reporter_rejects_a_spectrum_of_the_wrong_length(
    tmp_path: Path,
) -> None:
    integrator = _Integrator(spectra={0: [1.0, 1.0], 2: [1.0, 1.0]})
    with nqe.QTBFrictionReporter(tmp_path / "friction.log", 5,
                                 integrator) as reporter:
        with pytest.raises(ValueError, match="but the log was opened for 8"):
            reporter.report(_simulation(integrator, 5), _State(0.005))


def test_friction_reporter_close_is_idempotent(tmp_path: Path) -> None:
    reporter = nqe.QTBFrictionReporter(tmp_path / "friction.log", 5,
                                       _Integrator())
    reporter.close()
    reporter.close()
    reporter.__del__()
    assert (tmp_path / "friction.log").is_file()


# --------------------------------------------------------------------------
# Reading a friction log back
# --------------------------------------------------------------------------

def test_friction_spectra_round_trip_the_log(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    hydrogen = np.arange(12.0).reshape(3, 4)
    oxygen = np.arange(12.0).reshape(3, 4) * -1.0
    _write_friction_log(log, ["H", "O"], [hydrogen, oxygen])

    spectra = nqe.adqtb_friction_spectra(log)

    assert list(spectra) == ["H", "O"]
    np.testing.assert_allclose(spectra["H"], hydrogen)
    np.testing.assert_allclose(spectra["O"], oxygen)
    assert list(nqe.adqtb_friction_spectra(log, types="O")) == ["O"]
    assert list(nqe.adqtb_friction_spectra(log, types=["O", "H"])) == ["H", "O"]


def test_friction_spectra_reject_an_unknown_type(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    _write_friction_log(log, ["H"], [np.ones((2, 3))])
    with pytest.raises(ValueError, match="unknown friction log type"):
        nqe.adqtb_friction_spectra(log, types="D")


def test_friction_log_without_friction_columns_is_rejected(
    tmp_path: Path,
) -> None:
    log = tmp_path / "friction.log"
    log.write_text("Step\tTime(ps)\n5\t0.005\n")
    with pytest.raises(ValueError, match="no friction columns"):
        nqe.adqtb_friction_spectra(log)


def test_friction_log_with_a_missing_bin_is_rejected(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    log.write_text(
        "Step\tTime(ps)\tGamma_H_0000\tGamma_H_0002\n5\t0.005\t1.0\t1.0\n"
    )
    with pytest.raises(ValueError, match="gaps in the frequency bins"):
        nqe.adqtb_friction_spectra(log)


def test_frequencies_match_the_integrator_grid(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    # 5 steps per segment gives (3*5 + 1)//2 = 8 bins at a 1 fs step.
    _write_friction_log(log, ["H"], [np.ones((3, 8))], step_size=0.001,
                        segment_steps=5)

    frequencies = nqe.adqtb_frequencies(log)

    expected = np.arange(8) * np.pi / (8 * 0.001)
    np.testing.assert_allclose(frequencies, expected)
    assert frequencies[0] == 0.0
    assert frequencies[-1] < np.pi / 0.001


def test_frequencies_require_a_time_column(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    log.write_text("Step\tGamma_H_0000\tGamma_H_0001\n5\t1.0\t1.0\n")
    with pytest.raises(ValueError, match="Time\\(ps\\) column"):
        nqe.adqtb_frequencies(log)


def test_fdt_residual_is_the_per_segment_increment(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    hydrogen = np.array([[1.0, 1.0], [1.5, 0.5], [1.25, 0.75]])
    _write_friction_log(log, ["H"], [hydrogen])

    residual = nqe.adqtb_fdt_residual(log)

    np.testing.assert_allclose(
        residual["H"], [[0.5, -0.5], [-0.25, 0.25]], atol=1e-9,
    )


def test_fdt_residual_needs_two_segments(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    _write_friction_log(log, ["H"], [np.ones((1, 3))])
    with pytest.raises(ValueError, match="a residual needs at least two"):
        nqe.adqtb_fdt_residual(log)


# --------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------

def _random_walk(n_segments: int, n_freq: int, *, drift: float,
                 seed: int = 0) -> np.ndarray:
    """Return a spectrum diffusing about 1.0, optionally marching as well."""
    generator = np.random.default_rng(seed)
    steps = generator.normal(0.0, 0.01, size=(n_segments, n_freq)) + drift
    return 1.0 + np.cumsum(steps, axis=0)


def test_convergence_passes_a_spectrum_that_is_only_diffusing(
    tmp_path: Path,
) -> None:
    log = tmp_path / "friction.log"
    _write_friction_log(log, ["H"], [_random_walk(200, 6, drift=0.0)])

    verdict = nqe.adqtb_convergence(log)["H"]

    assert verdict.converged
    assert verdict.drift_ratio < 2.0
    assert verdict.clamped_fraction == 0.0
    assert verdict.step_rms > 0.0


def test_convergence_fails_a_spectrum_that_is_still_marching(
    tmp_path: Path,
) -> None:
    log = tmp_path / "friction.log"
    _write_friction_log(log, ["H"], [_random_walk(200, 6, drift=0.02)])

    verdict = nqe.adqtb_convergence(log)["H"]

    assert not verdict.converged
    assert verdict.drift_ratio > 2.0
    assert verdict.max_deviation > 1.0


def test_convergence_counts_bins_clamped_at_zero(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    spectrum = _random_walk(20, 4, drift=0.0)
    spectrum[10:, 0] = 0.0
    _write_friction_log(log, ["H"], [spectrum])

    verdict = nqe.adqtb_convergence(log, discard=0.5)["H"]

    assert verdict.clamped_fraction == pytest.approx(0.25)


def test_convergence_reports_a_frozen_spectrum_as_converged(
    tmp_path: Path,
) -> None:
    log = tmp_path / "friction.log"
    _write_friction_log(log, ["H"], [np.ones((5, 3))])

    verdict = nqe.adqtb_convergence(log)["H"]

    assert verdict.converged
    assert verdict.step_rms == 0.0
    assert verdict.drift_ratio == 0.0
    assert verdict.max_deviation == 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"discard": 1.0}, "discard must be in the interval"),
        ({"discard": -0.1}, "discard must be in the interval"),
        ({"discard": float("nan")}, "discard must be in the interval"),
        ({"tolerance": 0.0}, "tolerance must be finite and positive"),
        ({"tolerance": float("inf")}, "tolerance must be finite and positive"),
    ],
)
def test_convergence_validates_its_options(
    tmp_path: Path, kwargs: dict[str, float], message: str,
) -> None:
    log = tmp_path / "friction.log"
    _write_friction_log(log, ["H"], [np.ones((5, 3))])
    with pytest.raises(ValueError, match=message):
        nqe.adqtb_convergence(log, **kwargs)


def test_convergence_needs_three_segments(tmp_path: Path) -> None:
    log = tmp_path / "friction.log"
    _write_friction_log(log, ["H"], [np.ones((2, 3))])
    with pytest.raises(ValueError, match="needs at least three"):
        nqe.adqtb_convergence(log)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _plot_log(tmp_path: Path) -> Path:
    """Write a two-type friction log big enough to plot."""
    log = tmp_path / "friction.log"
    _write_friction_log(
        log,
        ["H", "O"],
        [_random_walk(20, 8, drift=0.0, seed=1),
         _random_walk(20, 8, drift=0.0, seed=2)],
    )
    return log


def test_plot_friction_spectra_draws_one_panel_per_type(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    log = _plot_log(tmp_path)
    figure, axes = nqe.plot_adqtb_friction_spectra(
        log, filename=tmp_path / "spectra.png",
    )

    assert len(axes) == 2
    assert axes[-1].get_xlabel() == "Frequency (cm$^{-1}$)"
    # six sampled segments plus the gamma_r = 1 reference line
    assert len(axes[0].lines) == 7
    assert (tmp_path / "spectra.png").is_file()
    plt.close(figure)


@pytest.mark.parametrize(
    "segments, expected",
    [(None, 6), (3, 3), ([0, 2, -1], 3), ([1, 1], 1)],
)
def test_plot_friction_spectra_selects_segments(
    tmp_path: Path, segments: Any, expected: int,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = nqe.plot_adqtb_friction_spectra(
        _plot_log(tmp_path), types="H", segments=segments,
    )

    assert len(axes[0].lines) == expected + 1
    plt.close(figure)


def test_plot_friction_spectra_rejects_an_out_of_range_segment(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="lies outside the 20 rows"):
        nqe.plot_adqtb_friction_spectra(_plot_log(tmp_path), segments=[99])


def test_plot_friction_spectra_rejects_an_unknown_frequency_unit(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="frequency_unit must be one of"):
        nqe.plot_adqtb_friction_spectra(_plot_log(tmp_path),
                                        frequency_unit="THz")


def test_plot_friction_spectra_rejects_non_finite_values(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    log = tmp_path / "friction.log"
    spectrum = np.ones((4, 3))
    spectrum[2, 1] = np.nan
    _write_friction_log(log, ["H"], [spectrum])
    with pytest.raises(ValueError, match="must be finite"):
        nqe.plot_adqtb_friction_spectra(log)


@pytest.mark.parametrize("frequency_unit, label",
                         [("1/ps", "1/ps"), ("rad/ps", "rad/ps")])
def test_plot_fdt_residual_stacks_spectrum_over_history(
    tmp_path: Path, frequency_unit: str, label: str,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = nqe.plot_adqtb_fdt_residual(
        _plot_log(tmp_path),
        frequency_unit=frequency_unit,
        filename=tmp_path / "residual.png",
    )

    spectrum_axis, history_axis = axes
    assert spectrum_axis.get_xlabel() == f"Frequency ({label})"
    assert history_axis.get_xlabel() == "Time (ps)"
    assert history_axis.get_yscale() == "log"
    # one line per type, plus the zero reference on the spectrum panel
    assert len(spectrum_axis.lines) == 3
    # a raw trace and a running mean per type on the history panel
    assert len(history_axis.lines) == 4
    assert (tmp_path / "residual.png").is_file()
    plt.close(figure)


def test_plot_fdt_residual_validates_discard(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="discard must be in the interval"):
        nqe.plot_adqtb_fdt_residual(_plot_log(tmp_path), discard=1.5)


# --------------------------------------------------------------------------
# The drivers, on a real CPU adQTB run
# --------------------------------------------------------------------------

class _WaterForceField:
    """Two hydrogens and one oxygen, each in its own harmonic well."""

    def createSystem(self, topology: app.Topology, **kwargs: Any) -> openmm.System:
        """Return a three-particle System, ignoring *kwargs*."""
        system = openmm.System()
        restraint = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
        restraint.addGlobalParameter(
            "k", 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        for atom in topology.atoms():
            system.addParticle(atom.element.mass)
            restraint.addParticle(atom.index, [])
        system.addForce(restraint)
        return system


@pytest.fixture
def water_system() -> tuple[app.Modeller, _WaterForceField]:
    """Return a Modeller of two hydrogens and one oxygen, plus its field."""
    topology = app.Topology()
    residue = topology.addResidue("HOH", topology.addChain())
    for name, symbol in (("H1", "H"), ("H2", "H"), ("O", "O")):
        topology.addAtom(name, app.Element.getBySymbol(symbol), residue)
    positions = [
        Vec3(0.05, 0.0, 0.0), Vec3(-0.05, 0.0, 0.0), Vec3(0.0, 0.05, 0.0),
    ] * unit.nanometer
    return app.Modeller(topology, positions), _WaterForceField()


def _run_equilibration(modeller: app.Modeller, forcefield: Any, prefix: Path,
                       **kwargs: Any) -> None:
    """Run a short adQTB equilibration with a segment of 50 steps."""
    nqe.run_openmm_adqtb_eq(
        modeller,
        forcefield,
        segment_length=0.05 * unit.picosecond,
        time_step=1.0 * unit.femtosecond,
        platform_name="CPU",
        n_report=200,
        steps=400,
        output_prefix=str(prefix),
        seed=7,
        **kwargs,
    )


def test_adqtb_equilibration_logs_one_spectrum_per_element(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    prefix = tmp_path / "adqtb_ready"

    _run_equilibration(modeller, forcefield, prefix)

    log = prefix.parent / f"{prefix.name}_friction.log"
    spectra = nqe.adqtb_friction_spectra(log)

    assert sorted(spectra) == ["H", "O"]
    # 400 steps at 50 steps per segment is 8 rows, of 76 bins each.
    assert spectra["H"].shape == (8, (3 * 50 + 1) // 2)
    assert not np.allclose(spectra["H"], 1.0)
    # The two hydrogens share a bath, so their spectra are identical and the
    # oxygen's is not.
    assert not np.allclose(spectra["H"][-1], spectra["O"][-1])
    assert nqe.adqtb_fdt_residual(log)["H"].shape[0] == 7
    assert set(nqe.adqtb_convergence(log)) == {"H", "O"}


def test_adqtb_progress_log_omits_the_classical_temperature(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    prefix = tmp_path / "adqtb_ready"

    _run_equilibration(modeller, forcefield, prefix)

    header = prefix.with_suffix(".log").read_text().splitlines()[0]
    assert "Temperature" not in header
    assert "Kinetic Energy" not in header
    assert "Total Energy" not in header
    assert "Potential Energy" in header


def test_adqtb_deuteration_splits_hydrogen_from_deuterium(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    prefix = tmp_path / "adqtb_ready"

    _run_equilibration(modeller, forcefield, prefix, deuterate=True,
                       deuterate_option="all")

    log = prefix.parent / f"{prefix.name}_friction.log"
    labels = sorted(nqe.adqtb_friction_spectra(log))

    assert labels == ["H2.014", "O"]


def test_adqtb_without_particle_types_skips_the_friction_log(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    prefix = tmp_path / "adqtb_ready"

    with pytest.warns(UserWarning, match="no adQTB particle types"):
        _run_equilibration(modeller, forcefield, prefix, particle_types="none")

    assert not (prefix.parent / f"{prefix.name}_friction.log").exists()


def test_adqtb_accepts_an_explicit_particle_type_map(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    prefix = tmp_path / "adqtb_ready"

    _run_equilibration(modeller, forcefield, prefix,
                       particle_types={0: 0, 1: 0, 2: 1})

    log = prefix.parent / f"{prefix.name}_friction.log"
    # Types come back unlabelled, because the caller supplied the mapping.
    assert list(nqe.adqtb_friction_spectra(log)) == ["T0", "T1"]


def test_adqtb_rejects_a_particle_type_mixing_masses(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    # OpenMM refuses a type spanning more than one mass, which is why
    # ``particle_types="element"`` splits a symbol whose masses differ.
    with pytest.raises(Exception, match="same type must have the same mass"):
        _run_equilibration(modeller, forcefield, tmp_path / "adqtb_ready",
                           particle_types={0: 0, 1: 0, 2: 0})


def test_adqtb_writes_no_friction_log_when_asked_not_to(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    prefix = tmp_path / "adqtb_ready"

    _run_equilibration(modeller, forcefield, prefix, friction_log=False)

    assert not (prefix.parent / f"{prefix.name}_friction.log").exists()
    assert prefix.with_suffix(".chk").is_file()


def test_adqtb_rejects_a_bad_particle_types_argument(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    with pytest.raises(ValueError, match="particle_types must be"):
        _run_equilibration(modeller, forcefield, tmp_path / "adqtb_ready",
                           particle_types="by-mass")


@pytest.mark.parametrize(
    "segment_length, message",
    [
        (0.0055 * unit.picosecond, "whole number of time steps"),
        (0.011 * unit.picosecond, "prime factors are 2, 3, 5 and 7"),
    ],
)
def test_adqtb_rejects_a_segment_openmm_cannot_transform(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
    segment_length: unit.Quantity, message: str,
) -> None:
    modeller, forcefield = water_system
    with pytest.raises(ValueError, match=message):
        nqe.run_openmm_adqtb_eq(
            modeller,
            forcefield,
            segment_length=segment_length,
            time_step=1.0 * unit.femtosecond,
            platform_name="CPU",
            steps=0,
            output_prefix=str(tmp_path / "adqtb_ready"),
        )


def test_live_friction_readback_matches_the_final_logged_row(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    prefix = tmp_path / "adqtb_ready"
    _run_equilibration(modeller, forcefield, prefix)

    system = forcefield.createSystem(modeller.topology)
    integrator = openmm.QTBIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond,
    )
    integrator.setSegmentLength(0.05 * unit.picosecond)
    type_to_label = nqe.set_adqtb_particle_types_by_element(
        integrator, topology=modeller.topology, system=system,
    )
    simulation = app.Simulation(
        modeller.topology, system, integrator,
        openmm.Platform.getPlatformByName("CPU"),
    )
    simulation.loadCheckpoint(str(prefix.with_suffix(".chk")))

    live = nqe.adqtb_friction(
        simulation, type_names={v: k for k, v in type_to_label.items()},
    )
    logged = nqe.adqtb_friction_spectra(
        prefix.parent / f"{prefix.name}_friction.log"
    )

    assert sorted(live) == ["H", "O"]
    for label in live:
        np.testing.assert_allclose(live[label], logged[label][-1], atol=1e-6)


def test_track_adqtb_friction_picks_the_segment_interval(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    system = forcefield.createSystem(modeller.topology)
    integrator = openmm.QTBIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond,
    )
    integrator.setSegmentLength(0.05 * unit.picosecond)
    nqe.set_adqtb_particle_types_by_element(
        integrator, topology=modeller.topology, system=system,
    )
    simulation = app.Simulation(
        modeller.topology, system, integrator,
        openmm.Platform.getPlatformByName("CPU"),
    )
    simulation.context.setPositions(modeller.positions)

    reporter = nqe.track_adqtb_friction(simulation, tmp_path / "friction.log",
                                        segments_per_report=2)
    try:
        assert simulation.reporters == [reporter]
        assert reporter.describeNextReport(simulation)[0] == 100
        simulation.step(200)
    finally:
        reporter.close()

    assert nqe.adqtb_friction_spectra(tmp_path / "friction.log")["T0"].shape[0] == 2


def test_track_adqtb_friction_rejects_a_bad_segment_count(
    tmp_path: Path, water_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = water_system
    simulation = SimpleNamespace(integrator=_Integrator(), reporters=[])
    with pytest.raises(TypeError, match="segments_per_report"):
        nqe.track_adqtb_friction(simulation, tmp_path / "friction.log",
                                 segments_per_report=1.5)


def test_plot_crops_the_frequency_axis_to_the_adapted_region(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    log = tmp_path / "friction.log"
    # Only the bottom eighth of the spectrum ever moves away from 1.0.
    spectrum = np.ones((5, 32))
    spectrum[1:, :4] = 1.5
    _write_friction_log(log, ["H"], [spectrum], segment_steps=21)
    full = nqe.adqtb_frequencies(log)[-1]

    figure, axes = nqe.plot_adqtb_friction_spectra(log,
                                                   frequency_unit="rad/ps")
    cropped = axes[0].get_xlim()[1]
    plt.close(figure)

    assert cropped < full / 2

    figure, axes = nqe.plot_adqtb_friction_spectra(
        log, frequency_unit="rad/ps", max_frequency=full,
    )
    assert axes[0].get_xlim()[1] == pytest.approx(full)
    plt.close(figure)


def test_plot_keeps_the_whole_axis_when_nothing_adapted(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    log = tmp_path / "friction.log"
    _write_friction_log(log, ["H"], [np.ones((4, 8))])

    figure, axes = nqe.plot_adqtb_friction_spectra(log,
                                                   frequency_unit="rad/ps")

    assert axes[0].get_xlim()[1] == pytest.approx(
        nqe.adqtb_frequencies(log)[-1]
    )
    plt.close(figure)


@pytest.mark.parametrize(
    "values, window, expected",
    [
        ([1.0, 2.0, 3.0], 1, [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0, 4.0], 2, [1.5, 1.5, 2.5, 3.5]),
        ([5.0, 5.0, 5.0], 10, [5.0, 5.0, 5.0]),
    ],
)
def test_running_mean_averages_over_the_window_that_fits(
    values: list[float], window: int, expected: list[float],
) -> None:
    smoothed = adqtb._running_mean(np.asarray(values), window)
    np.testing.assert_allclose(smoothed, expected)


def test_friction_log_with_unequal_type_widths_is_rejected(
    tmp_path: Path,
) -> None:
    log = tmp_path / "friction.log"
    log.write_text(
        "Step\tTime(ps)\tGamma_H_0000\tGamma_H_0001\tGamma_O_0000\n"
        "5\t0.005\t1.0\t1.0\t1.0\n"
    )
    with pytest.raises(ValueError, match="different numbers of frequency"):
        nqe.adqtb_friction_spectra(log)
