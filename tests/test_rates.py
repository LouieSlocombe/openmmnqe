"""Tests for the ring-polymer recrossing driver and rate assembly."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm

import openmmnqe as nqe
import openmmnqe.openmm as nqe_openmm
import openmmnqe.rates as nqe_rates

BARRIER_X = 0.30  # nm
WELL_HALF_WIDTH = 0.05  # nm
GAS_CONSTANT = 8.314462618e-3  # kJ/mol/K


def _double_well_topology() -> app.Topology:
    """Return a one-argon topology matching :func:`_double_well_system`."""
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    return topology


def _double_well_system() -> openmm.System:
    """Return one argon in a quartic double well along x, harmonic in y, z.

    Built by hand so there is no CMMotionRemover: a remover would zero the
    single particle's momentum and make every shot stall on the barrier.
    """
    system = openmm.System()
    system.addParticle(39.948 * unit.dalton)
    force = openmm.CustomExternalForce(
        "A*((x - x0)^2 - b2)^2 + 0.5*k*(y^2 + z^2)"
    )
    force.addGlobalParameter("A", 4.0e5)
    force.addGlobalParameter("x0", BARRIER_X)
    force.addGlobalParameter("b2", WELL_HALF_WIDTH**2)
    force.addGlobalParameter("k", 1000.0)
    force.addParticle(0, [])
    system.addForce(force)
    return system


def _x_coordinate(positions: np.ndarray) -> float:
    """The test collective variable: the particle's x position."""
    return float(positions[0, 0])


def _write_barrier_snapshots(n_beads: int, count: int,
                             prefix: str = "well") -> list[str]:
    """Write *count* bead archives with every bead near the barrier top."""
    integrator = openmm.RPMDIntegrator(
        n_beads,
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.2 * unit.femtoseconds,
    )
    simulation = app.Simulation(
        _double_well_topology(),
        _double_well_system(),
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    rng = np.random.default_rng(1234)
    paths = []
    for index in range(count):
        offset = 1.0e-4 * rng.standard_normal()
        for bead in range(n_beads):
            integrator.setPositions(
                bead,
                np.array([[BARRIER_X + offset, 0.0, 0.0]]) * unit.nanometer,
            )
            integrator.setVelocities(
                bead,
                np.zeros((1, 3)) * unit.nanometer / unit.picosecond,
            )
        path = f"{prefix}_snapshot_{index:03d}.npz"
        nqe_openmm._save_rpmd_restart(simulation, path, n_beads)
        paths.append(path)
    return paths


def _shoot(paths: list[str], *, prefix: str, n_beads: int, seed: int | None,
           n_children: int = 4, n_steps: int = 40, record_interval: int = 5,
           resume: bool = False, **kwargs: Any) -> list[str]:
    """Run the recrossing driver on the double well and list its logs."""
    modeller = app.Modeller(
        _double_well_topology(),
        [Vec3(BARRIER_X, 0.0, 0.0)] * unit.nanometer,
    )
    nqe.run_openmm_rpmd_recrossing(
        modeller,
        nqe.PreparedSystem(_double_well_system()),
        paths,
        _x_coordinate,
        output_prefix=prefix,
        n_beads=n_beads,
        temperature=300.0 * unit.kelvin,
        time_step=0.2 * unit.femtoseconds,
        n_children=n_children,
        n_steps=n_steps,
        record_interval=record_interval,
        platform_name="Reference",
        seed=seed,
        resume=resume,
        **kwargs,
    )
    return [
        f"{prefix}_recrossing_{index:05d}.log" for index in range(len(paths))
    ]


def _write_recrossing_log(path: str | Path, parent: int,
                          children: list[tuple[int, float, list[float]]],
                          *, record_interval: int = 5, dt: float = 0.001,
                          extra_column: str | None = None,
                          extra_value: float = 0.0) -> None:
    """Write a synthetic recrossing log with hand-chosen fluxes and series."""
    columns = list(nqe_rates._RECROSSING_REQUIRED_COLUMNS)
    if extra_column is not None:
        columns.append(extra_column)
    lines = ["\t".join(columns)]
    for child, flux0, series in children:
        for record, value in enumerate(series):
            step = record * record_interval
            fields = [
                str(step),
                f"{step * dt:.17g}",
                str(parent),
                str(child),
                f"{flux0:.17g}",
                f"{value:.17g}",
            ]
            if extra_column is not None:
                fields.append(f"{extra_value:.17g}")
            lines.append("\t".join(fields))
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Analysis on synthetic logs


def test_transmission_coefficient_matches_a_hand_computed_ratio() -> None:
    # Two parents, two children each, constant side functions, so every
    # number is checkable by hand: N(t) = 1 - 0.5 + 0 + 1 = 1.5 and
    # D = 1 + 0 + 2 + 1 = 4 at every time.
    _write_recrossing_log("a_00000.log", 0, [
        (0, 1.0, [0.1, 0.1, 0.1, 0.1]),
        (1, -0.5, [0.1, 0.1, 0.1, 0.1]),
    ])
    _write_recrossing_log("a_00001.log", 1, [
        (0, 2.0, [-0.1, -0.1, -0.1, -0.1]),
        (1, 1.0, [0.1, 0.1, 0.1, 0.1]),
    ])

    result = nqe.transmission_coefficient(
        ["a_00000.log", "a_00001.log"],
        s_dagger=0.0,
        blocks=2,
        plateau_fraction=0.5,
    )

    assert result.times_ps == pytest.approx([0.0, 0.005, 0.010, 0.015])
    assert result.kappa_t == pytest.approx([0.375] * 4)
    # Blocks are single parents: kappa_A = 0.5/1 and kappa_B = 1/3.
    expected_stderr = np.std([0.5, 1.0 / 3.0], ddof=1) / np.sqrt(2)
    assert result.kappa_stderr == pytest.approx([expected_stderr] * 4)
    assert result.plateau == pytest.approx(0.375)
    assert result.plateau_stderr == pytest.approx(expected_stderr)
    assert result.forward_flux == pytest.approx(1.0)
    assert result.s0_mean == pytest.approx(0.05)
    assert result.s0_std == pytest.approx(math.sqrt(0.0075))
    assert result.n_parents == 2
    assert result.n_parents_excluded == 0
    assert result.n_children == 4


def test_transmission_coefficient_prunes_parents_by_initial_deviation() -> None:
    good = [(0, 1.0, [0.01, 0.01]), (1, 1.0, [0.01, 0.01])]
    _write_recrossing_log("b_00000.log", 0, good)
    _write_recrossing_log("b_00001.log", 1, good)
    _write_recrossing_log("b_00002.log", 2, [
        (0, 1.0, [0.5, 0.5]),
        (1, 1.0, [0.5, 0.5]),
    ])

    files = ["b_00000.log", "b_00001.log", "b_00002.log"]
    kept = nqe.transmission_coefficient(
        files, s_dagger=0.0, blocks=2, plateau_fraction=1.0,
        max_s0_deviation=0.1,
    )
    assert kept.n_parents == 2
    assert kept.n_parents_excluded == 1
    assert kept.n_children == 4

    with pytest.raises(ValueError, match="excludes every parent"):
        nqe.transmission_coefficient(
            files, s_dagger=5.0, blocks=2, plateau_fraction=1.0,
            max_s0_deviation=0.1,
        )


def test_transmission_coefficient_requires_forward_flux() -> None:
    _write_recrossing_log("c_00000.log", 0, [
        (0, -1.0, [0.1, 0.1]),
        (1, 0.0, [0.1, 0.1]),
    ])
    _write_recrossing_log("c_00001.log", 1, [(0, -2.0, [0.1, 0.1])])

    with pytest.raises(ValueError, match="no forward-moving children"):
        nqe.transmission_coefficient(
            ["c_00000.log", "c_00001.log"], s_dagger=0.0, blocks=2,
        )


def test_transmission_coefficient_rejects_a_fluxless_block() -> None:
    _write_recrossing_log("d_00000.log", 0, [(0, 1.0, [0.1, 0.1])])
    _write_recrossing_log("d_00001.log", 1, [(0, -1.0, [0.1, 0.1])])

    with pytest.raises(ValueError, match="parent block carries no forward"):
        nqe.transmission_coefficient(
            ["d_00000.log", "d_00001.log"], s_dagger=0.0, blocks=2,
        )


def test_transmission_coefficient_rejects_mixed_record_grids() -> None:
    _write_recrossing_log("e_00000.log", 0, [(0, 1.0, [0.1, 0.1, 0.1])],
                          record_interval=5)
    _write_recrossing_log("e_00001.log", 1, [(0, 1.0, [0.1, 0.1])],
                          record_interval=7)

    with pytest.raises(ValueError, match="one record grid"):
        nqe.transmission_coefficient(
            ["e_00000.log", "e_00001.log"], s_dagger=0.0, blocks=2,
        )


def test_transmission_coefficient_rejects_mixed_time_steps() -> None:
    _write_recrossing_log("f_00000.log", 0, [(0, 1.0, [0.1, 0.1])], dt=0.001)
    _write_recrossing_log("f_00001.log", 1, [(0, 1.0, [0.1, 0.1])], dt=0.002)

    with pytest.raises(ValueError, match="disagree on record times"):
        nqe.transmission_coefficient(
            ["f_00000.log", "f_00001.log"], s_dagger=0.0, blocks=2,
        )


def test_transmission_coefficient_rejects_a_wandering_flux_column() -> None:
    lines = [
        "\t".join(nqe_rates._RECROSSING_REQUIRED_COLUMNS),
        "0\t0\t0\t0\t1.0\t0.1",
        "5\t0.005\t0\t0\t2.0\t0.1",
    ]
    Path("g_00000.log").write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="Flux0 varies"):
        nqe.transmission_coefficient("g_00000.log", s_dagger=0.0, blocks=2)


def test_transmission_coefficient_input_validation() -> None:
    _write_recrossing_log("h_00000.log", 0, [(0, 1.0, [0.1, 0.1])])
    _write_recrossing_log("h_00001.log", 1, [(0, 1.0, [0.1, 0.1])])
    files = ["h_00000.log", "h_00001.log"]

    with pytest.raises(ValueError, match="no recrossing logs"):
        nqe.transmission_coefficient([], s_dagger=0.0)
    with pytest.raises(ValueError, match="s_dagger"):
        nqe.transmission_coefficient(files, s_dagger=float("nan"))
    with pytest.raises(ValueError, match="blocks"):
        nqe.transmission_coefficient(files, s_dagger=0.0, blocks=1)
    with pytest.raises(ValueError, match="plateau_fraction"):
        nqe.transmission_coefficient(
            files, s_dagger=0.0, blocks=2, plateau_fraction=0.0,
        )
    with pytest.raises(ValueError, match="plateau_fraction"):
        nqe.transmission_coefficient(
            files, s_dagger=0.0, blocks=2, plateau_fraction=1.5,
        )
    with pytest.raises(ValueError, match="max_s0_deviation"):
        nqe.transmission_coefficient(
            files, s_dagger=0.0, blocks=2, max_s0_deviation=0.0,
        )
    with pytest.raises(ValueError, match="parents survive"):
        nqe.transmission_coefficient(files, s_dagger=0.0, blocks=3)


def test_recrossing_logs_are_validated_structurally() -> None:
    _write_recrossing_log("ok_00000.log", 0, [(0, 1.0, [0.1])])
    with pytest.raises(ValueError, match="single record"):
        nqe.transmission_coefficient("ok_00000.log", s_dagger=0.0, blocks=2)

    Path("bad_header.log").write_text(
        "Step\tTime(ps)\tParent\tChild\tCV\n0\t0\t0\t0\t0.1\n"
    )
    with pytest.raises(ValueError, match="lacks column"):
        nqe.transmission_coefficient("bad_header.log", s_dagger=0.0)

    _write_recrossing_log("bad_nan.log", 0, [(0, 1.0, [0.1, float("nan")])])
    with pytest.raises(ValueError, match="non-finite"):
        nqe.transmission_coefficient("bad_nan.log", s_dagger=0.0)


# --------------------------------------------------------------------------
# Rate assembly


def _harmonic_surface(k_f: float, minimum: float, cap: float,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """A harmonic reactant basin capped at the barrier free energy."""
    grid = np.linspace(-0.25, 0.05, 3001)
    surface = np.minimum(0.5 * k_f * (grid - minimum) ** 2, cap)
    return grid, surface


def test_rpmd_rate_matches_the_analytic_harmonic_reactant() -> None:
    # With a harmonic reactant window covering many sigma and a flat barrier,
    # the reactant integral is the full Gaussian and
    # k_QTST = flux * exp(-beta F) * sqrt(beta k_f / 2 pi).
    k_f, barrier = 5000.0, 20.0
    grid, surface = _harmonic_surface(k_f, -0.1, barrier)
    beta = 1.0 / (GAS_CONSTANT * 300.0)

    rate = nqe.rpmd_rate(
        0.5,
        grid,
        surface,
        temperature=300.0,
        s_dagger=0.02,
        reactant_window=(-0.185, -0.015),
        forward_flux=2.0,
        transmission_stderr=0.1,
    )

    expected_qtst = (
        2.0 * math.exp(-beta * barrier) * math.sqrt(beta * k_f / (2.0 * math.pi))
    )
    assert rate.qtst_rate == pytest.approx(expected_qtst, rel=2.0e-3)
    assert rate.rate == pytest.approx(0.5 * expected_qtst, rel=2.0e-3)
    assert rate.rate_stderr == pytest.approx(0.1 * expected_qtst, rel=2.0e-3)
    assert rate.transmission == 0.5
    assert rate.forward_flux == 2.0
    assert rate.dividing_surface_probability == pytest.approx(
        expected_qtst / 2.0, rel=2.0e-3,
    )


def test_rpmd_rate_converts_kilocalories() -> None:
    grid, surface = _harmonic_surface(5000.0, -0.1, 20.0)
    kwargs: dict[str, Any] = {
        "temperature": 300.0,
        "s_dagger": 0.02,
        "reactant_window": (-0.185, -0.015),
        "forward_flux": 1.0,
    }

    in_kj = nqe.rpmd_rate(1.0, grid, surface, **kwargs)
    in_kcal = nqe.rpmd_rate(
        1.0, grid, surface / 4.184, energy_unit="kilocalorie_per_mole",
        **kwargs,
    )

    assert in_kcal.rate == pytest.approx(in_kj.rate, rel=1.0e-12)


def test_rpmd_rate_takes_its_factors_from_a_transmission_result() -> None:
    grid, surface = _harmonic_surface(5000.0, -0.1, 20.0)
    transmission = nqe.TransmissionResult(
        times_ps=np.array([0.0, 1.0]),
        kappa_t=np.array([0.1, 0.6]),
        kappa_stderr=np.array([0.0, 0.05]),
        plateau=0.6,
        plateau_stderr=0.05,
        forward_flux=3.0,
        s0_mean=0.0,
        s0_std=0.01,
        n_parents=10,
        n_parents_excluded=0,
        n_children=100,
    )

    rate = nqe.rpmd_rate(
        transmission,
        grid,
        surface,
        temperature=300.0,
        s_dagger=0.02,
        reactant_window=(-0.185, -0.015),
    )
    assert rate.transmission == 0.6
    assert rate.transmission_stderr == 0.05
    assert rate.forward_flux == 3.0
    assert rate.rate == pytest.approx(0.6 * rate.qtst_rate)

    with pytest.raises(ValueError, match="ambiguous"):
        nqe.rpmd_rate(
            transmission, grid, surface, temperature=300.0, s_dagger=0.02,
            reactant_window=(-0.185, -0.015), forward_flux=1.0,
        )
    with pytest.raises(ValueError, match="needs forward_flux"):
        nqe.rpmd_rate(
            0.5, grid, surface, temperature=300.0, s_dagger=0.02,
            reactant_window=(-0.185, -0.015),
        )


def test_rpmd_rate_warns_when_the_surface_out_tops_the_dividing_point() -> None:
    grid = np.linspace(-0.2, 0.05, 501)
    surface = np.zeros_like(grid)
    surface[grid >= 0.0] = 20.0
    bump = (grid > -0.06) & (grid < -0.04)
    surface[bump] = 30.0

    with pytest.warns(UserWarning, match="probably not the barrier top"):
        nqe.rpmd_rate(
            1.0, grid, surface, temperature=300.0, s_dagger=0.02,
            reactant_window=(-0.19, -0.11), forward_flux=1.0,
        )


def test_rpmd_rate_input_validation() -> None:
    grid, surface = _harmonic_surface(5000.0, -0.1, 20.0)
    kwargs: dict[str, Any] = {
        "temperature": 300.0,
        "s_dagger": 0.02,
        "reactant_window": (-0.185, -0.015),
        "forward_flux": 1.0,
    }

    with pytest.raises(ValueError, match="strictly increasing"):
        nqe.rpmd_rate(1.0, grid[::-1], surface, **kwargs)
    with pytest.raises(ValueError, match="expected"):
        nqe.rpmd_rate(1.0, grid, surface[:-1], **kwargs)
    with pytest.raises(ValueError, match="outside the grid"):
        nqe.rpmd_rate(
            1.0, grid, surface, temperature=300.0, s_dagger=0.3,
            reactant_window=(-0.185, -0.015), forward_flux=1.0,
        )
    with pytest.raises(ValueError, match="must not contain s_dagger"):
        nqe.rpmd_rate(
            1.0, grid, surface, temperature=300.0, s_dagger=0.02,
            reactant_window=(-0.185, 0.03), forward_flux=1.0,
        )
    with pytest.raises(ValueError, match="low < high"):
        nqe.rpmd_rate(
            1.0, grid, surface, temperature=300.0, s_dagger=0.02,
            reactant_window=(-0.015, -0.185), forward_flux=1.0,
        )
    with pytest.raises(ValueError, match="fewer than two grid points"):
        nqe.rpmd_rate(
            1.0, grid, surface, temperature=300.0, s_dagger=0.02,
            reactant_window=(-0.10005, -0.09995), forward_flux=1.0,
        )
    with pytest.raises(ValueError, match="forward_flux must be positive"):
        nqe.rpmd_rate(
            1.0, grid, surface, temperature=300.0, s_dagger=0.02,
            reactant_window=(-0.185, -0.015), forward_flux=0.0,
        )
    with pytest.raises(ValueError, match="transmission_stderr"):
        nqe.rpmd_rate(
            1.0, grid, surface, temperature=300.0, s_dagger=0.02,
            reactant_window=(-0.185, -0.015), forward_flux=1.0,
            transmission_stderr=-0.1,
        )
    with pytest.raises(ValueError, match="energy_unit"):
        nqe.rpmd_rate(1.0, grid, surface, energy_unit="hartree", **kwargs)

    gapped = surface.copy()
    gapped[(grid > -0.1) & (grid < -0.05)] = float("inf")
    with pytest.raises(ValueError, match="inside reactant_window"):
        nqe.rpmd_rate(1.0, grid, gapped, **kwargs)
    unsampled = surface.copy()
    unsampled[grid > 0.0] = float("inf")
    with pytest.raises(ValueError, match="never sampled"):
        nqe.rpmd_rate(1.0, grid, unsampled, **kwargs)


# --------------------------------------------------------------------------
# Seed scheme and flux finite difference


def test_child_seed_streams_are_deterministic_distinct_and_stage_disjoint() -> None:
    def states(seqs: list[list[np.random.SeedSequence]]) -> list[tuple[int, ...]]:
        return [
            tuple(child.generate_state(4).tolist())
            for parent in seqs
            for child in parent
        ]

    first = states(nqe_rates._spawn_child_sequences(77, 3, 2))
    assert first == states(nqe_rates._spawn_child_sequences(77, 3, 2))
    assert len(set(first)) == len(first)
    assert states(nqe_rates._spawn_child_sequences(78, 3, 2)) != first

    # The regression this scheme exists for: a user reusing one master seed
    # must not correlate shooting momenta with any stage stream.
    stage_streams = np.random.SeedSequence(77).spawn(
        len(nqe_openmm._SEED_STREAMS)
    )
    stage_states = {
        tuple(stream.generate_state(4).tolist()) for stream in stage_streams
    }
    assert stage_states.isdisjoint(first)


def test_initial_flux_is_exact_for_a_linear_cv() -> None:
    positions = np.array([[[0.2, 0.0, 0.0]], [[0.4, 0.0, 0.0]]])
    velocities = np.array([[[1.5, 0.0, 0.0]], [[0.5, -1.0, 2.0]]])

    s0, flux = nqe_rates._initial_cv_and_flux(
        _x_coordinate, "centroid", positions, velocities, None, 1.0e-4, 0, 0,
    )
    assert s0 == pytest.approx(0.3)
    assert flux == pytest.approx(1.0)

    s0, flux = nqe_rates._initial_cv_and_flux(
        _x_coordinate, "bead-mean", positions, velocities, None, 1.0e-4, 0, 0,
    )
    assert s0 == pytest.approx(0.3)
    assert flux == pytest.approx(1.0)


def test_initial_flux_error_shrinks_quadratically_with_epsilon() -> None:
    # For s = x^3 the central difference carries an epsilon^2 v^3 error term,
    # so shrinking epsilon tenfold shrinks the error a hundredfold.
    positions = np.array([[[0.5, 0.0, 0.0]]])
    velocities = np.array([[[2.0, 0.0, 0.0]]])
    exact = 3.0 * 0.5**2 * 2.0

    def error(epsilon: float) -> float:
        _, flux = nqe_rates._initial_cv_and_flux(
            lambda p: float(p[0, 0]) ** 3, "centroid",
            positions, velocities, None, epsilon, 0, 0,
        )
        return abs(flux - exact)

    assert error(1.0e-3) == pytest.approx(1.0e-6 * 2.0**3, rel=1.0e-4)
    assert error(1.0e-2) / error(1.0e-3) == pytest.approx(100.0, rel=1.0e-3)


def test_cv_evaluation_errors_carry_their_context() -> None:
    positions = np.zeros((1, 3))

    with pytest.raises(ValueError, match=r"non-scalar .* parent 3, child 1"):
        nqe_rates._evaluate_cv(
            lambda p: np.array([1.0, 2.0]), positions, 3, 1, 0,
        )
    with pytest.raises(ValueError, match="non-finite value at parent 0"):
        nqe_rates._evaluate_cv(lambda p: float("nan"), positions, 0, 2, 5)


# --------------------------------------------------------------------------
# Real dynamics on the Reference platform


def test_classical_shots_from_the_barrier_top_never_recross() -> None:
    # A 1-D conservative classical particle (one bead) launched from the
    # barrier top of a double well commits to whichever side its velocity
    # points at and never returns, so the final-time transmission
    # coefficient is exactly one: numerator and denominator are literally
    # the same sum.
    paths = _write_barrier_snapshots(1, 6)
    logs = _shoot(paths, prefix="classical", n_beads=1, seed=7,
                  n_children=4, n_steps=40)

    result = nqe.transmission_coefficient(
        logs, s_dagger=BARRIER_X, blocks=3,
    )

    assert result.kappa_t[-1] == pytest.approx(1.0, abs=1.0e-12)
    assert result.forward_flux > 0.0
    assert result.s0_std < 1.0e-3
    assert result.n_children == 24


def test_ring_polymer_recrossing_produces_a_physical_plateau() -> None:
    paths = _write_barrier_snapshots(2, 6, prefix="ring")
    logs = _shoot(paths, prefix="ring", n_beads=2, seed=9,
                  n_children=3, n_steps=40)

    result = nqe.transmission_coefficient(logs, s_dagger=BARRIER_X, blocks=3)

    assert np.isfinite(result.kappa_t).all()
    assert 0.0 < result.plateau <= 1.0 + 1.0e-9
    assert result.forward_flux > 0.0


def test_recrossing_is_deterministic_and_resumable() -> None:
    paths = _write_barrier_snapshots(2, 4, prefix="det")

    def digests(prefix: str, seed: int, resume: bool = False) -> list[bytes]:
        logs = _shoot(paths, prefix=prefix, n_beads=2, seed=seed,
                      n_children=2, n_steps=20, resume=resume)
        return [Path(log).read_bytes() for log in logs]

    seeded = digests("seeded", 11)
    assert digests("repeated", 11) == seeded
    assert digests("reseeded", 12) != seeded

    # Interrupt the campaign: one parent log missing, one truncated. Resume
    # must redo exactly those and reproduce the uninterrupted bytes.
    Path("seeded_recrossing_00001.log").unlink()
    truncated = Path("seeded_recrossing_00002.log")
    truncated.write_text(
        "".join(truncated.read_text().splitlines(keepends=True)[:4])
    )
    assert digests("seeded", 11, resume=True) == seeded


def test_recrossing_records_a_conserved_ring_polymer_energy() -> None:
    paths = _write_barrier_snapshots(2, 2, prefix="energy")
    logs = _shoot(paths, prefix="energy", n_beads=2, seed=3,
                  n_children=2, n_steps=20, record_energy=True)

    header, values = nqe_rates._read_reporter_log(logs[0], "recrossing log")

    assert header[-1] == "E_ring(kJ/mol)"
    energies = values[:, -1]
    assert np.isfinite(energies).all()
    # Thermostat-off dynamics: within one child the ring-polymer Hamiltonian
    # only jitters with integration error.
    for child in (0, 1):
        child_energies = energies[values[:, 3] == child]
        assert np.ptp(child_energies) < 0.5


def test_harvested_snapshots_feed_the_recrossing_driver(
    one_particle_system: tuple[app.Modeller, Any],
) -> None:
    # The end-to-end seam: equilibrate, harvest snapshots from production,
    # shoot from them, and analyse. The harmonic well has no barrier, so
    # only the plumbing is under test, not the physics.
    modeller, forcefield = one_particle_system
    prepared = nqe.PreparedSystem(forcefield.createSystem(modeller.topology))

    nqe.run_openmm_rpmd_equilibration(
        modeller, prepared, n_beads=2, n_1=2, n_2=3, n_report=50,
        platform_name="Reference", seed=7,
    )
    nqe.run_openmm_rpmd_prod(
        modeller, prepared, checkpoint_file="rpmd_ready.chk", n_beads=2,
        steps=12, n_report=50, barostat_freq=None, snapshot_interval=4,
        platform_name="Reference", seed=7,
    )
    snapshots = sorted(nqe.list_files_with_pattern(".", "rpmd_prod_snapshot_*.npz"))
    assert len(snapshots) == 3

    # seed=2 gives every parent at least one forward-moving child; seed=1
    # happens to draw six backward launches for one parent, which the
    # analysis rightly refuses to block on.
    nqe.run_openmm_rpmd_recrossing(
        modeller, prepared, snapshots, _x_coordinate,
        output_prefix="well", n_beads=2, n_children=6, n_steps=10,
        record_interval=5, platform_name="Reference", seed=2,
    )
    result = nqe.transmission_coefficient(
        [f"well_recrossing_{index:05d}.log" for index in range(3)],
        s_dagger=0.0, blocks=2, plateau_fraction=1.0,
    )

    assert result.n_parents == 3
    assert result.n_children == 18
    assert np.isfinite(result.kappa_t).all()


def test_plot_transmission_coefficient_returns_figure_and_axis() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    _write_recrossing_log("p_00000.log", 0, [(0, 1.0, [0.1, 0.1, 0.1, 0.1])])
    _write_recrossing_log("p_00001.log", 1, [(0, 1.0, [0.1, 0.1, 0.1, 0.1])])

    figure, axis = nqe.plot_transmission_coefficient(
        ["p_00000.log", "p_00001.log"], s_dagger=0.0, blocks=2,
        plateau_fraction=0.5, filename="kappa.png",
    )

    assert Path("kappa.png").exists()
    assert axis.get_ylabel() == r"$\kappa(t)$"
    plt.close(figure)


# --------------------------------------------------------------------------
# Driver validation


def test_recrossing_driver_validates_before_touching_openmm() -> None:
    modeller = app.Modeller(
        _double_well_topology(),
        [Vec3(BARRIER_X, 0.0, 0.0)] * unit.nanometer,
    )
    Path("stub.npz").write_bytes(b"")

    with pytest.raises(ValueError, match="must not be empty"):
        nqe.run_openmm_rpmd_recrossing(modeller, object(), [], _x_coordinate)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        nqe.run_openmm_rpmd_recrossing(
            modeller, object(), ["missing.npz"], _x_coordinate,
        )
    with pytest.raises(ValueError, match="multiple of"):
        nqe.run_openmm_rpmd_recrossing(
            modeller, object(), ["stub.npz"], _x_coordinate,
            n_steps=10, record_interval=3,
        )
    with pytest.raises(TypeError, match="cv must be a callable"):
        nqe.run_openmm_rpmd_recrossing(
            modeller, object(), ["stub.npz"], "not-a-function",
        )
    with pytest.raises(ValueError, match="cv_mode"):
        nqe.run_openmm_rpmd_recrossing(
            modeller, object(), ["stub.npz"], _x_coordinate,
            cv_mode="beads",
        )
    with pytest.raises(ValueError, match="seed"):
        nqe.run_openmm_rpmd_recrossing(
            modeller, object(), ["stub.npz"], _x_coordinate, seed=-1,
        )
    with pytest.raises(ValueError, match="n_children"):
        nqe.run_openmm_rpmd_recrossing(
            modeller, object(), ["stub.npz"], _x_coordinate, n_children=0,
        )
    with pytest.raises(ValueError, match="flux_epsilon"):
        nqe.run_openmm_rpmd_recrossing(
            modeller, object(), ["stub.npz"], _x_coordinate,
            flux_epsilon=0.0,
        )


def test_recrossing_driver_rejects_snapshots_from_another_system() -> None:
    # The restart loader's identity gates are the KIE safety net: a
    # deuterated shooting system must refuse protiated snapshots.
    paths = _write_barrier_snapshots(2, 1, prefix="mismatch")

    with pytest.raises(ValueError, match="n_beads=3"):
        _shoot(paths, prefix="mismatch", n_beads=3, seed=1)