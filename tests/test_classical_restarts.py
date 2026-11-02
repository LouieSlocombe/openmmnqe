"""Classical production checkpoint restoration and continuation on Reference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import OpenMMException, Vec3, openmm

import openmmnqe as nqe


def _produce(modeller: app.Modeller, forcefield: Any, **kwargs: Any) -> None:
    nqe.run_openmm_prod(
        modeller,
        forcefield,
        barostat_freq=None,
        n_report=2,
        platform_name="Reference",
        seed=7,
        **kwargs,
    )


def _read_state(
    modeller: app.Modeller, forcefield: Any, checkpoint: str,
) -> openmm.State:
    simulation = app.Simulation(
        modeller.topology,
        forcefield.createSystem(modeller.topology),
        openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin,
            1.0 / unit.picosecond,
            1.0 * unit.femtosecond,
        ),
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.loadCheckpoint(checkpoint)
    return simulation.context.getState(getPositions=True, getVelocities=True)


def _assert_same_state(expected: openmm.State, actual: openmm.State) -> None:
    assert actual.getStepCount() == expected.getStepCount()
    assert actual.getTime() == expected.getTime()
    for getter, units in (
        ("getPositions", unit.nanometer),
        ("getVelocities", unit.nanometer / unit.picosecond),
        ("getPeriodicBoxVectors", unit.nanometer),
    ):
        np.testing.assert_array_equal(
            getattr(actual, getter)(asNumpy=True).value_in_unit(units),
            getattr(expected, getter)(asNumpy=True).value_in_unit(units),
        )


def test_zero_step_restart_restores_state_before_writing_binary_topology(
    one_particle_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = one_particle_system
    _produce(modeller, forcefield, output_prefix="first", steps=5)
    expected = _read_state(modeller, forcefield, "first.chk")
    previous_outputs = {
        path: path.read_bytes() for path in Path.cwd().glob("first*")
    }

    # Positions supplied by a newly constructed Modeller must not replace
    # the full-precision positions carried in the checkpoint.
    fresh_modeller = app.Modeller(
        modeller.topology, [Vec3(0.8, 0.9, 1.0)] * unit.nanometer,
    )
    _produce(
        fresh_modeller,
        forcefield,
        checkpoint_file="first.chk",
        output_prefix="restored",
        steps=0,
        trajectory="dcd",
    )

    restored = _read_state(modeller, forcefield, "restored.chk")
    _assert_same_state(expected, restored)
    topology = app.PDBFile("restored_topology.pdb")
    np.testing.assert_allclose(
        topology.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        expected.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        atol=5.1e-5,  # PDB coordinates have three decimal places in angstroms.
        rtol=0.0,
    )
    for path, contents in previous_outputs.items():
        assert path.read_bytes() == contents


def test_restart_runs_additional_steps_and_continues_report_clocks(
    one_particle_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = one_particle_system
    _produce(modeller, forcefield, output_prefix="first", steps=5)
    _produce(
        modeller,
        forcefield,
        checkpoint_file="first.chk",
        output_prefix="continued",
        steps=3,
        velocity_record_interval=2,
    )

    state = _read_state(modeller, forcefield, "continued.chk")
    assert state.getStepCount() == 8
    assert state.getTime().value_in_unit(unit.picosecond) == pytest.approx(0.008)
    log = np.loadtxt("continued.log", delimiter=",", ndmin=2)
    np.testing.assert_array_equal(log[:, 0], [6, 8])
    np.testing.assert_allclose(log[:, 1], [0.006, 0.008])
    with np.load("continued_velocities.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["times_ps"], [0.006, 0.008])
    assert app.PDBFile("continued_steps.pdb").getNumFrames() == 2


def test_split_langevin_production_matches_uninterrupted_continuation(
    one_particle_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = one_particle_system
    _produce(modeller, forcefield, output_prefix="first", steps=5)
    _produce(
        modeller, forcefield, checkpoint_file="first.chk",
        output_prefix="uninterrupted", steps=9,
    )
    _produce(
        modeller, forcefield, checkpoint_file="first.chk",
        output_prefix="split", steps=4,
    )
    _produce(
        modeller, forcefield, checkpoint_file="split.chk",
        output_prefix="continued", steps=5,
    )

    expected = _read_state(modeller, forcefield, "uninterrupted.chk")
    actual = _read_state(modeller, forcefield, "continued.chk")
    assert actual.getStepCount() == 14
    _assert_same_state(expected, actual)


@pytest.mark.parametrize("checkpoint_kind", ["missing", "incompatible"])
def test_unusable_checkpoint_does_not_overwrite_outputs(
    one_particle_system: tuple[app.Modeller, Any],
    checkpoint_kind: str,
) -> None:
    modeller, forcefield = one_particle_system
    checkpoint = "unusable.chk"
    if checkpoint_kind == "incompatible":
        system = openmm.System()
        system.addParticle(39.9 * unit.dalton)
        system.addParticle(39.9 * unit.dalton)
        integrator = openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond,
        )
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference"),
        )
        context.setPositions(
            [Vec3(0.1, 0.0, 0.0), Vec3(0.2, 0.0, 0.0)] * unit.nanometer,
        )
        Path(checkpoint).write_bytes(context.createCheckpoint())
        error = OpenMMException
        message = "number of particles"
    else:
        error = FileNotFoundError
        message = "not found"

    previous_outputs = {
        Path(f"continued{suffix}"): f"existing {suffix}".encode()
        for suffix in (".log", "_steps.pdb", ".chk", ".pdb")
    }
    for path, contents in previous_outputs.items():
        path.write_bytes(contents)

    with pytest.raises(error, match=message):
        _produce(
            modeller, forcefield, checkpoint_file=checkpoint,
            output_prefix="continued", steps=3,
        )

    for path, contents in previous_outputs.items():
        assert path.read_bytes() == contents
