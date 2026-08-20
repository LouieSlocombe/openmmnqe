"""CPU regression tests for continuity between the two adQTB stages."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import openmm

import openmmnqe as nqe


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
