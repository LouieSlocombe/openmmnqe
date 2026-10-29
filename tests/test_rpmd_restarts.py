"""Continue completed RPMD production runs from their full-bead archives."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest

import openmmnqe as nqe


@pytest.fixture(params=["full", "contracted"])
def production(request: pytest.FixtureRequest) -> Callable[..., None]:
    """Run either production driver with the same small Reference setup."""
    if request.param == "full":
        driver = nqe.run_openmm_rpmd_prod
        options: dict[str, Any] = {"time_step": 0.2 * unit.femtosecond}
    else:
        driver = nqe.run_openmm_rpmd_contracted
        options = {
            "timestep": 0.2 * unit.femtosecond,
            "contractions": {0: 1},
        }

    def run(modeller: app.Modeller, forcefield: Any, **kwargs: Any) -> None:
        driver(
            modeller,
            forcefield,
            n_beads=4,
            n_report=2,
            barostat_freq=None,
            platform_name="Reference",
            seed=19,
            **options,
            **kwargs,
        )

    return run


def _equilibrate(modeller: app.Modeller, forcefield: Any) -> None:
    nqe.run_openmm_rpmd_equilibration(
        modeller,
        forcefield,
        n_beads=4,
        n_1=2,
        n_2=3,
        timestep=0.1 * unit.femtosecond,
        n_report=2,
        platform_name="Reference",
        seed=7,
    )


def _read_checkpoint(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_completed_production_restores_every_bead_from_a_fresh_modeller(
    one_particle_system: tuple[app.Modeller, Any],
    production: Callable[..., None],
) -> None:
    modeller, forcefield = one_particle_system
    _equilibrate(modeller, forcefield)
    production(modeller, forcefield, output_prefix="first", steps=7)
    saved = _read_checkpoint("first.chk")
    previous_outputs = {
        path: path.read_bytes() for path in Path.cwd().glob("first*")
    }

    # Recreate the Modeller as a separate process would. Its PDB carries only
    # the centroid, so the checkpoint must supply every bead and velocity.
    pdb = app.PDBFile("first_final.pdb")
    fresh_modeller = app.Modeller(pdb.topology, pdb.positions)
    production(
        fresh_modeller,
        forcefield,
        checkpoint_file="first.chk",
        output_prefix="restored",
        steps=0,
    )
    restored = _read_checkpoint("restored.chk")
    for name, values in saved.items():
        np.testing.assert_array_equal(restored[name], values)
    assert np.ptp(restored["positions_nm"], axis=0).max() > 0.0
    assert np.ptp(restored["velocities_nm_per_ps"], axis=0).max() > 0.0

    production(
        fresh_modeller,
        forcefield,
        checkpoint_file="first.chk",
        output_prefix="continued",
        steps=3,
    )
    continued = _read_checkpoint("continued.chk")
    assert saved["step_count"].item() == 12
    assert continued["step_count"].item() == 15
    assert continued["time_ps"].item() == pytest.approx(
        saved["time_ps"].item() + 3 * 0.0002
    )
    for path, contents in previous_outputs.items():
        assert path.read_bytes() == contents


def test_split_nve_production_matches_uninterrupted_dynamics(
    one_particle_system: tuple[app.Modeller, Any],
    production: Callable[..., None],
) -> None:
    modeller, forcefield = one_particle_system
    _equilibrate(modeller, forcefield)
    production(
        modeller,
        forcefield,
        output_prefix="uninterrupted",
        steps=14,
        apply_thermostat=False,
    )
    production(
        modeller,
        forcefield,
        output_prefix="segment_one",
        steps=5,
        apply_thermostat=False,
    )
    production(
        modeller,
        forcefield,
        checkpoint_file="segment_one.chk",
        output_prefix="segment_two",
        steps=9,
        apply_thermostat=False,
    )

    uninterrupted = _read_checkpoint("uninterrupted.chk")
    restarted = _read_checkpoint("segment_two.chk")
    assert restarted["step_count"].item() == 19
    assert restarted["step_count"] == uninterrupted["step_count"]
    for name in (
        "positions_nm",
        "velocities_nm_per_ps",
        "box_vectors_nm",
        "time_ps",
    ):
        np.testing.assert_allclose(
            restarted[name], uninterrupted[name], rtol=1e-11, atol=1e-12
        )
