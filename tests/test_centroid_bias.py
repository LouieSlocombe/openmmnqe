"""The PLUMED bias has to act on the ring-polymer centroid, not on each bead.

``RPMDIntegrator`` evaluates a force group on every copy unless its
contractions map says otherwise, and ``PlumedForce`` knows nothing about ring
polymers -- so a bias merely added to the System is applied once per bead, at
each bead's own coordinates. What comes out of that is not the centroid
potential of mean force, which is the quantity a ring polymer exists to give.

The behaviour is checked by propagation rather than by inspecting the
contractions map, because ``getState(copy)`` evaluates every force at that
copy regardless of contraction and so cannot see the difference.

The set-up makes the answer unambiguous: two beads at x = +0.3 and -0.3 nm put
the centroid exactly at 0, and the bias is a harmonic restraint on x centred
at 0. Evaluated on the centroid it is exactly zero, so the run must be
*identical* to one with no bias at all; evaluated per bead it is -k*x on each,
which pulls them together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm

from openmmnqe.openmm import _centroid_bias_contraction, _load_plumed

KAPPA = 5_000.0
BEADS = 2
STEPS = 1_000
START = (0.3, -0.3)

Mode = Literal["no bias", "centroid", "per bead"]


def _propagate(mode: Mode, script: Path) -> np.ndarray:
    """Bead x positions after a deterministic microcanonical run."""
    system = openmm.System()
    system.addParticle(1.008 * unit.dalton)
    bias = _load_plumed(system, None if mode == "no bias" else str(script))
    contractions = _centroid_bias_contraction(
        system, bias, None, mode == "centroid"
    )
    settings = (BEADS, 300 * unit.kelvin, 0.0 / unit.picosecond, 0.2 * unit.femtoseconds)
    integrator = (
        openmm.RPMDIntegrator(*settings, contractions)
        if contractions
        else openmm.RPMDIntegrator(*settings)
    )
    integrator.setApplyThermostat(False)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatform("Reference")
    )
    context.setPositions([Vec3(0.0, 0.0, 0.0)] * unit.nanometer)
    for copy, x in enumerate(START):
        integrator.setPositions(copy, [Vec3(x, 0.0, 0.0)] * unit.nanometer)
        integrator.setVelocities(
            copy, [Vec3(0.0, 0.0, 0.0)] * unit.nanometer / unit.picosecond
        )
    integrator.step(STEPS)
    return np.array(
        [
            integrator.getState(copy, getPositions=True)
            .getPositions()
            .value_in_unit(unit.nanometer)[0]
            .x
            for copy in range(BEADS)
        ]
    )


@pytest.fixture
def restraint(tmp_path: Path) -> Path:
    path = tmp_path / "probe.dat"
    path.write_text(f"p: POSITION ATOM=1\nRESTRAINT ARG=p.x AT=0.0 KAPPA={KAPPA}\n")
    return path


def test_a_centroid_bias_that_is_zero_changes_nothing(restraint: Path) -> None:
    """The restraint is centred on the centroid, so it must do nothing at all.

    Exact equality is the point: anything else means some of the bias reached
    a bead directly.
    """
    unbiased = _propagate("no bias", restraint)
    centroid = _propagate("centroid", restraint)
    assert centroid == pytest.approx(unbiased, abs=1e-12)


def test_a_per_bead_bias_pulls_the_beads_together(restraint: Path) -> None:
    """What the default did before the contraction was added.

    Each bead feels -k times its *own* displacement, so over this many steps
    the two swing through each other -- a change of order the bead separation
    itself, not a rounding difference.
    """
    with pytest.warns(RuntimeWarning, match="not a centroid free energy"):
        per_bead = _propagate("per bead", restraint)
    unbiased = _propagate("no bias", restraint)
    assert np.abs(per_bead - unbiased).max() > 0.5
    # they have crossed: the bead that started positive is now negative
    assert per_bead[0] < 0.0 < per_bead[1]


def test_the_bias_gets_a_force_group_no_other_force_is_using(
    restraint: Path,
) -> None:
    """Contraction is keyed on the group, and group 0 is the bonded forces'."""
    system = openmm.System()
    system.addParticle(1.008 * unit.dalton)
    occupied = openmm.CustomExternalForce("0.0")
    occupied.setForceGroup(0)
    system.addForce(occupied)
    bias = _load_plumed(system, str(restraint))
    contractions = _centroid_bias_contraction(system, bias, None, True)
    assert contractions is not None
    assert bias.getForceGroup() != 0
    assert contractions == {bias.getForceGroup(): 1}


def test_contractions_the_caller_asked_for_are_kept(restraint: Path) -> None:
    """The contracted stage sorts its own forces into groups first."""
    system = openmm.System()
    system.addParticle(1.008 * unit.dalton)
    bias = _load_plumed(system, str(restraint))
    contractions = _centroid_bias_contraction(system, bias, {1: 4, 2: 8}, True)
    assert contractions is not None
    assert contractions[1] == 4
    assert contractions[2] == 8
    assert contractions[bias.getForceGroup()] == 1


def test_an_unbiased_run_asks_for_no_contraction(restraint: Path) -> None:
    system = openmm.System()
    system.addParticle(1.008 * unit.dalton)
    assert _centroid_bias_contraction(system, None, None, True) is None
