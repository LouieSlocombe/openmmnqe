"""
GPU-only guards for the RPMD reporters on a mixed ML/MM System.

``MLPotential.createMixedSystem`` attaches the ML model as an
``openmm.PythonForce``, so every force evaluation re-enters Python from the
platform's own threads. Both cases share a subprocess under a timeout, so a
report path that blocks on the GIL fails the suite instead of hanging it,
without paying for Python and CUDA startup twice.
``tests/test_reporters.py`` carries the platform-independent half of the
guard: a test double whose ``getTotalEnergy`` raises.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import openmm

#: Generous next to the few seconds the working path needs, and finite next
#: to the deadlock, which never returns.
_TIMEOUT_SECONDS = 180


def _cuda_is_usable() -> bool:
    """Whether a CUDA Context can actually be built on this machine."""
    try:
        platform = openmm.Platform.getPlatform("CUDA")
    except Exception:
        return False
    system = openmm.System()
    system.addParticle(1.0 * unit.dalton)
    try:
        openmm.Context(
            system,
            openmm.VerletIntegrator(0.001 * unit.picoseconds),
            platform,
        )
    except Exception:
        return False
    return True


_PREAMBLE = '''
import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3, openmm

from openmmnqe import step_rpmd
from openmmnqe.reporters import (
    RPMDBeadReporter,
    RPMDCentroidReporter,
    RPMDThermodynamicReporter,
)

N_BEADS = 4


def compute(state):
    """Stand in for the ML model: a PythonForce, which is what matters."""
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return 0.0, np.zeros_like(positions)


def build():
    """A small solvated-looking System whose one molecule has no bonds."""
    system = openmm.System()
    topology = app.Topology()
    chain = topology.addChain()
    positions = []

    # The "ML region": a molecule the Topology knows and the System does not,
    # exactly as createMixedSystem leaves it.
    residue = topology.addResidue("MOL", chain)
    previous = None
    for index in range(4):
        atom = topology.addAtom(f"C{index + 1}", app.element.carbon, residue)
        if previous is not None:
            topology.addBond(previous, atom)
        previous = atom
        system.addParticle(12.011 * unit.dalton)
        positions.append([-0.1 + 0.15 * index, 1.0, 1.0])

    bonds = openmm.HarmonicBondForce()
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(0.6 * unit.nanometer)
    for _ in range(4):
        nonbonded.addParticle(
            0.0, 0.3 * unit.nanometer, 0.2 * unit.kilojoule_per_mole
        )

    # Placed on a grid rather than at random: an overlapping start would
    # blow the box up and leave the log full of nonsense to assert against.
    sites = [
        np.array([0.25 + 0.35 * i, 0.25 + 0.35 * j, 0.25 + 0.35 * k])
        for i in range(5) for j in range(5) for k in range(5)
    ]
    sites = [site for site in sites if abs(site[1] - 1.0) > 0.2][:60]
    for base in sites:
        water = topology.addResidue("HOH", chain)
        oxygen = topology.addAtom("O", app.element.oxygen, water)
        hydrogen = topology.addAtom("H1", app.element.hydrogen, water)
        topology.addBond(oxygen, hydrogen)
        first = system.addParticle(15.999 * unit.dalton)
        system.addParticle(1.008 * unit.dalton)
        bonds.addBond(
            first, first + 1, 0.1 * unit.nanometer,
            100000.0 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        nonbonded.addParticle(
            0.0, 0.3 * unit.nanometer, 0.4 * unit.kilojoule_per_mole
        )
        nonbonded.addParticle(
            0.0, 0.2 * unit.nanometer, 0.2 * unit.kilojoule_per_mole
        )
        nonbonded.addException(first, first + 1, 0.0, 0.3 * unit.nanometer, 0.0)
        positions.append(base.tolist())
        positions.append((base + np.array([0.1, 0.0, 0.0])).tolist())

    system.addForce(bonds)
    system.addForce(nonbonded)

    python_force = openmm.PythonForce(compute)
    python_force.setUsesPeriodicBoundaryConditions(True)
    system.addForce(python_force)

    box = [Vec3(2.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0), Vec3(0.0, 0.0, 2.0)]
    box = box * unit.nanometer
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(box)

    integrator = openmm.RPMDIntegrator(
        N_BEADS, 300 * unit.kelvin, 1.0 / unit.picosecond,
        0.0002 * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(7)
    simulation = app.Simulation(
        topology, system, integrator,
        openmm.Platform.getPlatform("CUDA"),
    )
    simulation.context.setPositions(np.array(positions) * unit.nanometer)
    for bead in range(N_BEADS):
        integrator.setPositions(bead, np.array(positions) * unit.nanometer)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 3)
    return simulation
'''


def _run(tmp_path: Path, body: str) -> None:
    """Run *body* after the preamble in a subprocess, failing on a hang."""
    script = tmp_path / "probe.py"
    script.write_text(_PREAMBLE + textwrap.dedent(body))
    try:
        finished = subprocess.run(
            [sys.executable, "-u", str(script)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            cwd=tmp_path,
        )
    except subprocess.TimeoutExpired as expired:
        raise AssertionError(
            f"the RPMD reporters hung on CUDA for {_TIMEOUT_SECONDS} s; "
            "RPMDIntegrator.getTotalEnergy() has most likely come back into "
            f"the report path.\nstdout:\n{expired.stdout!r}"
        ) from None
    assert finished.returncode == 0, (
        f"stdout:\n{finished.stdout}\nstderr:\n{finished.stderr}"
    )


@pytest.fixture(scope="module")
def gpu_reporter_outputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Exercise both report paths once, each with a fresh CUDA simulation.

    CUDA is checked only when a GPU test runs, so collection and deselection
    do not create a Context. Selecting either test executes both probes.
    """
    if not _cuda_is_usable():
        pytest.skip("needs a working CUDA platform")
    output_dir = tmp_path_factory.mktemp("rpmd-gpu")
    _run(output_dir, """
        print("direct thermodynamic reporting")
        simulation = build()
        step_rpmd(simulation, 5)
        with RPMDThermodynamicReporter(
            file="direct_thermo.log", reportInterval=1,
        ) as reporter:
            reporter.report(simulation, state=None)
            reporter.report(simulation, state=None)
        del simulation

        print("scheduled bead, centroid and thermodynamic reporting")
        simulation = build()
        simulation.reporters.append(RPMDCentroidReporter(
            file_name="centroid.pdb", reportInterval=2,
            num_beads=N_BEADS, topology=simulation.topology,
        ))
        simulation.reporters.append(RPMDBeadReporter(
            file_base_name="beads", reportInterval=2,
            num_beads=N_BEADS, topology=simulation.topology,
        ))
        simulation.reporters.append(RPMDThermodynamicReporter(
            file="scheduled_thermo.log", reportInterval=2,
        ))
        step_rpmd(simulation, 4)
        for reporter in simulation.reporters:
            reporter.close()
    """)
    return output_dir


def test_thermodynamic_reporter_reports_on_a_python_force_system(
    gpu_reporter_outputs: Path,
) -> None:
    # One report falling due on a mixed ML/MM system must not stop the run.
    rows = [
        row.split("\t")
        for row in (gpu_reporter_outputs / "direct_thermo.log").read_text().splitlines()
    ]
    assert len(rows) == 3, rows
    # One step column plus the nine estimators, all of them finite.
    assert all(len(row) == 10 for row in rows), rows
    values = np.array([[float(cell) for cell in row] for row in rows[1:]])
    assert np.isfinite(values).all(), values


def test_every_rpmd_reporter_survives_a_scheduled_report(
    gpu_reporter_outputs: Path,
) -> None:
    # The whole attached set, driven through step_rpmd so the reports are
    # scheduled the way a driver schedules them rather than called by hand.
    assert len(
        (gpu_reporter_outputs / "scheduled_thermo.log").read_text().splitlines()
    ) == 3

    # The ML-region molecule must survive the round trip intact: its
    # bonds are absent from the System, so OpenMM's own wrapping would
    # have scattered its four atoms across the box.
    written = app.PDBFile(str(gpu_reporter_outputs / "beads_bead_0.pdb")).getPositions(
        asNumpy=True,
    ).value_in_unit(unit.nanometer)[:4]
    span = float(np.ptp(written[:, 0]))
    assert span < 1.0, span
