"""Behavioral guards for shared restraints, output setup and adQTB cadence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmm.unit as unit
import pytest
from openmm import Vec3, app, openmm

from openmmnqe import adqtb
from openmmnqe import openmm as stages


def _topology() -> app.Topology:
    topology = app.Topology()
    residue = topology.addResidue("MOL", topology.addChain())
    for name in ("CA", "SIDE", "N"):
        topology.addAtom(name, app.element.carbon, residue)
    return topology


@pytest.mark.parametrize("position_unit", [unit.nanometer, unit.angstrom])
@pytest.mark.parametrize("box_size", [2.0, 10.0])
@pytest.mark.parametrize("names, selected", [(None, [0, 2]), (["SIDE"], [1]), ([], [])])
def test_backbone_restraint_matches_periodic_quadratic_energy_and_force(
    position_unit: Any, box_size: float, names: list[str] | None, selected: list[int],
) -> None:
    references = np.array([[0.05, 0.1, 0.15], [0.8, 0.4, 0.1], [0.9, 0.2, 0.3]])
    positions = np.array([[1.95, 0.14, 0.12], [0.75, 0.5, 0.2], [1.0, 0.18, 0.35]])
    coefficient = 7.0
    quantity = (references * unit.nanometer).in_units_of(position_unit)
    restraint = stages._make_backbone_restraint(
        _topology(), quantity, names,
        coefficient * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    assert restraint.getGlobalParameterName(0) == "k"
    assert [restraint.getParticleParameters(i)[0] for i in range(restraint.getNumParticles())] == selected

    system = openmm.System()
    for _ in range(3):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(
        Vec3(box_size, 0, 0), Vec3(0, box_size, 0), Vec3(0, 0, box_size),
    )
    system.addForce(restraint)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)

    # Independent analytical oracle, including a selected atom across the box.
    displacement = positions - references
    displacement -= np.round(displacement / box_size) * box_size
    expected_forces = np.zeros_like(positions)
    expected_forces[selected] = -2.0 * coefficient * displacement[selected]
    expected_energy = coefficient * np.square(displacement[selected]).sum()
    assert state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == pytest.approx(expected_energy)
    np.testing.assert_allclose(
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
        expected_forces, atol=1e-12,
    )
    context.setParameter("k", 0.0)
    assert context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == 0.0


@pytest.mark.parametrize("selection", [None, [], ["UNMATCHED"]])
def test_relaxation_preserves_reference_access_and_legacy_units(
    monkeypatch: pytest.MonkeyPatch, selection: list[str] | None,
) -> None:
    topology = _topology()
    positions = [Vec3(1, 2, 3), Vec3(4, 5, 6), Vec3(7, 8, 9)] * unit.angstrom
    if selection is not None:
        # NumPy rows have no x/y/z attributes. No reference coordinates may
        # be accessed when there are no matching restraint atoms.
        positions = np.asarray(positions.value_in_unit(unit.angstrom)) * unit.angstrom
    modeller = app.Modeller(topology, positions)
    system = openmm.System()
    for _ in range(3):
        system.addParticle(12.0)
    simulation = SimpleNamespace(
        context=SimpleNamespace(setPositions=lambda _: None, setParameter=lambda *_: None),
        minimizeEnergy=lambda **_: None,
    )
    monkeypatch.setattr(stages, "_build_system", lambda *_: (system, object()))
    monkeypatch.setattr(stages.app, "Simulation", lambda *_: simulation)
    monkeypatch.setattr(stages, "_save_final_state", lambda *_, **__: None)

    stages.run_openmm_relaxation(modeller, object(), backbone_names=selection)

    restraint = system.getForce(0)
    # Historically relaxation strips the quantity before passing coordinates
    # to OpenMM; this consolidation must not silently fix that separate issue.
    if selection is None:
        assert restraint.getParticleParameters(0)[1] == (1.0, 2.0, 3.0)
        assert restraint.getParticleParameters(1)[1] == (7.0, 8.0, 9.0)
    else:
        assert restraint.getNumParticles() == 0


@pytest.mark.parametrize("kind", ["classical", "adqtb"])
@pytest.mark.parametrize("format_name", ["dcd", "h5", "none"])
def test_context_trajectory_setup_keeps_options_and_attachment_order(
    monkeypatch: pytest.MonkeyPatch, kind: str, format_name: str,
) -> None:
    trajectory = object()
    calls = []
    simulation = SimpleNamespace(reporters=[])
    options = stages.TrajectoryOptions(format_name, 50, [0, 2])

    def make_reporter(path: str, resolved: Any) -> object:
        calls.append((path, resolved))
        return trajectory

    def write_topology(actual: Any, prefix: str, resolved: Any) -> None:
        assert actual.reporters == [trajectory]
        calls.append((prefix, resolved))

    monkeypatch.setattr(stages, "_make_trajectory_reporter", make_reporter)
    monkeypatch.setattr(stages, "_write_trajectory_topology", write_topology)
    monkeypatch.setattr(stages.app, "StateDataReporter", lambda *_, **__: "progress")
    monkeypatch.setattr(stages.app, "CheckpointReporter", lambda *_: "checkpoint")
    kwargs = {"trajectory": options, "checkpoint_interval": 100}
    if kind == "classical":
        stages._add_standard_reporters(simulation, "run", 25, **kwargs)
    else:
        stages._add_adqtb_reporters(
            simulation, "run", 25, segment_steps=5, type_names=None,
            friction_log=False, **kwargs,
        )

    if format_name == "none":
        assert calls == []
        assert simulation.reporters == ["progress", "progress", "checkpoint"]
    else:
        assert calls == [(f"run_steps.{format_name}", options), ("run", options)]
        assert simulation.reporters == [trajectory, "progress", "progress", "checkpoint"]


def test_binary_topology_subset_keeps_atom_order_and_current_box(tmp_path: Path) -> None:
    topology = _topology()
    positions = [Vec3(0.1, 0.2, 0.3), Vec3(0.4, 0.5, 0.6), Vec3(0.7, 0.8, 0.9)] * unit.nanometer
    box = [Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3)] * unit.nanometer
    simulation = SimpleNamespace(
        topology=topology,
        system=SimpleNamespace(usesPeriodicBoundaryConditions=lambda: True),
        context=SimpleNamespace(getState=lambda **_: SimpleNamespace(
            getPositions=lambda: positions, getPeriodicBoxVectors=lambda: box,
        )),
    )
    prefix = str(tmp_path / "subset")
    stages._write_trajectory_topology(
        simulation, prefix, stages.TrajectoryOptions("dcd", 1, [2, 0]),
    )
    written = app.PDBFile(prefix + "_topology.pdb")
    assert [atom.name for atom in written.topology.atoms()] == ["CA", "N"]
    assert topology.getNumAtoms() == 3
    np.testing.assert_allclose(
        written.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]),
    )
    assert written.topology.getUnitCellDimensions().value_in_unit(unit.nanometer) == pytest.approx([3, 3, 3])


def _qtb_integrator(segment_ps: float, step_ps: float = 0.001) -> Any:
    return SimpleNamespace(
        getSegmentLength=lambda: segment_ps * unit.picosecond,
        getStepSize=lambda: step_ps * unit.picosecond,
    )


@pytest.mark.parametrize("offset", [-0.5e-9, 0.0, 0.5e-9])
def test_segment_step_tolerance_and_units_are_shared(offset: float) -> None:
    length_ps = 0.005 + offset
    assert stages._validate_adqtb_segment(length_ps * 1000 * unit.femtosecond, 1 * unit.femtosecond) == 5
    assert adqtb._segment_steps(_qtb_integrator(length_ps)) == 5


@pytest.mark.parametrize("length_ps", [0.0, 0.005 - 1.5e-9, 0.005 + 1.5e-9])
def test_segment_errors_keep_their_caller_specific_wording(length_ps: float) -> None:
    driver_message = "finite and positive" if length_ps == 0 else "segment_length must be a whole number of time steps"
    with pytest.raises(ValueError, match=driver_message):
        stages._validate_adqtb_segment(length_ps * unit.picosecond, 0.001 * unit.picosecond)
    with pytest.raises(ValueError, match="segment length must be a whole number of steps"):
        adqtb._segment_steps(_qtb_integrator(length_ps))


def test_only_the_driver_imposes_fft_factor_and_positive_timestep_checks() -> None:
    assert adqtb._segment_steps(_qtb_integrator(0.011)) == 11
    with pytest.raises(ValueError, match="prime factors are 2, 3, 5 and 7"):
        stages._validate_adqtb_segment(0.011 * unit.picosecond, 0.001 * unit.picosecond)
    with pytest.raises(ValueError, match="time_step must be finite and positive"):
        stages._validate_adqtb_segment(0.005 * unit.picosecond, 0.0 * unit.picosecond)
    with pytest.raises(ZeroDivisionError):
        adqtb._segment_steps(_qtb_integrator(0.005, 0.0))
