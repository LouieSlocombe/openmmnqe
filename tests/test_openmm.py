"""Tests for shared OpenMM workflow machinery and CPU integration seams."""

from types import SimpleNamespace

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm

import openmmnqe as nqe
import openmmnqe.openmm as nqe_openmm


def test_simple_relaxation_writes_parseable_state(one_particle_system):
    modeller, forcefield = one_particle_system

    nqe.run_openmm_relaxation_simple(
        modeller,
        forcefield,
        output_prefix="relaxed",
        platform_name="CPU",
    )

    output = app.PDBFile("relaxed.pdb")
    coordinates = output.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert output.topology.getNumAtoms() == 1
    assert np.isfinite(coordinates).all()
    assert nqe_openmm.os.path.getsize("relaxed.chk") > 0


@pytest.mark.forcefield
def test_nonstandard_ligand_forcefield_drives_relaxation(
    ligand_forcefield,
    data_dir,
):
    modeller, forcefield = ligand_forcefield(data_dir / "pdb" / "toluene.pdb")

    nqe.run_openmm_relaxation_simple(
        modeller,
        forcefield,
        output_prefix="ligand_relaxed",
        platform_name="CPU",
    )
    output = app.PDBFile("ligand_relaxed.pdb")

    assert output.topology.getNumAtoms() == modeller.topology.getNumAtoms()
    assert np.isfinite(
        output.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    ).all()


@pytest.mark.forcefield
def test_multiple_ligand_templates_build_one_system(ligand_forcefield, data_dir):
    modeller, forcefield = ligand_forcefield(data_dir / "pdb" / "gc.pdb")

    system = forcefield.createSystem(modeller.topology)

    assert {residue.name for residue in modeller.topology.residues()} == {
        "GGG",
        "CCC",
    }
    assert system.getNumParticles() == modeller.topology.getNumAtoms()


def test_maybe_deuterate_only_calls_helper_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nqe_openmm,
        "deuterate_system",
        lambda modeller, system, option: calls.append((modeller, system, option)),
    )
    modeller = object()
    system = object()

    nqe_openmm._maybe_deuterate(modeller, system, False, "water")
    nqe_openmm._maybe_deuterate(modeller, system, True, "protein")

    assert calls == [(modeller, system, "protein")]


def test_load_plumed_is_optional_and_adds_script_force(monkeypatch, tmp_path):
    constructed = []

    class FakePlumedForce:
        def __init__(self, script):
            self.script = script
            constructed.append(self)

    monkeypatch.setattr(nqe_openmm, "PlumedForce", FakePlumedForce)
    system = SimpleNamespace(forces=[], addForce=lambda force: system.forces.append(force))

    nqe_openmm._load_plumed(system, None)
    assert system.forces == []

    script = tmp_path / "plumed.dat"
    script.write_text("DISTANCE ATOMS=1,2\n")
    nqe_openmm._load_plumed(system, script)

    assert system.forces == constructed
    assert constructed[0].script == "DISTANCE ATOMS=1,2\n"


def test_add_standard_reporters_builds_requested_set(monkeypatch):
    monkeypatch.setattr(
        nqe_openmm.app,
        "PDBReporter",
        lambda *args, **kwargs: ("pdb", args, kwargs),
    )
    monkeypatch.setattr(
        nqe_openmm.app,
        "StateDataReporter",
        lambda *args, **kwargs: ("state", args, kwargs),
    )
    monkeypatch.setattr(
        nqe_openmm.app,
        "CheckpointReporter",
        lambda *args, **kwargs: ("checkpoint", args, kwargs),
    )
    simulation = SimpleNamespace(reporters=[])

    nqe_openmm._add_standard_reporters(
        simulation,
        output_prefix="run",
        n_report=25,
        pdb_steps=True,
        stdout_volume=True,
        checkpoint_interval=100,
    )

    assert [reporter[0] for reporter in simulation.reporters] == [
        "pdb",
        "state",
        "state",
        "checkpoint",
    ]
    assert simulation.reporters[0][1] == ("run_steps.pdb", 25)
    assert simulation.reporters[1][2]["volume"] is True
    assert simulation.reporters[2][1] == ("run.log", 25)
    assert simulation.reporters[3][1] == ("run.chk", 100)


def test_add_rpmd_reporters_builds_spread_centroid_and_beads(monkeypatch):
    monkeypatch.setattr(
        nqe_openmm,
        "RPMDQuantumSpreadReporter",
        lambda **kwargs: ("spread", kwargs),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "RPMDCentroidReporter",
        lambda **kwargs: ("centroid", kwargs),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "RPMDBeadReporter",
        lambda **kwargs: ("beads", kwargs),
    )
    simulation = SimpleNamespace(reporters=[])
    topology = object()

    nqe_openmm._add_rpmd_reporters(
        simulation,
        topology,
        output_prefix="rpmd",
        n_report=10,
        n_beads=4,
        atoms_to_watch=[1, 2],
    )

    assert [reporter[0] for reporter in simulation.reporters] == [
        "spread",
        "centroid",
        "beads",
    ]
    assert simulation.reporters[0][1]["atom_indices"] == [1, 2]
    assert simulation.reporters[1][1]["num_beads"] == 4
    assert simulation.reporters[2][1]["topology"] is topology


def test_add_rpmd_reporters_omits_spread_without_atom_selection(monkeypatch):
    monkeypatch.setattr(
        nqe_openmm,
        "RPMDCentroidReporter",
        lambda **kwargs: ("centroid", kwargs),
    )
    monkeypatch.setattr(
        nqe_openmm,
        "RPMDBeadReporter",
        lambda **kwargs: ("beads", kwargs),
    )
    simulation = SimpleNamespace(reporters=[])

    nqe_openmm._add_rpmd_reporters(simulation, object(), "rpmd", 10, 4, None)

    assert [reporter[0] for reporter in simulation.reporters] == [
        "centroid",
        "beads",
    ]


def test_save_final_state_checkpoints_and_uses_context_positions(monkeypatch, tmp_path):
    saved_checkpoints = []
    state_requests = []
    written = []
    final_positions = object()
    topology = object()

    def get_state(**kwargs):
        state_requests.append(kwargs)
        return SimpleNamespace(getPositions=lambda: final_positions)

    def write_file(actual_topology, positions, handle):
        written.append((actual_topology, positions, handle.name))
        handle.write("final state")

    monkeypatch.setattr(nqe_openmm.app.PDBFile, "writeFile", write_file)
    simulation = SimpleNamespace(
        topology=topology,
        context=SimpleNamespace(getState=get_state),
        saveCheckpoint=lambda path: saved_checkpoints.append(path),
    )
    prefix = tmp_path / "classical"

    nqe_openmm._save_final_state(simulation, str(prefix))

    assert saved_checkpoints == [str(prefix) + ".chk"]
    assert state_requests == [{"getPositions": True}]
    assert written == [(topology, final_positions, str(prefix) + ".pdb")]
    assert prefix.with_suffix(".pdb").read_text() == "final state"


def test_save_final_rpmd_state_uses_centroid_and_custom_suffix(monkeypatch, tmp_path):
    centroid_positions = object()
    centroid_calls = []
    restart_calls = []
    written = []
    topology = SimpleNamespace(getNumAtoms=lambda: 2)
    simulation = SimpleNamespace(
        topology=topology,
        context=SimpleNamespace(
            getState=lambda **kwargs: pytest.fail("context positions are bead 0")
        ),
        saveCheckpoint=lambda path: pytest.fail("checkpoint was disabled"),
    )

    monkeypatch.setattr(
        nqe_openmm,
        "centroid_positions",
        lambda *args: centroid_calls.append(args) or centroid_positions,
    )
    monkeypatch.setattr(
        nqe_openmm,
        "_save_rpmd_restart",
        lambda *args: restart_calls.append(args),
    )
    monkeypatch.setattr(
        nqe_openmm.app.PDBFile,
        "writeFile",
        lambda actual_topology, positions, handle: written.append(
            (actual_topology, positions, handle.name)
        ),
    )
    prefix = tmp_path / "rpmd"

    nqe_openmm._save_final_state(
        simulation,
        str(prefix),
        pdb_suffix="_centroid.pdb",
        n_beads=8,
    )

    assert restart_calls == [(simulation, str(prefix) + ".chk", 8)]
    assert centroid_calls == [(simulation, 2, 8)]
    assert written == [
        (topology, centroid_positions, str(prefix) + "_centroid.pdb")
    ]


def test_load_checkpoint_requires_existing_file(tmp_path):
    simulation = SimpleNamespace(loadCheckpoint=lambda path: pytest.fail(path))

    with pytest.raises(FileNotFoundError, match="Run the equilibration stage"):
        nqe_openmm._load_checkpoint(simulation, tmp_path / "missing.chk")


def test_load_checkpoint_delegates_to_simulation(tmp_path):
    checkpoint = tmp_path / "ready.chk"
    checkpoint.write_bytes(b"checkpoint")
    loaded = []
    simulation = SimpleNamespace(loadCheckpoint=lambda path: loaded.append(path))

    nqe_openmm._load_checkpoint(simulation, checkpoint)

    assert loaded == [checkpoint]


def _make_rpmd_restart_simulation(num_beads=2):
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)

    system = openmm.System()
    system.addParticle(39.9 * unit.dalton)
    system.setDefaultPeriodicBoxVectors(
        Vec3(2.0, 0.0, 0.0),
        Vec3(0.0, 2.0, 0.0),
        Vec3(0.0, 0.0, 2.0),
    )
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(0.5 * unit.nanometer)
    nonbonded.addParticle(0.0, 0.1, 0.0)
    system.addForce(nonbonded)

    integrator = openmm.RPMDIntegrator(
        num_beads,
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.1 * unit.femtosecond,
    )
    integrator.setApplyThermostat(False)
    simulation = app.Simulation(
        topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    return simulation


def test_rpmd_restart_round_trip_preserves_every_copy(tmp_path):
    source = _make_rpmd_restart_simulation()
    source_positions = np.asarray(
        [[[0.1, 0.2, 0.3]], [[1.1, 1.2, 1.3]]]
    )
    source_velocities = np.asarray(
        [[[0.4, 0.5, 0.6]], [[1.4, 1.5, 1.6]]]
    )
    for bead in range(2):
        source.integrator.setPositions(
            bead,
            source_positions[bead] * unit.nanometer,
        )
        source.integrator.setVelocities(
            bead,
            source_velocities[bead] * unit.nanometer / unit.picosecond,
        )
    source.context.setTime(1.25 * unit.picosecond)
    source.currentStep = 17

    checkpoint = tmp_path / "rpmd_ready.chk"
    nqe_openmm._save_rpmd_restart(source, checkpoint, n_beads=2)

    restored = _make_rpmd_restart_simulation()
    nqe_openmm._load_checkpoint(restored, checkpoint, n_beads=2)

    assert restored.currentStep == 17
    for bead in range(2):
        state = restored.integrator.getState(
            bead,
            getPositions=True,
            getVelocities=True,
        )
        assert state.getTime().value_in_unit(unit.picosecond) == pytest.approx(1.25)
        assert np.allclose(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            source_positions[bead],
        )
        assert np.allclose(
            state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            source_velocities[bead],
        )

    # Use the integrator directly: OpenMM 8.5's RPMD plugin advances time but
    # does not update Context.stepCount, which makes Simulation.step() loop.
    # The integrator step still exercises RPMD's first-step state handling.
    restored.integrator.step(1)
    stepped_positions = [
        restored.integrator.getState(bead, getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
        for bead in range(2)
    ]
    assert not np.allclose(stepped_positions[0], stepped_positions[1])


def test_rpmd_load_rejects_legacy_context_checkpoint(tmp_path):
    checkpoint = tmp_path / "legacy.chk"
    checkpoint.write_bytes(b"ordinary OpenMM Context checkpoint")

    with pytest.raises(ValueError, match="generic OpenMM checkpoint"):
        nqe_openmm._load_checkpoint(
            SimpleNamespace(),
            checkpoint,
            n_beads=2,
        )
