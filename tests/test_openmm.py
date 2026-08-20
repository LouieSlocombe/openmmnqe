"""Tests for shared OpenMM workflow machinery and CPU integration seams."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from collections.abc import Callable
from typing import Any, TextIO

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm

import openmmnqe as nqe
import openmmnqe.openmm as nqe_openmm


def test_simple_relaxation_writes_parseable_state(one_particle_system: tuple[app.Modeller, Any]) -> None:
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
    ligand_forcefield: Callable[..., tuple[app.Modeller, app.ForceField]],
    data_dir: Path,
) -> None:
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
def test_multiple_ligand_templates_build_one_system(ligand_forcefield: Callable[..., tuple[app.Modeller, app.ForceField]], data_dir: Path) -> None:
    modeller, forcefield = ligand_forcefield(data_dir / "pdb" / "gc.pdb")

    system = forcefield.createSystem(modeller.topology)

    assert {residue.name for residue in modeller.topology.residues()} == {
        "GGG",
        "CCC",
    }
    assert system.getNumParticles() == modeller.topology.getNumAtoms()


def test_rpmd_stages_run_on_a_prepared_system(one_particle_system: tuple[app.Modeller, Any]) -> None:
    # The in-repo regression for the pre-built-System seam: a System built
    # outside the stage runs equilibration, then production restarts from its
    # bead archive with the same PreparedSystem.
    modeller, forcefield = one_particle_system
    prepared = nqe.PreparedSystem(forcefield.createSystem(modeller.topology))

    nqe.run_openmm_rpmd_equilibration(
        modeller,
        prepared,
        n_beads=2,
        n_1=2,
        n_2=3,
        n_report=2,
        platform_name="Reference",
        seed=7,
    )

    with np.load("rpmd_ready.chk", allow_pickle=False) as archive:
        assert archive["kind"].item() == "openmmnqe-rpmd-restart"
        assert archive["num_beads"].item() == 2
        assert archive["step_count"].item() == 5
        assert np.isfinite(archive["positions_nm"]).all()

    nqe.run_openmm_rpmd_prod(
        modeller,
        prepared,
        checkpoint_file="rpmd_ready.chk",
        n_beads=2,
        steps=3,
        n_report=2,
        barostat_freq=None,
        platform_name="Reference",
    )

    with np.load("rpmd_prod.chk", allow_pickle=False) as archive:
        assert archive["num_beads"].item() == 2
        assert archive["step_count"].item() == 8


def test_maybe_deuterate_only_calls_helper_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_load_plumed_is_optional_and_adds_script_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    constructed = []

    class FakePlumedForce:
        def __init__(self, script: str) -> None:
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


def test_add_standard_reporters_builds_requested_set(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_add_rpmd_progress_reporters_omit_context_thermodynamics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nqe_openmm.app,
        "StateDataReporter",
        lambda *args, **kwargs: ("state", args, kwargs),
    )
    simulation = SimpleNamespace(reporters=[])

    nqe_openmm._add_rpmd_progress_reporters(
        simulation,
        output_prefix="rpmd",
        n_report=25,
    )

    assert [reporter[1] for reporter in simulation.reporters] == [
        (nqe_openmm.sys.stdout, 25),
        ("rpmd.log", 25),
    ]
    assert simulation.reporters[0][2] == {"step": True, "speed": True}
    assert simulation.reporters[1][2] == {
        "step": True,
        "time": True,
        "speed": True,
        "volume": True,
    }
    for _, _, options in simulation.reporters:
        assert not {
            "potentialEnergy",
            "kineticEnergy",
            "totalEnergy",
            "temperature",
        }.intersection(options)


def test_add_rpmd_reporters_builds_spread_centroid_and_beads(monkeypatch: pytest.MonkeyPatch) -> None:
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
    topology = SimpleNamespace(getNumAtoms=lambda: 4)

    nqe_openmm._add_rpmd_reporters(
        simulation,
        topology,
        output_prefix="rpmd",
        n_report=10,
        n_beads=4,
        atoms_to_watch=[1, 2],
        expansion_metric="mean",
        distance_pairs=[(0, 1), (3, 2)],
    )

    assert [reporter[0] for reporter in simulation.reporters] == [
        "spread",
        "centroid",
        "beads",
    ]
    assert simulation.reporters[0][1]["atom_indices"] == [1, 2]
    assert simulation.reporters[0][1]["metric"] == "mean"
    assert simulation.reporters[0][1]["distance_pairs"] == [(0, 1), (3, 2)]
    assert simulation.reporters[1][1]["num_beads"] == 4
    assert simulation.reporters[2][1]["topology"] is topology


def test_add_rpmd_reporters_omits_spread_without_atom_selection(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_add_rpmd_reporters_requires_spread_atoms_for_distances() -> None:
    simulation = SimpleNamespace(reporters=[])

    with pytest.raises(ValueError, match="require atoms_to_watch"):
        nqe_openmm._add_rpmd_reporters(
            simulation,
            object(),
            "rpmd",
            10,
            4,
            None,
            distance_pairs=[(0, 1)],
        )

    assert simulation.reporters == []


@pytest.mark.parametrize(
    ("atoms_to_watch", "distance_pairs", "message"),
    [
        ([], None, "must not be empty"),
        ([4], None, "outside topology"),
        ([0], [(0, 4)], "outside topology"),
    ],
)
def test_add_rpmd_reporters_validates_observable_indices(
    atoms_to_watch: list[int], distance_pairs: list[tuple[int, int]] | None, message: str,
) -> None:
    simulation = SimpleNamespace(reporters=[])
    topology = SimpleNamespace(getNumAtoms=lambda: 2)

    with pytest.raises(ValueError, match=message):
        nqe_openmm._add_rpmd_reporters(
            simulation,
            topology,
            "rpmd",
            10,
            4,
            atoms_to_watch,
            distance_pairs=distance_pairs,
        )

    assert simulation.reporters == []


def test_save_final_state_checkpoints_and_uses_context_positions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved_checkpoints = []
    state_requests = []
    written = []
    final_positions = object()
    topology = object()

    def get_state(**kwargs: Any) -> SimpleNamespace:
        state_requests.append(kwargs)
        return SimpleNamespace(getPositions=lambda: final_positions)

    def write_file(actual_topology: Any, positions: Any, handle: TextIO) -> None:
        written.append((actual_topology, positions, handle.name))
        handle.write("final state")

    monkeypatch.setattr(nqe_openmm.app.PDBFile, "writeFile", write_file)
    simulation = SimpleNamespace(
        topology=topology,
        system=SimpleNamespace(usesPeriodicBoundaryConditions=lambda: False),
        context=SimpleNamespace(getState=get_state),
        saveCheckpoint=lambda path: saved_checkpoints.append(path),
    )
    prefix = tmp_path / "classical"

    nqe_openmm._save_final_state(simulation, str(prefix))

    assert saved_checkpoints == [str(prefix) + ".chk"]
    assert state_requests == [{"getPositions": True}]
    assert written == [(topology, final_positions, str(prefix) + ".pdb")]
    assert prefix.with_suffix(".pdb").read_text() == "final state"


def test_save_final_rpmd_state_uses_centroid_and_custom_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    centroid_positions = object()
    centroid_calls = []
    restart_calls = []
    written = []
    topology = SimpleNamespace(getNumAtoms=lambda: 2)
    simulation = SimpleNamespace(
        topology=topology,
        system=SimpleNamespace(usesPeriodicBoundaryConditions=lambda: False),
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


def test_load_checkpoint_requires_existing_file(tmp_path: Path) -> None:
    simulation = SimpleNamespace(loadCheckpoint=lambda path: pytest.fail(path))

    with pytest.raises(FileNotFoundError, match="Run the equilibration stage"):
        nqe_openmm._load_checkpoint(simulation, tmp_path / "missing.chk")


def test_load_checkpoint_delegates_to_simulation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ready.chk"
    checkpoint.write_bytes(b"checkpoint")
    loaded = []
    simulation = SimpleNamespace(loadCheckpoint=lambda path: loaded.append(path))

    nqe_openmm._load_checkpoint(simulation, checkpoint)

    assert loaded == [checkpoint]


def _make_rpmd_restart_simulation(
    num_beads: int=2,
    particle_mass: float=39.9,
    atom_name: str="Ar",
    temperature: float=300.0,
) -> app.Simulation:
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom(atom_name, app.Element.getBySymbol("Ar"), residue)

    system = openmm.System()
    system.addParticle(particle_mass * unit.dalton)
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
        temperature * unit.kelvin,
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


def _write_test_rpmd_restart(checkpoint: Path) -> app.Simulation:
    simulation = _make_rpmd_restart_simulation()
    for bead in range(2):
        simulation.integrator.setPositions(
            bead,
            np.asarray([[0.1 + bead, 0.2, 0.3]]) * unit.nanometer,
        )
        simulation.integrator.setVelocities(
            bead,
            np.asarray([[0.4 + bead, 0.5, 0.6]])
            * unit.nanometer
            / unit.picosecond,
        )
    nqe_openmm._save_rpmd_restart(simulation, checkpoint, n_beads=2)
    return simulation


def _replace_test_restart_fields(checkpoint: Path, **updates: np.ndarray) -> None:
    with np.load(checkpoint, allow_pickle=False) as archive:
        fields = {name: np.array(archive[name]) for name in archive.files}
    fields.update(updates)
    with checkpoint.open("wb") as handle:
        np.savez(handle, **fields)


def test_rpmd_topology_signature_tracks_atom_order() -> None:
    def topology_with_order(names: list[str]) -> app.Topology:
        topology = app.Topology()
        residue = topology.addResidue("MIX", topology.addChain())
        for name in names:
            topology.addAtom(name, app.Element.getBySymbol(name), residue)
        return topology

    hydrogen_then_oxygen = topology_with_order(["H", "O"])
    oxygen_then_hydrogen = topology_with_order(["O", "H"])

    assert nqe_openmm._topology_identity_signature(
        hydrogen_then_oxygen
    ) != nqe_openmm._topology_identity_signature(oxygen_then_hydrogen)


def test_rpmd_topology_signature_survives_pdb_round_trip(tmp_path: Path) -> None:
    topology = app.Topology()
    chain = topology.addChain("X")
    residue = topology.addResidue("HOH", chain, id="42")
    oxygen = topology.addAtom(
        "O", app.Element.getBySymbol("O"), residue, id="9"
    )
    hydrogen_1 = topology.addAtom(
        "H1", app.Element.getBySymbol("H"), residue, id="12"
    )
    hydrogen_2 = topology.addAtom(
        "H2", app.Element.getBySymbol("H"), residue, id="14"
    )
    topology.addBond(oxygen, hydrogen_1, type=app.Single, order=1)
    topology.addBond(oxygen, hydrogen_2, type=app.Single, order=1)
    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0957, 0.0, 0.0],
        [-0.0240, 0.0927, 0.0],
    ]) * unit.nanometer

    pdb_path = tmp_path / "round_trip.pdb"
    with pdb_path.open("w") as handle:
        app.PDBFile.writeFile(topology, positions, handle)
    reloaded_topology = app.PDBFile(str(pdb_path)).topology

    assert nqe_openmm._topology_identity_signature(
        topology
    ) == nqe_openmm._topology_identity_signature(reloaded_topology)


@pytest.mark.parametrize(
    ("residue_name", "atom_name", "description"),
    [
        ("LONG", "C", "residue 0"),
        ("LIG", "CARBO", "atom 0"),
        ("LIG", "C Å", "atom 0"),
    ],
)
def test_rpmd_topology_signature_rejects_non_pdb_identity_names(
    residue_name: str,
    atom_name: str,
    description: str,
) -> None:
    topology = app.Topology()
    residue = topology.addResidue(residue_name, topology.addChain())
    topology.addAtom(atom_name, app.Element.getBySymbol("C"), residue)

    with pytest.raises(
        ValueError,
        match=rf"{description} name .* is not representable",
    ):
        nqe_openmm._topology_identity_signature(topology)


def test_rpmd_restart_round_trip_preserves_every_copy(tmp_path: Path) -> None:
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

    with np.load(checkpoint, allow_pickle=False) as archive:
        assert archive["format_version"].item() == 2
        assert archive["particle_masses_dalton"] == pytest.approx([39.9])
        assert archive["temperature_kelvin"].item() == pytest.approx(300.0)
        assert len(archive["topology_signature_sha256"].item()) == 64

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

    # Continue from the restored nonzero step through the RPMD-compatible
    # stepper, exercising both first-step state handling and reporter-count
    # bookkeeping on OpenMM versions whose RPMD kernel leaves it unchanged.
    nqe.step_rpmd(restored, 1)
    assert restored.currentStep == 18
    stepped_positions = [
        restored.integrator.getState(bead, getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
        for bead in range(2)
    ]
    assert not np.allclose(stepped_positions[0], stepped_positions[1])


def test_rpmd_load_rejects_particle_mass_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rpmd_ready.chk"
    _write_test_rpmd_restart(checkpoint)

    restored = _make_rpmd_restart_simulation(particle_mass=40.0)
    with pytest.raises(ValueError, match="particle mass 0 does not match"):
        nqe_openmm._load_checkpoint(restored, checkpoint, n_beads=2)


def test_rpmd_load_rejects_ordered_atom_identity_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rpmd_ready.chk"
    _write_test_rpmd_restart(checkpoint)

    restored = _make_rpmd_restart_simulation(atom_name="XAr")
    with pytest.raises(ValueError, match="atom/topology identity does not match"):
        nqe_openmm._load_checkpoint(restored, checkpoint, n_beads=2)


def test_rpmd_load_rejects_temperature_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rpmd_ready.chk"
    _write_test_rpmd_restart(checkpoint)

    restored = _make_rpmd_restart_simulation(temperature=310.0)
    with pytest.raises(ValueError, match="temperature 300.0 K does not match"):
        nqe_openmm._load_checkpoint(restored, checkpoint, n_beads=2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", np.asarray(2.9), "scalar integer"),
        ("num_beads", np.asarray(True), "scalar integer"),
        ("num_particles", np.asarray(1.9), "scalar integer"),
        ("periodic", np.asarray("False"), "scalar boolean"),
        ("step_count", np.asarray(17.9), "scalar integer"),
        ("temperature_kelvin", np.asarray(300), "scalar float"),
        (
            "topology_signature_sha256",
            np.asarray(["a" * 64]),
            "must be a scalar",
        ),
    ],
)
def test_rpmd_restart_rejects_malformed_scalar_schema(
    tmp_path: Path,
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    checkpoint = tmp_path / "rpmd_ready.chk"
    _write_test_rpmd_restart(checkpoint)
    _replace_test_restart_fields(checkpoint, **{field: value})

    with pytest.raises(ValueError, match=rf"{field}.*{message}"):
        nqe_openmm._read_rpmd_restart(checkpoint)


@pytest.mark.parametrize("n_beads", [True, np.bool_(False), 0, -1, 2.0])
def test_rpmd_restart_requires_positive_integer_n_beads(tmp_path: Path, n_beads: Any) -> None:
    simulation = _make_rpmd_restart_simulation()

    with pytest.raises(ValueError, match="n_beads must be a positive integer"):
        nqe_openmm._save_rpmd_restart(
            simulation,
            tmp_path / "rpmd_ready.chk",
            n_beads=n_beads,
        )
    with pytest.raises(ValueError, match="n_beads must be a positive integer"):
        nqe_openmm._load_rpmd_restart(
            SimpleNamespace(),
            tmp_path / "unused.chk",
            n_beads=n_beads,
        )


def test_rpmd_restart_wraps_initial_bad_zip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "corrupt.chk"
    checkpoint.touch()

    def raise_bad_zip(*args: Any, **kwargs: Any) -> None:
        raise nqe_openmm.zipfile.BadZipFile("broken central directory")

    monkeypatch.setattr(nqe_openmm.np, "load", raise_bad_zip)

    with pytest.raises(ValueError, match="corrupt or unreadable.*rerun"):
        nqe_openmm._read_rpmd_restart(checkpoint)


def test_rpmd_restart_wraps_lazy_bad_zip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class CorruptArchive:
        files = ["kind", "format_version"]

        def __init__(self) -> None:
            self.closed = False

        def __getitem__(self, name: str) -> None:
            raise nqe_openmm.zipfile.BadZipFile(f"bad CRC for {name}")

        def close(self) -> None:
            self.closed = True

    archive = CorruptArchive()
    monkeypatch.setattr(nqe_openmm.np, "load", lambda *args, **kwargs: archive)

    with pytest.raises(ValueError, match="corrupt or unreadable.*rerun"):
        nqe_openmm._read_rpmd_restart(tmp_path / "corrupt.chk")
    assert archive.closed


def test_rpmd_load_rejects_version_one_restart_with_rerun_message(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rpmd_v1.chk"
    with checkpoint.open("wb") as handle:
        np.savez(
            handle,
            kind=np.asarray(nqe_openmm._RPMD_RESTART_KIND),
            format_version=np.asarray(1, dtype=np.int64),
        )

    with pytest.raises(ValueError, match="version 1.*Rerun RPMD equilibration"):
        nqe_openmm._load_checkpoint(
            SimpleNamespace(),
            checkpoint,
            n_beads=2,
        )


def test_rpmd_load_rejects_legacy_context_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.chk"
    checkpoint.write_bytes(b"ordinary OpenMM Context checkpoint")

    with pytest.raises(ValueError, match="generic OpenMM checkpoint"):
        nqe_openmm._load_checkpoint(
            SimpleNamespace(),
            checkpoint,
            n_beads=2,
        )


def test_save_final_state_writes_current_box_for_periodic_system() -> None:
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    stale_box = [Vec3(2.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0), Vec3(0.0, 0.0, 2.0)]
    topology.setPeriodicBoxVectors(stale_box * unit.nanometer)

    system = openmm.System()
    system.addParticle(39.9 * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*(stale_box * unit.nanometer))
    nonbonded = openmm.NonbondedForce()
    nonbonded.addParticle(0.0, 0.3, 0.0)
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(0.9 * unit.nanometer)
    system.addForce(nonbonded)

    simulation = app.Simulation(
        topology,
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions([Vec3(0.1, 0.1, 0.1)] * unit.nanometer)
    moved_box = [Vec3(2.5, 0.0, 0.0), Vec3(0.0, 2.5, 0.0), Vec3(0.0, 0.0, 2.5)]
    simulation.context.setPeriodicBoxVectors(*(moved_box * unit.nanometer))

    nqe_openmm._save_final_state(simulation, "boxed", save_checkpoint=False)

    synced = simulation.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    assert [synced[i][i] for i in range(3)] == pytest.approx([2.5, 2.5, 2.5])
    written = app.PDBFile("boxed.pdb")
    cryst1 = written.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    assert [cryst1[i][i] for i in range(3)] == pytest.approx([2.5, 2.5, 2.5], abs=1e-3), (
        "the stage-final PDB must carry the post-barostat box, not the build-time one"
    )
