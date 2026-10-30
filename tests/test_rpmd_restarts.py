"""RPMD restart archive validation, round trips, and production continuity."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm

import openmmnqe as nqe
import openmmnqe.openmm as nqe_openmm


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


# ---------------------------------------------------------------------------
# RPMD restart: refusing to save or load what cannot be trusted
# ---------------------------------------------------------------------------


def test_rpmd_save_rejects_a_bead_count_the_integrator_disagrees_with(tmp_path: Path) -> None:
    simulation = _make_rpmd_restart_simulation()

    with pytest.raises(ValueError, match="n_beads=3 does not match RPMDIntegrator copies=2"):
        nqe_openmm._save_rpmd_restart(simulation, tmp_path / "x.chk", n_beads=3)

    assert not (tmp_path / "x.chk").exists()


def test_rpmd_save_rejects_a_topology_that_does_not_match_the_system(tmp_path: Path) -> None:
    simulation = _make_rpmd_restart_simulation()
    residue = next(simulation.topology.residues())
    simulation.topology.addAtom("Extra", app.Element.getBySymbol("Ar"), residue)

    with pytest.raises(ValueError, match="Topology atom count does not match"):
        nqe_openmm._save_rpmd_restart(simulation, tmp_path / "x.chk", n_beads=2)


def _fake_rpmd_simulation(positions: Any, velocities: Any) -> SimpleNamespace:
    """One bead of one argon whose state arrays are whatever the test says."""
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    system = openmm.System()
    system.addParticle(39.9 * unit.dalton)
    state = SimpleNamespace(
        getPositions=lambda asNumpy=False: np.asarray(positions, dtype=float) * unit.nanometer,
        getVelocities=lambda asNumpy=False: (
            np.asarray(velocities, dtype=float) * unit.nanometer / unit.picosecond
        ),
        getPeriodicBoxVectors=lambda asNumpy=False: np.eye(3) * unit.nanometer,
        getTime=lambda: 0.0 * unit.picosecond,
    )
    integrator = SimpleNamespace(
        getNumCopies=lambda: 1,
        getTemperature=lambda: 300.0 * unit.kelvin,
        getState=lambda bead, **kwargs: state,
    )
    return SimpleNamespace(
        topology=topology, system=system, integrator=integrator, currentStep=0,
    )


@pytest.mark.parametrize(
    ("positions", "velocities", "message"),
    [
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]], "position shape"),
        ([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "velocity shape"),
        ([[np.nan, 0.0, 0.0]], [[0.0, 0.0, 0.0]], "bead positions are not finite"),
        ([[0.0, 0.0, 0.0]], [[np.inf, 0.0, 0.0]], "bead velocities are not finite"),
    ],
)
def test_rpmd_save_rejects_malformed_bead_states(
    tmp_path: Path, positions: Any, velocities: Any, message: str,
) -> None:
    simulation = _fake_rpmd_simulation(positions, velocities)

    with pytest.raises(ValueError, match=message):
        nqe_openmm._save_rpmd_restart(simulation, tmp_path / "x.chk", n_beads=1)

    assert list(tmp_path.iterdir()) == []


def test_rpmd_save_removes_its_temporary_file_when_writing_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    simulation = _make_rpmd_restart_simulation()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(nqe_openmm.np, "savez", fail)

    with pytest.raises(OSError, match="disk full"):
        nqe_openmm._save_rpmd_restart(simulation, tmp_path / "x.chk", n_beads=2)

    assert list(tmp_path.iterdir()) == []


def test_rpmd_restart_rejects_an_archive_that_is_not_npz(tmp_path: Path) -> None:
    checkpoint = tmp_path / "plain.npy"
    np.save(checkpoint, np.zeros(3))

    with pytest.raises(ValueError, match="not an NPZ archive"):
        nqe_openmm._read_rpmd_restart(checkpoint)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"kind": np.asarray(nqe_openmm._RPMD_RESTART_KIND)},
            "missing fields: format_version",
        ),
        (
            {
                "kind": np.asarray("something-else"),
                "format_version": np.asarray(nqe_openmm._RPMD_RESTART_VERSION, dtype=np.int64),
            },
            "not an openmmnqe RPMD restart",
        ),
        (
            {
                "kind": np.asarray(nqe_openmm._RPMD_RESTART_KIND),
                "format_version": np.asarray(nqe_openmm._RPMD_RESTART_VERSION + 1, dtype=np.int64),
            },
            f"Unsupported RPMD restart version {nqe_openmm._RPMD_RESTART_VERSION + 1}",
        ),
        (
            {
                "kind": np.asarray(nqe_openmm._RPMD_RESTART_KIND),
                "format_version": np.asarray(nqe_openmm._RPMD_RESTART_VERSION, dtype=np.int64),
            },
            "missing fields: box_vectors_nm, num_beads, num_particles",
        ),
    ],
)
def test_rpmd_restart_rejects_header_schema_violations(
    tmp_path: Path, fields: dict[str, np.ndarray], message: str,
) -> None:
    checkpoint = tmp_path / "bad.chk"
    with checkpoint.open("wb") as handle:
        np.savez(handle, **fields)

    with pytest.raises(ValueError, match=message):
        nqe_openmm._read_rpmd_restart(checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_particles", np.asarray(-1, dtype=np.int64), "num_particles cannot be negative"),
        ("step_count", np.asarray(-1, dtype=np.int64), "step count cannot be negative"),
        ("topology_signature_sha256", np.asarray("Z" * 64), "64 lowercase hexadecimal"),
        ("topology_signature_sha256", np.asarray("abc"), "64 lowercase hexadecimal"),
    ],
)
def test_rpmd_restart_rejects_out_of_domain_metadata(
    tmp_path: Path, field: str, value: np.ndarray, message: str,
) -> None:
    checkpoint = tmp_path / "rpmd_ready.chk"
    _write_test_rpmd_restart(checkpoint)
    _replace_test_restart_fields(checkpoint, **{field: value})

    with pytest.raises(ValueError, match=message):
        nqe_openmm._read_rpmd_restart(checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "num_particles",
            np.asarray(2, dtype=np.int64),
            "contains 2 particles, but the current System contains 1",
        ),
        ("particle_masses_dalton", np.asarray([39.9, 39.9]), "particle masses have shape"),
        ("particle_masses_dalton", np.asarray([np.nan]), "masses contain non-finite"),
        ("particle_masses_dalton", np.asarray([-39.9]), "masses cannot be negative"),
        ("temperature_kelvin", np.asarray(np.nan), "temperature is not finite"),
        ("positions_nm", np.zeros((2, 2, 3)), "positions have shape"),
        ("velocities_nm_per_ps", np.zeros((2, 2, 3)), "velocities have shape"),
        ("box_vectors_nm", np.zeros((2, 3)), "box has shape"),
        ("periodic", np.asarray(False), "periodicity does not match"),
        ("positions_nm", np.full((2, 1, 3), np.nan), "positions_nm contains non-finite"),
        ("time_ps", np.asarray(np.nan), "time is not finite"),
    ],
)
def test_rpmd_load_refuses_a_restart_that_disagrees_with_the_simulation(
    tmp_path: Path, field: str, value: np.ndarray, message: str,
) -> None:
    checkpoint = tmp_path / "rpmd_ready.chk"
    _write_test_rpmd_restart(checkpoint)
    _replace_test_restart_fields(checkpoint, **{field: value})
    restored = _make_rpmd_restart_simulation()

    with pytest.raises(ValueError, match=message):
        nqe_openmm._load_rpmd_restart(restored, checkpoint, n_beads=2)


def test_rpmd_load_rejects_an_integrator_with_a_different_copy_count(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rpmd_ready.chk"
    _write_test_rpmd_restart(checkpoint)
    restored = _make_rpmd_restart_simulation(num_beads=3)

    with pytest.raises(ValueError, match="n_beads=2 does not match RPMDIntegrator copies=3"):
        nqe_openmm._load_rpmd_restart(restored, checkpoint, n_beads=2)


def test_rpmd_load_rejects_a_current_topology_that_does_not_match_its_system(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "rpmd_ready.chk"
    _write_test_rpmd_restart(checkpoint)
    restored = _make_rpmd_restart_simulation()
    residue = next(restored.topology.residues())
    restored.topology.addAtom("Extra", app.Element.getBySymbol("Ar"), residue)

    with pytest.raises(ValueError, match="Current Topology atom count does not match"):
        nqe_openmm._load_rpmd_restart(restored, checkpoint, n_beads=2)


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
