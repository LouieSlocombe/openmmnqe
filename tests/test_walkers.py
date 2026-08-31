"""Orchestration, seeding, resume, and worker behavior of multi-walker runs."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import Future
from pathlib import Path
from typing import Any, ClassVar

import openmm.app as app
import pytest

import openmmnqe as nqe
from openmmnqe import rates as nqe_rates
from openmmnqe import walkers
from openmmnqe.openmm import _derive_seeds

_RECORD = "stage_record.json"


def _recording_stage(**kwargs: Any) -> None:
    """Record where and with what a walker's stage was called."""
    record = {
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "cpu_threads": os.environ.get("OPENMM_CPU_THREADS"),
        "kwargs": {
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    with open(_RECORD, "w") as handle:
        json.dump(record, handle)


def _other_stage(**kwargs: Any) -> None:
    """A second module-level stage, for resume stage-mismatch checks."""
    _recording_stage(**kwargs)


def _fail_in_walker_001(**kwargs: Any) -> None:
    """Fail in walker_001 until an ``unbreak`` flag appears in the launch dir."""
    in_broken_walker = os.path.basename(os.getcwd()) == "walker_001"
    if in_broken_walker and not os.path.exists(os.path.join("..", "unbreak")):
        raise RuntimeError("boom")
    _recording_stage(**kwargs)


def _printing_stage(**kwargs: Any) -> None:
    print("stage says hello")


def _raising_stage(**kwargs: Any) -> None:
    raise RuntimeError("boom")


def _stage_records() -> dict[str, dict[str, Any]]:
    """Read every walker directory's stage record, keyed by directory name."""
    records = {}
    for path in sorted(Path(".").glob(f"walker_*/{_RECORD}")):
        records[path.parent.name] = json.loads(path.read_text())
    return records


class _InlineExecutor:
    """Runs submissions synchronously in-process, recording its configuration.

    Spawned children are invisible to coverage, so every orchestration branch
    is exercised through this stand-in; the real pool is exercised separately
    for behavior only.
    """

    instances: ClassVar[list[_InlineExecutor]] = []

    def __init__(self, max_workers: int | None = None, mp_context: Any = None,
                 max_tasks_per_child: int | None = None) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.max_tasks_per_child = max_tasks_per_child
        type(self).instances.append(self)

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:  # noqa: B036 - mirrored into the Future
            future.set_exception(error)
        return future


@pytest.fixture
def inline_pool(monkeypatch: pytest.MonkeyPatch) -> type[_InlineExecutor]:
    """Swap the process pool for the synchronous in-process stand-in."""
    _InlineExecutor.instances = []
    monkeypatch.setattr(walkers, "ProcessPoolExecutor", _InlineExecutor)
    return _InlineExecutor


# ---------------------------------------------------------------------------
# Seed derivation


def test_walker_seeds_are_deterministic_and_distinct() -> None:
    first = walkers._derive_walker_seeds(7, 4)
    assert walkers._derive_walker_seeds(7, 4) == first
    assert len(set(first)) == 4
    assert walkers._derive_walker_seeds(8, 4) != first


def test_walker_seeds_are_stable_when_the_ensemble_grows() -> None:
    # Resuming with more walkers must leave the existing walkers' seeds --
    # and therefore their recorded, completed runs -- exactly as they were.
    assert walkers._derive_walker_seeds(7, 8)[:4] == \
        walkers._derive_walker_seeds(7, 4)


def test_walker_seeds_are_disjoint_from_stage_and_recrossing_streams() -> None:
    walker_seeds = set(walkers._derive_walker_seeds(7, 4))
    stage_seeds = set(
        _derive_seeds(7, "initialization", "thermostat", "velocities",
                      "barostat")
    )
    assert not walker_seeds & stage_seeds
    assert walkers._WALKER_SEED_TAG != nqe_rates._RECROSSING_SEED_TAG


def test_unseeded_walkers_still_get_recorded_integer_seeds() -> None:
    first = walkers._derive_walker_seeds(None, 3)
    assert all(isinstance(seed, int) for seed in first)
    assert walkers._derive_walker_seeds(None, 3) != first


def test_walker_seed_rejects_bool_and_negative_values() -> None:
    for bad_seed in (True, -1, 2.5):
        with pytest.raises(ValueError, match="seed must be"):
            walkers._derive_walker_seeds(bad_seed, 2)


# ---------------------------------------------------------------------------
# Orchestration through the inline executor


def test_each_walker_runs_in_its_own_directory(
    inline_pool: type[_InlineExecutor],
) -> None:
    ensemble = nqe.run_openmm_walkers(_recording_stage, 3, seed=1)

    assert ensemble.directories == ("walker_000", "walker_001", "walker_002")
    assert ensemble.resumed == (False, False, False)
    records = _stage_records()
    assert set(records) == set(ensemble.directories)
    for directory, record in records.items():
        assert record["cwd"] == os.path.abspath(directory)


def test_stage_kwargs_gain_an_injected_per_walker_seed(
    inline_pool: type[_InlineExecutor],
) -> None:
    stage_kwargs = {"steps": 5}
    ensemble = nqe.run_openmm_walkers(_recording_stage, 2,
                                      stage_kwargs, seed=3)

    records = _stage_records()
    for index, directory in enumerate(ensemble.directories):
        assert records[directory]["kwargs"]["seed"] == ensemble.seeds[index]
        assert records[directory]["kwargs"]["steps"] == 5
    # The caller's mapping is merged from, never written to.
    assert stage_kwargs == {"steps": 5}


def test_per_walker_overrides_merge_over_shared_kwargs(
    inline_pool: type[_InlineExecutor],
) -> None:
    nqe.run_openmm_walkers(
        _recording_stage, 2, {"steps": 5, "label": "shared"},
        per_walker_kwargs=[{}, {"label": "special"}], seed=3,
    )

    records = _stage_records()
    assert records["walker_000"]["kwargs"]["label"] == "shared"
    assert records["walker_001"]["kwargs"]["label"] == "special"
    assert records["walker_001"]["kwargs"]["steps"] == 5


def test_inline_plumed_text_is_written_into_each_walker_directory(
    inline_pool: type[_InlineExecutor],
) -> None:
    script = "p: POSITION ATOM=1\nPRINT ARG=p.x STRIDE=5 FILE=COLVAR\n"
    ensemble = nqe.run_openmm_walkers(_recording_stage, 2, seed=1,
                                      plumed_input=script)

    for directory in ensemble.directories:
        assert Path(directory, "plumed.dat").read_text() == script
    records = _stage_records()
    for record in records.values():
        assert record["kwargs"]["plumed_script_path"] == "plumed.dat"


def test_plumed_path_is_copied_into_each_walker_directory(
    inline_pool: type[_InlineExecutor],
) -> None:
    script = "p: POSITION ATOM=1\nPRINT ARG=p.x STRIDE=5 FILE=COLVAR\n"
    Path("bias.dat").write_text(script)

    ensemble = nqe.run_openmm_walkers(_recording_stage, 2, seed=1,
                                      plumed_input="bias.dat")

    for directory in ensemble.directories:
        assert Path(directory, "plumed.dat").read_text() == script


def test_missing_plumed_path_is_an_error(
    inline_pool: type[_InlineExecutor],
) -> None:
    with pytest.raises(FileNotFoundError):
        nqe.run_openmm_walkers(_recording_stage, 2, seed=1,
                               plumed_input=Path("missing") / "bias.dat")


def test_shared_inputs_are_copied_by_basename(
    inline_pool: type[_InlineExecutor],
) -> None:
    payload = Path("inputs") / "rpmd_ready.chk"
    payload.parent.mkdir()
    payload.write_bytes(b"restart bytes")

    ensemble = nqe.run_openmm_walkers(_recording_stage, 2, seed=1,
                                      shared_inputs=[payload])

    for directory in ensemble.directories:
        assert Path(directory, "rpmd_ready.chk").read_bytes() == \
            b"restart bytes"


def test_missing_shared_input_fails_before_any_walker_runs(
    inline_pool: type[_InlineExecutor],
) -> None:
    with pytest.raises(FileNotFoundError, match="shared input"):
        nqe.run_openmm_walkers(_recording_stage, 2, seed=1,
                               shared_inputs=["nowhere.chk"])
    with pytest.raises(ValueError, match="is not a file"):
        os.mkdir("a_directory")
        nqe.run_openmm_walkers(_recording_stage, 2, seed=1,
                               shared_inputs=["a_directory"])
    assert not os.path.exists("walker_000")
    assert not _stage_records()


def test_caller_supplied_seed_kwarg_is_rejected(
    inline_pool: type[_InlineExecutor],
) -> None:
    with pytest.raises(ValueError, match="must not set 'seed'"):
        nqe.run_openmm_walkers(_recording_stage, 2, {"seed": 4})
    with pytest.raises(ValueError, match="must not set 'seed'"):
        nqe.run_openmm_walkers(_recording_stage, 2,
                               per_walker_kwargs=[{}, {"seed": 4}])


def test_plumed_script_path_conflicts_with_plumed_input(
    inline_pool: type[_InlineExecutor],
) -> None:
    with pytest.raises(ValueError, match="plumed_script_path"):
        nqe.run_openmm_walkers(_recording_stage, 2,
                               {"plumed_script_path": "own.dat"},
                               plumed_input="p: POSITION ATOM=1\n")
    # Without plumed_input the caller may route their own script.
    nqe.run_openmm_walkers(_recording_stage, 1,
                           {"plumed_script_path": "own.dat"}, seed=1)
    assert _stage_records()["walker_000"]["kwargs"]["plumed_script_path"] == \
        "own.dat"


def test_per_walker_kwargs_length_must_match_n_walkers(
    inline_pool: type[_InlineExecutor],
) -> None:
    with pytest.raises(ValueError, match="per_walker_kwargs has 1 entries"):
        nqe.run_openmm_walkers(_recording_stage, 2, per_walker_kwargs=[{}])


def test_counts_and_stage_are_validated(
    inline_pool: type[_InlineExecutor],
) -> None:
    with pytest.raises(ValueError, match="n_walkers must be"):
        nqe.run_openmm_walkers(_recording_stage, 0)
    with pytest.raises(TypeError, match="n_walkers must be"):
        nqe.run_openmm_walkers(_recording_stage, True)
    with pytest.raises(ValueError, match="max_workers must be"):
        nqe.run_openmm_walkers(_recording_stage, 2, max_workers=0)
    with pytest.raises(ValueError, match="cpu_threads_per_walker must be"):
        nqe.run_openmm_walkers(_recording_stage, 2, cpu_threads_per_walker=0)
    with pytest.raises(TypeError, match="stage must be a callable"):
        nqe.run_openmm_walkers("not a stage", 2)
    with pytest.raises(ValueError, match="walker_dir_prefix"):
        nqe.run_openmm_walkers(_recording_stage, 2, walker_dir_prefix="")


def test_unpicklable_stage_or_kwargs_fail_fast_with_guidance(
    inline_pool: type[_InlineExecutor],
) -> None:
    with pytest.raises(TypeError, match="picklable"):
        nqe.run_openmm_walkers(lambda **kwargs: None, 1, seed=1)
    with open("handle.txt", "w") as handle:
        with pytest.raises(TypeError, match="picklable"):
            nqe.run_openmm_walkers(_recording_stage, 1, {"stream": handle},
                                   seed=1)
    assert not inline_pool.instances
    assert not os.path.exists("walker_000")


def test_walker_failures_are_aggregated_and_survivors_kept(
    inline_pool: type[_InlineExecutor],
) -> None:
    with pytest.raises(RuntimeError) as failure:
        nqe.run_openmm_walkers(_fail_in_walker_001, 3, seed=5)

    message = str(failure.value)
    assert "1 of 3 walkers failed" in message
    assert "walker_001: RuntimeError: boom" in message
    assert "resume=True" in message
    assert Path("walker_000", "walker.json").exists()
    assert not Path("walker_001", "walker.json").exists()
    assert Path("walker_002", "walker.json").exists()

    # The failed walker's traceback is in its screen.out, not lost.
    assert "RuntimeError: boom" in Path("walker_001", "screen.out").read_text()

    Path("unbreak").touch()
    ensemble = nqe.run_openmm_walkers(_fail_in_walker_001, 3, seed=5,
                                      resume=True)
    assert ensemble.resumed == (True, False, True)
    assert Path("walker_001", "walker.json").exists()


def test_resume_skips_completed_walkers(
    inline_pool: type[_InlineExecutor],
) -> None:
    first = nqe.run_openmm_walkers(_recording_stage, 2, seed=9)
    for directory in first.directories:
        Path(directory, _RECORD).unlink()

    again = nqe.run_openmm_walkers(_recording_stage, 2, seed=9, resume=True)

    assert again.resumed == (True, True)
    assert again.seeds == first.seeds
    assert not _stage_records()
    # With nothing pending, no pool is ever constructed.
    assert len(inline_pool.instances) == 1


def test_resume_rejects_a_different_master_seed(
    inline_pool: type[_InlineExecutor],
) -> None:
    nqe.run_openmm_walkers(_recording_stage, 1, seed=9)
    with pytest.raises(ValueError, match="different master seed"):
        nqe.run_openmm_walkers(_recording_stage, 1, seed=10, resume=True)


def test_resume_rejects_a_different_stage(
    inline_pool: type[_InlineExecutor],
) -> None:
    nqe.run_openmm_walkers(_recording_stage, 1, seed=9)
    with pytest.raises(ValueError, match="resume with the same stage"):
        nqe.run_openmm_walkers(_other_stage, 1, seed=9, resume=True)


def test_completed_walker_without_resume_is_an_error(
    inline_pool: type[_InlineExecutor],
) -> None:
    nqe.run_openmm_walkers(_recording_stage, 1, seed=9)
    with pytest.raises(ValueError, match="resume=True"):
        nqe.run_openmm_walkers(_recording_stage, 1, seed=9)


def test_corrupt_completion_record_is_reported(
    inline_pool: type[_InlineExecutor],
) -> None:
    os.mkdir("walker_000")
    Path("walker_000", "walker.json").write_text("not json")
    with pytest.raises(ValueError, match="unreadable walker record"):
        nqe.run_openmm_walkers(_recording_stage, 1, seed=9, resume=True)

    Path("walker_000", "walker.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="not an object"):
        nqe.run_openmm_walkers(_recording_stage, 1, seed=9, resume=True)


def test_unseeded_resume_reruns_only_incomplete_walkers(
    inline_pool: type[_InlineExecutor],
) -> None:
    first = nqe.run_openmm_walkers(_recording_stage, 2)
    for directory in first.directories:
        Path(directory, _RECORD).unlink()
    Path("walker_001", "walker.json").unlink()

    # A fresh entropy root cannot re-derive what the first launch drew, so
    # the completed walker is kept without a seed check.
    again = nqe.run_openmm_walkers(_recording_stage, 2, resume=True)

    assert again.resumed == (True, False)
    assert set(_stage_records()) == {"walker_001"}


def test_executor_is_configured_for_spawn_one_task_per_child(
    inline_pool: type[_InlineExecutor],
) -> None:
    nqe.run_openmm_walkers(_recording_stage, 3, seed=1)

    (pool,) = inline_pool.instances
    assert pool.max_workers == 3
    assert pool.max_tasks_per_child == 1
    assert pool.mp_context.get_start_method() == "spawn"

    nqe.run_openmm_walkers(_recording_stage, 3, seed=1,
                           walker_dir_prefix="capped_", max_workers=2)
    assert inline_pool.instances[-1].max_workers == 2


# ---------------------------------------------------------------------------
# The worker, called directly in-process


def _spec(directory: str, stage: Any, *, cpu_threads: int | None = None,
          redirect_output: bool = True) -> walkers._WalkerSpec:
    os.makedirs(directory, exist_ok=True)
    return walkers._WalkerSpec(
        walker=0,
        directory=os.path.abspath(directory),
        stage=stage,
        stage_kwargs={},
        seed=42,
        stage_name=walkers._stage_name(stage),
        cpu_threads=cpu_threads,
        redirect_output=redirect_output,
    )


def test_worker_redirects_stage_output_to_screen_out_and_restores_fds(
    capfd: pytest.CaptureFixture[str],
) -> None:
    walkers._run_single_walker(_spec("walker_000", _printing_stage))

    assert "stage says hello" in Path("walker_000", "screen.out").read_text()
    assert "stage says hello" not in capfd.readouterr().out
    print("back on the console")
    assert "back on the console" in capfd.readouterr().out


def test_worker_restores_cwd_and_environment_after_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMM_CPU_THREADS", "5")
    launch_directory = os.getcwd()

    with pytest.raises(RuntimeError, match="boom"):
        walkers._run_single_walker(
            _spec("walker_000", _raising_stage, cpu_threads=2))

    assert os.getcwd() == launch_directory
    assert os.environ["OPENMM_CPU_THREADS"] == "5"
    assert "RuntimeError: boom" in Path("walker_000", "screen.out").read_text()
    assert not Path("walker_000", "walker.json").exists()


def test_worker_sets_openmm_cpu_threads_for_the_stage() -> None:
    assert "OPENMM_CPU_THREADS" not in os.environ

    walkers._run_single_walker(
        _spec("walker_000", _recording_stage, cpu_threads=2))

    record = json.loads(Path("walker_000", _RECORD).read_text())
    assert record["cpu_threads"] == "2"
    assert "OPENMM_CPU_THREADS" not in os.environ


def test_worker_writes_the_completion_record() -> None:
    walkers._run_single_walker(_spec("walker_000", _recording_stage))

    record = json.loads(Path("walker_000", "walker.json").read_text())
    assert record == {
        "walker": 0,
        "seed": 42,
        "stage": f"{__name__}:_recording_stage",
    }


def test_worker_without_redirect_leaves_stdout_alone(
    capfd: pytest.CaptureFixture[str],
) -> None:
    walkers._run_single_walker(
        _spec("walker_000", _printing_stage, redirect_output=False))

    assert "stage says hello" in capfd.readouterr().out
    assert not Path("walker_000", "screen.out").exists()


# ---------------------------------------------------------------------------
# Real processes and real PLUMED (behavioral; coverage comes from the above)


def test_walkers_run_in_separate_spawned_processes() -> None:
    ensemble = nqe.run_openmm_walkers(_recording_stage, 3, seed=1,
                                      max_workers=2)

    records = _stage_records()
    pids = {record["pid"] for record in records.values()}
    assert len(pids) == 3
    assert os.getpid() not in pids
    for directory in ensemble.directories:
        assert Path(directory, "walker.json").exists()


def test_seeded_walker_production_is_reproducible_across_real_pools(
    one_particle_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = one_particle_system
    prepared = nqe.PreparedSystem(forcefield.createSystem(modeller.topology))
    stage_kwargs = {
        "modeller": modeller,
        "forcefield": prepared,
        "barostat_freq": None,
        "steps": 20,
        "n_report": 10,
        "output_prefix": "prod",
        "platform_name": "Reference",
    }

    def trajectories() -> list[bytes]:
        ensemble = nqe.run_openmm_walkers(nqe.run_openmm_prod, 2,
                                          stage_kwargs, seed=11)
        return [Path(directory, "prod_steps.pdb").read_bytes()
                for directory in ensemble.directories]

    first = trajectories()
    assert first[0] != first[1]

    for directory in ("walker_000", "walker_001"):
        shutil.rmtree(directory)
    assert trajectories() == first


def test_real_plumed_walkers_write_per_directory_colvars(
    inline_pool: type[_InlineExecutor],
    one_particle_system: tuple[app.Modeller, Any],
) -> None:
    modeller, forcefield = one_particle_system
    prepared = nqe.PreparedSystem(forcefield.createSystem(modeller.topology))

    ensemble = nqe.run_openmm_walkers(
        nqe.run_openmm_prod,
        2,
        {
            "modeller": modeller,
            "forcefield": prepared,
            "barostat_freq": None,
            "steps": 10,
            "n_report": 5,
            "platform_name": "Reference",
        },
        plumed_input="p: POSITION ATOM=1\nPRINT ARG=p.x STRIDE=5 FILE=COLVAR\n",
        seed=11,
    )

    for directory in ensemble.directories:
        colvar = Path(directory, "COLVAR")
        assert colvar.exists()
        assert colvar.stat().st_size > 0
