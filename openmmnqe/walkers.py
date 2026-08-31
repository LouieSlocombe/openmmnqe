"""
Independent multi-walker orchestration for biased production stages.

:func:`run_openmm_walkers` runs N copies of one ``run_openmm_*`` stage as
parallel *processes*, one working directory per walker, each with its own
derived random seeds and (optionally) one shared PLUMED script. The walkers
are fully independent: this project's PLUMED is built without MPI, so
``WALKERS_MPI`` is unavailable, and ``OPES_METAD`` has no file-based
multiple-walkers scheme -- independent runs combined afterwards by
reweighting are the supported route. Merge the per-walker ``COLVAR`` files
with ``reactiontools.combine_colvar_files`` and reconstruct the surface with
``reactiontools.run_opes_reweighting``.

Because every walker executes a whole driver, everything a stage does
internally happens per walker unchanged -- the reporters, the seed streams,
and for RPMD stages the centroid-bias contraction that keeps PLUMED acting
on the centroid rather than on each bead.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import shutil
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np

from ._validation import require_integer
from .openmm import _is_inline_plumed_input

# Mixed into the root SeedSequence alongside the master seed, so walker
# streams are disjoint from both the stage streams (openmm._derive_seeds
# spawns from the bare seed) and the recrossing streams (rates mixes in
# 0x52504D44, "RPMD") for the same master seed. 0x57414C4B is "WALK".
_WALKER_SEED_TAG = 0x57414C4B

#: Name the shared PLUMED script is written under inside each walker
#: directory, and the ``plumed_script_path`` injected into the stage.
_PLUMED_FILENAME = "plumed.dat"

#: Completion record a walker writes after its stage returns. Its presence
#: is the completion flag ``resume=True`` checks.
_RECORD_FILENAME = "walker.json"

#: Where a walker's stdout and stderr go, so N concurrent stages do not
#: interleave on the parent's console.
_SCREEN_FILENAME = "screen.out"


class WalkerEnsemble(NamedTuple):
    """
    What a multi-walker launch produced.

    Attributes
    ----------
    directories : tuple of str
        Walker directory names relative to the launch directory, in walker
        order -- ``('walker_000', 'walker_001', ...)`` by default. Each
        holds that walker's stage outputs, its ``plumed.dat`` and PLUMED
        outputs, its ``screen.out``, and its ``walker.json`` completion
        record.
    seeds : tuple of int
        The per-walker master seeds actually used, recorded even when the
        launch itself was unseeded.
    resumed : tuple of bool
        True where an already-complete walker was skipped rather than run.
    """

    directories: tuple[str, ...]
    seeds: tuple[int, ...]
    resumed: tuple[bool, ...]


class _WalkerSpec(NamedTuple):
    """Everything one walker's worker process needs, all of it picklable."""

    walker: int
    directory: str
    stage: Callable[..., object]
    stage_kwargs: dict[str, object]
    seed: int
    stage_name: str
    cpu_threads: int | None
    redirect_output: bool


def _stage_name(stage: Callable[..., object]) -> str:
    """Identify a stage callable the way its completion record spells it."""
    return f"{getattr(stage, '__module__', '?')}:{getattr(stage, '__qualname__', repr(stage))}"


def _derive_walker_seeds(seed: int | None, n_walkers: int) -> tuple[int, ...]:
    """
    Derive one independent master seed per walker.

    The root sequence carries :data:`_WALKER_SEED_TAG` alongside the master
    seed, so walker streams are disjoint from every stage stream
    ``openmmnqe.openmm._derive_seeds`` hands out for the same master seed,
    and from the recrossing streams in :mod:`openmmnqe.rates`. Each walker's
    seed depends only on ``(seed, walker index)``, never on execution order
    or on how many walkers there are -- growing the ensemble on a resumed
    launch leaves every existing walker's seed what it was.

    Parameters
    ----------
    seed : int or None
        Non-negative master seed, or None to draw the root from entropy.
        Either way every walker gets a concrete recorded seed, so an
        unseeded launch is still reproducible from its records.
    n_walkers : int
        Number of walker seeds wanted.

    Returns
    -------
    tuple of int
        One seed per walker, each a valid master seed for any stage.

    Raises
    ------
    ValueError
        If *seed* is a bool, not an integer, or negative.
    """
    if seed is not None and (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")
    if seed is None:
        root = np.random.SeedSequence()
    else:
        root = np.random.SeedSequence([int(seed), _WALKER_SEED_TAG])
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in root.spawn(n_walkers)
    )


def _resolve_plumed_text(plumed_input: str | os.PathLike[str] | None) -> str | None:
    """
    Turn the *plumed_input* argument into script text, or None.

    Inline text and paths are told apart exactly as
    :func:`openmmnqe.openmm.run_openmm_steered` tells them apart; a path is
    read once here, so every walker gets identical content even if the file
    changes mid-launch.

    Parameters
    ----------
    plumed_input : str, os.PathLike, or None
        Inline PLUMED script text, a path to one, or None for no bias.

    Returns
    -------
    str or None
        The script text, or None when *plumed_input* is None.

    Raises
    ------
    FileNotFoundError
        If *plumed_input* names a file that does not exist.
    """
    if plumed_input is None:
        return None
    if _is_inline_plumed_input(plumed_input):
        return str(plumed_input)
    with open(plumed_input) as handle:
        return handle.read()


@contextmanager
def _working_directory(directory: str) -> Iterator[None]:
    """Run the body with *directory* as the process working directory."""
    previous = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _cpu_thread_limit(threads: int | None) -> Iterator[None]:
    """
    Cap OpenMM's CPU platform threads for the body, or do nothing.

    Parameters
    ----------
    threads : int or None
        Value for ``OPENMM_CPU_THREADS``, or None to leave the environment
        alone. The variable is read when a CPU Context is created, so it
        has to be in place before the stage builds its Simulation.
    """
    if threads is None:
        yield
        return
    previous = os.environ.get("OPENMM_CPU_THREADS")
    os.environ["OPENMM_CPU_THREADS"] = str(threads)
    try:
        yield
    finally:
        if previous is None:
            del os.environ["OPENMM_CPU_THREADS"]
        else:
            os.environ["OPENMM_CPU_THREADS"] = previous


@contextmanager
def _output_to_file(filename: str, enabled: bool) -> Iterator[None]:
    """
    Send the body's stdout and stderr into *filename*, or do nothing.

    The redirection happens at both levels: the raw descriptors, because
    OpenMM and PLUMED write from C++ straight to those, and the
    ``sys.stdout``/``sys.stderr`` objects, because whatever currently
    holds them -- a console wrapper, a test harness's capture -- need not
    go through descriptor 1 at all.

    Parameters
    ----------
    filename : str
        File the output is appended to, relative to the current working
        directory.
    enabled : bool
        False leaves both streams alone, for callers that already manage
        them.
    """
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    screen = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(screen, 1)
    os.dup2(screen, 2)
    os.close(screen)
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    replacement = os.fdopen(os.dup(1), "w")
    sys.stdout = replacement
    sys.stderr = replacement
    try:
        yield
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        replacement.close()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def _run_single_walker(spec: _WalkerSpec) -> None:
    """
    Run one walker's stage inside its directory, then record completion.

    This is the function each worker process executes. Every environment
    change is restored on the way out, so it is equally safe to call
    in-process.

    Parameters
    ----------
    spec : _WalkerSpec
        The walker to run. ``spec.directory`` must already exist and hold
        whatever files the stage expects.
    """
    with _working_directory(spec.directory):
        with _cpu_thread_limit(spec.cpu_threads), \
                _output_to_file(_SCREEN_FILENAME, spec.redirect_output):
            try:
                spec.stage(**dict(spec.stage_kwargs))
            except Exception:
                # The parent only gets the pickled exception; the full
                # traceback belongs next to the walker's own outputs.
                traceback.print_exc()
                raise
        record = {"walker": spec.walker, "seed": spec.seed,
                  "stage": spec.stage_name}
        with open(_RECORD_FILENAME, "w") as handle:
            json.dump(record, handle)
            handle.write("\n")


def _read_walker_record(directory: str) -> dict[str, object] | None:
    """
    Read a walker directory's completion record, if there is one.

    Parameters
    ----------
    directory : str
        The walker directory.

    Returns
    -------
    dict or None
        The record, or None when the walker never completed.

    Raises
    ------
    ValueError
        If the record file exists but is not valid JSON.
    """
    path = os.path.join(directory, _RECORD_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            record = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"unreadable walker record {path!r}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"unreadable walker record {path!r}: not an object")
    return record


def _check_resumable(record: dict[str, object], directory: str,
                     stage_name: str, walker_seed: int,
                     validate_seed: bool) -> None:
    """
    Reject a completion record that does not belong to this launch.

    Parameters
    ----------
    record : dict
        The record read from *directory*.
    directory : str
        The walker directory, for error messages.
    stage_name : str
        ``module:qualname`` of the stage this launch runs.
    walker_seed : int
        The seed this launch derived for the walker.
    validate_seed : bool
        Whether the recorded seed has to match -- only when the launch
        gave a master seed, since an unseeded launch cannot re-derive what
        an earlier one drew.

    Raises
    ------
    ValueError
        If the record names a different stage or, when *validate_seed*,
        a different seed.
    """
    if record.get("stage") != stage_name:
        raise ValueError(
            f"{directory} was run with stage {record.get('stage')!r}, not "
            f"{stage_name!r}; resume with the same stage or remove the "
            "directory"
        )
    if validate_seed and record.get("seed") != walker_seed:
        raise ValueError(
            f"{directory} was run with a different master seed; walkers "
            "from different seeds are different ensembles, so rerun with "
            "the original seed or remove the directory"
        )


def run_openmm_walkers(
    stage: Callable[..., object],
    n_walkers: int,
    stage_kwargs: Mapping[str, object] | None = None,
    *,
    plumed_input: str | os.PathLike[str] | None = None,
    seed: int | None = None,
    walker_dir_prefix: str = "walker_",
    shared_inputs: Sequence[str | os.PathLike[str]] = (),
    per_walker_kwargs: Sequence[Mapping[str, object]] | None = None,
    max_workers: int | None = None,
    cpu_threads_per_walker: int | None = None,
    resume: bool = False,
) -> WalkerEnsemble:
    """
    Run *n_walkers* independent copies of a stage as parallel processes.

    Each walker gets its own directory under the current one, its own
    derived master seed, a copy of the shared PLUMED script and input
    files, and a fresh worker process to run the stage in -- so the
    CWD-relative outputs of concurrent walkers never collide, and PLUMED's
    ``COLVAR``, ``STATE`` and ``KERNELS`` land per walker.

    Parameters
    ----------
    stage : callable
        The driver to run, e.g. :func:`openmmnqe.openmm.run_openmm_prod` or
        :func:`openmmnqe.openmm.run_openmm_rpmd_prod`. It must be a
        module-level function, because worker processes import it by name.
        Its return value is discarded -- stages report through the files
        they write.
    n_walkers : int
        How many walkers to run.
    stage_kwargs : mapping, optional
        Keyword arguments for every walker's stage call. ``seed`` may not
        appear here -- each walker gets its own derived seed -- and
        ``plumed_script_path`` may not appear when *plumed_input* is given.
        Every value must be picklable; ``Modeller``, ``ForceField`` and
        :class:`openmmnqe.openmm.PreparedSystem` all are.
    plumed_input : str, os.PathLike, or None, optional
        Inline PLUMED script text or a path to one, told apart as
        :func:`openmmnqe.openmm.run_openmm_steered` tells them apart. The
        script is written into each walker directory as ``plumed.dat`` and
        the stage is called with ``plumed_script_path='plumed.dat'``.
        Default is None, no bias.
    seed : int or None, optional
        Master seed the per-walker seeds are derived from, disjoint from
        what the same value seeds in any single stage. None, the default,
        draws from entropy; the seeds actually used are still recorded and
        returned.
    walker_dir_prefix : str, optional
        Prefix of the per-walker directory names, completed with the
        zero-padded walker index. Default is ``'walker_'``.
    shared_inputs : sequence of path-like, optional
        Files copied by basename into every walker directory before the
        launch -- a restart archive such as ``rpmd_ready.chk``, reference
        structures, anything the stage reads from its working directory.
    per_walker_kwargs : sequence of mapping, optional
        One mapping per walker, merged over *stage_kwargs* for that walker
        -- different starting ``Modeller`` objects is the usual use. Must
        have exactly *n_walkers* entries.
    max_workers : int or None, optional
        How many walkers run at once. None, the default, runs all pending
        walkers concurrently; ``max_workers=1`` serialises them, which on
        a single GPU costs little wall-clock since concurrent contexts
        time-slice it anyway.
    cpu_threads_per_walker : int or None, optional
        Sets ``OPENMM_CPU_THREADS`` inside each worker, so CPU-platform
        walkers divide the machine instead of oversubscribing it. Default
        is None, leave OpenMM's own choice alone.
    resume : bool, optional
        Skip walkers whose directory holds a completion record from an
        identical launch, and run only the rest. A walker directory
        without a record -- a crashed or killed walker -- is rerun in
        place. Default is False, in which case an existing record is an
        error rather than silently ignored work.

    Returns
    -------
    WalkerEnsemble
        The walker directories, the per-walker seeds, and which walkers
        were resumed rather than run.

    Raises
    ------
    TypeError
        If *stage* is not callable, or a walker's stage and arguments do
        not pickle -- lambdas, open files, live Simulations and most ML
        calculators do not.
    ValueError
        If counts or *seed* are invalid, reserved keyword arguments are
        set, *per_walker_kwargs* has the wrong length, a completed walker
        is found without ``resume=True``, or a completion record belongs
        to a different stage or master seed.
    FileNotFoundError
        If *plumed_input* names a missing file, or a shared input is
        missing.

    Notes
    -----
    The walkers are independent because that is what this build can do
    honestly: PLUMED's ``OPES_METAD`` shares bias between walkers only
    through ``WALKERS_MPI``, which needs an MPI-enabled PLUMED and an MPI
    communicator neither openmm-plumed nor this package provides. N
    independent walkers still cut the wall-clock per effective sample and,
    combined by reweighting, give the cross-walker scatter that a single
    run cannot -- merge the ``COLVAR`` files with
    ``reactiontools.combine_colvar_files`` and rebuild the surface with
    ``reactiontools.run_opes_reweighting``.

    Worker processes are spawned, not forked, so CUDA contexts stay valid;
    each walker gets a fresh interpreter, and everything sent to one --
    the stage and its keyword arguments -- crosses by pickle, which is
    what the module-level-function requirement and the pickling preflight
    are about. A walker's stdout and stderr go to ``screen.out`` in its
    directory.

    When some walkers fail, the rest are still run to completion, the
    failures are then raised together, and every completed walker keeps
    its results -- so ``resume=True`` on the next call runs only what is
    missing.
    """
    n_walkers = require_integer(n_walkers, name="n_walkers", minimum=1)
    if max_workers is not None:
        max_workers = require_integer(max_workers, name="max_workers",
                                      minimum=1)
    if cpu_threads_per_walker is not None:
        cpu_threads_per_walker = require_integer(
            cpu_threads_per_walker, name="cpu_threads_per_walker", minimum=1)
    if not callable(stage):
        raise TypeError("stage must be a callable run_openmm_* driver")
    if not isinstance(walker_dir_prefix, str) or not walker_dir_prefix:
        raise ValueError("walker_dir_prefix must be a non-empty string")
    if per_walker_kwargs is not None and len(per_walker_kwargs) != n_walkers:
        raise ValueError(
            f"per_walker_kwargs has {len(per_walker_kwargs)} entries for "
            f"{n_walkers} walkers"
        )

    plumed_text = _resolve_plumed_text(plumed_input)

    merged_kwargs: list[dict[str, object]] = []
    for index in range(n_walkers):
        merged = dict(stage_kwargs or {})
        if per_walker_kwargs is not None:
            merged.update(per_walker_kwargs[index])
        if "seed" in merged:
            raise ValueError(
                "stage_kwargs must not set 'seed'; run_openmm_walkers "
                "derives one per walker from its own seed argument"
            )
        if plumed_text is not None and "plumed_script_path" in merged:
            raise ValueError(
                "stage_kwargs must not set 'plumed_script_path' when "
                "plumed_input is given; the script is placed in each "
                "walker directory"
            )
        merged_kwargs.append(merged)

    shared_files = [os.fspath(path) for path in shared_inputs]
    for path in shared_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"shared input {path!r} does not exist")
        if not os.path.isfile(path):
            raise ValueError(f"shared input {path!r} is not a file")

    seeds = _derive_walker_seeds(seed, n_walkers)
    stage_name = _stage_name(stage)
    pad = max(3, len(str(n_walkers - 1)))
    directories = tuple(
        f"{walker_dir_prefix}{index:0{pad}d}" for index in range(n_walkers)
    )

    resumed = []
    for index, directory in enumerate(directories):
        record = _read_walker_record(directory)
        if record is None:
            resumed.append(False)
            continue
        if not resume:
            raise ValueError(
                f"{directory} already holds a completed walker; pass "
                "resume=True to keep it, or remove the directory to rerun"
            )
        _check_resumable(record, directory, stage_name, seeds[index],
                         validate_seed=seed is not None)
        resumed.append(True)

    specs = []
    for index, directory in enumerate(directories):
        if resumed[index]:
            continue
        walker_kwargs = dict(merged_kwargs[index])
        walker_kwargs["seed"] = seeds[index]
        if plumed_text is not None:
            walker_kwargs["plumed_script_path"] = _PLUMED_FILENAME
        spec = _WalkerSpec(
            walker=index,
            directory=os.path.abspath(directory),
            stage=stage,
            stage_kwargs=walker_kwargs,
            seed=seeds[index],
            stage_name=stage_name,
            cpu_threads=cpu_threads_per_walker,
            redirect_output=True,
        )
        try:
            pickle.dumps((spec.stage, spec.stage_kwargs))
        except Exception as exc:
            raise TypeError(
                f"walker {index} cannot be sent to a worker process: "
                f"{exc}. The stage must be a module-level function and "
                "every stage_kwargs value picklable; Modeller, ForceField "
                "and PreparedSystem are, while lambdas, open files, live "
                "Simulations and most ML calculators are not."
            ) from exc
        specs.append(spec)

    for spec in specs:
        os.makedirs(spec.directory, exist_ok=True)
        if plumed_text is not None:
            plumed_path = os.path.join(spec.directory, _PLUMED_FILENAME)
            with open(plumed_path, "w") as handle:
                handle.write(plumed_text)
        for path in shared_files:
            shutil.copy2(path, spec.directory)

    if specs:
        failures: list[tuple[_WalkerSpec, BaseException]] = []
        with ProcessPoolExecutor(
            max_workers=max_workers or len(specs),
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        ) as pool:
            futures = [(spec, pool.submit(_run_single_walker, spec))
                       for spec in specs]
            for spec, future in futures:
                error = future.exception()
                if error is not None:
                    failures.append((spec, error))
        if failures:
            details = "\n".join(
                f"  {directories[spec.walker]}: "
                f"{type(error).__name__}: {error}"
                for spec, error in failures
            )
            raise RuntimeError(
                f"{len(failures)} of {len(specs)} walkers failed:\n"
                f"{details}\n"
                "See screen.out in each failed directory; completed "
                "walkers keep their results and are skipped by resume=True."
            ) from failures[0][1]

    return WalkerEnsemble(
        directories=directories,
        seeds=seeds,
        resumed=tuple(resumed),
    )
