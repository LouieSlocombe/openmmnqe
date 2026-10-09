"""Shared trajectory/velocity selection policy and distinct observable policy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmmnqe import RPMDVelocityReporter, TrajectoryOptions, VelocityArchiveReporter
from openmmnqe.openmm import _resolve_trajectory_options
from openmmnqe.reporters import (
    RPMDQuantumSpreadReporter,
    _validate_atom_indices,
    _validate_distance_pairs,
)


@pytest.fixture(params=["trajectory", "classical_velocity", "rpmd_velocity"])
def normalize_selection(
    request: pytest.FixtureRequest, tmp_path: Path,
) -> Callable[[Iterable[Any] | None], list[int] | None]:
    """Exercise each entry point through its existing selection argument."""
    def normalize(indices: Iterable[Any] | None) -> list[int] | None:
        if request.param == "trajectory":
            return _resolve_trajectory_options(
                TrajectoryOptions(atom_indices=indices), 10,
            ).atom_indices
        reporter_type = (
            VelocityArchiveReporter if request.param == "classical_velocity"
            else RPMDVelocityReporter
        )
        with reporter_type(tmp_path / "velocities.npz", 10, indices) as reporter:
            return reporter._atom_indices

    return normalize


def test_unique_selection_keeps_order_normalizes_numpy_ints_and_consumes_once(
    normalize_selection: Callable[[Iterable[Any] | None], list[int] | None],
) -> None:
    original = [np.int64(2), np.int32(0), np.int64(1)]
    consumed = []

    def indices() -> Iterable[np.integer]:
        for index in original:
            consumed.append(index)
            yield index

    selected = normalize_selection(indices())

    assert selected == [2, 0, 1]
    assert selected is not None and all(type(index) is int for index in selected)
    assert consumed == original
    assert original == [2, 0, 1]


def test_absent_selection_still_means_every_atom(
    normalize_selection: Callable[[Iterable[Any] | None], list[int] | None],
) -> None:
    assert normalize_selection(None) is None


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        ([], ValueError, "atom_indices must not be empty"),
        ([1, 1], ValueError, "atom_indices contains duplicate indices"),
        ([True], TypeError, "atom_indices[0] must be an integer"),
        ([np.bool_(False)], TypeError, "atom_indices[0] must be an integer"),
        ([1.0], TypeError, "atom_indices[0] must be an integer"),
        ([0, np.nan], TypeError, "atom_indices[1] must be an integer"),
        ([-1], ValueError, "atom_indices[0] must be a non-negative integer"),
        ([-1, 0.5], ValueError, "atom_indices[0] must be a non-negative integer"),
        ([0.5, -1], TypeError, "atom_indices[0] must be an integer"),
        ([0, 0, 0.5], TypeError, "atom_indices[2] must be an integer"),
    ],
)
def test_unique_selection_preserves_errors_and_validation_order(
    normalize_selection: Callable[[Iterable[Any] | None], list[int] | None],
    indices: list[Any], error: type[Exception], message: str,
) -> None:
    with pytest.raises(error) as failure:
        normalize_selection(iter(indices))
    assert str(failure.value) == message


def test_observables_still_allow_repeated_atoms_and_distance_pairs(
    tmp_path: Path,
) -> None:
    assert _validate_atom_indices([np.int64(1), 1]) == [1, 1]
    assert _validate_distance_pairs([(0, 1), (0, 1)]) == [(0, 1), (0, 1)]
    with RPMDQuantumSpreadReporter(
        tmp_path / "spread.tsv", 10, [0, 0], names=["first", "second"],
    ) as reporter:
        assert reporter._atom_indices == [0, 0]


def test_observable_validation_keeps_its_type_before_range_error_policy() -> None:
    with pytest.raises(TypeError) as failure:
        _validate_atom_indices([-1, 0.5])
    assert str(failure.value) == "atom_indices must be integers"
