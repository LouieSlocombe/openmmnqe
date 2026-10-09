"""Small, shared runtime validators for public numeric arguments."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
import openmm.unit as unit


def require_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    """Return *value* as an ``int`` after strict integer validation.

    Booleans are rejected explicitly: Python treats ``bool`` as an integer,
    but accepting ``True`` for a step count or bead count is almost always a
    caller mistake.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")

    result = int(value)
    if minimum is not None and result < minimum:
        if minimum == 0:
            requirement = "a non-negative integer"
        elif minimum == 1:
            requirement = "a positive integer"
        else:
            requirement = f"an integer greater than or equal to {minimum}"
        raise ValueError(f"{name} must be {requirement}")
    return result


def require_scalar_in_unit(
    value: object,
    expected_unit: Any,
    *,
    name: str,
) -> float:
    """Convert a quantity or bare number to a scalar in *expected_unit*."""
    try:
        if unit.is_quantity(value):
            converted = cast(Any, value).value_in_unit(expected_unit)
        else:
            converted = value
        if isinstance(converted, (bool, np.bool_)) or not isinstance(
            converted,
            Real,
        ):
            raise TypeError
        return float(converted)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a scalar with units compatible with "
            f"{expected_unit}"
        ) from exc


def require_positive_finite_scalar_in_unit(
    value: object,
    expected_unit: Any,
    *,
    name: str,
) -> float:
    """Return a scalar in *expected_unit* after physical-domain validation."""
    result = require_scalar_in_unit(value, expected_unit, name=name)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def require_seed(seed: object) -> int | None:
    """Validate a NumPy/OpenMM master seed without changing stream derivation.

    These entry points historically raise ValueError for all invalid seeds;
    require_integer has a different type/error contract.
    """
    if seed is None:
        return None
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")
    return int(seed)


def require_rpmd_n_beads(n_beads: object) -> int:
    """Return a positive bead count, preserving the RPMD ValueError contract."""
    if (
        isinstance(n_beads, (bool, np.bool_))
        or not isinstance(n_beads, (int, np.integer))
        or n_beads <= 0
    ):
        raise ValueError("n_beads must be a positive integer")
    return int(n_beads)


def require_discard_fraction(discard: object) -> float:
    """Validate a log's leading discard fraction without choosing a row policy."""
    if isinstance(discard, bool) or not isinstance(discard, Real):
        raise ValueError("discard must be a number in [0, 1)")
    result = float(discard)
    if not np.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("discard must be a number in [0, 1)")
    return result
