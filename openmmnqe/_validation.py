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
