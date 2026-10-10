"""Every plot helper names the ``plot`` extra when matplotlib is missing."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import pytest

import openmmnqe as nqe


@pytest.mark.parametrize(
    ("plot", "kwargs"),
    [
        (nqe.plot_rpmd_atom_expansion, {}),
        (nqe.plot_rpmd_thermodynamics, {}),
        (nqe.plot_rpmd_kinetic_decomposition, {"temperature": 300.0}),
        (nqe.plot_transmission_coefficient, {"s_dagger": 0.0}),
        (nqe.plot_adqtb_friction_spectra, {}),
        (nqe.plot_adqtb_fdt_residual, {}),
    ],
)
def test_plot_helpers_point_at_the_plot_extra_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
    plot: Callable[..., Any],
    kwargs: dict[str, Any],
) -> None:
    # A None entry in sys.modules makes the import fail even though
    # matplotlib is installed, which is what a bare install looks like.
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.raises(
        ImportError,
        match=rf"^{plot.__name__} requires matplotlib; install the 'plot' optional dependency$",
    ):
        plot("absent.log", **kwargs)
