"""Shared text-reporter file ownership, including failed construction."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openmm.unit as unit
import pytest

import openmmnqe._reporting as reporting
from openmmnqe import (
    QTBFrictionReporter,
    RPMDKineticDecompositionReporter,
    RPMDQuantumSpreadReporter,
    RPMDThermodynamicReporter,
)


@pytest.fixture(params=["spread", "thermodynamic", "kinetic", "friction"])
def reporter_constructor(request: pytest.FixtureRequest) -> tuple[type, tuple[Any, ...]]:
    """The public text reporters, with valid arguments but no simulation."""
    if request.param == "spread":
        return RPMDQuantumSpreadReporter, (5, [0])
    if request.param == "thermodynamic":
        return RPMDThermodynamicReporter, (5,)
    if request.param == "kinetic":
        return RPMDKineticDecompositionReporter, (5, [0])
    integrator = SimpleNamespace(
        getParticleTypes=lambda: {0: 0},
        getStepSize=lambda: 0.001 * unit.picosecond,
        getSegmentLength=lambda: 0.005 * unit.picosecond,
    )
    return QTBFrictionReporter, (5, integrator)


def test_text_reporters_close_when_the_context_body_fails(
    tmp_path: Path, reporter_constructor: tuple[type, tuple[Any, ...]],
) -> None:
    reporter_type, arguments = reporter_constructor
    reporter = reporter_type(tmp_path / "log.tsv", *arguments)

    with pytest.raises(RuntimeError, match="simulation failed"):
        with reporter as entered:
            assert entered is reporter
            raise RuntimeError("simulation failed")

    assert reporter._out.closed
    reporter.close()
    reporter.__del__()


def test_text_reporters_tolerate_cleanup_before_initialization(
    reporter_constructor: tuple[type, tuple[Any, ...]],
) -> None:
    reporter_type, _ = reporter_constructor
    reporter = reporter_type.__new__(reporter_type)
    reporter.close()
    reporter.__del__()


def test_text_reporters_release_a_file_after_header_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reporter_constructor: tuple[type, tuple[Any, ...]],
) -> None:
    class FailingHeader(StringIO):
        def write(self, text: str) -> int:
            raise OSError("header write failed")

    output = FailingHeader()
    monkeypatch.setattr(reporting, "open", lambda *args: output, raising=False)
    reporter_type, arguments = reporter_constructor
    reporter = reporter_type.__new__(reporter_type)

    with pytest.raises(OSError, match="header write failed"):
        reporter_type.__init__(reporter, tmp_path / "log.tsv", *arguments)
    # Explicitly exercise the fallback on the partially initialized object;
    # there is no reliance on garbage-collection timing in this assertion.
    reporter.__del__()

    assert output.closed


def test_tabular_rows_are_flushed_and_not_partially_written_on_format_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "log.tsv"
    with reporting._TabularReporter(path, "Step\tValue") as reporter:
        reporter._write_row(7, [1.23456789])
        assert path.read_text() == "Step\tValue\n7\t1.234568\n"
        with pytest.raises(ValueError):
            reporter._write_row(8, [1.0, "unformattable"])
        assert path.read_text() == "Step\tValue\n7\t1.234568\n"


def test_text_reporters_swallow_close_errors_during_del(tmp_path: Path) -> None:
    reporter = reporting._TabularReporter(tmp_path / "log.tsv", "Step\tValue")
    handle = reporter._out

    def refuse() -> None:
        raise OSError("flush failed")

    reporter._out = SimpleNamespace(closed=False, close=refuse)
    reporter.__del__()
    with pytest.raises(OSError, match="flush failed"):
        reporter.close()
    handle.close()
