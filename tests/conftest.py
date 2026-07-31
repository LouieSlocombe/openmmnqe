"""Shared setup for the test suite.

Two things every test module relies on and none of them should have to arrange
for itself.

**Working directory.**  The tests name their inputs as ``tests/data/...`` and
write their outputs into the current directory, so they only work when pytest
is launched from the repository root.  Running the session from there makes
that true however pytest was invoked.

**Matplotlib backend.**  Many of these tests double as research scripts and end
in ``plt.show()``.  Under pytest there is nobody to close the window, so the
backend is forced to Agg for the whole session and ``show()`` becomes a no-op.
This used to happen by accident -- ``test_plotting`` switched the backend at
import time, which pytest does for every module during collection -- so it took
effect only because that module existed.  It is now deliberate.
"""
import os
from pathlib import Path

import matplotlib

REPO_ROOT = Path(__file__).resolve().parent.parent

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

import pytest  # noqa: E402


def pytest_configure(config):
    os.chdir(REPO_ROOT)


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test leaves behind."""
    yield
    plt.close("all")
