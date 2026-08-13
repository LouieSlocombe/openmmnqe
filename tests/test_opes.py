"""Regression tests for the standalone OPES reweighting scripts."""

import subprocess
import sys
from pathlib import Path

import numpy as np


def test_two_dimensional_reweighting_supports_rectangular_grids(tmp_path):
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time cv_x cv_y\n"
        "0.0 0.0 0.0\n"
        "1.0 0.2 0.8\n"
        "2.0 0.5 0.4\n"
        "3.0 0.8 0.2\n"
        "4.0 1.0 1.0\n"
    )
    output = tmp_path / "fes.dat"
    script = (
        Path(__file__).resolve().parents[1]
        / "openmmnqe"
        / "opes"
        / "FES_from_Reweighting.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--colvar", str(colvar),
            "--outfile", str(output),
            "--cv", "cv_x,cv_y",
            "--bias", "NO",
            "--sigma", "0.2,0.3",
            "--kt", "1.0",
            "--min", "0.0,0.0",
            "--max", "1.0,1.0",
            "--bin", "2,3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    grid = np.loadtxt(output, comments="#!")
    assert grid.shape == (12, 3)
    assert len(np.unique(grid[:, 0])) == 3
    assert len(np.unique(grid[:, 1])) == 4
