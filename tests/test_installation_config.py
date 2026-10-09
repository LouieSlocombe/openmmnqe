"""Installation checks without creating environments or downloading packages."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tomllib
from itertools import zip_longest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_TOOLS = PROJECT_ROOT / "build_tools"


@pytest.mark.parametrize("interpreter", ["python", "python3"])
@pytest.mark.parametrize("failure", [None, 0, 1, 2])
@pytest.mark.parametrize("conditional", [False, True])
def test_plumed_checks_use_the_selected_interpreter_and_stop_on_failure(
    tmp_path: Path,
    interpreter: str,
    failure: int | None,
    conditional: bool,
) -> None:
    commands = [
        "plumed --no-mpi config -q module opes",
        f"{interpreter} -c import plumed; plumed.Plumed()",
        f"{interpreter} -c from openmmplumed import PlumedForce",
    ]
    messages = [
        "PLUMED opes module: OK",
        "py-plumed kernel load: OK",
        "openmm-plumed: OK",
    ]
    stub_dir = tmp_path / "commands"
    stub_dir.mkdir()
    for name in ("plumed", "python", "python3"):
        stub = stub_dir / name
        stub.write_text(
            '#!/bin/bash\n'
            'command="${0##*/} $*"\n'
            'printf "%s\\n" "$command" >> "$PLUMED_CHECK_LOG"\n'
            'if [[ "$command" == "$PLUMED_CHECK_FAIL" ]]; then exit 47; fi\n'
        )
        stub.chmod(0o755)
    log = tmp_path / "commands.log"
    environment = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "PLUMED_CHECK_LOG": str(log),
        "PLUMED_CHECK_FAIL": "" if failure is None else commands[failure],
    }
    invocation = 'check_plumed_installation "$2"'
    if conditional:
        invocation = f'if {invocation}; then exit 0; else exit "$?"; fi'
    result = subprocess.run(
        [
            "bash", "-c", f'set -e\nsource "$1"\n{invocation}',
            "plumed-check", str(BUILD_TOOLS / "build_plumed.sh"), interpreter,
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == (0 if failure is None else 47), result.stderr
    assert log.read_text().splitlines() == commands[:3 if failure is None else failure + 1]
    assert result.stdout.splitlines() == messages[:3 if failure is None else failure]


def _environment_requirements(filename: str) -> dict[str, str]:
    """Read package entries from these simple conda files, including pip entries.

    This deliberately handles only their dependency list, not general YAML;
    shlex removes quotes and comments without adding a YAML test dependency.
    """
    dependencies = (BUILD_TOOLS / filename).read_text().split("dependencies:\n", 1)[1]
    requirements = {}
    for line in dependencies.splitlines():
        if not line.lstrip().startswith("- "):
            continue
        spec = shlex.split(line.lstrip()[2:], comments=True)[0]
        if spec == "pip:":
            continue
        match = re.match(r"[\w-]+", spec)
        assert match is not None, spec
        requirements[match.group()] = spec
    return requirements


def _assert_same_release(first: str, second: str) -> None:
    # The metadata uses 1.0 where CI uses 1.0.0; both denote the same release.
    assert all(
        int(left) == int(right)
        for left, right in zip_longest(first.split("."), second.split("."), fillvalue="0")
    ), (first, second)


def test_installers_and_ci_share_the_declared_runtime_baselines() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    main = _environment_requirements("environment.yml")
    ci = _environment_requirements("environment_ci.yml")
    custom = _environment_requirements("environment_custom.yml")
    sol = shlex.split((BUILD_TOOLS / "custom_install_sol.sh").read_text(), comments=True)
    source_build = (BUILD_TOOLS / "custom_install.sh").read_text()
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text()

    for conda_name, package_name, source_variable in (
        ("openmm", "openmm", "OPENMM_VERSION"),
        ("openmm-ml", "openmmml", "OPENMM_ML_VERSION"),
    ):
        requirement, = [
            spec for spec in project["dependencies"]
            if spec.startswith(f"{package_name}>=")
        ]
        floor = requirement.split(">=", 1)[1]
        assert main[conda_name] == f"{conda_name}>={floor}"
        assert ci[conda_name] == custom[conda_name] == f"{conda_name}={floor}"
        assert main[conda_name] in sol
        assert f'{source_variable}="{floor}"' in source_build
        assert f"assert version('{package_name}') == '{floor}'" in workflow

    for name in ("forcefill", "reactiontools"):
        requirement, = [spec for spec in project["dependencies"] if spec.startswith(f"{name}>=")]
        assert ci[name].startswith(f"{name}==")
        _assert_same_release(requirement.split(">=", 1)[1], ci[name].split("==", 1)[1])
        # Local installers use editable sibling checkouts instead.
        assert name not in main
        assert name not in custom

    python_floor = project["requires-python"].removeprefix(">=")
    assert main["python"] == f"python>={python_floor}"
    assert ci["python"] == f"python={python_floor}"
    assert ci["python"] in sol
    for name in ("pymace", "ase"):
        assert main[name] == ci[name]
        assert main[name] in sol


def test_environment_differences_are_intentional() -> None:
    main = _environment_requirements("environment.yml")
    ci = _environment_requirements("environment_ci.yml")
    custom = _environment_requirements("environment_custom.yml")
    sol = shlex.split((BUILD_TOOLS / "custom_install_sol.sh").read_text(), comments=True)

    assert ci["pytorch"] == "pytorch=*=cpu*"
    assert "cuda-version" not in ci
    assert "nnpops" not in ci
    assert main["pytorch"] == custom["pytorch"] == "pytorch=*=cuda*"
    assert main["pytorch"] in sol
    assert main["cuda-version"] == "cuda-version<=13.2"
    assert "CUDA_VERSION=12.6" in sol
    assert "cuda-version<=${CUDA_VERSION}" in sol
    assert "nnpops" in main
    assert "nnpops" not in sol

    # The source-build recipe intentionally selects a different Python/MACE
    # stack and pins openmmforcefields; it is not a copy of the primary env.
    assert custom["python"] == "python=3.13"
    assert custom["pymace"] == "pymace=0.3.15"
    assert custom["openmmforcefields"] == "openmmforcefields=0.15.1"
    for requirements in (main, ci, custom):
        assert requirements["ambertools"] == "ambertools"
        assert requirements["cython"] == "cython"
    assert {"ambertools", "cython"}.issubset(sol)
