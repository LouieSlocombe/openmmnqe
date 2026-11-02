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


def _run_plumed_build(
    tmp_path: Path,
    *,
    launcher: str | None,
    compiler: str | None = None,
    failure: str = "",
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run the build recipe against recording tools, without network or compilation."""
    stub_dir = tmp_path / "commands"
    stub_dir.mkdir()
    stub_script = (
        '#!/bin/bash\n'
        'command="${0##*/}"\n'
        'printf "%q " "$command" "${CXX-__unset__}" "$@" >> "$PLUMED_BUILD_LOG"\n'
        'printf "\\n" >> "$PLUMED_BUILD_LOG"\n'
        'if [[ "$command" == "$PLUMED_BUILD_FAIL" ]]; then exit 47; fi\n'
        'if [[ "$command" == git ]]; then\n'
        '    source_dir="${@: -1}"\n'
        '    source_dir="${source_dir##*/}"\n'
        '    source_dir="${source_dir%.git}"\n'
        '    mkdir -p "$source_dir"\n'
        '    cp "$0" "$source_dir/configure"\n'
        'elif [[ "$command" == cmake ]]; then\n'
        '    mkdir python\n'
        'elif [[ "$command" == make && "$1" == PythonInstall ]]; then\n'
        '    exit 1\n'
        'fi\n'
    )
    for name in ("git", "cmake", "make", "pip"):
        stub = stub_dir / name
        stub.write_text(stub_script)
        stub.chmod(0o755)
    work_dir = tmp_path / "build sources"
    work_dir.mkdir()
    log = tmp_path / "commands.log"
    environment = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "CONDA_PREFIX": str(tmp_path / "conda environment"),
        "PLUMED_BUILD_LOG": str(log),
        "PLUMED_BUILD_FAIL": failure,
    }
    if compiler is None:
        environment.pop("CXX", None)
    else:
        environment["CXX"] = compiler
    result = subprocess.run(
        [
            "bash", "-c", 'set -e\nsource "$1"\nshift\nbuild_plumed "$@"',
            "plumed-build", str(BUILD_TOOLS / "build_plumed.sh"), str(work_dir),
            *([] if launcher is None else [launcher]),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, [shlex.split(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize("launcher", [None, "ccache"])
@pytest.mark.parametrize("compiler", [None, "conda-c++ -std=c++17"])
def test_plumed_build_compiler_cache_is_opt_in(
    tmp_path: Path, launcher: str | None, compiler: str | None,
) -> None:
    result, calls = _run_plumed_build(tmp_path, launcher=launcher, compiler=compiler)

    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == [
        "git", "configure", "make", "make", "git", "cmake", "make", "make",
        "make", "pip",
    ]
    # The launcher is local to each build system, not exported over conda's CXX.
    assert all(call[1] == (compiler or "__unset__") for call in calls)
    configure = calls[1][2:]
    cmake = calls[5][2:]
    assert [arg for arg in configure if arg.startswith("CXX=")] == (
        [] if launcher is None else [f"CXX={launcher} {compiler or 'c++'}"]
    )
    assert [arg for arg in cmake if arg.startswith("-DCMAKE_CXX_COMPILER_LAUNCHER=")] == (
        [] if launcher is None else [f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}"]
    )
    assert "--enable-modules=opes" in configure
    assert "--disable-python" in configure
    assert "--disable-pycv" in configure
    # Cache hits still install both libraries and the generated Python wrapper.
    assert [call[2:] for call in calls if call[0] == "make"].count(["install"]) == 2
    assert calls[-1][2:] == ["install", ".", "--no-build-isolation"]


@pytest.mark.parametrize("failure", ["git", "configure", "make", "cmake", "pip"])
def test_plumed_build_with_a_compiler_cache_stops_on_failure(
    tmp_path: Path, failure: str,
) -> None:
    result, calls = _run_plumed_build(tmp_path, launcher="ccache", failure=failure)

    assert result.returncode == 47, result.stderr
    assert calls[-1][0] == failure
    assert [call[0] for call in calls].count(failure) == 1


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
    # stack; it is not a copy of the primary env.
    assert custom["python"] == "python=3.13"
    assert custom["pymace"] == "pymace=0.3.15"
    for requirements in (main, ci, custom):
        assert requirements["ambertools"] == "ambertools"
        assert requirements["cython"] == "cython"
        # forcefill declares openmmforcefields>=0.16, so an exact conda pin
        # below that floor would be replaced from PyPI on install.
        assert requirements["openmmforcefields"] == "openmmforcefields"
        # Nothing in this package or its siblings imports MDAnalysis.
        assert "mdanalysis" not in requirements
    assert {"ambertools", "cython", "openmmforcefields"}.issubset(sol)
    assert "mdanalysis" not in sol
