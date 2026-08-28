"""Regression tests for the root API and package metadata."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import openmmnqe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PUBLIC_API = {
    "PreparedSystem",
    "QTBConvergence",
    "QTBFrictionReporter",
    "RPMDBeadReporter",
    "RPMDCentroidReporter",
    "RPMDQuantumSpreadReporter",
    "RPMDThermodynamicReporter",
    "__version__",
    "adqtb_convergence",
    "adqtb_fdt_residual",
    "adqtb_frequencies",
    "adqtb_friction",
    "adqtb_friction_spectra",
    "angle_between_atoms",
    "atom_indices_from_vmd_picks",
    "center_in_box",
    "centroid_positions",
    "check_platform",
    "convert_sdfs_to_pdb",
    "copy_and_rename_file",
    "count_dna_and_estimate_charge",
    "deuterate_system",
    "distance_between_atoms",
    "fix_pdb",
    "fix_pdb_atom_labels",
    "fix_pdb_chains",
    "get_atoms_in_residue",
    "get_thermal_de_broglie_wavelength",
    "init_beads",
    "list_files_with_pattern",
    "move_pdb_to_origin",
    "openmm_nqe_dir",
    "plot_adqtb_fdt_residual",
    "plot_adqtb_friction_spectra",
    "plot_rpmd_atom_expansion",
    "plot_rpmd_thermodynamics",
    "relabel_residues_in_pdb",
    "remove_directory",
    "remove_file",
    "remove_file_pattern",
    "remove_residues_in_pdb",
    "rpmd_thermodynamic_averages",
    "rpmd_thermodynamics",
    "run_openmm_adqtb_eq",
    "run_openmm_adqtb_prod",
    "run_openmm_heating",
    "run_openmm_npt",
    "run_openmm_prod",
    "run_openmm_relaxation",
    "run_openmm_relaxation_simple",
    "run_openmm_rpmd_contracted",
    "run_openmm_rpmd_equilibration",
    "run_openmm_rpmd_prod",
    "run_openmm_steered",
    "save_only_index_atoms",
    "save_pdb_selection",
    "set_adqtb_particle_types_by_element",
    "step_rpmd",
    "track_adqtb_friction",
    "track_rpmd_atom_expansion",
    "write_multimodel_pdb",
    "xyz_to_sdf",
    "zero_velocities",
}


def _project_metadata() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_root_public_api_is_explicit_and_complete() -> None:
    assert len(openmmnqe.__all__) == len(set(openmmnqe.__all__))
    assert set(openmmnqe.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(openmmnqe, name) for name in openmmnqe.__all__)


def test_runtime_version_matches_project_and_installed_metadata() -> None:
    project = _project_metadata()
    assert openmmnqe.__version__ == project["version"]

    try:
        installed_version = version("openmmnqe")
    except PackageNotFoundError:
        return
    assert openmmnqe.__version__ == installed_version


def test_critical_dependency_minimums_are_declared() -> None:
    dependencies = set(_project_metadata()["dependencies"])
    assert "openmm>=8.5.2" in dependencies
    assert "openmmml>=1.6" in dependencies


def test_docs_requirements_match_docs_extra() -> None:
    # Read the Docs installs docs/requirements.txt rather than the [docs]
    # extra, because it cannot request an extra without also installing the
    # project -- and openmmnqe's runtime dependencies are not pip-installable.
    # That leaves two lists that have to say the same thing, so pin them here.
    extra = _project_metadata()["optional-dependencies"]["docs"]
    requirements = [
        line.strip()
        for line in (PROJECT_ROOT / "docs" / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == list(extra)
