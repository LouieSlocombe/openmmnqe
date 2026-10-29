"""
openmmnqe: OpenMM workflows for nuclear quantum effects and enhanced sampling.

Bundles structure edits (:mod:`openmmnqe.io`), OpenMM simulation stages
including RPMD and adQTB nuclear-quantum-effect integrators and ML/MM
potentials (:mod:`openmmnqe.openmm`), RPMD reporters
(:mod:`openmmnqe.reporters`), equilibrium isotope effects
(:mod:`openmmnqe.isotopes`), ring-polymer rate dynamics
(:mod:`openmmnqe.rates`), independent multi-walker orchestration
(:mod:`openmmnqe.walkers`) and assorted simulation-setup utilities
(:mod:`openmmnqe.tools`).

This package is the simulation itself. Ligand parameters come from
`forcefill <https://github.com/LouieSlocombe/forcefill>`_, and everything
reaction-side -- reaction paths, the PLUMED collective variables that bias
them, ORCA, and free-energy plotting -- from
`reactiontools <https://github.com/LouieSlocombe/reactiontools>`_. The README
and the guides at https://openmmnqe.readthedocs.io show how the three fit
together.
"""

import os as _os
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .adqtb import (
    QTBConvergence,
    QTBFrictionReporter,
    adqtb_convergence,
    adqtb_fdt_residual,
    adqtb_frequencies,
    adqtb_friction,
    adqtb_friction_spectra,
    plot_adqtb_fdt_residual,
    plot_adqtb_friction_spectra,
    track_adqtb_friction,
)
from .io import (
    center_in_box,
    convert_sdfs_to_pdb,
    copy_and_rename_file,
    fix_pdb,
    fix_pdb_atom_labels,
    fix_pdb_chains,
    list_files_with_pattern,
    move_pdb_to_origin,
    relabel_residues_in_pdb,
    remove_directory,
    remove_file,
    remove_file_pattern,
    remove_residues_in_pdb,
    save_only_index_atoms,
    save_pdb_selection,
    xyz_to_sdf,
)
from .isotopes import (
    IsotopeFreeEnergy,
    MassIntegrationNodes,
    rpmd_fractionation_factor,
    rpmd_isotope_free_energy,
    rpmd_mass_integration_nodes,
)
from .openmm import (
    PreparedSystem,
    TrajectoryOptions,
    run_openmm_adqtb_eq,
    run_openmm_adqtb_prod,
    run_openmm_heating,
    run_openmm_npt,
    run_openmm_prod,
    run_openmm_relaxation,
    run_openmm_relaxation_simple,
    run_openmm_rpmd_contracted,
    run_openmm_rpmd_equilibration,
    run_openmm_rpmd_prod,
    run_openmm_steered,
)
from .rates import (
    RPMDRate,
    TransmissionResult,
    plot_transmission_coefficient,
    rpmd_rate,
    run_openmm_rpmd_recrossing,
    transmission_coefficient,
)
from .reporters import (
    RPMDBeadReporter,
    RPMDCentroidReporter,
    RPMDEnergyConservation,
    RPMDKineticDecompositionReporter,
    RPMDQuantumSpreadReporter,
    RPMDThermodynamicReporter,
    RPMDVelocityReporter,
    VelocityArchiveReporter,
    plot_rpmd_atom_expansion,
    plot_rpmd_kinetic_decomposition,
    plot_rpmd_thermodynamics,
    rpmd_energy_conservation,
    rpmd_kinetic_decomposition,
    rpmd_kinetic_decomposition_averages,
    rpmd_thermodynamic_averages,
    rpmd_thermodynamics,
    track_rpmd_atom_expansion,
    velocity_autocorrelation,
    vibrational_spectrum,
)
from .tools import (
    angle_between_atoms,
    atom_indices_from_vmd_picks,
    centroid_positions,
    check_platform,
    count_dna_and_estimate_charge,
    deuterate_system,
    distance_between_atoms,
    get_atoms_in_residue,
    get_thermal_de_broglie_wavelength,
    init_beads,
    sample_rpmd_velocities,
    set_adqtb_particle_types_by_element,
    step_rpmd,
    write_multimodel_pdb,
    zero_velocities,
)
from .walkers import (
    WalkerEnsemble,
    run_openmm_walkers,
)

__all__ = [
    "IsotopeFreeEnergy",
    "MassIntegrationNodes",
    "PreparedSystem",
    "QTBConvergence",
    "QTBFrictionReporter",
    "RPMDBeadReporter",
    "RPMDCentroidReporter",
    "RPMDEnergyConservation",
    "RPMDKineticDecompositionReporter",
    "RPMDQuantumSpreadReporter",
    "RPMDRate",
    "RPMDThermodynamicReporter",
    "RPMDVelocityReporter",
    "TrajectoryOptions",
    "TransmissionResult",
    "VelocityArchiveReporter",
    "WalkerEnsemble",
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
    "plot_rpmd_kinetic_decomposition",
    "plot_rpmd_thermodynamics",
    "plot_transmission_coefficient",
    "relabel_residues_in_pdb",
    "remove_directory",
    "remove_file",
    "remove_file_pattern",
    "remove_residues_in_pdb",
    "rpmd_energy_conservation",
    "rpmd_fractionation_factor",
    "rpmd_isotope_free_energy",
    "rpmd_kinetic_decomposition",
    "rpmd_kinetic_decomposition_averages",
    "rpmd_mass_integration_nodes",
    "rpmd_rate",
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
    "run_openmm_rpmd_recrossing",
    "run_openmm_steered",
    "run_openmm_walkers",
    "sample_rpmd_velocities",
    "save_only_index_atoms",
    "save_pdb_selection",
    "set_adqtb_particle_types_by_element",
    "step_rpmd",
    "track_adqtb_friction",
    "track_rpmd_atom_expansion",
    "transmission_coefficient",
    "velocity_autocorrelation",
    "vibrational_spectrum",
    "write_multimodel_pdb",
    "xyz_to_sdf",
    "zero_velocities",
]

try:
    __version__ = _distribution_version("openmmnqe")
except _PackageNotFoundError:
    # Importing directly from an unpacked source tree has no installed metadata.
    __version__ = "1.2.0"

openmm_nqe_dir = _os.path.dirname(_os.path.realpath(__file__))
