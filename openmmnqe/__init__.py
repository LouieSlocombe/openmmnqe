"""
openmmnqe: OpenMM workflows for nuclear quantum effects and enhanced sampling.

Bundles structure edits (:mod:`openmmnqe.io`), OpenMM simulation stages
including RPMD and adQTB nuclear-quantum-effect integrators and ML/MM
potentials (:mod:`openmmnqe.openmm`), RPMD reporters
(:mod:`openmmnqe.reporters`), ring-polymer rate dynamics
(:mod:`openmmnqe.rates`) and assorted simulation-setup utilities
(:mod:`openmmnqe.tools`).

This package is the simulation itself, and it has two dependencies that are
the rest of the workflow.

Ligand parameters come from
`forcefill <https://github.com/LouieSlocombe/forcefill>`_: it asks the base
force field which residues it cannot match, parameterises those with GAFF and
AM1-BCC, and writes an ffxml that loads underneath the standard files. The
force field every ``run_openmm_*`` stage takes is built from that::

    import forcefill as ff
    import openmm.app as app

    names = ("amber14-all.xml", "amber14/tip3pfb.xml")
    result = ff.build_forcefield_xml(input_pdb, "ligands.xml", base_forcefield=names)
    if result.skipped:
        raise RuntimeError(f"forcefill skipped residues: {result.skipped}")
    extra = [] if result.forcefield_xml is None else [result.forcefield_xml]

    pdb_data = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    forcefield = app.ForceField(*names, *extra)

A skipped residue would otherwise surface later, as a template error at
``createSystem``, and ``forcefield_xml`` is ``None`` when the base files
already matched every residue, which ``ForceField()`` will not take.

Nothing may edit the topology between those two blocks -- the templates
describe the residues exactly as ``input_pdb`` spells them. A raw crystal
structure is repaired first, with :func:`openmmnqe.io.fix_pdb`: forcefill
decides what needs parameters by asking what the base force field cannot
match, and a protein missing its hydrogens matches nothing.

Everything upstream of the simulation -- building a reaction path with NEB,
refining a transition state, running ORCA -- and everything alongside or
downstream of it -- the PLUMED collective variables that bias a proton
transfer, turning a steered trajectory into a reference path, and plotting the
free-energy surface that comes out -- lives in
`reactiontools <https://github.com/LouieSlocombe/reactiontools>`_::

    import openmmnqe as nqe
    import reactiontools as rt

    product = rt.swap_bonding_configuration(reactant, 0, 8, 1)
    neb_path = rt.quick_guess_path(reactant, product)
    plumed_input, fes_command = rt.plumed_input_1pt(modeller, idx, temperature)
    nqe.run_openmm_prod(modeller, forcefield, plumed_script_path=plumed_input)
    rt.plot_plumed_fes("fes.dat", filename="fes")

The reactiontools builders take the ``openmm.app.Modeller`` and
``openmm.unit.Quantity`` this package works in, so they can be called with what
is already to hand.
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
from .openmm import (
    PreparedSystem,
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
    RPMDQuantumSpreadReporter,
    RPMDThermodynamicReporter,
    RPMDVelocityReporter,
    plot_rpmd_atom_expansion,
    plot_rpmd_thermodynamics,
    rpmd_energy_conservation,
    rpmd_thermodynamic_averages,
    rpmd_thermodynamics,
    rpmd_velocity_autocorrelation,
    rpmd_vibrational_spectrum,
    track_rpmd_atom_expansion,
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

__all__ = [
    "PreparedSystem",
    "QTBConvergence",
    "QTBFrictionReporter",
    "RPMDBeadReporter",
    "RPMDCentroidReporter",
    "RPMDEnergyConservation",
    "RPMDQuantumSpreadReporter",
    "RPMDRate",
    "RPMDThermodynamicReporter",
    "RPMDVelocityReporter",
    "TransmissionResult",
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
    "plot_transmission_coefficient",
    "relabel_residues_in_pdb",
    "remove_directory",
    "remove_file",
    "remove_file_pattern",
    "remove_residues_in_pdb",
    "rpmd_energy_conservation",
    "rpmd_rate",
    "rpmd_thermodynamic_averages",
    "rpmd_thermodynamics",
    "rpmd_velocity_autocorrelation",
    "rpmd_vibrational_spectrum",
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
    "sample_rpmd_velocities",
    "save_only_index_atoms",
    "save_pdb_selection",
    "set_adqtb_particle_types_by_element",
    "step_rpmd",
    "track_adqtb_friction",
    "track_rpmd_atom_expansion",
    "transmission_coefficient",
    "write_multimodel_pdb",
    "xyz_to_sdf",
    "zero_velocities",
]

try:
    __version__ = _distribution_version("openmmnqe")
except _PackageNotFoundError:
    # Importing directly from an unpacked source tree has no installed metadata.
    __version__ = "0.1.0"

openmm_nqe_dir = _os.path.dirname(_os.path.realpath(__file__))
