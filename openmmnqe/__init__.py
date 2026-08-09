"""
openmmnqe: OpenMM workflows for nuclear quantum effects and enhanced sampling.

Bundles system preparation (:mod:`openmmnqe.io`), OpenMM simulation stages
including RPMD and adQTB nuclear-quantum-effect integrators and ML/MM
potentials (:mod:`openmmnqe.openmm`), PLUMED-based collective variables and
enhanced sampling (:mod:`openmmnqe.plumed`), reference-path estimation
(:mod:`openmmnqe.path`), RPMD reporters (:mod:`openmmnqe.reporters`) and
assorted simulation-setup utilities (:mod:`openmmnqe.tools`).

Everything upstream of the simulation -- building a reaction path with NEB,
refining a transition state, running ORCA -- and everything downstream of it
that plots a free-energy surface lives in `reactiontools
<https://github.com/LouieSlocombe/reactiontools>`_, which is a dependency.
Import those from there rather than from here::

    import openmmnqe as nqe
    import reactiontools as rt

    product = rt.swap_bonding_configuration(reactant, 0, 8, 1)
    neb_path = rt.quick_guess_path(reactant, product)
    ...
    rt.plot_plumed_fes("fes.dat", filename="fes")
"""

__version__ = "0.1.0"

from .io import (remove_directory,
                 copy_and_rename_file,
                 list_files_with_pattern,
                 search_fes_files,
                 load_fes_data,
                 xyz_to_sdf,
                 extract_nonstandard_res,
                 get_non_standard_residues,
                 list_non_standard_residues,
                 clean_ions_in_pdb,
                 relabel_residues_in_pdb,
                 remove_residues_in_pdb,
                 remove_water_residues_in_pdb,
                 fix_pdb,
                 make_sdf,
                 pdb_patcher,
                 combine_sdf_pdb,
                 convert_sdfs_to_pdb,
                 prepare_lig_system,
                 prepare_ligand_ff,
                 save_pdb_selection,
                 remove_file_pattern,
                 remove_file,
                 move_pdb_to_origin,
                 center_in_box,
                 fix_pdb_chains,
                 fix_pdb_atom_labels,
                 convert_xyz_to_pdb,
                 convert_pdb_to_xyz,
                 convert_xyz_to_plumed_ref,
                 save_only_index_atoms,
                 pdb_remove_ter_index,
                 strip_hydrogens_keep_indices,
                 )
from .openmm import (run_openmm_relaxation,
                     run_openmm_relaxation_simple,
                     run_openmm_heating,
                     run_openmm_npt,
                     run_openmm_prod,
                     run_openmm_rpmd_equilibration,
                     run_openmm_rpmd_contracted,
                     run_openmm_rpmd_prod,
                     run_openmm_adqtb_eq,
                     run_openmm_adqtb_prod,
                     run_openmm_steered,
                     )
from .path import (cv_from_colvar,
                   path_from_steered_md,
                   select_frames_by_cv,
                   select_frames_by_msd,
                   )
from .plumed import (estimate_path_lambda,
                     plumed_input_1pt,
                     plumed_input_2pt_1d,
                     plumed_input_2pt_2d,
                     plumed_input_wob_1,
                     plumed_input_wob_2,
                     plumed_input_wob_3,
                     plumed_input_wob_4,
                     plumed_input_neb_path,
                     plumed_input_neb_path_wob,
                     plumed_input_steered,
                     plumed_input_steered_pt,
                     )
from .reporters import (RPMDQuantumSpreadReporter,
                        RPMDBeadReporter,
                        RPMDCentroidReporter,
                        )
from .tools import (zero_velocities,
                    write_multimodel_pdb,
                    centroid_positions,
                    init_beads,
                    get_thermal_de_broglie_wavelength,
                    init_beads_scaled,
                    count_dna_and_estimate_charge,
                    deuterate_system,
                    get_atoms_in_residue,
                    set_adqtb_particle_types_by_element,
                    atom_indices_from_vmd_picks,
                    atom_indices_to_plumed,
                    distance_between_atoms,
                    angle_between_atoms,
                    check_platform,
                    temperature_to_kbt,
                    )

import os

openmm_nqe_dir = os.path.dirname(os.path.realpath(__file__))
