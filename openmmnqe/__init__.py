"""
openmmnqe: OpenMM workflows for nuclear quantum effects and enhanced sampling.

Bundles system preparation (:mod:`openmmnqe.io`), OpenMM simulation stages
including RPMD and adQTB nuclear-quantum-effect integrators and ML/MM
potentials (:mod:`openmmnqe.openmm`), RPMD reporters
(:mod:`openmmnqe.reporters`) and assorted simulation-setup utilities
(:mod:`openmmnqe.tools`).

This package is the simulation itself. Everything upstream of it -- building a
reaction path with NEB, refining a transition state, running ORCA -- and
everything alongside or downstream of it -- the PLUMED collective variables
that bias a proton transfer, turning a steered trajectory into a reference
path, and plotting the free-energy surface that comes out -- lives in
`reactiontools <https://github.com/LouieSlocombe/reactiontools>`_, which is a
dependency. Import those from there rather than from here::

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

__version__ = "0.1.0"

from .io import (remove_directory,
                 copy_and_rename_file,
                 list_files_with_pattern,
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
                 save_only_index_atoms,
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
                    distance_between_atoms,
                    angle_between_atoms,
                    check_platform,
                    )

import os

openmm_nqe_dir = os.path.dirname(os.path.realpath(__file__))
