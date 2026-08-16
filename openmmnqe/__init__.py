"""
openmmnqe: OpenMM workflows for nuclear quantum effects and enhanced sampling.

Bundles structure edits (:mod:`openmmnqe.io`), OpenMM simulation stages
including RPMD and adQTB nuclear-quantum-effect integrators and ML/MM
potentials (:mod:`openmmnqe.openmm`), RPMD reporters
(:mod:`openmmnqe.reporters`) and assorted simulation-setup utilities
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

    pdb_data = app.PDBFile(input_pdb)
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    forcefield = app.ForceField(*names, result.forcefield_xml)

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
                 relabel_residues_in_pdb,
                 remove_residues_in_pdb,
                 fix_pdb,
                 convert_sdfs_to_pdb,
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
                        track_rpmd_atom_expansion,
                        plot_rpmd_atom_expansion,
                        )
from .tools import (zero_velocities,
                    write_multimodel_pdb,
                    centroid_positions,
                    init_beads,
                    step_rpmd,
                    get_thermal_de_broglie_wavelength,
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
