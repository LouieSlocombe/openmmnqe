"""Regression checks for complete OpenMM NQE workflows."""

from ase.io import read, write
from mace.calculators.foundations_models import mace_off
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
import pytest

import openmmnqe as nqe
import reactiontools as rt


@pytest.mark.pipeline
@pytest.mark.forcefield
def test_estimate_path_lambda(ligand_forcefield):
    """A generated malonaldehyde path has a physically meaningful scale."""
    input_pdb = "tests/data/pdb/malonaldehyde.pdb"
    ml_model = "mace-off23-small"
    potential = MLPotential(ml_model)
    modeller, forcefield = ligand_forcefield(input_pdb)

    modeller.addSolvent(
        forcefield,
        padding=1.5 * unit.nanometer,
        boxShape="cube",
    )
    nqe.center_in_box(modeller)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    nqe.run_openmm_relaxation_simple(
        modeller,
        forcefield,
        potential=potential,
        ml_idx=ml_atoms,
    )

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.save_only_index_atoms(modeller, ml_atoms, file_idx="index_atoms.pdb")

    reactant = read("index_atoms.pdb")
    product = rt.swap_bonding_configuration(reactant, 0, 8, 1)
    calc = mace_off(model_name=ml_model, device="cuda")
    product = rt.optimise_geom(product, calc, fmax=0.01)
    neb_path = rt.quick_guess_path(reactant, product, n_images=5)
    write("neb_path.xyz", neb_path)
    rt.convert_xyz_to_plumed_ref("neb_path.xyz", "index_atoms.pdb", "neb_path.pdb")

    assert rt.estimate_path_lambda("neb_path.pdb") > 0.0
