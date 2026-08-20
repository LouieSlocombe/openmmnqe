"""Relax structures with native MACE, ASE MACE, ORCA, or mixed ML/MM."""

from __future__ import annotations

import argparse
import os

from ase.calculators.orca import ORCA, OrcaProfile
from mace.calculators.foundations_models import mace_off
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

import openmmnqe as nqe


def run_openmm_ml() -> None:
    """Relax a peptide with MACE driving the whole system."""
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    potential = MLPotential("mace-off23-small")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_relaxation_simple(modeller, potential)
    nqe.remove_file_pattern("minimized*")


def run_ase_mace() -> None:
    """Relax the same peptide with MACE through the ASE bridge instead."""
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    potential = MLPotential("ase")
    calculator = mace_off("small", default_dtype="float32")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_relaxation_simple(
        modeller,
        potential,
        calculator=calculator,
    )
    nqe.remove_file_pattern("minimized*")


def run_ase_orca() -> None:
    """Relax malonaldehyde with ORCA as the ASE calculator; needs ORCA_PATH."""
    if "ORCA_PATH" not in os.environ:
        raise RuntimeError("set ORCA_PATH to the ORCA executable")

    pdb = app.PDBFile("tests/data/pdb/malonaldehyde.pdb")
    potential = MLPotential("ase")
    profile = OrcaProfile(command=os.environ["ORCA_PATH"])
    calculator = ORCA(profile=profile, orcasimpleinput="ENGRAD")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_relaxation_simple(
        modeller,
        potential,
        calculator=calculator,
    )
    nqe.remove_file_pattern("minimized*")
    nqe.remove_file_pattern("orca*")


def run_openmm_ml_mixed_system() -> None:
    """Relax a solvated peptide with MACE on the solute and MM on the water."""
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    potential = MLPotential("mace-off23-small")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    modeller.addSolvent(
        forcefield,
        padding=1.5 * unit.nanometer,
        boxShape="dodecahedron",
    )

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    nqe.run_openmm_relaxation(
        modeller,
        forcefield,
        potential=potential,
        ml_idx=ml_atoms,
    )
    nqe.remove_file_pattern("minimized*")


EXAMPLES = {
    name.removeprefix("run_"): function
    for name, function in list(globals().items())
    if name.startswith("run_") and callable(function)
}


def main() -> None:
    """Run whichever example is named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    args = parser.parse_args()
    EXAMPLES[args.example]()


if __name__ == "__main__":
    main()
