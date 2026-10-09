"""Assemble matching structures and force fields after parameterisation."""

from __future__ import annotations

import os
from collections.abc import Sequence

import openmm.app as app


def load_parameterized_structure(
    input_pdb: str | os.PathLike[str],
    base_forcefield: Sequence[str],
    forcefield_xml: str | os.PathLike[str] | None,
) -> tuple[app.Modeller, app.ForceField]:
    """Read the parameterised PDB unchanged and load its matching templates.

    Callers parameterise and check for skipped residues first. Generated
    templates match the original PDB's topology, so this adapter deliberately
    performs no structure edits. A missing XML means the base files already
    cover the structure.
    """
    extra = [] if forcefield_xml is None else [os.fspath(forcefield_xml)]
    pdb = app.PDBFile(os.fspath(input_pdb))
    return (
        app.Modeller(pdb.topology, pdb.positions),
        app.ForceField(*base_forcefield, *extra),
    )
