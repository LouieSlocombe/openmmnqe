"""Shared paths, output isolation, and ligand parameterisation fixtures."""

import itertools
import os
import shutil
from pathlib import Path

import forcefill as ff
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3, openmm

TEST_DATA = Path(__file__).resolve().parent / "data"

#: The force fields a ligand system is built on top of, unless a test says
#: otherwise.  Matches what the examples use.
BASE_FORCEFIELD = ("amber14-all.xml", "amber14/tip3pfb.xml")


class OneParticleForceField:
    """Deterministic force field for cheap public-workflow tests."""

    def createSystem(self, topology, **kwargs):
        system = openmm.System()
        system.addParticle(39.9 * unit.dalton)
        restraint = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
        restraint.addGlobalParameter(
            "k",
            100.0 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        restraint.addParticle(0, [])
        system.addForce(restraint)
        return system


@pytest.fixture(scope="session")
def data_dir():
    """Return the absolute path to immutable test inputs."""
    return TEST_DATA


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    """Contain every test's generated files in its own temporary directory."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def one_particle_system():
    """Return a one-argon Modeller and deterministic harmonic force field."""
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    modeller = app.Modeller(
        topology,
        [Vec3(0.1, 0.0, 0.0)] * unit.nanometer,
    )
    return modeller, OneParticleForceField()


@pytest.fixture
def ligand_forcefield(tmp_path):
    """Build ``(modeller, forcefield)`` for a PDB whose ligands need parameters.

    ``forcefill.build_forcefield_xml`` asks the base force field which residues
    it cannot match, parameterises those with GAFF2/AM1-BCC, and writes an
    ffxml; the force field is that file loaded underneath the standard ones.

    The Modeller is built from the *same* file forcefill read, and nothing is
    allowed between the two.  The generated templates describe those residues
    exactly as that file spells them, so any edit in between -- adding
    hydrogens, deleting water, relabelling -- stops them matching, and OpenMM
    reports that much later as an opaque "no template found".

    A charged ligand needs ``net_charges={"RES": q}``: a residue extracted from
    a PDB carries no formal charge for forcefill to read, so it assumes 0
    without warning, and a wrong net charge is the classic source of
    plausible-but-wrong AM1-BCC charges.

    The AmberTools guard lives here rather than on each test, where it would
    drift. A pip-only install without the executables skips these integration
    checks instead of failing inside a subprocess.
    """
    if shutil.which("antechamber") is None or shutil.which("parmchk2") is None:
        if os.environ.get("CI"):
            pytest.fail(
                "CI must provide AmberTools (antechamber and parmchk2) for "
                "forcefield integration tests"
            )
        pytest.skip("forcefill's gaff backend needs AmberTools "
                    "(antechamber, parmchk2) on PATH")

    build_number = itertools.count()

    def _build(input_pdb, base_forcefield=BASE_FORCEFIELD, **kwargs):
        build_dir = tmp_path / f"forcefill-{next(build_number)}"
        build_dir.mkdir()
        result = ff.build_forcefield_xml(input_pdb,
                                         build_dir / "ligands.xml",
                                         base_forcefield=base_forcefield,
                                         workdir=build_dir / "work",
                                         **kwargs)
        # A skipped residue also suppresses forcefill's own whole-structure
        # check, so without this the failure surfaces at createSystem instead
        # of here, where the reason is still to hand.
        assert not result.skipped, f"forcefill skipped residues: {result.skipped}"
        # None when the base force field already covered everything, and
        # ForceField() will not take it.
        extra = [] if result.forcefield_xml is None else [result.forcefield_xml]
        pdb = app.PDBFile(str(input_pdb))
        return (app.Modeller(pdb.topology, pdb.positions),
                app.ForceField(*base_forcefield, *extra))

    return _build
