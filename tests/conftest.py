"""Shared setup for the test suite.

Three things every test module relies on and none of them should have to
arrange for itself.

**Working directory.**  The tests name their inputs as ``tests/data/...`` and
write their outputs into the current directory, so they only work when pytest
is launched from the repository root.  Running the session from there makes
that true however pytest was invoked.

**Matplotlib backend.**  Many of these tests double as research scripts and end
in ``plt.show()``.  Under pytest there is nobody to close the window, so the
backend is forced to Agg for the whole session and ``show()`` becomes a no-op.
This used to happen by accident -- ``test_plotting`` switched the backend at
import time, which pytest does for every module during collection -- so it took
effect only because that module existed.  It is now deliberate.

**Ligand force fields.**  Every workflow that starts from a PDB with a ligand
in it needs the same two objects, built the same way; :func:`ligand_forcefield`
is that, and the reason none of them build it by hand is written on the
fixture.
"""
import os
import shutil
from pathlib import Path

import matplotlib

REPO_ROOT = Path(__file__).resolve().parent.parent

# An argparse CLI for multiple-walker metadynamics, not a test module.
collect_ignore = ["run_walker.py"]

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

import forcefill as ff  # noqa: E402
import openmm.app as app  # noqa: E402
import pytest  # noqa: E402

#: The force fields a ligand system is built on top of, unless a test says
#: otherwise.  Matches what the examples use.
BASE_FORCEFIELD = ("amber14-all.xml", "amber14/tip3pfb.xml")


def pytest_configure(config):
    os.chdir(REPO_ROOT)


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test leaves behind."""
    yield
    plt.close("all")


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
    drift.  conda-forge's openmmforcefields depends on ambertools, so these run
    in CI and in any environment built from build_tools/; a pip-only install
    has neither executable and skips instead of failing inside a subprocess.
    """
    if shutil.which("antechamber") is None or shutil.which("parmchk2") is None:
        pytest.skip("forcefill's gaff backend needs AmberTools "
                    "(antechamber, parmchk2) on PATH")

    def _build(input_pdb, base_forcefield=BASE_FORCEFIELD, **kwargs):
        result = ff.build_forcefield_xml(input_pdb,
                                         tmp_path / "ligands.xml",
                                         base_forcefield=base_forcefield,
                                         workdir=tmp_path / "forcefill",
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
