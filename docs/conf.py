"""Sphinx configuration for the openmmnqe documentation.

openmmnqe cannot be imported for real here. Every module imports ``openmm`` at
module scope; ``openmmnqe/openmm.py`` also imports ``openmmml`` and
``openmmplumed``, and ``openmmnqe/io.py`` imports ``pdbfixer``, ``rdkit`` and
``reactiontools``. None of those are installable from PyPI at the versions this
project pins -- ``build_tools/README.md`` explains why the runtime environment
is conda plus a PLUMED source build. The docs build therefore puts the checkout
on ``sys.path``, mocks those distributions, and installs only the doc toolchain
plus numpy and scipy for real.
"""

from __future__ import annotations

import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from sphinx.ext.autodoc.mock import _MockObject

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Import openmmnqe from the checkout: nothing installs the package, and the
# checkout must win over any openmmnqe already in the ambient environment.
# ``build/lib/openmmnqe`` (a stale setuptools copy) is never reachable, because
# the repository root, not ``build/lib``, is what goes on the path.
sys.path.insert(0, str(_REPO_ROOT))

with (_REPO_ROOT / "pyproject.toml").open("rb") as _pyproject:
    _PROJECT = tomllib.load(_pyproject)["project"]


# -- Mock arithmetic ---------------------------------------------------------
#
# Sphinx's mock objects implement no numeric protocol, and openmmnqe evaluates
# openmm.unit arithmetic at import time in two places:
#
#   openmmnqe/reporters.py:52     unit.kilojoule_per_mole / unit.kelvin
#   every run_openmm_* signature  300.0 * unit.kelvin, 1.0 / unit.picosecond
#
# Without these four operators ``import openmmnqe`` raises TypeError under
# mocking and autodoc documents nothing at all.
#
# ``__or__`` is deliberately NOT patched. Annotations are strings under
# ``from __future__ import annotations``, so they are only evaluated by
# ``typing.get_type_hints()``, which Sphinx already guards. A patched
# ``__or__`` would make that evaluation *succeed* and silently collapse
# ``app.ForceField | MLPotential | PreparedSystem`` to its first operand.


def _mock_binary_op(self: _MockObject, other: object) -> _MockObject:
    """Return the left operand so unit arithmetic survives mocking."""
    return self


for _operator in ("__mul__", "__rmul__", "__truediv__", "__rtruediv__"):
    setattr(_MockObject, _operator, _mock_binary_op)


# -- Project information -----------------------------------------------------

project = "openmmnqe"
author = "Louie Slocombe"
project_copyright = f"{datetime.now(tz=UTC):%Y}, {author}"

# pyproject.toml stays the single source of truth for the version: openmmnqe
# reads it back through importlib.metadata and tests/test_package_metadata.py
# asserts the two agree. Reading the same file here adds no third source.
release = str(_PROJECT["version"])
version = release


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    # Not loaded by default in Sphinx 9, and reporters.py has .. math:: blocks.
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"

# Relative to this directory. The repository-root ``build/`` tree is not
# reachable from here at all, so it needs no pattern.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- autodoc -----------------------------------------------------------------

autodoc_mock_imports = [
    "ase",
    "forcefill",
    "mace",
    "matplotlib",
    "openmm",
    "openmmml",
    "openmmplumed",
    "pdbfixer",
    "rdkit",
    "reactiontools",
]

# Every parameter already carries an authoritative NumPy "Parameters" type line,
# and the mocked third-party types render inconsistently in signatures -- a
# union of mocks cannot be evaluated, so it falls back to half-qualified source
# text like ``app.ForceField``. Dropping the inline hints removes the only place
# a mock can misinform the reader, and shortens signatures that run to 23
# parameters.
autodoc_typehints = "none"

# Read default values from the source with ast instead of repr()-ing the
# evaluated object. Without this, ``1.0 * unit.bar`` renders as
# ``openmm.unit.bar`` and ``1.0 / unit.picosecond`` as
# ``openmm.unit.picosecond``: the number silently disappears.
autodoc_preserve_defaults = True

# openmmnqe/openmm.py defines the stages in workflow order, and its module
# docstring numbers them in that same order.
autodoc_member_order = "bysource"
autodoc_class_signature = "mixed"


# -- napoleon ----------------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False  # constructor args live in the class docstring
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False
napoleon_preprocess_types = False


# -- intersphinx -------------------------------------------------------------
#
# No OpenMM entry. https://docs.openmm.org/latest/api-python/objects.inv does
# exist but indexes implementation paths -- openmm.openmm.System,
# openmm.app.modeller.Modeller, openmm.unit.quantity.Quantity -- while every
# docstring here uses the public aliases openmm.System, openmm.app.Modeller,
# openmm.unit.Quantity. It also publishes no py:module entries at all, so it
# would resolve nothing and cost a network fetch per build. The nitpick_ignore
# entry below lets those render as plain text instead.
#
# scipy is imported (scipy.constants in tools.py) but never named in a docstring
# type, so it earns no mapping yet.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}


# -- Cross-reference strictness ----------------------------------------------
#
# nitpicky is on so that a typo in one of openmmnqe's own :func:/:class: roles
# fails the build. Sphinx matches these patterns with re.fullmatch against the
# raw reftarget, hence the explicit ``(\..*)?`` suffix.

nitpicky = True
nitpick_ignore_regex = [
    # Third-party APIs with no usable Sphinx inventory. Most of these come from
    # the napoleon "Parameters" type fields and are openmm.* names.
    (
        r"py:.*",
        r"(ase|forcefill|mace|matplotlib|mdtraj|openmm|openmmml|openmmplumed"
        r"|pdbfixer|rdkit|reactiontools)(\..*)?",
    ),
    # Private helpers named in the narrative parts of docstrings but
    # deliberately not published.
    (r"py:.*", r"_\w+"),
    # NumPy docstrings write prose into the type line ("int, optional",
    # "{'all', 'water'}, optional"). Napoleon splits that line on commas and
    # the Python domain then tries to resolve each fragment as a class. These
    # two patterns drop the fragments that were never type names: bare prose
    # words, and anything containing a character a dotted Python name cannot
    # hold -- a quote, a brace, a space or a hyphen. A genuine typo in an
    # openmmnqe name is still a valid identifier, so it still fails the build.
    (r"py:class", r"optional|iterable|sequence|pair|scalar|callable"),
    (r"py:class", r".*[^\w.].*"),
]


# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"openmmnqe {release}"
html_theme_options = {
    "source_repository": "https://github.com/LouieSlocombe/openmmnqe/",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_static_path: list[str] = []


# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
