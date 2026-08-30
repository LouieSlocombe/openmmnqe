"""The OpenMM simulation stages, from minimisation through to production.

Each ``run_openmm_*`` function is one stage of a workflow, and they are meant
to be run in order using the structure written by the preceding stage:

1. :func:`run_openmm_relaxation` or :func:`run_openmm_relaxation_simple` --
   take the strain out of the starting structure,
2. :func:`run_openmm_heating` -- warm it to temperature under restraints,
3. :func:`run_openmm_npt` -- relax the box density,
4. :func:`run_openmm_prod` -- classical production, optionally biased,
5. :func:`run_openmm_rpmd_equilibration` then
   :func:`run_openmm_rpmd_prod` or :func:`run_openmm_rpmd_contracted`, or
   :func:`run_openmm_adqtb_eq` then :func:`run_openmm_adqtb_prod` -- the two
   routes to nuclear quantum effects.

Ring-polymer MD is the reference method and converges to the exact quantum
statistics with enough beads, but costs a force evaluation per bead;
the adaptive quantum thermal bath costs no more than a classical run but is
an approximation. :func:`run_openmm_steered` sits outside the sequence and
pulls a collective variable to generate a reference path (see
:mod:`reactiontools.tools_path`). The RPMD production stages can also run
with their thermostat off (``apply_thermostat=False``), which is the
microcanonical ring-polymer dynamics that RPMD time-correlation observables
are defined in; :func:`openmmnqe.rates.run_openmm_rpmd_recrossing` builds a
transmission-coefficient calculation on top of that.

Every stage takes the same shape: build the system, optionally deuterate it,
attach a PLUMED bias and the reporters, run, then save its final structure.
Classical stages initialize new velocities when started from that structure;
the paired RPMD and adQTB stages additionally consume their equilibration
restart, preserving the state their production integrator needs. RPMD restart
files contain every bead rather than an ordinary single-Context checkpoint.
The shared arguments behave the same throughout -- *potential* with *ml_idx*
runs an ML/MM mixed system and forces the CUDA platform,
*plumed_script_path* attaches a bias, *output_prefix* names every file the
stage writes, and *seed* fixes every random stream the stage draws, so that
the run reproduces bit for bit.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import warnings
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal, overload

import numpy as np
import openmm.unit as unit
from openmm import app, openmm
from openmmml import MLPotential
from openmmplumed import PlumedForce

from ._validation import require_integer, require_positive_finite_scalar_in_unit
from .adqtb import QTBFrictionReporter
from .reporters import (
    RPMDBeadReporter,
    RPMDCentroidReporter,
    RPMDKineticDecompositionReporter,
    RPMDQuantumSpreadReporter,
    RPMDThermodynamicReporter,
    RPMDVelocityReporter,
    _validate_observable_indices,
)
from .tools import (
    WorkflowDeuterationOption,
    _particle_masses_dalton,
    centroid_positions,
    check_platform,
    deuterate_system,
    init_beads,
    set_adqtb_particle_types_by_element,
    step_rpmd,
)

_RPMD_RESTART_KIND = "openmmnqe-rpmd-restart"
_RPMD_RESTART_VERSION = 2


def _validate_rpmd_n_beads(n_beads: int) -> int:
    """
    Require an RPMD bead count to be a positive, non-boolean integer.

    Parameters
    ----------
    n_beads : int
        Bead count to check.

    Returns
    -------
    int
        The bead count as a plain int.

    Raises
    ------
    ValueError
        If *n_beads* is a bool, not an integer, or not positive.
    """
    if (
        isinstance(n_beads, (bool, np.bool_))
        or not isinstance(n_beads, (int, np.integer))
        or n_beads <= 0
    ):
        raise ValueError("n_beads must be a positive integer")
    return int(n_beads)


def _validate_ml_indices(ml_idx: Iterable[int], n_atoms: int) -> list[int]:
    """Return a normalized, unique, in-bounds ML atom selection."""
    raw_indices = list(ml_idx)
    if not raw_indices:
        raise ValueError("ml_idx is empty; a mixed system needs at least one ML atom.")

    normalized = []
    seen = set()
    for position, index in enumerate(raw_indices):
        index = require_integer(
            index,
            name=f"ml_idx[{position}]",
            minimum=0,
        )
        if index >= n_atoms:
            raise ValueError(
                f"ml_idx[{position}]={index} is outside the topology with "
                f"{n_atoms} atoms"
            )
        if index in seen:
            raise ValueError(f"ml_idx contains duplicate atom index {index}")
        normalized.append(index)
        seen.add(index)
    return normalized


def _validate_rpmd_contractions(
    contractions: Mapping[int, int] | None,
    n_beads: int,
) -> dict[int, int]:
    """Normalize an RPMD force-group contraction map for *n_beads*."""
    if contractions is None:
        return {1: min(8, n_beads), 2: 1}
    if not isinstance(contractions, Mapping):
        raise TypeError("contractions must be a mapping of force groups to copy counts")

    normalized = {}
    for raw_group, raw_count in contractions.items():
        group = require_integer(raw_group, name="contraction force group", minimum=0)
        if group > 31:
            raise ValueError("contraction force group must be between 0 and 31")
        count = require_integer(
            raw_count,
            name=f"contractions[{group}]",
            minimum=1,
        )
        if count > n_beads:
            raise ValueError(
                f"contractions[{group}]={count} cannot exceed n_beads={n_beads}"
            )
        normalized[group] = count
    return normalized


# Ordered registry of the independent random streams a stage can draw from.
# Two consumers sharing a stream would correlate -- a thermostat driven by the
# same numbers that chose the starting velocities is not the same run twice --
# so each gets its own child of the master seed's SeedSequence. The order is
# part of the derivation, since a stream is identified by its index here:
# appending a name leaves every seed already in use meaning what it did, and
# reordering silently changes all of them.
_SEED_STREAMS: tuple[str, ...] = (
    "initialization",  # NumPy generator placing the RPMD beads
    "thermostat",      # Langevin, RPMD PILE, or QTB integrator noise
    "velocities",      # Context.setVelocitiesToTemperature
    "barostat",        # Monte Carlo barostat volume moves
)

# NumPy takes any 32-bit value; OpenMM reads zero as a request for a
# non-deterministic seed, so its streams are mapped onto the positive range.
_NUMPY_SEED_STREAMS = frozenset({"initialization"})


def _derive_seeds(seed: int | None, *streams: str) -> tuple[int | None, ...]:
    """
    Derive one independent seed per named stream from a master seed.

    Parameters
    ----------
    seed : int or None
        Non-negative master seed, or None to leave every stream
        non-deterministic.
    *streams : str
        Names from :data:`_SEED_STREAMS`, one per seed wanted, returned in the
        order given.

    Returns
    -------
    tuple of (int or None)
        One seed per requested stream, or all None if *seed* is None. Seeds
        bound for OpenMM are never zero, which OpenMM would read as a request
        for a non-deterministic seed.

    Raises
    ------
    KeyError
        If a name is not a registered stream.
    ValueError
        If *seed* is a bool, not an integer, or negative.
    """
    for name in streams:
        if name not in _SEED_STREAMS:
            raise KeyError(f"unknown seed stream {name!r}")
    if seed is None:
        return tuple(None for _ in streams)
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")

    # Spawning the whole registry rather than only what was asked for is what
    # ties a stream's value to its index, so two stages given the same master
    # seed agree on what "velocities" means however many streams either draws.
    children = np.random.SeedSequence(int(seed)).spawn(len(_SEED_STREAMS))
    derived: list[int] = []
    for name in streams:
        child = children[_SEED_STREAMS.index(name)]
        raw = int(child.generate_state(1, dtype=np.uint32)[0])
        if name in _NUMPY_SEED_STREAMS:
            derived.append(raw)
        else:
            # Zero is reserved for the seed=None path, so that every explicit
            # master seed stays reproducible.
            derived.append(raw % 2_147_483_647 or 1)
    return tuple(derived)


def _seed_random_stream(target: Any, seed: int | None) -> None:
    """
    Fix an OpenMM integrator's or barostat's random stream.

    Parameters
    ----------
    target : object
        Any OpenMM object carrying ``setRandomNumberSeed``, i.e. an integrator
        or a Monte Carlo barostat.
    seed : int or None
        Seed to set, or None to leave OpenMM's own non-deterministic choice in
        place.
    """
    if seed is not None:
        target.setRandomNumberSeed(seed)


def _set_velocities_to_temperature(simulation: app.Simulation,
                                   temperature: unit.Quantity,
                                   seed: int | None) -> None:
    """
    Draw fresh Maxwell-Boltzmann velocities at *temperature*.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation whose Context is given the new velocities.
    temperature : openmm.unit.Quantity
        Temperature the velocities are drawn at.
    seed : int or None
        Seed for the draw. None is not passed through as a seed at all, so
        that OpenMM falls back to its own entropy source.
    """
    if seed is None:
        simulation.context.setVelocitiesToTemperature(temperature)
    else:
        simulation.context.setVelocitiesToTemperature(temperature, seed)


def _validate_pdb_identity_name(name: str, description: str,
                                max_length: int) -> None:
    """
    Reject topology identity names that PDB output cannot preserve.

    Parameters
    ----------
    name : str
        Atom or residue name to check.
    description : str
        What *name* labels, used in the error message.
    max_length : int
        Width of the PDB field the name has to fit, 4 for an atom and 3 for
        a residue.

    Raises
    ------
    ValueError
        If *name* is empty, over-long, non-ASCII, contains whitespace, or is
        not a string.
    """
    if (
        not isinstance(name, str)
        or not name
        or len(name) > max_length
        or not name.isascii()
        or any(character.isspace() for character in name)
    ):
        raise ValueError(
            f"Cannot create a PDB-stable RPMD restart: {description} name "
            f"{name!r} is not representable in a {max_length}-character "
            "PDB identity field"
        )


def _topology_identity_signature(topology: app.Topology) -> str:
    """
    Return a PDB-round-trip-stable ordered topology signature.

    PDB serialization is allowed to renumber atom, residue, and chain IDs and
    does not reliably preserve bond type/order metadata.  The restart identity
    therefore uses atom order, chemically meaningful names/elements, the
    ordered chain/residue grouping, and bond endpoints.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology to fingerprint.

    Returns
    -------
    str
        Hex SHA-256 digest of the signature, stable across a PDB round trip
        of the same structure.

    Raises
    ------
    ValueError
        If an atom or residue name would not survive PDB output.
    """
    chain_ordinals = {
        chain: ordinal for ordinal, chain in enumerate(topology.chains())
    }
    residue_ordinals = {
        residue: ordinal for ordinal, residue in enumerate(topology.residues())
    }
    for residue in topology.residues():
        _validate_pdb_identity_name(
            residue.name,
            f"residue {residue_ordinals[residue]}",
            3,
        )
    atoms = []
    for atom in topology.atoms():
        residue = atom.residue
        chain = residue.chain
        element = atom.element
        _validate_pdb_identity_name(atom.name, f"atom {atom.index}", 4)
        atoms.append({
            "name": atom.name,
            "element": None if element is None else element.symbol,
            "atomic_number": (
                None if element is None else element.atomic_number
            ),
            "residue_ordinal": residue_ordinals[residue],
            "residue_name": residue.name,
            "chain_ordinal": chain_ordinals[chain],
        })

    bonds = sorted(
        (
            min(bond.atom1.index, bond.atom2.index),
            max(bond.atom1.index, bond.atom2.index),
        )
        for bond in topology.bonds()
    )
    payload = json.dumps(
        {"atoms": atoms, "bonds": bonds},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rpmd_temperature_kelvin(integrator: openmm.RPMDIntegrator) -> float:
    """
    Return an RPMD integrator's finite, positive temperature in kelvin.

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        Integrator to read the temperature from.

    Returns
    -------
    float
        Temperature in kelvin.

    Raises
    ------
    ValueError
        If the temperature is not finite and positive.
    """
    temperature = float(integrator.getTemperature().value_in_unit(unit.kelvin))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("RPMDIntegrator temperature must be positive and finite")
    return temperature


@overload
def _restart_scalar(
    archive: np.lib.npyio.NpzFile,
    name: str,
    scalar_type: Literal["integer"],
) -> int: ...


@overload
def _restart_scalar(
    archive: np.lib.npyio.NpzFile,
    name: str,
    scalar_type: Literal["float"],
) -> float: ...


@overload
def _restart_scalar(
    archive: np.lib.npyio.NpzFile,
    name: str,
    scalar_type: Literal["boolean"],
) -> bool: ...


@overload
def _restart_scalar(
    archive: np.lib.npyio.NpzFile,
    name: str,
    scalar_type: Literal["string"],
) -> str: ...


def _restart_scalar(
    archive: np.lib.npyio.NpzFile,
    name: str,
    scalar_type: Literal["integer", "float", "boolean", "string"],
) -> int | float | bool | str:
    """
    Read a restart scalar only when its shape and dtype match the schema.

    Parameters
    ----------
    archive : numpy.lib.npyio.NpzFile
        Opened restart archive.
    name : str
        Field to read.
    scalar_type : {"integer", "float", "boolean", "string"}
        Kind the field is required to hold.

    Returns
    -------
    int or float or bool or str
        The stored value, as a Python scalar.

    Raises
    ------
    ValueError
        If the field is not a scalar, or does not hold *scalar_type*.
    """
    value = archive[name]
    if value.shape != ():
        raise ValueError(f"RPMD restart field {name} must be a scalar")

    dtype_kind = value.dtype.kind
    expected_kinds = {
        "integer": {"i", "u"},
        "float": {"f"},
        "boolean": {"b"},
        "string": {"U"},
    }
    if dtype_kind not in expected_kinds[scalar_type]:
        raise ValueError(
            f"RPMD restart field {name} must be a scalar {scalar_type}"
        )
    return value.item()


class PreparedSystem:
    """
    Stand-in force field that hands a pre-built ``openmm.System`` to a stage.

    Every ``run_openmm_*`` stage builds its System by calling
    ``forcefield.createSystem(topology, **kwargs)``.  Wrapping an existing
    System in this class routes it through that seam unchanged, so a System
    prepared elsewhere -- for example a QM/MM System carrying an
    ``openmm.PythonForce`` -- runs under the classical, RPMD, and adQTB
    stages without any stage signature changing.

    Parameters
    ----------
    system : openmm.System
        The System to hand out.

    Raises
    ------
    TypeError
        If *system* is not an ``openmm.System``.

    Notes
    -----
    ``createSystem`` returns the held System as-is, never a copy: a copy
    would sever externally held references (an ``openmm.PythonForce``
    callback, say) from the System the stage actually runs.  Stage options
    that mutate the System -- ``deuterate``, a non-None ``barostat_freq``,
    ``plumed_script_path`` -- therefore mutate this instance too, and
    re-running a mutating stage against the same ``PreparedSystem``
    accumulates their forces.  Build a fresh System and ``PreparedSystem``
    per stage when using those options.

    A ``PreparedSystem`` can also serve as the MM base of an ML/MM mixed
    system by passing *potential* and *ml_idx* to a stage as usual.
    """

    def __init__(self, system: openmm.System) -> None:
        if not isinstance(system, openmm.System):
            raise TypeError(
                "PreparedSystem wraps an existing openmm.System, got "
                f"{type(system).__name__}. Build the System first, or pass "
                "a force field to the stage instead."
            )
        self._system = system

    @property
    def system(self) -> openmm.System:
        """
        The held System that :meth:`createSystem` returns.

        Returns
        -------
        openmm.System
            The System this instance was built around.
        """
        return self._system

    def createSystem(self, topology: app.Topology,
                     **kwargs: Any) -> openmm.System:
        """
        Return the held System after checking it matches *topology*.

        Parameters
        ----------
        topology : openmm.app.Topology
            The topology the stage is about to simulate.
        **kwargs
            System-construction options, accepted and ignored: they describe
            how to build a System, and this one is already built.

        Returns
        -------
        openmm.System
            The held System, as-is.

        Raises
        ------
        ValueError
            If the topology's atom count differs from the held System's
            particle count, meaning the System was prepared for a different
            structure than the stage was given.
        """
        num_atoms = topology.getNumAtoms()
        num_particles = self._system.getNumParticles()
        if num_atoms != num_particles:
            raise ValueError(
                f"PreparedSystem holds {num_particles} particles but the "
                f"stage topology has {num_atoms} atoms; the System was "
                "prepared for a different structure. Rebuild the System from "
                "the same structure the Modeller was built from."
            )
        return self._system


def _build_system(modeller: app.Modeller,
                  forcefield: app.ForceField | MLPotential | PreparedSystem,
                  platform_name: str | None,
                  potential: Any,
                  ml_idx: list[int] | None,
                  calculator: Any,
                  ) -> tuple[openmm.System, openmm.Platform]:
    """
    Construct the system and platform shared by every ``run_openmm_*`` driver.

    Three configurations are supported:

    * **Pure MM** -- *forcefield* is a plain ``openmm.app.ForceField`` and no
      ML arguments are given.
    * **Pure ML** -- *forcefield* is an ``openmmml.MLPotential`` standing in
      for a force field; *ml_idx* stays None because every atom is ML. An ASE
      *calculator* may be supplied for ``MLPotential('ase')`` and is forwarded
      to its ``createSystem``.
    * **ML/MM mixed** -- *potential* (or *calculator*, which implies
      ``MLPotential('ase')``) together with *ml_idx* promotes the MM system to
      a mixed system via ``createMixedSystem``, and the platform is forced to
      CUDA.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField or openmmml.MLPotential
        The force field used to parameterise the system, or an ML potential
        used in its place for a pure-ML system.
    platform_name : str or None
        OpenMM platform name. None auto-detects via ``check_platform``.
    potential : object or None
        ML potential object with a ``createMixedSystem`` method.
    ml_idx : list of int or None
        Atom indices for the ML region of a mixed system.
    calculator : object or None
        Optional ASE calculator. Only ``MLPotential('ase')`` consumes it;
        other potentials silently ignore it.

    Returns
    -------
    system : openmm.System
        The parameterised system.
    platform : openmm.Platform
        The platform to run it on.

    Raises
    ------
    ValueError
        If *potential* is given without *ml_idx*, or if *calculator* is given
        without *ml_idx* when *forcefield* is not an ``MLPotential``. In both
        cases the ML region is undefined, and ``ForceField.createSystem``
        absorbs unknown keywords into ``**args``, so passing a calculator
        through would quietly run the whole simulation at pure MM instead.
        Also raised when *ml_idx* is empty or is supplied without either an ML
        potential or calculator, or when an ML atom index is negative,
        duplicated, or outside the topology.
    """
    if potential is not None and ml_idx is None:
        raise ValueError(
            "An ML potential was given but ml_idx is None, so the ML region is "
            "undefined. Pass ml_idx with the indices of the atoms the ML potential "
            "should cover, or pass the MLPotential as forcefield to run pure ML."
        )
    if (calculator is not None and ml_idx is None
            and not isinstance(forcefield, MLPotential)):
        raise ValueError(
            "A calculator was given without ml_idx, and forcefield is not an "
            "MLPotential, so ForceField.createSystem would silently ignore it. "
            "Pass ml_idx to define an ML/MM region, or pass MLPotential('ase') "
            "as forcefield to run pure ML."
        )
    if ml_idx is not None and potential is None and calculator is None:
        raise ValueError(
            "ml_idx was given without an ML potential or calculator. Pass a "
            "potential/calculator for a mixed system, or omit ml_idx for pure MM."
        )
    run_mixed = ml_idx is not None and (potential is not None or calculator is not None)
    if run_mixed:
        assert ml_idx is not None
        ml_idx = _validate_ml_indices(
            ml_idx,
            modeller.topology.getNumAtoms(),
        )
        print("Adding ML potential to the system...", flush=True)
        platform_name = 'CUDA'
        print("ML potential in use: forcing platform to CUDA.", flush=True)

        # An explicit potential wins; 'ase' is only the fallback that wraps a
        # bare calculator.
        if calculator is not None and potential is None:
            potential = MLPotential('ase')

    platform = openmm.Platform.getPlatformByName(check_platform(platform_name))
    has_box = modeller.topology.getUnitCellDimensions() is not None

    system_kwargs = {
        'nonbondedMethod': app.PME if has_box else app.CutoffNonPeriodic,
        'nonbondedCutoff': 1.0 * unit.nanometer,
        'constraints': None,
        'rigidWater': False,
        'removeCMMotion': True,
        # RPMD and adQTB depend on the physical vibrational frequencies, so
        # hydrogen mass repartitioning must not be used.  Keep this explicit:
        # it is too important for the NQE workflows to rely on an OpenMM
        # default, and it also keeps the MM part of a mixed system unchanged.
        'hydrogenMass': None,
    }

    if not run_mixed:
        # Pure MM, or pure ML with the MLPotential standing in as forcefield.
        if calculator is not None:
            system_kwargs['calculator'] = calculator
        return forcefield.createSystem(modeller.topology, **system_kwargs), platform

    mm_system = forcefield.createSystem(modeller.topology, **system_kwargs)
    if calculator is not None:
        system_kwargs['calculator'] = calculator
    system = potential.createMixedSystem(modeller.topology, mm_system, ml_idx, **system_kwargs)

    return system, platform


def _maybe_deuterate(modeller: app.Modeller, system: openmm.System,
                     deuterate: bool,
                     deuterate_option: WorkflowDeuterationOption) -> None:
    """
    Deuterate the system in place when requested.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller whose topology names the hydrogens to convert.
    system : openmm.System
        System whose particle masses are edited in place.
    deuterate : bool
        Whether to deuterate at all. False makes this a no-op.
    deuterate_option : str
        Selection passed through to :func:`openmmnqe.tools.deuterate_system`.
    """
    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)


def _validate_barostat_frequency(
    system: openmm.System,
    barostat_freq: int | None,
) -> int | None:
    """Validate a requested barostat frequency and system periodicity."""
    if barostat_freq is None:
        return None

    frequency = require_integer(
        barostat_freq,
        name="barostat_freq",
        minimum=1,
    )
    if not system.usesPeriodicBoundaryConditions():
        raise ValueError(
            "A barostat requires a periodic System with at least one force "
            "configured for periodic boundaries (for example PME or "
            "CutoffPeriodic); otherwise pass barostat_freq=None"
        )
    return frequency


def _warn_barostat_on_python_force(system: openmm.System) -> None:
    """
    Warn when a barostat is added to a System carrying a ``PythonForce``.

    OpenMM builds its molecule list from constraints and the bonded pairs each
    force reports, and a ``PythonForce`` reports none, so every atom it covers
    becomes its own molecule and the barostat scales those atoms individually
    instead of translating molecules rigidly.  That is still a valid volume
    move -- the Jacobian follows whatever is scaled -- but acceptance falls off
    once stiff covalent bonds are being strained, and a callback that ignores
    the box vectors it is handed contributes nothing to the energy change at
    all.  Both are worth knowing about; neither is worth refusing to run.

    Parameters
    ----------
    system : openmm.System
        System about to receive a barostat force.

    Warns
    -----
    UserWarning
        If any force on *system* is an ``openmm.PythonForce``.
    """
    if any(isinstance(force, openmm.PythonForce)
           for force in system.getForces()):
        warnings.warn(
            "Adding a barostat to a System carrying an external PythonForce "
            "potential: OpenMM sees no bonds through that force, so its atoms "
            "are scaled one at a time rather than as molecules. Watch the "
            "barostat acceptance rate, and check that the potential responds "
            "to the periodic box vectors it is passed; pass barostat_freq=None "
            "to run at fixed volume instead.",
            stacklevel=3,
        )


def _load_plumed(system: openmm.System,
                 plumed_script_path: str | None) -> None:
    """
    Attach a PLUMED bias force to the system if a script path is given.

    Parameters
    ----------
    system : openmm.System
        System the bias force is added to, in place.
    plumed_script_path : str or None
        Path to a PLUMED input script. None makes this a no-op.
    """
    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path) as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)


def _is_inline_plumed_input(plumed_input: str | os.PathLike[str]) -> bool:
    """Distinguish inline PLUMED text from a path without opening either."""
    if isinstance(plumed_input, os.PathLike):
        return False
    if "\n" in plumed_input or "\r" in plumed_input:
        return True
    if os.path.exists(plumed_input):
        return False
    if "=" in plumed_input:
        return True
    if os.path.sep in plumed_input or (
        os.path.altsep is not None and os.path.altsep in plumed_input
    ):
        return False
    if os.path.splitext(plumed_input)[1]:
        return False
    if any(character.isspace() for character in plumed_input):
        return True
    return plumed_input.isupper()


def _add_standard_reporters(simulation: app.Simulation, output_prefix: str,
                            n_report: int, pdb_steps: bool = False,
                            stdout_volume: bool = False,
                            checkpoint_interval: int | None = None) -> None:
    """
    Append the standard reporter set shared by the classical drivers.

    Order: optional PDB trajectory, stdout state data, ``.log`` state data,
    optional periodic checkpoint.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation the reporters are appended to.
    output_prefix : str
        Prefix for the files written, giving ``<prefix>.log`` and friends.
    n_report : int
        Interval between reports, in steps.
    pdb_steps : bool, optional
        Also write a ``<prefix>_steps.pdb`` trajectory. Default is False.
    stdout_volume : bool, optional
        Include box volume in the stdout report, which is worth having under
        a barostat. Default is False.
    checkpoint_interval : int or None, optional
        Interval between ``<prefix>.chk`` checkpoints. With None, no
        checkpoint reporter is added. Default is None.
    """
    if pdb_steps:
        simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True,
                                                      volume=stdout_volume))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))
    if checkpoint_interval is not None:
        simulation.reporters.append(app.CheckpointReporter(f'{output_prefix}.chk',
                                                           checkpoint_interval))


def _add_rpmd_progress_reporters(simulation: app.Simulation,
                                 output_prefix: str, n_report: int) -> None:
    """
    Append Context-independent progress reporters for an RPMD run.

    An RPMD integrator's ordinary Context state is not a bead average and is
    not guaranteed to mirror any particular copy.  Step, time, speed, and box
    volume remain meaningful, but Context energy and kinetic-temperature
    fields do not, so they are deliberately omitted here.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation the reporters are appended to.
    output_prefix : str
        Prefix for the ``<prefix>.log`` progress log.
    n_report : int
        Interval between reports, in steps.
    """
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout,
        n_report,
        step=True,
        speed=True,
    ))
    simulation.reporters.append(app.StateDataReporter(
        f'{output_prefix}.log',
        n_report,
        step=True,
        time=True,
        speed=True,
        volume=True,
    ))


def _add_rpmd_reporters(simulation: app.Simulation, topology: app.Topology,
                        output_prefix: str, n_report: int, n_beads: int,
                        atoms_to_watch: list[int] | None,
                        expansion_metric: Literal["rms", "mean"] = "rms",
                        distance_pairs: Iterable[tuple[int, int]] | None = None,
                        velocity_record_interval: int | None = None,
                        velocity_atom_indices: Sequence[int] | None = None,
                        kinetic_decomposition: bool = False,
                        ) -> None:
    """
    Append the RPMD reporter trio: optional spread, then centroid and beads.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation the reporters are appended to.
    topology : openmm.app.Topology
        Topology written into the PDB output and used to bounds-check the
        watched atoms.
    output_prefix : str
        Prefix for ``<prefix>_spread.log``, ``<prefix>_centroid.pdb``, the
        per-bead ``<prefix>_bead_<i>.pdb`` files and ``<prefix>_thermo.log``.
    n_report : int
        Interval between reports, in steps.
    n_beads : int
        Number of ring-polymer beads.
    atoms_to_watch : list of int or None
        Atoms whose expansion is logged. With None, no spread reporter is
        added and *distance_pairs* may not be given either.
    expansion_metric : {"rms", "mean"}, optional
        Spread metric recorded for *atoms_to_watch*. Default is ``"rms"``.
    distance_pairs : iterable of pair of int or None, optional
        Atom pairs whose centroid distance is logged alongside the expansion.
        Default is None.
    velocity_record_interval : int or None, optional
        If set, also attach an :class:`~openmmnqe.reporters.RPMDVelocityReporter`
        writing centroid velocities to ``<prefix>_velocities.npz`` every this
        many steps. Default is None.
    velocity_atom_indices : sequence of int or None, optional
        Atoms whose centroid velocities are recorded. Requires
        *velocity_record_interval*. Default is None, which records every
        atom.
    kinetic_decomposition : bool, optional
        If True, also attach an
        :class:`~openmmnqe.reporters.RPMDKineticDecompositionReporter`
        writing per-atom centroid-virial kinetic energies for
        *atoms_to_watch* to ``<prefix>_kinetic.log``. Requires
        *atoms_to_watch*. Default is False.

    Raises
    ------
    ValueError
        If *distance_pairs* is given without *atoms_to_watch*, or an index
        lies outside *topology*, or *velocity_atom_indices* is given without
        *velocity_record_interval*, or *kinetic_decomposition* is set without
        *atoms_to_watch*.

    Notes
    -----
    The thermodynamic reporter is always attached: an RPMD Context carries no
    meaningful energy or temperature, so without it a run leaves no energy
    trace at all.  Each of its reports reads every bead once, costing roughly
    one RPMD step; at the drivers' default *n_report* of 1000 that is a
    fraction of a percent.

    Under ring-polymer contraction its estimators use the full, uncontracted
    forces evaluated at each bead, so they describe the full potential rather
    than the contracted one that drives the dynamics.

    The kinetic decomposition is opt-in rather than automatic on
    *atoms_to_watch*, because it reads the beads a second time: turning it on
    for every caller of that argument would quietly double the per-report
    cost of a run that only wanted the spread log.
    """
    if distance_pairs is not None and atoms_to_watch is None:
        raise ValueError("distance_pairs require atoms_to_watch")
    if kinetic_decomposition and atoms_to_watch is None:
        raise ValueError("kinetic_decomposition requires atoms_to_watch")
    if atoms_to_watch is not None:
        atoms_to_watch, distance_pairs = _validate_observable_indices(
            atoms_to_watch,
            distance_pairs,
            n_atoms=topology.getNumAtoms(),
        )
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
            metric=expansion_metric,
            distance_pairs=distance_pairs,
        ))
        if kinetic_decomposition:
            simulation.reporters.append(RPMDKineticDecompositionReporter(
                file=f'{output_prefix}_kinetic.log',
                reportInterval=n_report,
                atom_indices=atoms_to_watch,
            ))

    simulation.reporters.append(RPMDCentroidReporter(
        topology=topology,
        file_name=f"{output_prefix}_centroid.pdb",
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDBeadReporter(
        topology=topology,
        file_base_name=output_prefix,
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDThermodynamicReporter(
        file=f'{output_prefix}_thermo.log',
        reportInterval=n_report,
    ))

    if velocity_atom_indices is not None and velocity_record_interval is None:
        raise ValueError(
            "velocity_atom_indices require velocity_record_interval"
        )
    if velocity_record_interval is not None:
        simulation.reporters.append(RPMDVelocityReporter(
            file=f'{output_prefix}_velocities.npz',
            reportInterval=velocity_record_interval,
            atom_indices=velocity_atom_indices,
        ))


# The adQTB noise buffer is transformed with a mixed-radix FFT, so a segment
# has to span a number of steps whose only prime factors are these.
_ADQTB_SEGMENT_FACTORS = (2, 3, 5, 7)


def _validate_adqtb_segment(segment_length: unit.Quantity,
                            time_step: unit.Quantity) -> int:
    """
    Return the number of steps in an adQTB segment, rejecting bad lengths.

    OpenMM raises on both of these conditions, but only once the Context is
    being built, which is after the System has been parameterised and any ML
    potential loaded.  Checking here costs nothing and fails in the second it
    takes to read the arguments.

    Parameters
    ----------
    segment_length : openmm.unit.Quantity
        Length of one adaptation segment.
    time_step : openmm.unit.Quantity
        Integration step size.

    Returns
    -------
    int
        Number of integration steps in one segment.

    Raises
    ------
    ValueError
        If the segment is not a whole number of steps, or that number has a
        prime factor larger than seven.
    """
    step_size = require_positive_finite_scalar_in_unit(
        time_step, unit.picosecond, name="time_step",
    )
    length = require_positive_finite_scalar_in_unit(
        segment_length, unit.picosecond, name="segment_length",
    )
    steps = int(round(length / step_size))
    if steps < 1 or abs(steps * step_size - length) > 1e-9:
        raise ValueError(
            f"segment_length must be a whole number of time steps, but "
            f"{length} ps is not a multiple of {step_size} ps"
        )
    remainder = steps
    for factor in _ADQTB_SEGMENT_FACTORS:
        while remainder % factor == 0:
            remainder //= factor
    if remainder != 1:
        raise ValueError(
            f"segment_length must span a number of steps whose only prime "
            f"factors are 2, 3, 5 and 7, but it spans {steps} steps"
        )
    return steps


def _adqtb_particle_labels(topology: app.Topology,
                           system: openmm.System) -> list[str]:
    """
    Label every particle by element, splitting a symbol when masses differ.

    ``deuterate_system`` edits masses without touching elements, so grouping
    on the element symbol alone would put hydrogen and deuterium in the same
    bath.  An adapted spectrum is mass-dependent -- that is the whole point
    of it -- so a particle whose mass has been moved away from its element's
    standard value gets a label, and therefore a bath, of its own.  OpenMM
    enforces this too: a Context refuses to build when one particle type
    spans more than one mass, so without the split a deuterated run would
    not start at all.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology naming the elements, in particle order.
    system : openmm.System
        System the masses are read from.

    Returns
    -------
    list of str
        One label per particle, e.g. ``["H", "H2.014", "O"]``.

    Raises
    ------
    ValueError
        If the topology and System disagree on the particle count, which
        means extra particles the elements cannot describe.
    """
    masses = _particle_masses_dalton(system)
    elements = [atom.element for atom in topology.atoms()]
    if len(elements) != len(masses):
        raise ValueError(
            f"topology has {len(elements)} atoms but the System has "
            f"{len(masses)} particles; pass particle_types explicitly"
        )
    labels = []
    for element, mass in zip(elements, masses, strict=True):
        if element is None:
            labels.append("X")
            continue
        standard = element.mass.value_in_unit(unit.dalton)
        if abs(mass - standard) > 1e-3:
            labels.append(f"{element.symbol}{mass:.3f}")
        else:
            labels.append(element.symbol)
    return labels


def _assign_adqtb_particle_types(
    integrator: openmm.QTBIntegrator,
    topology: app.Topology,
    system: openmm.System,
    particle_types: Literal["element", "none"] | Mapping[int, int] | None,
) -> dict[int, str] | None:
    """
    Assign the integrator's particle types before its Context is created.

    Without this every particle adapts a spectrum of its own, which is both
    far noisier at a given adaptation rate and unreadable once logged.

    Parameters
    ----------
    integrator : openmm.QTBIntegrator
        Integrator whose types are set, in place.
    topology : openmm.app.Topology
        Topology naming the elements.
    system : openmm.System
        System the masses are read from.
    particle_types : {"element", "none"} or mapping of int to int or None
        ``"element"`` groups by element and mass, ``"none"`` and None leave
        the types alone, and a mapping assigns particle index to type index
        directly.

    Returns
    -------
    dict of int to str or None
        Labels for each assigned type, for the friction log's column names,
        or None when the caller supplied the types itself.

    Raises
    ------
    ValueError
        If *particle_types* is neither a recognised keyword nor a mapping.
    """
    if particle_types is None or particle_types == "none":
        return None
    if particle_types == "element":
        label_to_type = set_adqtb_particle_types_by_element(
            integrator,
            particle_elements=_adqtb_particle_labels(topology, system),
            system=system,
        )
        return {index: label for label, index in label_to_type.items()}
    if isinstance(particle_types, Mapping):
        for particle, type_index in particle_types.items():
            integrator.setParticleType(
                require_integer(particle, name="particle index", minimum=0),
                require_integer(type_index, name="particle type", minimum=0),
            )
        return None
    raise ValueError(
        "particle_types must be 'element', 'none', or a mapping of particle "
        "index to type index"
    )


def _add_adqtb_progress_reporters(simulation: app.Simulation,
                                  output_prefix: str, n_report: int) -> None:
    """
    Append Context-independent progress reporters for an adQTB run.

    An adQTB thermostat drives the velocities to a quantum distribution, so
    the standard estimators of temperature and pressure, which assume a
    classical one, do not describe the run; OpenMM's own documentation warns
    that they "do not produce correct results for an adQTB simulation".
    Step, time, potential energy, speed and box volume remain meaningful, so
    the kinetic and total energies and the temperature are deliberately
    omitted here, exactly as they are for RPMD.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation the reporters are appended to.
    output_prefix : str
        Prefix for the ``<prefix>.log`` progress log.
    n_report : int
        Interval between reports, in steps.
    """
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout,
        n_report,
        step=True,
        potentialEnergy=True,
        speed=True,
    ))
    simulation.reporters.append(app.StateDataReporter(
        f'{output_prefix}.log',
        n_report,
        step=True,
        time=True,
        potentialEnergy=True,
        volume=True,
        speed=True,
    ))


def _add_adqtb_reporters(simulation: app.Simulation, output_prefix: str,
                         n_report: int, *, segment_steps: int,
                         type_names: dict[int, str] | None,
                         friction_log: bool,
                         checkpoint_interval: int | None) -> None:
    """
    Append the adQTB reporter set: trajectory, progress, friction spectra.

    The friction reporter runs at the adaptation cadence rather than at
    *n_report*: OpenMM adapts exactly once per segment, and a row per
    segment is what makes each row-to-row difference one
    fluctuation-dissipation correction.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation the reporters are appended to.
    output_prefix : str
        Prefix for ``<prefix>_steps.pdb``, ``<prefix>.log``,
        ``<prefix>.chk`` and ``<prefix>_friction.log``.
    n_report : int
        Interval between trajectory and progress reports, in steps.
    segment_steps : int
        Number of steps in one adaptation segment.
    type_names : dict of int to str or None
        Labels for the friction log's columns, one per particle type.
    friction_log : bool
        Whether to write ``<prefix>_friction.log`` at all.
    checkpoint_interval : int or None
        Interval between ``<prefix>.chk`` checkpoints, or None for no
        checkpoint reporter.

    Warns
    -----
    UserWarning
        If a friction log was asked for but no particle types are assigned,
        in which case every particle would adapt its own spectrum and the
        log would carry one block of columns per atom.
    """
    simulation.reporters.append(
        app.PDBReporter(f'{output_prefix}_steps.pdb', n_report)
    )
    _add_adqtb_progress_reporters(simulation, output_prefix, n_report)
    if checkpoint_interval is not None:
        simulation.reporters.append(
            app.CheckpointReporter(f'{output_prefix}.chk', checkpoint_interval)
        )
    if not friction_log:
        return
    if not dict(simulation.integrator.getParticleTypes()):
        warnings.warn(
            "no adQTB particle types are assigned, so every particle adapts "
            "its own noise spectrum; skipping the friction log",
            UserWarning,
            stacklevel=2,
        )
        return
    simulation.reporters.append(QTBFrictionReporter(
        f'{output_prefix}_friction.log',
        segment_steps,
        simulation.integrator,
        type_names,
    ))


def _close_output_reporters(
    simulation: app.Simulation,
    *,
    suppress_errors: bool,
) -> None:
    """Close every package-owned reporter attached to *simulation*."""
    first_error: Exception | None = None
    reporter_types = (
        QTBFrictionReporter,
        RPMDQuantumSpreadReporter,
        RPMDCentroidReporter,
        RPMDBeadReporter,
        RPMDThermodynamicReporter,
    )
    for reporter in simulation.reporters:
        if not isinstance(reporter, reporter_types):
            continue
        try:
            reporter.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None and not suppress_errors:
        raise first_error


@contextmanager
def _finalize_reporters(
    simulation: app.Simulation,
) -> Iterator[None]:
    """Ensure package output reporters close on normal and exceptional exits."""
    try:
        yield
    except BaseException:
        _close_output_reporters(simulation, suppress_errors=True)
        raise
    else:
        _close_output_reporters(simulation, suppress_errors=False)


def _save_rpmd_restart(simulation: app.Simulation, checkpoint_file: str,
                       n_beads: int) -> None:
    """
    Atomically save every RPMD copy to a portable restart archive.

    OpenMM's ordinary ``Context`` checkpoint only sees the copy currently
    mirrored into the Context.  The other copies live in private arrays owned
    by ``RPMDIntegrator``, so they must be collected through its copy-specific
    API.  Positions are stored in nanometres and velocities in nanometres per
    picosecond.  The archive also carries the ordered particle masses, atom
    and topology signature, source temperature, box, time, and step count
    needed to validate and continue in a new ``Simulation``.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``.
    checkpoint_file : str
        Path the ``.npz`` archive is written to. It is written via a
        temporary file and moved into place, so an interrupted save cannot
        leave a half-written restart behind.
    n_beads : int
        Number of ring-polymer beads, checked against the integrator.

    Raises
    ------
    ValueError
        If *n_beads* is not a positive integer or disagrees with the
        integrator, or the topology cannot be given a PDB-stable signature.
    """
    n_beads = _validate_rpmd_n_beads(n_beads)

    integrator = simulation.integrator
    actual_beads = integrator.getNumCopies()
    if actual_beads != n_beads:
        raise ValueError(
            f"n_beads={n_beads} does not match RPMDIntegrator copies={actual_beads}"
        )

    n_particles = simulation.system.getNumParticles()
    if simulation.topology.getNumAtoms() != n_particles:
        raise ValueError(
            "Cannot save RPMD restart: Topology atom count does not match "
            "System particle count"
        )
    particle_masses = _particle_masses_dalton(simulation.system)
    topology_signature = _topology_identity_signature(simulation.topology)
    temperature_kelvin = _rpmd_temperature_kelvin(integrator)
    position_frames = []
    velocity_frames = []
    first_state = None
    for bead in range(n_beads):
        state = integrator.getState(
            bead,
            getPositions=True,
            getVelocities=True,
        )
        if first_state is None:
            first_state = state
        bead_positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer
        )
        bead_velocities = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        if bead_positions.shape != (n_particles, 3):
            raise ValueError(
                f"Bead {bead} has position shape {bead_positions.shape}; "
                f"expected {(n_particles, 3)}"
            )
        if bead_velocities.shape != (n_particles, 3):
            raise ValueError(
                f"Bead {bead} has velocity shape {bead_velocities.shape}; "
                f"expected {(n_particles, 3)}"
            )
        position_frames.append(np.asarray(bead_positions, dtype=np.float64))
        velocity_frames.append(np.asarray(bead_velocities, dtype=np.float64))

    positions = np.stack(position_frames)
    velocities = np.stack(velocity_frames)
    if not np.isfinite(positions).all():
        raise ValueError("Cannot save RPMD restart: bead positions are not finite")
    if not np.isfinite(velocities).all():
        raise ValueError("Cannot save RPMD restart: bead velocities are not finite")

    if first_state is None:  # Defensive: n_beads validation makes this unreachable.
        raise RuntimeError("RPMD restart has no bead states to save")
    box_vectors = first_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
        unit.nanometer
    )
    time_ps = first_state.getTime().value_in_unit(unit.picosecond)
    step_count = simulation.currentStep
    periodic = simulation.system.usesPeriodicBoundaryConditions()

    checkpoint_file = os.fspath(checkpoint_file)
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_file))
    file_descriptor, temporary_file = tempfile.mkstemp(
        prefix=f".{os.path.basename(checkpoint_file)}.",
        suffix=".tmp",
        dir=checkpoint_dir,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.savez(
                handle,
                kind=np.asarray(_RPMD_RESTART_KIND),
                format_version=np.asarray(_RPMD_RESTART_VERSION, dtype=np.int64),
                num_beads=np.asarray(n_beads, dtype=np.int64),
                num_particles=np.asarray(n_particles, dtype=np.int64),
                particle_masses_dalton=particle_masses,
                topology_signature_sha256=np.asarray(topology_signature),
                temperature_kelvin=np.asarray(
                    temperature_kelvin, dtype=np.float64
                ),
                positions_nm=positions,
                velocities_nm_per_ps=velocities,
                periodic=np.asarray(periodic, dtype=np.bool_),
                box_vectors_nm=np.asarray(box_vectors, dtype=np.float64),
                time_ps=np.asarray(time_ps, dtype=np.float64),
                step_count=np.asarray(step_count, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_file, checkpoint_file)
    except BaseException:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)
        raise


def _read_rpmd_restart(checkpoint_file: str) -> dict[str, Any]:
    """
    Read and validate the format-independent portion of an RPMD restart.

    Parameters
    ----------
    checkpoint_file : str
        Path to a ``.npz`` archive written by :func:`_save_rpmd_restart`.

    Returns
    -------
    dict
        The restart contents: ``num_beads``, ``num_particles``,
        ``particle_masses_dalton``, ``topology_signature_sha256``,
        ``temperature_kelvin``, ``positions_nm``, ``velocities_nm_per_ps``,
        ``periodic``, ``box_vectors_nm``, and the run's time and step count.

    Raises
    ------
    ValueError
        If the file is corrupt, is a generic OpenMM checkpoint rather than a
        bead-aware one, or fails the restart schema.
    """
    try:
        archive = np.load(checkpoint_file, allow_pickle=False)
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"RPMD restart {checkpoint_file} is corrupt or unreadable; "
            "rerun RPMD equilibration to create a new checkpoint."
        ) from exc
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(
            f"{checkpoint_file} is not a bead-aware openmmnqe RPMD restart. "
            "A generic OpenMM checkpoint cannot safely restore all RPMD copies; "
            "rerun RPMD equilibration to create a new checkpoint."
        ) from exc

    try:
        try:
            if not hasattr(archive, "files"):
                raise ValueError("restart is not an NPZ archive")
            header = {"kind", "format_version"}
            missing_header = header.difference(archive.files)
            if missing_header:
                raise ValueError(
                    "RPMD restart is missing fields: "
                    + ", ".join(sorted(missing_header))
                )
            kind = _restart_scalar(archive, "kind", "string")
            if kind != _RPMD_RESTART_KIND:
                raise ValueError("checkpoint is not an openmmnqe RPMD restart")
            version = _restart_scalar(archive, "format_version", "integer")
            if version == 1:
                raise ValueError(
                    "RPMD restart version 1 lacks particle identity, mass, and "
                    "temperature metadata. Rerun RPMD equilibration to create a "
                    "compatible checkpoint."
                )
            if version != _RPMD_RESTART_VERSION:
                raise ValueError(
                    f"Unsupported RPMD restart version {version}; "
                    f"expected {_RPMD_RESTART_VERSION}. Rerun RPMD equilibration "
                    "to create a compatible checkpoint."
                )

            required = header | {
                "num_beads",
                "num_particles",
                "particle_masses_dalton",
                "topology_signature_sha256",
                "temperature_kelvin",
                "positions_nm",
                "velocities_nm_per_ps",
                "periodic",
                "box_vectors_nm",
                "time_ps",
                "step_count",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    "RPMD restart is missing fields: "
                    + ", ".join(sorted(missing))
                )
            num_beads = _validate_rpmd_n_beads(
                _restart_scalar(archive, "num_beads", "integer")
            )
            num_particles = _restart_scalar(
                archive, "num_particles", "integer"
            )
            if num_particles < 0:
                raise ValueError(
                    "RPMD restart num_particles cannot be negative"
                )
            step_count = _restart_scalar(archive, "step_count", "integer")
            if step_count < 0:
                raise ValueError("RPMD restart step count cannot be negative")
            topology_signature = _restart_scalar(
                archive, "topology_signature_sha256", "string"
            )
            if (
                len(topology_signature) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in topology_signature
                )
            ):
                raise ValueError(
                    "RPMD restart topology signature must be 64 lowercase "
                    "hexadecimal characters"
                )
            return {
                "num_beads": num_beads,
                "num_particles": int(num_particles),
                "particle_masses_dalton": np.array(
                    archive["particle_masses_dalton"], dtype=np.float64
                ),
                "topology_signature_sha256": topology_signature,
                "temperature_kelvin": float(
                    _restart_scalar(archive, "temperature_kelvin", "float")
                ),
                "positions_nm": np.array(
                    archive["positions_nm"], dtype=np.float64
                ),
                "velocities_nm_per_ps": np.array(
                    archive["velocities_nm_per_ps"], dtype=np.float64
                ),
                "periodic": bool(
                    _restart_scalar(archive, "periodic", "boolean")
                ),
                "box_vectors_nm": np.array(
                    archive["box_vectors_nm"], dtype=np.float64
                ),
                "time_ps": float(
                    _restart_scalar(archive, "time_ps", "float")
                ),
                "step_count": int(step_count),
            }
        except (OSError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(
                f"RPMD restart {checkpoint_file} is corrupt or unreadable; "
                "rerun RPMD equilibration to create a new checkpoint."
            ) from exc
    finally:
        if hasattr(archive, "close"):
            archive.close()


def _load_rpmd_restart(simulation: app.Simulation, checkpoint_file: str,
                       n_beads: int) -> None:
    """
    Restore every RPMD copy before the integrator takes its first step.

    The restart is checked against the Simulation it is being loaded into:
    a mismatched bead count, particle count, mass ordering, topology
    signature, or temperature means the archive belongs to a different run
    and is refused rather than silently reinterpreted.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation driven by an ``RPMDIntegrator``, restored in place.
    checkpoint_file : str
        Path to a restart written by :func:`_save_rpmd_restart`.
    n_beads : int
        Number of ring-polymer beads the run expects.

    Raises
    ------
    ValueError
        If the restart is corrupt, or does not match the current bead count,
        ordered topology, particle masses, or temperature.
    """
    n_beads = _validate_rpmd_n_beads(n_beads)
    restart = _read_rpmd_restart(checkpoint_file)
    integrator = simulation.integrator
    integrator_beads = integrator.getNumCopies()
    n_particles = simulation.system.getNumParticles()
    periodic = simulation.system.usesPeriodicBoundaryConditions()
    if simulation.topology.getNumAtoms() != n_particles:
        raise ValueError(
            "Current Topology atom count does not match System particle count"
        )
    current_masses = _particle_masses_dalton(simulation.system)
    current_signature = _topology_identity_signature(simulation.topology)
    current_temperature = _rpmd_temperature_kelvin(integrator)

    if restart["num_beads"] != n_beads:
        raise ValueError(
            f"RPMD restart contains {restart['num_beads']} beads, but "
            f"n_beads={n_beads} was requested"
        )
    if integrator_beads != n_beads:
        raise ValueError(
            f"n_beads={n_beads} does not match RPMDIntegrator copies={integrator_beads}"
        )
    if restart["num_particles"] != n_particles:
        raise ValueError(
            f"RPMD restart contains {restart['num_particles']} particles, but "
            f"the current System contains {n_particles}"
        )
    if restart["particle_masses_dalton"].shape != (n_particles,):
        raise ValueError(
            "RPMD restart particle masses have shape "
            f"{restart['particle_masses_dalton'].shape}; expected {(n_particles,)}"
        )
    if not np.isfinite(restart["particle_masses_dalton"]).all():
        raise ValueError("RPMD restart particle masses contain non-finite values")
    if np.any(restart["particle_masses_dalton"] < 0.0):
        raise ValueError("RPMD restart particle masses cannot be negative")
    if not np.allclose(
        restart["particle_masses_dalton"],
        current_masses,
        rtol=1e-12,
        atol=1e-12,
    ):
        mismatch = int(np.flatnonzero(~np.isclose(
            restart["particle_masses_dalton"],
            current_masses,
            rtol=1e-12,
            atol=1e-12,
        ))[0])
        raise ValueError(
            f"RPMD restart particle mass {mismatch} does not match the current "
            "System; check atom ordering and deuteration settings"
        )
    if restart["topology_signature_sha256"] != current_signature:
        raise ValueError(
            "RPMD restart atom/topology identity does not match the current "
            "Topology; check atom ordering and source structure"
        )
    if not np.isfinite(restart["temperature_kelvin"]):
        raise ValueError("RPMD restart temperature is not finite")
    if not np.isclose(
        restart["temperature_kelvin"],
        current_temperature,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"RPMD restart temperature {restart['temperature_kelvin']} K does "
            f"not match the current RPMDIntegrator temperature "
            f"{current_temperature} K"
        )
    expected_shape = (n_beads, n_particles, 3)
    if restart["positions_nm"].shape != expected_shape:
        raise ValueError(
            f"RPMD restart positions have shape {restart['positions_nm'].shape}; "
            f"expected {expected_shape}"
        )
    if restart["velocities_nm_per_ps"].shape != expected_shape:
        raise ValueError(
            "RPMD restart velocities have shape "
            f"{restart['velocities_nm_per_ps'].shape}; expected {expected_shape}"
        )
    if restart["box_vectors_nm"].shape != (3, 3):
        raise ValueError(
            f"RPMD restart box has shape {restart['box_vectors_nm'].shape}; "
            "expected (3, 3)"
        )
    if restart["periodic"] != periodic:
        raise ValueError(
            "RPMD restart periodicity does not match the current System"
        )
    for name in ("positions_nm", "velocities_nm_per_ps", "box_vectors_nm"):
        if not np.isfinite(restart[name]).all():
            raise ValueError(f"RPMD restart field {name} contains non-finite values")
    if not np.isfinite(restart["time_ps"]):
        raise ValueError("RPMD restart time is not finite")

    if periodic:
        box_vectors = [
            openmm.Vec3(*row) * unit.nanometer
            for row in restart["box_vectors_nm"]
        ]
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setTime(restart["time_ps"] * unit.picosecond)
    simulation.currentStep = restart["step_count"]
    for bead in range(n_beads):
        integrator.setPositions(
            bead,
            restart["positions_nm"][bead] * unit.nanometer,
        )
        integrator.setVelocities(
            bead,
            restart["velocities_nm_per_ps"][bead]
            * unit.nanometer
            / unit.picosecond,
        )


def _save_final_state(simulation: app.Simulation, output_prefix: str,
                      pdb_suffix: str = '.pdb', save_checkpoint: bool = True,
                      n_beads: int | None = None) -> None:
    """
    Save an optional checkpoint and write the final structure to PDB.

    Pass *n_beads* for an RPMD run: all copy positions and velocities are
    saved in a bead-aware restart archive, and the final PDB positions are
    averaged over the copies via
    :func:`openmmnqe.tools.centroid_positions`. Without a bead count, an
    ordinary OpenMM checkpoint and Context structure are written. For a
    periodic system the topology's box is refreshed from the Context first,
    so the PDB's CRYST1 record reflects any barostat moves.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation to save.
    output_prefix : str
        Prefix for the files written, giving ``<prefix>.chk`` and
        ``<prefix><pdb_suffix>``.
    pdb_suffix : str, optional
        Suffix for the final structure file. Default is ``'.pdb'``.
    save_checkpoint : bool, optional
        Whether to write a checkpoint alongside the structure. Default is
        True.
    n_beads : int or None, optional
        Number of ring-polymer beads for an RPMD run, or None for a
        classical one. Default is None.
    """
    if save_checkpoint:
        checkpoint_file = f'{output_prefix}.chk'
        if n_beads is None:
            simulation.saveCheckpoint(checkpoint_file)
        else:
            _save_rpmd_restart(simulation, checkpoint_file, n_beads)
    if simulation.system.usesPeriodicBoundaryConditions():
        box_vectors = simulation.context.getState().getPeriodicBoxVectors()
        simulation.topology.setPeriodicBoxVectors(box_vectors)
    if n_beads is None:
        positions = simulation.context.getState(getPositions=True).getPositions()
    else:
        positions = centroid_positions(simulation,
                                       simulation.topology.getNumAtoms(),
                                       n_beads)
    with open(f'{output_prefix}{pdb_suffix}', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, positions, f)


def _load_checkpoint(simulation: app.Simulation, checkpoint_file: str,
                     n_beads: int | None = None) -> None:
    """
    Load equilibration restart data into *simulation*.

    With *n_beads*, require the bead-aware openmmnqe RPMD format and restore
    every copy. Without it, delegate to OpenMM's ordinary Context checkpoint
    loader.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation restored in place.
    checkpoint_file : str
        Path to the checkpoint written by the preceding stage.
    n_beads : int or None, optional
        Number of ring-polymer beads for an RPMD run, or None for a
        classical one. Default is None.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist. Returning quietly here would let a
        batch job exit successfully having run no simulation at all.
    ValueError
        If an RPMD restart is legacy, corrupt, or incompatible with the current
        bead count, ordered topology, particle masses, or temperature.
    """
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_file} not found. Run the equilibration stage first, "
            "or pass checkpoint_file with the path it was written to."
        )

    print(f"Loading state from {checkpoint_file}...", flush=True)
    if n_beads is None:
        simulation.loadCheckpoint(checkpoint_file)
    else:
        _load_rpmd_restart(simulation, checkpoint_file, n_beads)


def run_openmm_relaxation(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        output_prefix: str = 'minimized',
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 1.0 * unit.femtoseconds,
        n_1: int = 1_000,
        n_2: int = 1_000,
        n_3: int = 2_000,
        backbone_names: list[str] | None = None,
        ks_1: float = 100.0,
        ks_2: float = 10.0,
        ks_3: float = 0.0,
        platform_name: str | None = None,
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        seed: int | None = None,
) -> None:
    """
    Minimise in stages, easing the backbone restraints as it goes.

    Three successive minimisation stages are executed with decreasing quadratic
    restraint coefficients on backbone atoms, allowing the structure to relax
    gently. An optional ML/MM mixed potential can be used.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for the output PDB file. Default is ``'minimized'``.
    temperature : openmm.unit.Quantity, optional
        Temperature for the Langevin integrator. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    n_1 : int, optional
        Maximum iterations for stage 1 (strong restraints). Default is 1000.
    n_2 : int, optional
        Maximum iterations for stage 2 (weak restraints). Default is 1000.
    n_3 : int, optional
        Maximum iterations for stage 3 (unrestrained). Default is 2000.
    backbone_names : list of str or None, optional
        Atom names considered backbone for restraints. If None, defaults to
        ``['CA', 'C', 'N', 'P', 'O3']``.
    ks_1 : float, optional
        Coefficient multiplying the squared displacement in stage 1, in
        kJ/mol/nm^2. Default is 100.0.
    ks_2 : float, optional
        Coefficient multiplying the squared displacement in stage 2, in
        kJ/mol/nm^2. Default is 10.0.
    ks_3 : float, optional
        Coefficient multiplying the squared displacement in stage 3, in
        kJ/mol/nm^2. Default is 0.0.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    seed : int or None, optional
        Master random seed, fixing the Langevin integrator's random stream.
        Minimisation draws no random numbers, so this changes nothing in the
        structure written here; it is accepted so that one seed can be handed
        to every stage of a workflow alike. Default is None.
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    thermostat_seed, = _derive_seeds(seed, "thermostat")
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    current_positions = modeller.positions
    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 0.0)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    atom_indices = []
    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            pos = current_positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
            atom_indices.append(atom.index)

    system.addForce(restraint)
    print(f"Restraints applied to {len(atom_indices)} backbone atoms.", flush=True)
    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    _seed_random_stream(integrator, thermostat_seed)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(current_positions)

    print(f"\n--- Stage 1: Strong Backbone Restraints ({ks_1} kJ/mol/nm^2) ---", flush=True)
    k_strong = ks_1 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_strong)
    simulation.minimizeEnergy(maxIterations=n_1)

    print(f"\n--- Stage 2: Weak Backbone Restraints ({ks_2} kJ/mol/nm^2) ---", flush=True)
    k_weak = ks_2 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_weak)
    simulation.minimizeEnergy(maxIterations=n_2)

    print(f"\n--- Stage 3: Unrestrained Relaxation ({ks_3} kJ/mol/nm^2) ---", flush=True)
    k_vweak = ks_3 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_vweak)
    simulation.minimizeEnergy(maxIterations=n_3)

    _save_final_state(simulation, output_prefix, save_checkpoint=False)
    print(f"\nProcess complete. Saved to {output_prefix}", flush=True)


def run_openmm_relaxation_simple(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        output_prefix: str = 'minimized',
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 1.0 * unit.femtoseconds,
        platform_name: str | None = None,
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        seed: int | None = None,
) -> None:
    """
    Perform a simple, unrestrained energy minimisation.

    Sets up a system with a Langevin integrator, runs ``minimizeEnergy``,
    and saves the minimised structure and checkpoint. Optionally uses an
    ML/MM mixed potential.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for output files (PDB, checkpoint, log). Default is ``'minimized'``.
    temperature : openmm.unit.Quantity, optional
        Temperature for the Langevin integrator. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    seed : int or None, optional
        Master random seed, fixing the Langevin integrator's random stream.
        Minimisation itself draws no random numbers, but that stream is part
        of the checkpoint this stage writes, so a seed is what makes the
        checkpoint reproduce byte for byte. Default is None.
    """
    thermostat_seed, = _derive_seeds(seed, "thermostat")
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    _seed_random_stream(integrator, thermostat_seed)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print("Minimizing energy", flush=True)
    simulation.minimizeEnergy()

    _save_final_state(simulation, output_prefix)
    print(f"\nProcess complete. Saved to {output_prefix}", flush=True)


def run_openmm_heating(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        output_prefix: str = 'equilibrate',
        k1: float = 100.0,
        backbone_names: list[str] | None = None,
        target_temp: unit.Quantity = 300.0 * unit.kelvin,
        temp_step: unit.Quantity = 50.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 1.0 * unit.femtoseconds,
        n_report: int = 1_000,
        steps_per_stage: int = 5_000,
        steps_final: int = 5_000,
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        seed: int | None = None,
) -> None:
    """
    Heat a system from 0 K to temperature, under backbone restraints.

    The temperature is incremented in steps of ``temp_step`` until
    ``target_temp`` is reached, followed by a final equilibration stage at the
    target temperature. Backbone atoms are harmonically restrained throughout.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for output files. Default is ``'equilibrate'``.
    k1 : float, optional
        Coefficient multiplying the squared backbone displacement, in
        kJ/mol/nm^2. Default is 100.0.
    backbone_names : list of str or None, optional
        Atom names considered backbone for restraints. If None, defaults to
        ``['CA', 'C', 'N', 'P', 'O3']``.
    target_temp : openmm.unit.Quantity, optional
        Target temperature. Default is 300.0 K.
    temp_step : openmm.unit.Quantity, optional
        Temperature increment per heating stage. Default is 50.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps_per_stage : int, optional
        Number of MD steps per heating stage. Default is 5000.
    steps_final : int, optional
        Number of MD steps for the final equilibration at target temperature.
        Default is 5000.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate (e.g. ``'water'``, ``'all'``).
        Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for the starting velocities and the Langevin thermostat, making the
        run reproducible. If None, OpenMM chooses both non-deterministically.
        Default is None.
    """
    thermostat_seed, velocity_seed = _derive_seeds(
        seed, "thermostat", "velocities"
    )
    target_temp_kelvin = require_positive_finite_scalar_in_unit(
        target_temp,
        unit.kelvin,
        name="target_temp",
    )
    temp_step_kelvin = require_positive_finite_scalar_in_unit(
        temp_step,
        unit.kelvin,
        name="temp_step",
    )
    target_temp = target_temp_kelvin * unit.kelvin
    temp_step = temp_step_kelvin * unit.kelvin

    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    print("Applying backbone restraints for heating...", flush=True)
    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", k1 * unit.kilojoules_per_mole / (unit.nanometer ** 2))
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            restraint.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(restraint)

    current_temp = 0 * unit.kelvin
    integrator = openmm.LangevinMiddleIntegrator(current_temp,
                                                 gamma,
                                                 time_step)
    _seed_random_stream(integrator, thermostat_seed)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print(f"\n--- Starting Gentle Heating (0K -> {target_temp}) ---", flush=True)
    _add_standard_reporters(simulation, output_prefix, n_report, pdb_steps=True)

    # The ramp stops one step short of the target and the target gets a stage of
    # its own, so that a target which is not a whole multiple of temp_step is
    # still reached exactly rather than being left at the multiple below it.
    temp = temp_step
    while temp < target_temp:
        print(f"\n-> Heating to {temp}...", flush=True)
        integrator.setTemperature(temp)
        if temp == temp_step:
            _set_velocities_to_temperature(simulation, temp, velocity_seed)
        simulation.step(steps_per_stage)
        temp += temp_step

    print(f"\n-> Heating to {target_temp}...", flush=True)
    integrator.setTemperature(target_temp)
    if target_temp <= temp_step:
        # The ramp never ran, so this stage is also where the velocities start.
        _set_velocities_to_temperature(simulation, target_temp, velocity_seed)
    simulation.step(steps_per_stage)
    print("\n--- Heating Complete ---", flush=True)
    print(f"Running final equilibration at {target_temp} for {steps_final} steps...", flush=True)
    simulation.step(steps_final)

    _save_final_state(simulation, output_prefix)
    print(f"Saved equilibrated structure to {output_prefix}", flush=True)


def run_openmm_npt(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        output_prefix: str = 'npt_equilibrate',
        pressure: unit.Quantity = 1.0 * unit.bar,
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 1.0 * unit.femtoseconds,
        barostat_freq: int | None = 50,
        backbone_names: list[str] | None = None,
        k: float = 10.0,
        n_report: int = 500,
        n_1: int = 5_000,
        n_2: int = 15_000,
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        seed: int | None = None,
) -> None:
    """
    Run a two-phase NPT density equilibration.

    Phase 1 applies backbone restraints while relaxing the box density under a
    Monte Carlo barostat. Phase 2 removes restraints and continues the NPT
    simulation. An optional ML/MM mixed potential is supported.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for output files. Default is ``'npt_equilibrate'``.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the barostat. Default is 1.0 bar.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    barostat_freq : int or None, optional
        Positive barostat attempt frequency in steps. The System must be
        periodic when set. If None, no barostat is added. Default is 50.
    backbone_names : list of str or None, optional
        Atom names considered backbone for restraints. If None, defaults to
        ``['CA', 'C', 'N', 'P', 'O3']``.
    k : float, optional
        Coefficient multiplying the squared backbone displacement, in
        kJ/mol/nm^2. Default is 10.0.
    n_report : int, optional
        Reporter interval in steps. Default is 500.
    n_1 : int, optional
        Number of steps for restrained NPT (Phase 1). Default is 5000.
    n_2 : int, optional
        Number of steps for unrestrained NPT (Phase 2). Default is 15000.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for the starting velocities, the Langevin thermostat, and the
        barostat's volume moves, making the run reproducible. If None, OpenMM
        chooses each non-deterministically. Default is None.
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    thermostat_seed, velocity_seed, barostat_seed = _derive_seeds(
        seed, "thermostat", "velocities", "barostat"
    )
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    barostat_freq = _validate_barostat_frequency(system, barostat_freq)
    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_freq)
        _seed_random_stream(barostat, barostat_seed)
        system.addForce(barostat)

    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", k * unit.kilojoules_per_mole / (unit.nanometer ** 2))
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            restraint.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(restraint)

    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    _seed_random_stream(integrator, thermostat_seed)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    _set_velocities_to_temperature(simulation, temperature, velocity_seed)

    _add_standard_reporters(simulation, output_prefix, n_report, pdb_steps=True,
                            stdout_volume=True)

    print("\n--- Phase 1: Restrained NPT (Relaxing Density) ---", flush=True)
    simulation.step(n_1)

    print("\n--- Phase 2: Removing Restraints (Unrestrained NPT) ---", flush=True)
    simulation.context.setParameter("k", 0.0)
    simulation.step(n_2)

    _save_final_state(simulation, output_prefix)

    print(f"\nDensity equilibration complete. Saved to {output_prefix}", flush=True)


def run_openmm_prod(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        plumed_script_path: str | None = None,
        pressure: unit.Quantity = 1.0 * unit.bar,
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 1.0 * unit.femtoseconds,
        barostat_freq: int | None = 50,
        n_report: int = 1_000,
        steps: int = 500_000,
        output_prefix: str = 'prod',
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        seed: int | None = None,
) -> None:
    """
    Run an NPT production MD simulation, optionally with PLUMED enhanced sampling.

    Sets up the system with a Langevin integrator and optional Monte Carlo
    barostat, loads an optional PLUMED bias script, and runs the production
    trajectory.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_script_path : str or None, optional
        Path to a PLUMED input script. If None, no PLUMED bias is applied.
        Default is None.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the barostat. Default is 1.0 bar.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    barostat_freq : int or None, optional
        Positive barostat attempt frequency in steps. The System must be
        periodic when set. If None, no barostat is added. Default is 50.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps : int, optional
        Total number of production MD steps. Default is 500000.
    output_prefix : str, optional
        Prefix for output files (PDB, checkpoint, log). Default is ``'prod'``.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for the starting velocities, the Langevin thermostat, and the
        barostat's volume moves, making the run reproducible. If None, OpenMM
        chooses each non-deterministically. Default is None.
    """
    thermostat_seed, velocity_seed, barostat_seed = _derive_seeds(
        seed, "thermostat", "velocities", "barostat"
    )
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    barostat_freq = _validate_barostat_frequency(system, barostat_freq)
    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_freq)
        _seed_random_stream(barostat, barostat_seed)
        system.addForce(barostat)

    _load_plumed(system, plumed_script_path)
    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    _seed_random_stream(integrator, thermostat_seed)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    _set_velocities_to_temperature(simulation, temperature, velocity_seed)

    _add_standard_reporters(simulation, output_prefix, n_report, pdb_steps=True,
                            checkpoint_interval=n_report * 10)
    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    _save_final_state(simulation, output_prefix)


def run_openmm_steered(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        plumed_input: str | os.PathLike[str],
        steps: int,
        output_prefix: str = 'smd',
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 0.5 * unit.femtoseconds,
        n_report: int = 100,
        pressure: unit.Quantity = 1.0 * unit.bar,
        barostat_freq: int | None = None,
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        seed: int | None = None,
) -> str:
    """
    Run a steered MD simulation, dragging a collective variable with PLUMED.

    This is :func:`run_openmm_prod` with the settings a pulling run wants: no
    barostat, a short time step, and frequent reporting so the trajectory has
    enough frames to pick a path out of. The PLUMED script comes from
    :func:`reactiontools.plumed_input_steered` or one of its wrappers, and
    the trajectory feeds :func:`reactiontools.path_from_steered_md`.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions, equilibrated at
        the reactant.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_input : str or os.PathLike
        The PLUMED script itself, or the path to a file holding one. Anything
        containing a newline is taken to be a script and written to
        ``'{output_prefix}_plumed.dat'``. Existing files are always treated as
        paths; a one-line PLUMED command is accepted as inline input too. A
        path-like object is always treated as a path and can disambiguate an
        unusual filename from script text.
    steps : int
        Number of MD steps to run. Use the step count returned alongside the
        script by the ``plumed_input_steered*`` builders, so the run covers the
        whole pulling schedule.
    output_prefix : str, optional
        Prefix for output files. Default is ``'smd'``.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 0.5 fs, since pulling a proton
        across a hydrogen bond is not gentle.
    n_report : int, optional
        Reporter interval in steps. Default is 100. Match this to the PLUMED
        ``PRINT`` stride so that every frame has a CV value.
    pressure : openmm.unit.Quantity, optional
        Target pressure, only used if a barostat is asked for. Default is
        1.0 bar.
    barostat_freq : int or None, optional
        Barostat attempt frequency in steps. Default is None, i.e. pull at
        constant volume.
    platform_name : str, optional
        OpenMM platform name. Default is None, which auto-detects.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    seed : int or None, optional
        Master random seed, handed to :func:`run_openmm_prod`. A value makes
        the pulling run reproducible, which is what lets a set of paths differ
        only in the schedule that pulled them. Default is None.

    Returns
    -------
    str
        Path to the trajectory written by the run.
    """
    if _is_inline_plumed_input(plumed_input):
        assert isinstance(plumed_input, str)
        plumed_script_path = f'{output_prefix}_plumed.dat'
        with open(plumed_script_path, 'w') as f:
            f.write(plumed_input)
    else:
        plumed_script_path = os.fspath(plumed_input)

    print(f"Starting steered MD for {steps} steps...", flush=True)
    run_openmm_prod(modeller,
                    forcefield,
                    plumed_script_path=plumed_script_path,
                    pressure=pressure,
                    temperature=temperature,
                    gamma=gamma,
                    time_step=time_step,
                    barostat_freq=barostat_freq,
                    n_report=n_report,
                    steps=steps,
                    output_prefix=output_prefix,
                    platform_name=platform_name,
                    deuterate=deuterate,
                    deuterate_option=deuterate_option,
                    potential=potential,
                    ml_idx=ml_idx,
                    calculator=calculator,
                    seed=seed)

    traj_file = f'{output_prefix}_steps.pdb'
    print(f"Steered trajectory written to {traj_file}", flush=True)
    return traj_file


def run_openmm_rpmd_equilibration(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        output_prefix: str = 'rpmd_ready',
        n_beads: int = 32,
        temperature: unit.Quantity = 300 * unit.kelvin,
        friction: unit.Quantity = 1.0 / unit.picosecond,
        timestep: unit.Quantity = 0.5 * unit.femtoseconds,
        n_report: int = 1_000,
        n_1: int = 1_000,
        n_2: int = 5_000,
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        atoms_to_watch: list[int] | None = None,
        scale_factor: float = 1.0,
        seed: int | None = None,
        expansion_metric: Literal["rms", "mean"] = "rms",
        distance_pairs_to_watch: Iterable[tuple[int, int]] | None = None,
        kinetic_decomposition: bool = False,
) -> None:
    """
    Equilibrate a ring-polymer molecular dynamics (RPMD) simulation.

    Performs a two-stage equilibration: stage 1 uses a reduced time step for
    gentle bead expansion, and stage 2 runs at the full time step. A
    checkpoint is saved at the end for use by subsequent production runs.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    output_prefix : str, optional
        Prefix for output files. Default is ``'rpmd_ready'``.
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300 K.
    friction : openmm.unit.Quantity, optional
        RPMD friction coefficient. Default is 1.0 / ps.
    timestep : openmm.unit.Quantity, optional
        Integration time step. Default is 0.5 fs.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    n_1 : int, optional
        Number of steps for stage 1 (bead expansion). Default is 1000.
    n_2 : int, optional
        Number of steps for stage 2 (relaxation). Default is 5000.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    atoms_to_watch : list of int or None, optional
        Atom indices for quantum spread monitoring. Default is None.
    scale_factor : float, optional
        Multiplier applied to the exact free-ring-polymer normal-mode position
        amplitudes. Default is 1.0.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for initial positions/velocities and the PILE thermostat. If None,
        NumPy and OpenMM select independent seeds. Default is None.
    expansion_metric : {"rms", "mean"}, optional
        Spread metric written for *atoms_to_watch*. ``"mean"`` is the mean
        bead-centroid degree of expansion; ``"rms"`` preserves the existing
        radius-of-gyration output. Default is ``"rms"``.
    distance_pairs_to_watch : iterable of pair of int or None, optional
        Atom pairs whose centroid distances are written alongside the spread
        values. Requires *atoms_to_watch*. Default is None.
    kinetic_decomposition : bool, optional
        If True, also log the per-atom centroid-virial kinetic energy of
        *atoms_to_watch* to ``<output_prefix>_kinetic.log``, which
        :func:`~openmmnqe.isotopes.rpmd_isotope_free_energy` integrates over
        mass into an equilibrium isotope effect. Reads the beads a second
        time per report, so it is opt-in. Requires *atoms_to_watch*. Default
        is False.
    """
    initialization_seed, thermostat_seed = _derive_seeds(
        seed, "initialization", "thermostat"
    )
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep)
    _seed_random_stream(integrator, thermostat_seed)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    with _finalize_reporters(simulation):
        _add_rpmd_reporters(
            simulation,
            modeller.topology,
            output_prefix,
            n_report,
            n_beads,
            atoms_to_watch,
            expansion_metric=expansion_metric,
            distance_pairs=distance_pairs_to_watch,
            kinetic_decomposition=kinetic_decomposition,
        )
        _add_rpmd_progress_reporters(simulation, output_prefix, n_report)

        init_beads(
            modeller,
            simulation,
            n_beads,
            scale_factor=scale_factor,
            seed=initialization_seed,
        )

        print("\n--- Stage 1: Bead Expansion  ---", flush=True)
        integrator.setStepSize(timestep * 0.5)
        step_rpmd(simulation, n_1)

        print(
            f"\n--- Stage 2: Relaxation at full timestep ({timestep}) ---",
            flush=True,
        )
        integrator.setStepSize(timestep)
        step_rpmd(simulation, n_2)

        print("\n--- Saving State ---", flush=True)
        # Not '_centroid.pdb': RPMDCentroidReporter owns that name and is still
        # holding it open, so writing here would truncate the trajectory it spent
        # the whole run building.
        _save_final_state(
            simulation,
            output_prefix,
            pdb_suffix='_final.pdb',
            n_beads=n_beads,
        )
        print(
            f"Saved final centroid structure to {output_prefix}_final.pdb",
            flush=True,
        )


def run_openmm_rpmd_contracted(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        plumed_script_path: str | None = None,
        checkpoint_file: str = 'rpmd_ready.chk',
        output_prefix: str = 'rpmd_prod_contracted',
        n_beads: int = 32,
        temperature: unit.Quantity = 300 * unit.kelvin,
        pressure: unit.Quantity = 1.0 * unit.bar,
        barostat_freq: int | None = 50,
        friction: unit.Quantity = 1.0 / unit.picosecond,
        timestep: unit.Quantity = 0.5 * unit.femtoseconds,
        steps: int = 100_000,
        n_report: int = 1_000,
        contractions: Mapping[int, int] | None = None,
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        atoms_to_watch: list[int] | None = None,
        calculator: Any = None,
        expansion_metric: Literal["rms", "mean"] = "rms",
        distance_pairs_to_watch: Iterable[tuple[int, int]] | None = None,
        kinetic_decomposition: bool = False,
        seed: int | None = None,
        apply_thermostat: bool = True,
) -> None:
    """
    Run a contracted ring-polymer MD (RPMD) production simulation.

    Uses the ring-polymer contraction scheme to evaluate expensive force
    components (e.g. PME reciprocal space) on fewer bead copies, reducing
    computational cost. A checkpoint from a prior RPMD equilibration is
    required. An optional PLUMED bias and ML/MM mixed potential are supported.
    With *apply_thermostat* set to False the run is microcanonical
    ring-polymer dynamics on the contracted forces.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_script_path : str or None, optional
        Path to a PLUMED input script. If None, no bias is applied.
        Default is None.
    checkpoint_file : str, optional
        Path to the equilibration checkpoint. Default is ``'rpmd_ready.chk'``.
    output_prefix : str, optional
        Prefix for output files. Default is ``'rpmd_prod_contracted'``.
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300 K.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the RPMD barostat. Default is 1.0 bar.
    barostat_freq : int or None, optional
        Positive RPMD barostat attempt frequency. The System must be periodic
        when set. If None, no barostat is added. Default is 50.
    friction : openmm.unit.Quantity, optional
        RPMD friction coefficient. Default is 1.0 / ps.
    timestep : openmm.unit.Quantity, optional
        Integration time step. Default is 0.5 fs.
    steps : int, optional
        Total number of production steps. Default is 100000.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    contractions : dict or None, optional
        Mapping of force group to the number of contracted copies. If None,
        direct-space forces use up to 8 copies and reciprocal-space forces use
        the centroid alone. Default is None.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    atoms_to_watch : list of int or None, optional
        Atom indices for quantum spread monitoring. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    expansion_metric : {"rms", "mean"}, optional
        Spread metric written for *atoms_to_watch*. Use ``"mean"`` for the
        Figure-7 degree of expansion. Default is ``"rms"``.
    distance_pairs_to_watch : iterable of pair of int or None, optional
        Atom pairs whose centroid distances are written alongside the spread
        values. Requires *atoms_to_watch*. Default is None.
    kinetic_decomposition : bool, optional
        If True, also log the per-atom centroid-virial kinetic energy of
        *atoms_to_watch* to ``<output_prefix>_kinetic.log``, which
        :func:`~openmmnqe.isotopes.rpmd_isotope_free_energy` integrates over
        mass into an equilibrium isotope effect. Reads the beads a second
        time per report, so it is opt-in. Requires *atoms_to_watch*. Default
        is False.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for the PILE thermostat and the barostat's volume moves, making the
        run reproducible. The ring polymer itself comes from
        *checkpoint_file*, so it is unaffected. If None, OpenMM chooses both
        non-deterministically. Default is None.
    apply_thermostat : bool, optional
        If False, disable the PILE thermostat and run microcanonical
        (constant-energy) ring-polymer dynamics. *temperature* is still
        required -- it sets the ring-polymer spring constants and so defines
        the Hamiltonian, not a thermostat target -- and *barostat_freq* must
        be None. *friction* is ignored by the dynamics. Default is True.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist.
    ValueError
        If an ML potential or calculator is given without *ml_idx*, or if
        a contraction is invalid, or *barostat_freq* is set on a nonperiodic
        System, or *seed* is negative or not an integer, or
        *apply_thermostat* is False while *barostat_freq* is set.

    Warns
    -----
    UserWarning
        If *barostat_freq* is set on a System carrying a ``PythonForce``.
    """
    if not apply_thermostat and barostat_freq is not None:
        raise ValueError(
            "apply_thermostat=False is microcanonical; a Monte Carlo "
            "barostat samples no ensemble without a thermostat. Pass "
            "barostat_freq=None."
        )
    n_beads = _validate_rpmd_n_beads(n_beads)
    contractions = _validate_rpmd_contractions(contractions, n_beads)
    thermostat_seed, barostat_seed = _derive_seeds(
        seed, "thermostat", "barostat"
    )
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    barostat_freq = _validate_barostat_frequency(system, barostat_freq)
    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        _warn_barostat_on_python_force(system)
        barostat = openmm.RPMDMonteCarloBarostat(pressure, barostat_freq)
        _seed_random_stream(barostat, barostat_seed)
        system.addForce(barostat)

    _load_plumed(system, plumed_script_path)

    # Contraction is keyed on force group, so the forces have to be sorted into
    # the groups `contractions` names: the costlier the force, the fewer beads
    # it should be evaluated on.
    print("Assigning force groups for contraction...", flush=True)

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            # One force object covering two costs, so its direct and reciprocal
            # halves are split across groups and contracted separately.
            force.setForceGroup(1)
            force.setReciprocalSpaceForceGroup(2)
            print(f"  - {force.__class__.__name__}: Direct->Group 1, Reciprocal->Group 2")

        elif isinstance(force, (openmm.HarmonicBondForce,
                                openmm.HarmonicAngleForce,
                                openmm.PeriodicTorsionForce,
                                openmm.RBTorsionForce,
                                openmm.CMAPTorsionForce)):
            force.setForceGroup(0)
            print(f"  - {force.__class__.__name__}: Group 0")

        else:
            # An unrecognised force (an external PythonForce potential, a
            # PLUMED bias) keeps its group: groups absent from the
            # contractions dict run on every bead.
            print(f"  - {force.__class__.__name__}: keeping group "
                  f"{force.getForceGroup()}")

    print(f"\nInitializing RPMDIntegrator with contractions: {contractions}", flush=True)
    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep, contractions)
    _seed_random_stream(integrator, thermostat_seed)
    if not apply_thermostat:
        integrator.setApplyThermostat(False)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    _load_checkpoint(simulation, checkpoint_file, n_beads=n_beads)

    with _finalize_reporters(simulation):
        _add_rpmd_reporters(
            simulation,
            modeller.topology,
            output_prefix,
            n_report,
            n_beads,
            atoms_to_watch,
            expansion_metric=expansion_metric,
            distance_pairs=distance_pairs_to_watch,
            kinetic_decomposition=kinetic_decomposition,
        )

        _add_rpmd_progress_reporters(simulation, output_prefix, n_report)

        print(f"\nStarting Production Run ({steps} steps)...")
        step_rpmd(simulation, steps)
        print("Done.", flush=True)

        print("\n--- Saving State ---", flush=True)
        # The centroid reporter owns '_centroid.pdb'; save separately.
        _save_final_state(
            simulation,
            output_prefix,
            pdb_suffix='_final.pdb',
            n_beads=n_beads,
        )
        print(
            f"Saved final centroid structure to {output_prefix}_final.pdb",
            flush=True,
        )


def run_openmm_rpmd_prod(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        plumed_script_path: str | None = None,
        checkpoint_file: str = 'rpmd_ready.chk',
        output_prefix: str = 'rpmd_prod',
        n_beads: int = 32,
        pressure: unit.Quantity = 1.0 * unit.bar,
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 1.0 * unit.femtoseconds,
        barostat_freq: int | None = 50,
        n_report: int = 1_000,
        steps: int = 500_000,
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        atoms_to_watch: list[int] | None = None,
        calculator: Any = None,
        expansion_metric: Literal["rms", "mean"] = "rms",
        distance_pairs_to_watch: Iterable[tuple[int, int]] | None = None,
        kinetic_decomposition: bool = False,
        seed: int | None = None,
        apply_thermostat: bool = True,
        snapshot_interval: int | None = None,
        velocity_record_interval: int | None = None,
        velocity_atom_indices: Sequence[int] | None = None,
) -> None:
    """
    Run a full ring-polymer MD (RPMD) production simulation.

    Loads a checkpoint from a prior RPMD equilibration and continues with a
    production run using the ``RPMDIntegrator``. An optional PLUMED bias,
    RPMD barostat, and ML/MM mixed potential are supported. With
    *apply_thermostat* set to False the run is microcanonical ring-polymer
    dynamics, the ensemble every RPMD time-correlation observable is defined
    in; *snapshot_interval* harvests full-bead restart archives along the way,
    which is how :func:`openmmnqe.rates.run_openmm_rpmd_recrossing` gets its
    dividing-surface configurations.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_script_path : str or None, optional
        Path to a PLUMED input script. If None, no bias is applied.
        Default is None.
    checkpoint_file : str, optional
        Path to the equilibration checkpoint. Default is ``'rpmd_ready.chk'``.
    output_prefix : str, optional
        Prefix for output files. Default is ``'rpmd_prod'``.
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the RPMD barostat. Default is 1.0 bar.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    barostat_freq : int or None, optional
        Positive RPMD barostat attempt frequency. The System must be periodic
        when set. If None, no barostat is added. Default is 50.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps : int, optional
        Total number of production steps. Default is 500000.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    atoms_to_watch : list of int or None, optional
        Atom indices for quantum spread monitoring. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    expansion_metric : {"rms", "mean"}, optional
        Spread metric written for *atoms_to_watch*. Use ``"mean"`` for the
        Figure-7 degree of expansion. Default is ``"rms"``.
    distance_pairs_to_watch : iterable of pair of int or None, optional
        Atom pairs whose centroid distances are written alongside the spread
        values. Requires *atoms_to_watch*. Default is None.
    kinetic_decomposition : bool, optional
        If True, also log the per-atom centroid-virial kinetic energy of
        *atoms_to_watch* to ``<output_prefix>_kinetic.log``, which
        :func:`~openmmnqe.isotopes.rpmd_isotope_free_energy` integrates over
        mass into an equilibrium isotope effect. Reads the beads a second
        time per report, so it is opt-in. Requires *atoms_to_watch*. Default
        is False.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for the PILE thermostat and the barostat's volume moves, making the
        run reproducible. The ring polymer itself comes from
        *checkpoint_file*, so it is unaffected. If None, OpenMM chooses both
        non-deterministically. Default is None.
    apply_thermostat : bool, optional
        If False, disable the PILE thermostat and run microcanonical
        (constant-energy) ring-polymer dynamics. *temperature* is still
        required -- it sets the ring-polymer spring constants and so defines
        the Hamiltonian, not a thermostat target -- and *barostat_freq* must
        be None. *gamma* is ignored by the dynamics. The conserved quantity
        is the ``E_ring(kJ/mol)`` column of the thermodynamic log; check it
        with :func:`openmmnqe.reporters.rpmd_energy_conservation`.
        Default is True.
    snapshot_interval : int or None, optional
        If set, write a full-bead RPMD restart archive
        ``<output_prefix>_snapshot_<i>.npz`` after every *snapshot_interval*
        steps, numbered from zero. The archives are ordinary RPMD restarts:
        each holds every bead's positions and velocities and can seed
        :func:`openmmnqe.rates.run_openmm_rpmd_recrossing`. Steps left over
        after the last whole interval still run. Default is None, which
        writes no snapshots.
    velocity_record_interval : int or None, optional
        If set, record the bead-averaged (centroid) velocities every this
        many steps and write them to ``<output_prefix>_velocities.npz`` via
        :class:`openmmnqe.reporters.RPMDVelocityReporter`, for the
        correlation-function readers. The frames accumulate in memory until
        the run ends. Default is None, which records nothing.
    velocity_atom_indices : sequence of int or None, optional
        Atoms whose centroid velocities are recorded. Requires
        *velocity_record_interval*. Default is None, which records every
        atom.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist.
    ValueError
        If an ML potential or calculator is given without *ml_idx*, or if
        *barostat_freq* is set on a nonperiodic System, or *seed* is negative
        or not an integer, or *apply_thermostat* is False while
        *barostat_freq* is set, or *velocity_atom_indices* is given without
        *velocity_record_interval*, or *snapshot_interval* or
        *velocity_record_interval* is not a positive integer.

    Warns
    -----
    UserWarning
        If *barostat_freq* is set on a System carrying a ``PythonForce``, or
        if *snapshot_interval* exceeds *steps* so no snapshot is written.
    """
    if not apply_thermostat and barostat_freq is not None:
        raise ValueError(
            "apply_thermostat=False is microcanonical; a Monte Carlo "
            "barostat samples no ensemble without a thermostat. Pass "
            "barostat_freq=None."
        )
    if snapshot_interval is not None:
        snapshot_interval = require_integer(
            snapshot_interval,
            name="snapshot_interval",
            minimum=1,
        )
        if snapshot_interval > steps:
            warnings.warn(
                f"snapshot_interval={snapshot_interval} exceeds "
                f"steps={steps}; no snapshot will be written",
                UserWarning,
                stacklevel=2,
            )
    thermostat_seed, barostat_seed = _derive_seeds(
        seed, "thermostat", "barostat"
    )
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    barostat_freq = _validate_barostat_frequency(system, barostat_freq)
    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        _warn_barostat_on_python_force(system)
        barostat = openmm.RPMDMonteCarloBarostat(pressure, barostat_freq)
        _seed_random_stream(barostat, barostat_seed)
        system.addForce(barostat)

    _load_plumed(system, plumed_script_path)
    integrator = openmm.RPMDIntegrator(n_beads,
                                       temperature,
                                       gamma,
                                       time_step)
    _seed_random_stream(integrator, thermostat_seed)
    if not apply_thermostat:
        integrator.setApplyThermostat(False)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    _load_checkpoint(simulation, checkpoint_file, n_beads=n_beads)

    with _finalize_reporters(simulation):
        _add_rpmd_reporters(
            simulation,
            modeller.topology,
            output_prefix,
            n_report,
            n_beads,
            atoms_to_watch,
            expansion_metric=expansion_metric,
            distance_pairs=distance_pairs_to_watch,
            velocity_record_interval=velocity_record_interval,
            velocity_atom_indices=velocity_atom_indices,
            kinetic_decomposition=kinetic_decomposition,
        )

        _add_rpmd_progress_reporters(simulation, output_prefix, n_report)

        print(f"Starting production run for {steps} steps...", flush=True)
        if snapshot_interval is None:
            step_rpmd(simulation, steps)
        else:
            completed = 0
            snapshot_index = 0
            while completed + snapshot_interval <= steps:
                step_rpmd(simulation, snapshot_interval)
                completed += snapshot_interval
                _save_rpmd_restart(
                    simulation,
                    f'{output_prefix}_snapshot_{snapshot_index:05d}.npz',
                    n_beads,
                )
                snapshot_index += 1
            if completed < steps:
                step_rpmd(simulation, steps - completed)
        print("Production run complete.", flush=True)

        _save_final_state(
            simulation,
            output_prefix,
            pdb_suffix='_final.pdb',
            n_beads=n_beads,
        )


def run_openmm_adqtb_eq(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 1.0 * unit.femtoseconds,
        segment_length: unit.Quantity = 0.5 * unit.picosecond,
        adaptation_rate: float = 0.5,
        n_report: int = 1_000,
        steps: int = 500_000,
        output_prefix: str = 'adqtb_ready',
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        seed: int | None = None,
        particle_types: Literal["element", "none"] | Mapping[int, int] | None = "element",
        friction_log: bool = True,
) -> None:
    """
    Run an adaptive quantum thermal bath (adQTB) equilibration simulation.

    Uses the ``QTBIntegrator`` to thermalise the system with quantum thermal
    noise. A checkpoint is saved at the end for subsequent production runs.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    segment_length : openmm.unit.Quantity, optional
        Segment length for the QTB integrator. Default is 0.5 ps.
    adaptation_rate : float, optional
        Adaptation rate for the QTB integrator. Default is 0.5.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps : int, optional
        Total number of equilibration steps. Default is 500000.
    output_prefix : str, optional
        Prefix for output files. Default is ``'adqtb_ready'``.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for the starting velocities and the quantum thermal bath, making the
        run reproducible. If None, OpenMM chooses both non-deterministically.
        Default is None.
    particle_types : {"element", "none"} or dict of int to int or None, optional
        How the bath groups particles. ``"element"`` gives every element its
        own adapted noise spectrum, splitting a symbol when masses differ so
        that deuterium does not share hydrogen's bath. ``"none"`` and None
        leave the types unset, which makes every particle adapt a spectrum of
        its own. A mapping assigns particle index to type index directly.
        Default is ``"element"``.
    friction_log : bool, optional
        Write ``<prefix>_friction.log``, one row of adapted friction spectra
        per adaptation segment, for :mod:`openmmnqe.adqtb` to read back.
        Skipped with a warning when no particle types are assigned. Default
        is True.
    """
    segment_steps = _validate_adqtb_segment(segment_length, time_step)
    thermostat_seed, velocity_seed = _derive_seeds(
        seed, "thermostat", "velocities"
    )
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    integrator = openmm.QTBIntegrator(temperature, gamma, time_step)
    integrator.setSegmentLength(segment_length)
    integrator.setDefaultAdaptationRate(adaptation_rate)
    _seed_random_stream(integrator, thermostat_seed)
    type_names = _assign_adqtb_particle_types(
        integrator, modeller.topology, system, particle_types,
    )

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    _set_velocities_to_temperature(simulation, temperature, velocity_seed)

    _add_adqtb_reporters(simulation, output_prefix, n_report,
                         segment_steps=segment_steps, type_names=type_names,
                         friction_log=friction_log,
                         checkpoint_interval=n_report * 10)
    with _finalize_reporters(simulation):
        print(f"Starting production run for {steps} steps...", flush=True)
        simulation.step(steps)
        print("Production run complete.", flush=True)

    _save_final_state(simulation, output_prefix)


def run_openmm_adqtb_prod(
        modeller: app.Modeller,
        forcefield: app.ForceField | MLPotential | PreparedSystem,
        plumed_script_path: str | None = None,
        pressure: unit.Quantity = 1.0 * unit.bar,
        barostat_freq: int | None = 50,
        temperature: unit.Quantity = 300.0 * unit.kelvin,
        gamma: unit.Quantity = 1.0 / unit.picosecond,
        time_step: unit.Quantity = 1.0 * unit.femtoseconds,
        segment_length: unit.Quantity = 0.5 * unit.picosecond,
        adaptation_rate: float = 0.5,
        n_report: int = 1_000,
        steps: int = 500_000,
        output_prefix: str = 'adqtb_prod',
        platform_name: str | None = None,
        deuterate: bool = False,
        deuterate_option: WorkflowDeuterationOption = 'water',
        potential: Any = None,
        ml_idx: list[int] | None = None,
        calculator: Any = None,
        checkpoint_file: str = 'adqtb_ready.chk',
        seed: int | None = None,
        particle_types: Literal["element", "none"] | Mapping[int, int] | None = "element",
        friction_log: bool = True,
) -> None:
    """
    Run an adaptive quantum thermal bath (adQTB) production simulation.

    Uses the ``QTBIntegrator`` with optional PLUMED enhanced-sampling bias
    and a Monte Carlo barostat for NPT conditions.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM Modeller containing topology and positions.
    forcefield : openmm.app.ForceField
        The force field used to parameterise the system.
    plumed_script_path : str or None, optional
        Path to a PLUMED input script. If None, no bias is applied.
        Default is None.
    pressure : openmm.unit.Quantity, optional
        Target pressure for the barostat. Default is 1.0 bar.
    barostat_freq : int or None, optional
        Positive barostat attempt frequency in steps. The System must be
        periodic when set. If None, no barostat is added. Default is 50.
    temperature : openmm.unit.Quantity, optional
        Simulation temperature. Default is 300.0 K.
    gamma : openmm.unit.Quantity, optional
        Friction coefficient. Default is 1.0 / ps.
    time_step : openmm.unit.Quantity, optional
        Integration time step. Default is 1.0 fs.
    segment_length : openmm.unit.Quantity, optional
        Segment length for the QTB integrator. Default is 0.5 ps.
    adaptation_rate : float, optional
        Adaptation rate for the QTB integrator. Default is 0.5.
    n_report : int, optional
        Reporter interval in steps. Default is 1000.
    steps : int, optional
        Total number of production steps. Default is 500000.
    output_prefix : str, optional
        Prefix for output files. Default is ``'adqtb_prod'``.
    platform_name : str or None, optional
        OpenMM platform name. Default is None, which auto-detects via
        ``check_platform``. Forced to ``'CUDA'`` when a mixed/ML potential
        is active.
    deuterate : bool, optional
        If True, deuterate the system before simulation. Default is False.
    deuterate_option : str, optional
        Subset of the system to deuterate. Default is ``'water'``.
    potential : object or None, optional
        ML potential object with a ``createMixedSystem`` method. Default is None.
    ml_idx : list of int or None, optional
        Atom indices for the ML region. Default is None.
    calculator : object or None, optional
        Optional calculator object to pass to the ML potential. Default is None.
    checkpoint_file : str, optional
        Path to the adQTB equilibration checkpoint. The checkpoint contains
        the adapted friction spectrum as well as coordinates and velocities.
        Default is ``'adqtb_ready.chk'``.
    seed : int or None, optional
        Master random seed. A value derives independent deterministic streams
        for the quantum thermal bath and the barostat's volume moves, making
        the run reproducible. Velocities and the adapted friction spectrum
        come from *checkpoint_file*, so they are unaffected. If None, OpenMM
        chooses both non-deterministically. Default is None.
    particle_types : {"element", "none"} or dict of int to int or None, optional
        How the bath groups particles. ``"element"`` gives every element its
        own adapted noise spectrum, splitting a symbol when masses differ so
        that deuterium does not share hydrogen's bath. ``"none"`` and None
        leave the types unset, which makes every particle adapt a spectrum of
        its own. A mapping assigns particle index to type index directly.
        Default is ``"element"``.
    friction_log : bool, optional
        Write ``<prefix>_friction.log``, one row of adapted friction spectra
        per adaptation segment, for :mod:`openmmnqe.adqtb` to read back.
        Skipped with a warning when no particle types are assigned. Default
        is True.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_file* does not exist.
    ValueError
        If *barostat_freq* is set on a nonperiodic System, or *seed* is
        negative or not an integer.

    Warns
    -----
    UserWarning
        If *barostat_freq* is set on a System carrying a ``PythonForce``.
    """
    segment_steps = _validate_adqtb_segment(segment_length, time_step)
    thermostat_seed, barostat_seed = _derive_seeds(
        seed, "thermostat", "barostat"
    )
    system, platform = _build_system(modeller, forcefield, platform_name,
                                     potential, ml_idx, calculator)

    barostat_freq = _validate_barostat_frequency(system, barostat_freq)
    _maybe_deuterate(modeller, system, deuterate, deuterate_option)

    if barostat_freq is not None:
        _warn_barostat_on_python_force(system)
        barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_freq)
        _seed_random_stream(barostat, barostat_seed)
        system.addForce(barostat)

    _load_plumed(system, plumed_script_path)

    integrator = openmm.QTBIntegrator(temperature, gamma, time_step)
    integrator.setSegmentLength(segment_length)
    integrator.setDefaultAdaptationRate(adaptation_rate)
    _seed_random_stream(integrator, thermostat_seed)
    type_names = _assign_adqtb_particle_types(
        integrator, modeller.topology, system, particle_types,
    )

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    _load_checkpoint(simulation, checkpoint_file)

    _add_adqtb_reporters(simulation, output_prefix, n_report,
                         segment_steps=segment_steps, type_names=type_names,
                         friction_log=friction_log,
                         checkpoint_interval=n_report * 10)
    with _finalize_reporters(simulation):
        print(f"Starting production run for {steps} steps...", flush=True)
        simulation.step(steps)
        print("Production run complete.", flush=True)

    _save_final_state(simulation, output_prefix)
