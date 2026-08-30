"""Equilibrium isotope effects from ring-polymer kinetic energies.

The centroid-virial kinetic energy of an atom is not only a diagnostic.  It
is the exact derivative of the free energy with respect to that atom's log
mass,

.. math::

    \\frac{\\partial F}{\\partial \\ln m_i} = -\\left<K_i\\right>,

so integrating :func:`~openmmnqe.reporters.rpmd_kinetic_decomposition` over
mass gives the free-energy change of an isotope substitution -- an
equilibrium isotope effect -- without ever running a separate deuterated
trajectory and differencing two large numbers.

The integration is over ``ln m``, where the integrand is smooth enough that
two Gauss-Legendre nodes are converged well below any achievable sampling
error.  :func:`rpmd_mass_integration_nodes` says which masses to run at and
:func:`rpmd_isotope_free_energy` combines their averaged kinetic energies;
:func:`rpmd_fractionation_factor` turns two such results into the
fractionation ratio between two sites.

What this costs is one short RPMD run per node, at a *fictitious* mass
between the two isotopes.  That is not the same as one trajectory: for a
3600 cm^-1 O-H stretch at 300 K the exact H-to-D free energy is
-18.898 kJ/mol, and evaluating the integrand only at the physical hydrogen
mass gives -22.356, wrong by 18%.  Two Gauss-Legendre nodes, at 1.167 and
1.740 daltons, give -18.898.

Mass *perturbation* -- reweighting a single trajectory from one mass to
another -- is deliberately absent.  Mass enters the ring-polymer weight
through the free-particle normalisation and the spring term rather than as a
potential-energy difference, and the overlap between the light and heavy
ring-polymer distributions degrades as the bead count grows, so the
reweighted estimator is both harder to write and badly behaved where it
matters.  A wrong number that looks plausible is worse than an extra run.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import openmm.unit as unit

from ._validation import (
    require_integer,
    require_positive_finite_scalar_in_unit,
    require_scalar_in_unit,
)
from .reporters import _BOLTZMANN_KJ_PER_MOL_K


class MassIntegrationNodes(NamedTuple):
    """
    The masses to run at for one mass thermodynamic integration.

    Produced by :func:`rpmd_mass_integration_nodes` and consumed by
    :func:`rpmd_isotope_free_energy`.  The nodes are Gauss-Legendre points
    in ``ln m``, so they lie strictly between the two isotope masses and
    none of them is a physical isotope.

    Attributes
    ----------
    masses : numpy.ndarray
        Node masses in daltons, shaped ``(nodes,)``. Run one RPMD trajectory
        at each, with the substituted atoms set to that mass.
    ln_masses : numpy.ndarray
        ``log(masses)``, the variable actually integrated over.
    weights : numpy.ndarray
        Quadrature weights in ``d(ln m)``, shaped ``(nodes,)``. They sum to
        ``log(mass_heavy / mass_light)``.
    mass_light : float
        The starting mass in daltons.
    mass_heavy : float
        The finishing mass in daltons.
    """

    masses: npt.NDArray[np.float64]
    ln_masses: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    mass_light: float
    mass_heavy: float


class IsotopeFreeEnergy(NamedTuple):
    """
    The free-energy change of an isotope substitution, and its parts.

    Produced by :func:`rpmd_isotope_free_energy`.  Energies are in kJ/mol
    and are for the substitution as a whole, light to heavy.

    Attributes
    ----------
    free_energy : float
        ``-sum(weights * kinetic_energies)``, in kJ/mol. Negative for a
        heavier isotope, because a heavier nucleus is more tightly bound.
    free_energy_stderr : float
        ``sqrt(sum(weights**2 * kinetic_stderr**2))``, in kJ/mol. The nodes
        are independent trajectories, so this propagation is exact rather
        than approximate. It carries no quadrature-truncation term.
    free_energy_excess : float
        *free_energy* with the classical contribution removed, in kJ/mol.
        A wholly classical particle still has a mass-dependent free energy,
        through its momentum partition function; this is the part that
        survives it, and it is the part an isotope effect is really about.
    free_energy_excess_stderr : float
        Same as *free_energy_stderr*: the classical term is exact.
    kinetic_energies : numpy.ndarray
        The integrand at each node, in kJ/mol.
    kinetic_stderr : numpy.ndarray
        Standard error of each node's integrand, in kJ/mol.
    masses : numpy.ndarray
        The node masses in daltons.
    nodes : int
        Number of quadrature nodes.
    """

    free_energy: float
    free_energy_stderr: float
    free_energy_excess: float
    free_energy_excess_stderr: float
    kinetic_energies: npt.NDArray[np.float64]
    kinetic_stderr: npt.NDArray[np.float64]
    masses: npt.NDArray[np.float64]
    nodes: int


def _as_kilojoule_array(values: Any, *, name: str, length: int,
                        ) -> npt.NDArray[np.float64]:
    """
    Normalise an energy input to a finite float array in kJ/mol.

    Parameters
    ----------
    values : array-like or openmm.unit.Quantity or float
        Energies, either as bare numbers in kJ/mol or carrying units. A
        scalar is broadcast to *length*.
    name : str
        Argument name, used in error messages.
    length : int
        Number of entries required.

    Returns
    -------
    numpy.ndarray
        The values in kJ/mol, shaped ``(length,)``.

    Raises
    ------
    ValueError
        If the values are not numeric, not finite, or the wrong length.
    """
    if unit.is_quantity(values):
        values = values.value_in_unit(unit.kilojoule_per_mole)
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, in kJ/mol") from exc
    if array.ndim == 0:
        array = np.full(length, float(array))
    if array.shape != (length,):
        raise ValueError(
            f"{name} must hold one entry per node, {length} of them"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def rpmd_mass_integration_nodes(mass_light: unit.Quantity | float,
                                mass_heavy: unit.Quantity | float,
                                *,
                                nodes: int = 2) -> MassIntegrationNodes:
    """
    Choose the masses to run at for a mass thermodynamic integration.

    The integration variable is ``ln m``, and the nodes are Gauss-Legendre
    points on ``[ln mass_light, ln mass_heavy]``.  Picking them is the step
    that is easy to get wrong -- the endpoints themselves are a poor choice,
    and the physical light mass alone is not a quadrature at all -- so it is
    split out here rather than left to the caller.

    Parameters
    ----------
    mass_light : openmm.unit.Quantity or float
        Mass of the starting isotope. A bare number is read as daltons.
    mass_heavy : openmm.unit.Quantity or float
        Mass of the finishing isotope. A bare number is read as daltons.
    nodes : int, optional
        Number of quadrature nodes, one RPMD trajectory each. Default is 2,
        which is converged far below any achievable sampling error.

    Returns
    -------
    MassIntegrationNodes
        The node masses, log masses, weights and the two endpoint masses.

    Raises
    ------
    ValueError
        If either mass is not finite and positive, the two are equal, or
        *nodes* is below 1.
    TypeError
        If *nodes* is not an integer.

    See Also
    --------
    rpmd_isotope_free_energy : Combine the nodes into a free energy.

    Notes
    -----
    Accuracy for a 3600 cm^-1 O-H stretch at 300 K, substituting H for D,
    against the exact -18.898 kJ/mol:

    ==========================  =============  ==============  =========
    scheme                      trajectories   masses (Da)     error
    ==========================  =============  ==============  =========
    integrand at the H mass     1              1.008           3.5
    ``nodes=1``                 1              1.425           0.09
    trapezoid on the endpoints  2              1.008, 2.014    0.19
    ``nodes=2``                 2              1.167, 1.740    0.00006
    ==========================  =============  ==============  =========

    Two nodes cost the same as evaluating both endpoints and are some three
    thousand times more accurate, which is why the endpoints are not offered.
    ``nodes=1`` puts a single trajectory at the geometric mean of the two
    masses and is a legitimate cheap estimate; a single trajectory at the
    physical light mass is not.

    Examples
    --------
    Run one trajectory per node::

        from openmmnqe import rpmd_mass_integration_nodes

        plan = rpmd_mass_integration_nodes(1.008, 2.014)
        for index, mass in enumerate(plan.masses):
            for atom in substituted:
                system.setParticleMass(atom, mass * unit.dalton)
            run_openmm_rpmd_prod(..., output_prefix=f"node_{index}")
    """
    light = require_positive_finite_scalar_in_unit(
        mass_light, unit.dalton, name="mass_light",
    )
    heavy = require_positive_finite_scalar_in_unit(
        mass_heavy, unit.dalton, name="mass_heavy",
    )
    if light == heavy:
        raise ValueError("mass_light and mass_heavy must differ")
    node_count = require_integer(nodes, name="nodes", minimum=1)

    start, end = np.log(light), np.log(heavy)
    points, weights = np.polynomial.legendre.leggauss(node_count)
    ln_masses = start + (end - start) * (points + 1.0) / 2.0
    scaled = weights * (end - start) / 2.0
    return MassIntegrationNodes(
        masses=np.exp(ln_masses),
        ln_masses=ln_masses,
        weights=scaled,
        mass_light=light,
        mass_heavy=heavy,
    )


def rpmd_isotope_free_energy(nodes: MassIntegrationNodes,
                             kinetic_energies: Any,
                             *,
                             temperature: unit.Quantity | float,
                             kinetic_stderr: Any = 0.0,
                             n_substituted: int = 1) -> IsotopeFreeEnergy:
    r"""
    Integrate per-atom kinetic energies over mass into a free energy.

    Uses the exact identity

    .. math::

        \frac{\partial F}{\partial \ln m} = -\left<K\right>,

    so that the free energy of substituting one isotope for another is

    .. math::

        \Delta F = -\int_{\ln m_{\mathrm{light}}}^{\ln m_{\mathrm{heavy}}}
                    \left<K\right>_m \,\mathrm{d}\ln m,

    evaluated by the Gauss-Legendre rule that
    :func:`rpmd_mass_integration_nodes` laid out.

    Parameters
    ----------
    nodes : MassIntegrationNodes
        The quadrature the trajectories were run at.
    kinetic_energies : array-like or openmm.unit.Quantity
        Mean centroid-virial kinetic energy at each node, summed over every
        substituted atom, in kJ/mol. One entry per node.
    temperature : openmm.unit.Quantity or float
        Simulation temperature. A bare number is read as kelvin.
    kinetic_stderr : array-like or openmm.unit.Quantity or float, optional
        Standard error of each entry of *kinetic_energies*, in kJ/mol. A
        single number applies to every node. Default is 0.0.
    n_substituted : int, optional
        Number of atoms substituted, which sets the classical reference
        removed in ``free_energy_excess``. Must match the number of atoms
        summed into *kinetic_energies*. Default is 1.

    Returns
    -------
    IsotopeFreeEnergy
        The free-energy change and its parts, in kJ/mol.

    Raises
    ------
    ValueError
        If *temperature* is not finite and positive, the energy arrays are
        not finite or not one entry per node, or *n_substituted* is below 1.
    TypeError
        If *n_substituted* is not an integer.

    See Also
    --------
    openmmnqe.reporters.rpmd_kinetic_decomposition_averages : Supplies the
        integrand and its error straight from a log.
    rpmd_fractionation_factor : Compare two sites.

    Notes
    -----
    ``free_energy_stderr`` propagates only the sampling error of the nodes,
    which are independent runs, so the quadrature sum is exact.  It says
    nothing about quadrature truncation -- for two nodes that term is far
    below any reachable sampling error, and for one node it is not, which is
    the trade :func:`rpmd_mass_integration_nodes` documents.

    The estimator inherits the centroid-virial estimator's constraint bias:
    run the ring polymer flexible.

    Examples
    --------
    Combine two nodes read back from their logs::

        from openmmnqe import (
            rpmd_isotope_free_energy,
            rpmd_kinetic_decomposition_averages,
            rpmd_mass_integration_nodes,
        )

        plan = rpmd_mass_integration_nodes(1.008, 2.014)
        means, errors = [], []
        for index in range(len(plan.masses)):
            averages = rpmd_kinetic_decomposition_averages(
                f"node_{index}_kinetic.log", discard=0.2,
            )
            mean, error = averages["Kcv_H1(kJ/mol)"]
            means.append(mean)
            errors.append(error)

        result = rpmd_isotope_free_energy(
            plan, means, kinetic_stderr=errors, temperature=300.0,
        )
        print(result.free_energy, result.free_energy_stderr)
    """
    temperature_k = require_positive_finite_scalar_in_unit(
        temperature, unit.kelvin, name="temperature",
    )
    substituted = require_integer(
        n_substituted, name="n_substituted", minimum=1,
    )
    count = len(nodes.masses)
    kinetic = _as_kilojoule_array(
        kinetic_energies, name="kinetic_energies", length=count,
    )
    errors = _as_kilojoule_array(
        kinetic_stderr, name="kinetic_stderr", length=count,
    )
    if np.any(errors < 0.0):
        raise ValueError("kinetic_stderr must not be negative")

    free_energy = -float(np.sum(nodes.weights * kinetic))
    stderr = float(np.sqrt(np.sum((nodes.weights * errors) ** 2)))

    # A classical particle's free energy still moves with its mass, through
    # the momentum partition function: 3/2 kT per atom per unit of ln m.
    span = np.log(nodes.mass_heavy / nodes.mass_light)
    classical = -span * substituted * 1.5 * _BOLTZMANN_KJ_PER_MOL_K * temperature_k
    return IsotopeFreeEnergy(
        free_energy=free_energy,
        free_energy_stderr=stderr,
        free_energy_excess=free_energy - float(classical),
        free_energy_excess_stderr=stderr,
        kinetic_energies=kinetic,
        kinetic_stderr=errors,
        masses=nodes.masses,
        nodes=count,
    )


def _free_energy_and_error(value: Any, *, name: str) -> tuple[float, float]:
    """
    Read a free energy and its error out of whatever the caller passed.

    Parameters
    ----------
    value : IsotopeFreeEnergy or tuple of float or float
        A result from :func:`rpmd_isotope_free_energy`, a
        ``(free_energy, standard_error)`` pair, or a bare free energy in
        kJ/mol.
    name : str
        Argument name, used in error messages.

    Returns
    -------
    free_energy : float
        The free energy in kJ/mol.
    stderr : float
        Its standard error in kJ/mol, zero if none was supplied.

    Raises
    ------
    ValueError
        If *value* is none of those shapes, or is not finite.
    """
    if isinstance(value, IsotopeFreeEnergy):
        return value.free_energy, value.free_energy_stderr
    if isinstance(value, (tuple, list)) and len(value) == 2:
        pair = _as_kilojoule_array(value, name=name, length=2)
        return float(pair[0]), float(pair[1])
    scalar = require_scalar_in_unit(
        value, unit.kilojoule_per_mole, name=name,
    )
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar, 0.0


def rpmd_fractionation_factor(site_a: Any, site_b: Any, *,
                              temperature: unit.Quantity | float,
                              ) -> tuple[float, float]:
    r"""
    Compare two sites' isotope free energies as a fractionation factor.

    For the exchange ``A-H + B-D <-> A-D + B-H`` the equilibrium constant is
    the fractionation factor

    .. math::

        \ln\alpha_{A/B} = -\frac{\Delta F_A - \Delta F_B}{k_B T},

    the ratio of heavy-to-light isotope enrichment at site ``A`` against
    site ``B``.  A positive ``ln_alpha`` means site ``A`` prefers the heavy
    isotope.

    Parameters
    ----------
    site_a : IsotopeFreeEnergy or tuple of float or float
        Substitution free energy at the first site, from
        :func:`rpmd_isotope_free_energy`. A ``(value, standard_error)`` pair
        or a bare value in kJ/mol is also accepted.
    site_b : IsotopeFreeEnergy or tuple of float or float
        The same for the second site.
    temperature : openmm.unit.Quantity or float
        Simulation temperature. A bare number is read as kelvin.

    Returns
    -------
    ln_alpha : float
        Natural log of the fractionation factor, dimensionless.
    ln_alpha_stderr : float
        Its standard error, propagating the two sites' independent errors.

    Raises
    ------
    ValueError
        If *temperature* is not finite and positive, or either site is not a
        recognised free-energy shape.

    Notes
    -----
    Both sites must have been computed over the same isotope pair and the
    same number of substituted atoms.  When they were, the classical term
    cancels in the difference, so ``free_energy`` and ``free_energy_excess``
    give the same answer and either may be passed.

    Examples
    --------
    Ask which of two sites concentrates deuterium::

        from openmmnqe import rpmd_fractionation_factor

        ln_alpha, error = rpmd_fractionation_factor(
            donor, acceptor, temperature=300.0,
        )
    """
    temperature_k = require_positive_finite_scalar_in_unit(
        temperature, unit.kelvin, name="temperature",
    )
    free_a, error_a = _free_energy_and_error(site_a, name="site_a")
    free_b, error_b = _free_energy_and_error(site_b, name="site_b")

    kt = _BOLTZMANN_KJ_PER_MOL_K * temperature_k
    ln_alpha = -(free_a - free_b) / kt
    stderr = float(np.hypot(error_a, error_b)) / kt
    return ln_alpha, stderr
