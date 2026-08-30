"""Unit tests for mass thermodynamic integration and isotope free energies."""

from __future__ import annotations

from typing import Any

import numpy as np
import openmm.unit as unit
import pytest
from scipy import constants

from openmmnqe.isotopes import (
    IsotopeFreeEnergy,
    rpmd_fractionation_factor,
    rpmd_isotope_free_energy,
    rpmd_mass_integration_nodes,
)

_BOLTZMANN = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin
)

# hbar in the package's MD units, kJ*ps/mol.
_HBAR = constants.hbar * constants.Avogadro * 1.0e-3 * 1.0e12

_MASS_H = 1.008
_MASS_D = 2.014


def _oh_force_constant(wavenumber: float = 3600.0) -> float:
    """Harmonic force constant giving a hydrogen stretch at *wavenumber*."""
    omega = 2.0 * np.pi * wavenumber * constants.c * 1.0e2 * 1.0e-12
    return float(_MASS_H * omega ** 2)


def _harmonic_free_energy(mass: float, force_constant: float,
                          temperature: float) -> float:
    """Exact free energy of a 3D quantum harmonic oscillator, in kJ/mol."""
    omega = np.sqrt(force_constant / mass)
    beta = 1.0 / (_BOLTZMANN * temperature)
    return float(
        3.0 * _BOLTZMANN * temperature
        * np.log(2.0 * np.sinh(beta * _HBAR * omega / 2.0))
    )


def _harmonic_kinetic(mass: float, force_constant: float,
                      temperature: float) -> float:
    """Exact kinetic energy of a 3D quantum harmonic oscillator, in kJ/mol."""
    omega = np.sqrt(force_constant / mass)
    beta = 1.0 / (_BOLTZMANN * temperature)
    return float(
        3.0 * (_HBAR * omega / 4.0) / np.tanh(beta * _HBAR * omega / 2.0)
    )


def test_mass_derivative_identity_holds_for_a_harmonic_oscillator() -> None:
    # The identity the whole module rests on: dF/d(ln m) = -<K>.
    force_constant = _oh_force_constant()
    mass, step = 1.5, 1.0e-5

    derivative = (
        _harmonic_free_energy(mass * np.exp(step), force_constant, 300.0)
        - _harmonic_free_energy(mass * np.exp(-step), force_constant, 300.0)
    ) / (2.0 * step)

    assert derivative == pytest.approx(
        -_harmonic_kinetic(mass, force_constant, 300.0), rel=1e-8
    )


def test_nodes_are_gauss_legendre_points_in_log_mass() -> None:
    single = rpmd_mass_integration_nodes(_MASS_H, _MASS_D, nodes=1)

    # One node lands on the geometric mean, which is the midpoint in ln m.
    assert single.masses == pytest.approx([np.sqrt(_MASS_H * _MASS_D)])
    assert single.weights.sum() == pytest.approx(np.log(_MASS_D / _MASS_H))

    pair = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)
    points, weights = np.polynomial.legendre.leggauss(2)
    start, end = np.log(_MASS_H), np.log(_MASS_D)
    assert pair.ln_masses == pytest.approx(
        start + (end - start) * (points + 1.0) / 2.0
    )
    assert pair.weights == pytest.approx(weights * (end - start) / 2.0)
    assert pair.masses == pytest.approx(np.exp(pair.ln_masses))
    assert pair.weights.sum() == pytest.approx(end - start)

    # Every node is strictly between the two isotopes: none is physical.
    assert np.all(pair.masses > _MASS_H)
    assert np.all(pair.masses < _MASS_D)
    assert (pair.mass_light, pair.mass_heavy) == (_MASS_H, _MASS_D)


def test_nodes_run_downhill_when_the_heavy_mass_is_given_first() -> None:
    forward = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)
    backward = rpmd_mass_integration_nodes(_MASS_D, _MASS_H)

    assert backward.masses == pytest.approx(forward.masses[::-1])
    assert backward.weights.sum() == pytest.approx(-forward.weights.sum())


def test_nodes_accept_quantities() -> None:
    plan = rpmd_mass_integration_nodes(
        _MASS_H * unit.dalton, _MASS_D * unit.dalton,
    )

    assert plan.mass_light == pytest.approx(_MASS_H)
    assert plan.mass_heavy == pytest.approx(_MASS_D)


@pytest.mark.parametrize(
    ("light", "heavy"),
    [(0.0, 2.0), (-1.0, 2.0), (1.0, float("nan"))],
)
def test_nodes_reject_unphysical_masses(light: float, heavy: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        rpmd_mass_integration_nodes(light, heavy)


def test_nodes_reject_equal_masses() -> None:
    with pytest.raises(ValueError, match="must differ"):
        rpmd_mass_integration_nodes(1.008, 1.008)


def test_nodes_reject_bad_node_counts() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        rpmd_mass_integration_nodes(_MASS_H, _MASS_D, nodes=2.0)
    with pytest.raises(ValueError, match="must be a positive"):
        rpmd_mass_integration_nodes(_MASS_H, _MASS_D, nodes=0)


def test_mass_integration_reproduces_the_exact_isotope_free_energy() -> None:
    force_constant, temperature = _oh_force_constant(), 300.0
    exact = (
        _harmonic_free_energy(_MASS_D, force_constant, temperature)
        - _harmonic_free_energy(_MASS_H, force_constant, temperature)
    )
    # A 3600 cm-1 O-H stretch: strongly quantum, hbar*omega/kT is about 17.
    assert exact == pytest.approx(-18.898, abs=1.0e-3)

    def integrate(nodes: int) -> float:
        plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D, nodes=nodes)
        kinetic = [
            _harmonic_kinetic(mass, force_constant, temperature)
            for mass in plan.masses
        ]
        return rpmd_isotope_free_energy(
            plan, kinetic, temperature=temperature,
        ).free_energy

    assert integrate(2) == pytest.approx(exact, abs=1.0e-3)
    assert integrate(3) == pytest.approx(exact, abs=1.0e-3)
    # One node is a cheap but real estimate, good to a few tenths.
    assert integrate(1) == pytest.approx(exact, abs=0.15)

    # Evaluating the integrand only at the physical hydrogen mass -- which is
    # the tempting "one trajectory" shortcut -- is wrong by 18%, which is why
    # the module offers a quadrature rather than that.
    span = np.log(_MASS_D / _MASS_H)
    shortcut = -_harmonic_kinetic(_MASS_H, force_constant, temperature) * span
    assert abs(shortcut - exact) == pytest.approx(3.46, abs=0.01)


def test_free_energy_is_the_negative_weighted_sum() -> None:
    plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)
    kinetic = [10.0, 8.0]

    result = rpmd_isotope_free_energy(plan, kinetic, temperature=300.0)

    assert result.nodes == 2
    assert result.free_energy == pytest.approx(
        -float(np.sum(plan.weights * np.array(kinetic)))
    )
    assert result.kinetic_energies == pytest.approx(kinetic)
    assert result.masses == pytest.approx(plan.masses)
    assert result.kinetic_stderr == pytest.approx([0.0, 0.0])
    assert result.free_energy_stderr == 0.0


def test_error_propagation_is_the_weighted_quadrature_sum() -> None:
    plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)
    errors = [0.3, 0.5]

    result = rpmd_isotope_free_energy(
        plan, [10.0, 8.0], kinetic_stderr=errors, temperature=300.0,
    )

    assert result.free_energy_stderr == pytest.approx(
        float(np.sqrt(np.sum((plan.weights * np.array(errors)) ** 2)))
    )
    assert result.free_energy_excess_stderr == result.free_energy_stderr


def test_a_scalar_standard_error_applies_to_every_node() -> None:
    plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D, nodes=3)

    result = rpmd_isotope_free_energy(
        plan, [10.0, 9.0, 8.0], kinetic_stderr=0.2, temperature=300.0,
    )

    assert result.kinetic_stderr == pytest.approx([0.2, 0.2, 0.2])


def test_excess_removes_the_classical_mass_dependence() -> None:
    plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)
    temperature = 300.0
    kt = _BOLTZMANN * temperature
    span = np.log(_MASS_D / _MASS_H)

    for substituted in (1, 2):
        result = rpmd_isotope_free_energy(
            plan,
            [10.0, 8.0],
            temperature=temperature,
            n_substituted=substituted,
        )
        assert result.free_energy_excess - result.free_energy == pytest.approx(
            span * substituted * 1.5 * kt
        )

    # A wholly classical particle has 3kT/2 of kinetic energy at every mass,
    # so its excess is exactly zero.
    classical = rpmd_isotope_free_energy(
        plan, [1.5 * kt, 1.5 * kt], temperature=temperature,
    )
    assert classical.free_energy_excess == pytest.approx(0.0, abs=1e-12)
    assert classical.free_energy == pytest.approx(-span * 1.5 * kt)


def test_free_energy_accepts_quantities() -> None:
    plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)

    bare = rpmd_isotope_free_energy(plan, [10.0, 8.0], temperature=300.0)
    quantified = rpmd_isotope_free_energy(
        plan,
        np.array([10.0, 8.0]) * unit.kilojoule_per_mole,
        temperature=300 * unit.kelvin,
    )

    assert quantified.free_energy == pytest.approx(bare.free_energy)


def test_free_energy_validates_its_inputs() -> None:
    plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)

    with pytest.raises(ValueError, match="one entry per node"):
        rpmd_isotope_free_energy(plan, [10.0], temperature=300.0)
    with pytest.raises(ValueError, match="one entry per node"):
        rpmd_isotope_free_energy(
            plan, [10.0, 8.0], kinetic_stderr=[0.1, 0.2, 0.3],
            temperature=300.0,
        )
    with pytest.raises(ValueError, match="must be numeric, in kJ/mol"):
        rpmd_isotope_free_energy(plan, ["warm", "cold"], temperature=300.0)
    with pytest.raises(ValueError, match="kinetic_energies must be finite"):
        rpmd_isotope_free_energy(
            plan, [10.0, float("inf")], temperature=300.0,
        )
    with pytest.raises(ValueError, match="must not be negative"):
        rpmd_isotope_free_energy(
            plan, [10.0, 8.0], kinetic_stderr=[-0.1, 0.2], temperature=300.0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        rpmd_isotope_free_energy(plan, [10.0, 8.0], temperature=0.0)
    with pytest.raises(TypeError, match="must be an integer"):
        rpmd_isotope_free_energy(
            plan, [10.0, 8.0], temperature=300.0, n_substituted=1.5,
        )
    with pytest.raises(ValueError, match="must be a positive"):
        rpmd_isotope_free_energy(
            plan, [10.0, 8.0], temperature=300.0, n_substituted=0,
        )


def test_fractionation_factor_favours_the_more_negative_site() -> None:
    plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)
    temperature = 300.0
    kt = _BOLTZMANN * temperature

    donor = rpmd_isotope_free_energy(
        plan, [12.0, 10.0], kinetic_stderr=0.3, temperature=temperature,
    )
    acceptor = rpmd_isotope_free_energy(
        plan, [10.0, 8.0], kinetic_stderr=0.4, temperature=temperature,
    )

    ln_alpha, error = rpmd_fractionation_factor(
        donor, acceptor, temperature=temperature,
    )

    assert ln_alpha == pytest.approx(
        -(donor.free_energy - acceptor.free_energy) / kt
    )
    # The donor's larger kinetic energies make its substitution the more
    # favourable one, so it is the site that concentrates deuterium.
    assert donor.free_energy < acceptor.free_energy
    assert ln_alpha > 0.0
    assert error == pytest.approx(
        np.hypot(donor.free_energy_stderr, acceptor.free_energy_stderr) / kt
    )

    # Swapping the sites flips the sign and leaves the error alone.
    reversed_alpha, reversed_error = rpmd_fractionation_factor(
        acceptor, donor, temperature=temperature,
    )
    assert reversed_alpha == pytest.approx(-ln_alpha)
    assert reversed_error == pytest.approx(error)


def test_fractionation_factor_ignores_the_cancelling_classical_term() -> None:
    plan = rpmd_mass_integration_nodes(_MASS_H, _MASS_D)
    temperature = 300.0
    donor = rpmd_isotope_free_energy(
        plan, [12.0, 10.0], temperature=temperature,
    )
    acceptor = rpmd_isotope_free_energy(
        plan, [10.0, 8.0], temperature=temperature,
    )

    from_total, _ = rpmd_fractionation_factor(
        donor, acceptor, temperature=temperature,
    )
    from_excess, _ = rpmd_fractionation_factor(
        (donor.free_energy_excess, 0.0),
        (acceptor.free_energy_excess, 0.0),
        temperature=temperature,
    )

    assert from_excess == pytest.approx(from_total)


@pytest.mark.parametrize(
    "site", [-20.0, (-20.0, 0.5), [-20.0, 0.5], -20.0 * unit.kilojoule_per_mole],
)
def test_fractionation_factor_accepts_plain_free_energies(site: Any) -> None:
    ln_alpha, error = rpmd_fractionation_factor(
        site, -18.0, temperature=300.0,
    )

    kt = _BOLTZMANN * 300.0
    assert ln_alpha == pytest.approx(2.0 / kt)
    expected_error = 0.5 / kt if isinstance(site, (tuple, list)) else 0.0
    assert error == pytest.approx(expected_error)


def test_fractionation_factor_validates_its_inputs() -> None:
    result = IsotopeFreeEnergy(
        free_energy=-20.0,
        free_energy_stderr=0.1,
        free_energy_excess=-17.0,
        free_energy_excess_stderr=0.1,
        kinetic_energies=np.array([10.0, 8.0]),
        kinetic_stderr=np.array([0.1, 0.1]),
        masses=np.array([1.17, 1.74]),
        nodes=2,
    )

    with pytest.raises(ValueError, match="finite and positive"):
        rpmd_fractionation_factor(result, -18.0, temperature=0.0)
    with pytest.raises(ValueError, match="site_b must be finite"):
        rpmd_fractionation_factor(
            result, float("nan"), temperature=300.0,
        )
    with pytest.raises(ValueError, match="site_a"):
        rpmd_fractionation_factor("cold", -18.0, temperature=300.0)
