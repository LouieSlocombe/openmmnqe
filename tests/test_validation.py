"""Compatibility checks for shared validation at existing API boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from openmmnqe import tools
from openmmnqe._validation import require_discard_fraction
from openmmnqe.openmm import _SEED_STREAMS, _derive_seeds, _validate_rpmd_n_beads
from openmmnqe.rates import _spawn_child_sequences, _validate_recrossing_seed
from openmmnqe.walkers import _derive_walker_seeds


@pytest.mark.parametrize("seed", [True, np.bool_(False), -1, 1.5, "7", np.nan])
@pytest.mark.parametrize("entry", [
    lambda seed: _derive_seeds(seed, "thermostat"),
    _validate_recrossing_seed,
    lambda seed: _derive_walker_seeds(seed, 2),
    lambda seed: tools.sample_rpmd_velocities(None, 300, 2, seed=seed),
    lambda seed: tools.init_beads(None, None, 2, seed=seed),
])
def test_invalid_seeds_keep_the_same_error_at_every_boundary(entry, seed):
    with pytest.raises(ValueError, match="^seed must be a non-negative integer or None$"):
        entry(seed)


@pytest.mark.parametrize("beads", [True, np.bool_(False), 0, -1, 2.0, "2"])
@pytest.mark.parametrize("entry", [
    _validate_rpmd_n_beads,
    lambda beads: tools.sample_rpmd_velocities(None, 300, beads),
    lambda beads: tools.init_beads(None, None, beads),
])
def test_invalid_bead_counts_fail_before_accessing_the_system(entry, beads):
    with pytest.raises(ValueError, match="^n_beads must be a positive integer$"):
        entry(beads)


def test_seed_streams_match_preconsolidation_values():
    # Captured before extraction: these pin stream identity, not merely that
    # two calls through the new implementation agree with each other.
    assert _derive_seeds(np.int64(7), *_SEED_STREAMS) == (
        1201125462, 1471499524, 1684166798, 1694716536,
    )
    assert _derive_walker_seeds(7, 4) == (
        965725788, 3568268479, 3908098321, 2533909489,
    )
    children = _spawn_child_sequences(7, 2, 3)
    assert [[int(child.generate_state(1)[0]) for child in row] for row in children] == [
        [3353181588, 239119527, 1322153692],
        [4120870950, 2582759846, 1140642528],
    ]
    assert _derive_seeds(None, *_SEED_STREAMS) == (None,) * len(_SEED_STREAMS)
    assert _validate_recrossing_seed(2**100) == 2**100
    # Unknown stream names were checked before invalid seeds in the old code.
    with pytest.raises(KeyError, match="unknown seed stream"):
        _derive_seeds(-1, "missing")


@pytest.mark.parametrize("discard", [True, np.bool_(False), "0.2", -0.1, 1, np.nan, np.inf])
def test_discard_validation_preserves_scalar_and_range_contract(discard):
    with pytest.raises(ValueError, match=r"^discard must be a number in \[0, 1\)$"):
        require_discard_fraction(discard)


@pytest.mark.parametrize("discard", [0, np.int64(0), np.float32(0.25), 0.999])
def test_discard_validation_accepts_real_scalars(discard):
    assert require_discard_fraction(discard) == float(discard)


@pytest.mark.parametrize("mass", [-1.0, np.nan, np.inf])
def test_shared_mass_reader_preserves_sampling_and_restart_error_context(mass):
    from openmm import unit

    system = SimpleNamespace(
        getNumParticles=lambda: 1,
        getParticleMass=lambda index: mass * unit.dalton,
    )
    with pytest.raises(ValueError, match="^RPMD System particle masses must be finite and non-negative$"):
        tools._particle_masses_dalton(system)
    with pytest.raises(ValueError, match="^particle masses must be finite and non-negative$"):
        tools.sample_rpmd_velocities(system, 300, 2, seed=7)
