import numpy as np

from rapidai_tools.delta_spectrum import energy_at_rank


def test_rank1_deltas_fully_captured_at_rank1():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(4, 8, 1)), rng.normal(size=(4, 1, 6))
    deltas = u @ v  # each expert delta is exactly rank 1
    assert energy_at_rank(deltas, 1) > 0.999


def test_full_rank_noise_needs_full_rank():
    rng = np.random.default_rng(0)
    deltas = rng.normal(size=(4, 8, 6))
    assert energy_at_rank(deltas, 1) < 0.6
    assert energy_at_rank(deltas, 6) > 0.999


def test_monotonic_in_rank():
    rng = np.random.default_rng(1)
    deltas = rng.normal(size=(3, 10, 10))
    vals = [energy_at_rank(deltas, r) for r in (1, 3, 5, 10)]
    assert vals == sorted(vals)
