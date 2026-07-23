import numpy as np

from rapidai_tools.routing_stats import overlap_at_lag, random_baseline


def test_identical_tokens_full_overlap():
    # 3 tokens, 1 layer, k=2, always experts {1,2} -> overlap 1.0 at lag 1
    tr = np.tile(np.array([[[1, 2]]], dtype=np.int16), (3, 1, 1))
    assert overlap_at_lag(tr, 1) == 1.0


def test_disjoint_tokens_zero_overlap():
    tr = np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.int16)
    assert overlap_at_lag(tr, 1) == 0.0


def test_half_overlap_at_lag_two():
    # token t and t+2 share exactly one of two experts
    tr = np.array([[[1, 2]], [[9, 9]], [[2, 7]]], dtype=np.int16)
    assert overlap_at_lag(tr, 2) == 0.5


def test_lag_longer_than_trace_raises():
    tr = np.zeros((2, 1, 2), dtype=np.int16)
    try:
        overlap_at_lag(tr, 5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_random_baseline():
    # k of n uniformly-random experts: expected shared fraction = k/n
    assert abs(random_baseline(8, 128) - 8 / 128) < 1e-9
