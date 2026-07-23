"""Expert-routing overlap statistics for lookahead-decay analysis (Experiment 0.1)."""

import numpy as np


def overlap_at_lag(trace: np.ndarray, lag: int) -> float:
    """Mean fraction of token t's experts also used by token t+lag, averaged
    over tokens and layers.

    trace: (T, L, K) int array of top-k expert ids per token per layer.
    """
    t_count, layers, k = trace.shape
    if t_count <= lag:
        raise ValueError(f"trace length {t_count} too short for lag {lag}")
    hits = 0
    total = 0
    for layer in range(layers):
        a = trace[:-lag, layer, :]
        b = trace[lag:, layer, :]
        for i in range(a.shape[0]):
            hits += np.intersect1d(a[i], b[i]).size
            total += k
    return hits / total


def random_baseline(k: int, n: int) -> float:
    """Expected overlap fraction under uniform random routing: k/n.

    (Each of token t's k experts appears in t+lag's uniform k-subset with
    probability k/n, so the expected shared fraction is k/n.)
    """
    return k / n
