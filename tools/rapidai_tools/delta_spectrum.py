"""SVD energy-capture analysis for expert-delta decomposition (Experiment 0.3)."""

import numpy as np


def energy_at_rank(deltas: np.ndarray, rank: int) -> float:
    """Mean fraction of squared-Frobenius energy captured by a rank-`rank`
    SVD truncation, averaged over experts.

    deltas: (E, out, in) — one delta matrix per expert.
    """
    fracs = []
    for d in deltas:
        s = np.linalg.svd(d, compute_uv=False)
        fracs.append((s[:rank] ** 2).sum() / (s**2).sum())
    return float(np.mean(fracs))
