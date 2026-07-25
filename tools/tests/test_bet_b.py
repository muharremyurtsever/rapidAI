"""Unit tests for Bet B predictability capture + analysis."""

import numpy as np

from rapidai_tools import bet_b_predict as bp
from rapidai_tools.bet_b_capture import BOUNDARY, build_arrays


def test_build_arrays_skips_prefill_and_splits():
    # 2 layers, k=2, d=3. One prefill (2 tokens) then 3 decode steps.
    store = [(BOUNDARY, None, None)]
    # prefill: multi-token gate call (shape (1,2,d)) -> skipped
    for layer in range(2):
        store.append((layer, np.zeros((1, 2, 3), np.float16),
                      np.zeros((1, 2, 2), np.int16)))
    # 3 decode steps
    for t in range(3):
        for layer in range(2):
            store.append((layer, np.full((1, 1, 3), t, np.float16),
                          np.array([[[layer, t]]], np.int16)))
    X, E = build_arrays(store, n_layers=2)
    assert X.shape == (3, 2, 3)
    assert E.shape == (3, 2, 2)
    assert E[2, 1, 0] == 1 and E[2, 1, 1] == 2


def test_persistence_recall_perfect_when_static():
    # experts never change -> persistence recall = 1.0 at every lag
    E = np.tile(np.array([[0, 1, 2]]), (10, 1))[:, None, :]  # (10,1,3)
    assert bp.persistence_recall(E, 1, 1) == 1.0
    assert bp.persistence_recall(E, 3, 1) == 1.0


def test_persistence_recall_zero_when_disjoint_alternating():
    # alternate between two disjoint sets -> lag-1 overlap 0, lag-2 overlap 1
    a = np.array([0, 1]); b = np.array([2, 3])
    seq = np.stack([a if t % 2 == 0 else b for t in range(12)])[:, None, :]
    assert bp.persistence_recall(seq, 1, 1) == 0.0
    assert bp.persistence_recall(seq, 2, 1) == 1.0


def test_trained_recall_learns_planted_signal():
    # Plant: expert set is a deterministic function of a hidden feature at t,
    # revealed at t+1. A predictor from X[t] should beat chance strongly.
    rng = np.random.default_rng(0)
    T, d, n_exp, k = 400, 32, 16, 4
    X = rng.standard_normal((T, 1, d)).astype(np.float32)
    E = np.zeros((T, 1, k), np.int16)
    for t in range(T):
        # next-token experts determined by sign pattern of X[t]'s first 4 dims
        base = int((X[t, 0, 0] > 0)) * 4 + int((X[t, 0, 1] > 0)) * 2 \
            + int((X[t, 0, 2] > 0))
        E[min(t + 1, T - 1), 0] = [(base + j) % n_exp for j in range(k)]
    r, per_layer, preds, evt = bp.trained_recall(
        X, E, n_exp, lag=1, kprime_mult=1, n_pca=16)
    # planted deterministic signal -> recall well above k/n_exp = 0.25
    assert r > 0.6, r
    assert len(preds[0]) == len(evt)


def test_miss_coverage_bounds_and_shape():
    rng = np.random.default_rng(1)
    T, L, k, n_exp = 60, 2, 3, 20
    E = rng.integers(0, n_exp, size=(T, L, k)).astype(np.int16)
    X = rng.standard_normal((T, L, 8)).astype(np.float32)
    _, _, preds, evt = bp.trained_recall(X, E, n_exp, 1, 1, n_pca=8)
    cov = bp.miss_coverage(E, X, n_exp, entry_bytes=1000,
                           budget_bytes=5000, lag=1,
                           eval_preds=preds, eval_true_t=evt, kprime_mult=1)
    assert 0.0 <= cov["persistence_miss_coverage"] <= 1.0
    assert 0.0 <= cov["trained_miss_coverage"] <= 1.0
    assert cov["cache_capacity_experts"] == 5
