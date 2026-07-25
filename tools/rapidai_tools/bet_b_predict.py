"""Bet B predictability analysis: persistence vs trained predictor of t+lag experts.

Given a captured decode trace X (T, L, d) of gate-input hidden states and E (T, L, k)
of routed expert ids, measure how well a future token's expert set can be predicted
from the current token's per-layer hidden state, versus the free "persistence"
baseline (predict next = current). Also simulates a byte-budget SLRU cache to report
the metric that matters: fraction of on-critical-path expert MISSES the predictor
would have prefetched in time.
"""

import glob
import os

import numpy as np


# ---------------------------------------------------------------------------
# recall helpers
# ---------------------------------------------------------------------------

def _recall_sets(pred_sets, true_sets) -> float:
    """Mean |pred ∩ true| / |true| over aligned lists of id-arrays."""
    num = den = 0
    for p, t in zip(pred_sets, true_sets):
        ps = set(int(x) for x in p)
        ts = set(int(x) for x in t)
        num += len(ps & ts)
        den += len(ts)
    return num / max(den, 1)


def persistence_recall(E: np.ndarray, lag: int, kprime_mult: int) -> float:
    """Persistence predictor recall of t+lag experts.

    kprime_mult=1: predicted set = experts of token t (size k).
    kprime_mult=2: predicted set = union of tokens t and t-1 (~2k experts).
    Averaged over layers and eval tokens.
    """
    T, L, k = E.shape
    recalls = []
    for layer in range(L):
        preds, trues = [], []
        for t in range(0, T - lag):
            if kprime_mult == 1:
                pred = E[t, layer]
            else:
                lo = max(0, t - 1)
                pred = np.concatenate([E[t, layer], E[lo, layer]])
            preds.append(pred)
            trues.append(E[t + lag, layer])
        recalls.append(_recall_sets(preds, trues))
    return float(np.mean(recalls))


def _ridge_fit(Xtr, Ytr, lam):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(d, dtype=np.float32)
    B = Xtr.T @ Ytr
    return np.linalg.solve(A, B)  # (d, n_experts)


def _pca_fit(Xtr, n_comp):
    """Train-only PCA: returns (mean, components (d, P)). Keeps n_comp <= rank."""
    mu = Xtr.mean(axis=0)
    Xc = Xtr - mu
    # economy SVD; components are right-singular vectors
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = min(n_comp, Vt.shape[0])
    return mu, Vt[:P].T  # (d, P)


def trained_recall(X: np.ndarray, E: np.ndarray, n_experts: int, lag: int,
                   kprime_mult: int, train_frac: float = 0.7, lam: float = 10.0,
                   n_pca: int = 256, include_current_experts: bool = False):
    """Per-layer ridge predictor X[t] -> multihot(E[t+lag]); recall on held-out tail.

    A train-only PCA projection (n_pca comps) conditions the problem so it is not
    underdetermined when samples < hidden dim. Optionally appends token t's
    current-expert multi-hot as features (a hybrid diagnostic that subsumes
    persistence). Returns (mean_recall, per_layer_recall, eval_predictions, eval_t).
    """
    T, L, k = E.shape
    kprime = k * kprime_mult
    per_layer = []
    eval_preds = {}
    eval_true_t = None
    for layer in range(L):
        Xl = X[:, layer, :].astype(np.float32)
        # input token index t in [0, T-lag); target = E[t+lag]
        Xin = Xl[: T - lag]
        n = Xin.shape[0]
        n_tr = int(n * train_frac)
        # standardize on train stats, then PCA-project (train-only fit)
        mu = Xin[:n_tr].mean(axis=0)
        sd = Xin[:n_tr].std(axis=0) + 1e-6
        Xn = (Xin - mu) / sd
        pmu, comps = _pca_fit(Xn[:n_tr], n_pca)
        Xp = (Xn - pmu) @ comps  # (n, P)
        feats = [Xp]
        if include_current_experts:
            cur = np.zeros((n, n_experts), np.float32)
            for i in range(n):
                cur[i, E[i, layer].astype(np.int64)] = 1.0
            feats.append(cur)
        Xf = np.concatenate(feats + [np.ones((n, 1), np.float32)], axis=1)
        Y = np.zeros((n, n_experts), np.float32)
        for i in range(n):
            Y[i, E[i + lag, layer].astype(np.int64)] = 1.0
        W = _ridge_fit(Xf[:n_tr], Y[:n_tr], lam)
        scores = Xf[n_tr:] @ W  # (n_eval, n_experts)
        topk = np.argpartition(scores, -kprime, axis=1)[:, -kprime:]
        preds, trues = [], []
        for j in range(scores.shape[0]):
            preds.append(topk[j])
            trues.append(E[n_tr + j + lag, layer])
        per_layer.append(_recall_sets(preds, trues))
        eval_preds[layer] = [topk[j] for j in range(scores.shape[0])]
        if eval_true_t is None:
            # global token indices (into E) of eval targets: n_tr..n-1 -> t+lag
            eval_true_t = list(range(n_tr + lag, n + lag))
    return float(np.mean(per_layer)), per_layer, eval_preds, eval_true_t


# ---------------------------------------------------------------------------
# real per-expert entry bytes (sum of 3 projections' one-expert rows)
# ---------------------------------------------------------------------------

def expert_entry_bytes(model_dir: str) -> int:
    """Bytes of one expert across gate/up/down proj (all quant parts), layer 0.

    Mirrors what the streaming DiskExpertStore caches per (layer, expert): the 3
    projections share the routing indices and are fetched together.
    """
    from .st_reader import STReader
    readers = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        r = STReader(f)
        for n in r.tensors:
            readers[n] = r
    total = 0
    parts = ("weight", "scales", "biases", "bias")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for layout in (
            f"model.layers.0.mlp.switch_mlp.{proj}",
            f"model.layers.0.mlp.experts.{proj}",
        ):
            hit = False
            for part in parts:
                name = f"{layout}.{part}"
                if name in readers:
                    m = readers[name].tensors[name]
                    total += m.nbytes // m.shape[0]  # one expert row
                    hit = True
            if hit:
                break
    return total


# ---------------------------------------------------------------------------
# miss-aware prefetch coverage
# ---------------------------------------------------------------------------

def miss_coverage(E: np.ndarray, X: np.ndarray, n_experts: int,
                  entry_bytes: int, budget_bytes: int, lag: int,
                  eval_preds: dict, eval_true_t, kprime_mult: int,
                  train_frac: float = 0.7):
    """Fraction of on-critical-path expert MISSES a prefetcher would cover.

    Replays a byte-budget SLRU over the full sequence feeding actual routed
    experts (key = (layer, expert)); records, per eval token/layer, which experts
    were misses. Then for persistence-prefetch (set = E[t-lag]) and
    trained-prefetch (set = eval_preds top-k' from X[t-lag]) reports
    |misses ∩ prefetch| / |misses|.
    """
    from .slru import SLRUCache
    T, L, k = E.shape
    cache = SLRUCache(budget_bytes)
    # miss_set[(t, layer)] = frozenset of missed experts at that access
    miss_set = {}
    eval_start = min(eval_true_t) if eval_true_t else T
    for t in range(T):
        for layer in range(L):
            missed = []
            for e in E[t, layer]:
                e = int(e)
                key = (layer, e)
                if cache.get(key) is None:
                    missed.append(e)
                    cache.put(key, True, entry_bytes)
            if t >= eval_start:
                miss_set[(t, layer)] = missed

    # persistence-prefetch coverage
    p_num = p_den = 0
    for (t, layer), missed in miss_set.items():
        if not missed:
            continue
        if kprime_mult == 1:
            pref = set(int(x) for x in E[t - lag, layer])
        else:
            pref = set(int(x) for x in E[t - lag, layer]) | set(
                int(x) for x in E[max(0, t - lag - 1), layer])
        p_num += len(set(missed) & pref)
        p_den += len(missed)

    # trained-prefetch coverage: map global token t -> eval prediction row
    # eval_preds[layer][j] predicts token eval_true_t[j]
    t_to_j = {tt: j for j, tt in enumerate(eval_true_t)}
    tr_num = tr_den = 0
    covered_tokens = 0
    for (t, layer), missed in miss_set.items():
        if not missed:
            continue
        j = t_to_j.get(t)
        if j is None:
            continue
        pref = set(int(x) for x in eval_preds[layer][j])
        tr_num += len(set(missed) & pref)
        tr_den += len(missed)
        covered_tokens += 1

    total_miss = sum(len(m) for m in miss_set.values())
    total_access = len(miss_set) * k
    return {
        "eval_hit_rate": round(1 - total_miss / max(total_access, 1), 4),
        "cache_capacity_experts": budget_bytes // max(entry_bytes, 1),
        "persistence_miss_coverage": round(p_num / max(p_den, 1), 4),
        "trained_miss_coverage": round(tr_num / max(tr_den, 1), 4),
        "misses_scored": tr_den,
    }
