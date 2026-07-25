"""Bet B predictability capture: per-decode-step (hidden_state, routed_experts).

Generalizes the Experiment-0.1 GateTap to record, for every MoE layer and every
DECODE step, both the router/gate INPUT hidden state (the signal available during
token t's compute) and the actually-routed top-k expert ids. Works for:

  - Qwen3 MoE   : block exposes `.gate` (nn.Linear) + `.top_k`
  - GPT-OSS MoE : block exposes `.router` (nn.Linear) + `.num_experts_per_tok`

The tap wraps the gate/router Linear and returns its logits unchanged, so model
math is untouched. It records x as float16 and expert ids as int16 to keep the
trace compact enough to hold thousands of tokens in RAM.
"""

import numpy as np
import mlx.core as mx

BOUNDARY = "BOUNDARY"


class HiddenGateTap:
    """Wraps a MoE block's gate/router Linear; records (x_hidden, top_k ids)."""

    def __init__(self, linear, k: int, layer_idx: int, store: list):
        self.linear = linear
        self.k = k
        self.layer = layer_idx
        self.store = store

    def __call__(self, x):
        logits = self.linear(x)
        idx = mx.argpartition(logits, kth=-self.k, axis=-1)[..., -self.k :]
        # x: (..., d). Cast to f32 in MLX first (bf16 has no numpy dtype), then
        # store compactly as f16. Force materialization once (already needed
        # downstream by the router/expert path).
        x_np = np.array(x.astype(mx.float32), copy=False).astype(np.float16)
        idx_np = np.array(idx.astype(mx.int32), copy=False).astype(np.int16)
        self.store.append((self.layer, x_np, idx_np))
        return logits


def install_hidden_taps(model, store: list) -> int:
    """Replace each MoE block's gate/router Linear with a HiddenGateTap.

    Returns the number of taps installed.
    """
    count = 0
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if hasattr(mlp, "gate") and hasattr(mlp, "top_k"):
            mlp.gate = HiddenGateTap(mlp.gate, mlp.top_k, i, store)
            count += 1
        elif hasattr(mlp, "router") and hasattr(mlp, "num_experts_per_tok"):
            mlp.router = HiddenGateTap(
                mlp.router, mlp.num_experts_per_tok, i, store
            )
            count += 1
    return count


def build_arrays(store: list, n_layers: int):
    """Group a flat store into per-generation decode traces.

    Returns (X, E):
      X: float16 (T, L, d)  gate-input hidden state per token per layer
      E: int16   (T, L, k)  routed top-k expert ids per token per layer
    Only DECODE steps (single-token gate calls) are kept; prefill (multi-token)
    calls are dropped. Segments are split on BOUNDARY markers; each segment's
    complete layer-cycles become tokens.
    """
    segments, cur = [], []
    for rec in store:
        if rec[0] == BOUNDARY:
            if cur:
                segments.append(cur)
            cur = []
        else:
            cur.append(rec)
    if cur:
        segments.append(cur)

    all_X, all_E = [], []
    for seg in segments:
        current_x, current_e = {}, {}
        for layer, x_np, idx_np in seg:
            xf = x_np.reshape(-1, x_np.shape[-1])
            ef = idx_np.reshape(-1, idx_np.shape[-1])
            if xf.shape[0] != 1:
                continue  # prefill chunk
            current_x[layer] = xf[0]
            current_e[layer] = ef[0]
            if len(current_x) == n_layers:
                layers = sorted(current_x)
                all_X.append(np.stack([current_x[l] for l in layers]))
                all_E.append(np.stack([current_e[l] for l in layers]))
                current_x, current_e = {}, {}
    if not all_X:
        raise ValueError("no complete decode tokens captured")
    X = np.stack(all_X)  # (T, L, d)
    E = np.stack(all_E)  # (T, L, k)
    return X, E
