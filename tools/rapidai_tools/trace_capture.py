"""MLX router-logging harness for Experiment 0.1.

Works with any mlx_lm MoE model whose MoE block exposes `.gate` (nn.Linear
producing per-expert logits) and `.top_k` — verified for Qwen3MoeSparseMoeBlock
(qwen3_moe.py) and OlmoeSparseMoeBlock (olmoe.py).
"""

import numpy as np
import mlx.core as mx

BOUNDARY = "BOUNDARY"


class GateTap:
    """Wraps a MoE block's gate Linear; records top-k expert ids per call."""

    def __init__(self, gate, k: int, layer_idx: int, store: list):
        self.gate = gate
        self.k = k
        self.layer = layer_idx
        self.store = store

    def __call__(self, x):
        logits = self.gate(x)
        idx = mx.argpartition(logits, kth=-self.k, axis=-1)[..., -self.k :]
        self.store.append((self.layer, np.array(idx, copy=True)))
        return logits


def install_taps(model, store: list) -> int:
    """Replace each MoE block's .gate with a GateTap. Returns #taps installed."""
    count = 0
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "top_k"):
            mlp.gate = GateTap(mlp.gate, mlp.top_k, i, store)
            count += 1
    return count


def split_on_boundaries(store: list) -> list:
    """Group store entries into segments delimited by (BOUNDARY, None) markers."""
    segments, cur = [], []
    for layer, ids in store:
        if layer == BOUNDARY:
            if cur:
                segments.append(cur)
            cur = []
        else:
            cur.append((layer, ids))
    if cur:
        segments.append(cur)
    return segments


def to_trace_array(segment: list) -> np.ndarray:
    """Stack one segment's decode-step records into (T, L, K) int16.

    Decode steps are entries whose id array has exactly one token position
    (prefill entries carry the whole prompt at once and are skipped).
    """
    layers = sorted({layer for layer, _ in segment})
    layer_pos = {l: p for p, l in enumerate(layers)}
    per_token: list = []  # list of (L, K) rows, appended layer by layer
    current: dict = {}
    for layer, ids in segment:
        flat = ids.reshape(-1, ids.shape[-1])
        if flat.shape[0] != 1:
            continue  # prefill chunk — skip
        current[layer_pos[layer]] = flat[0]
        if len(current) == len(layers):
            row = np.stack([current[p] for p in range(len(layers))])
            per_token.append(row)
            current = {}
    if not per_token:
        raise ValueError("no decode steps captured in segment")
    return np.stack(per_token).astype(np.int16)
