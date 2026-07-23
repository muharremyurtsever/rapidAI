"""Experiment 0.3 — expert delta spectrum.

For sampled layers and each expert projection, decompose experts as
mean-base + delta and measure how much delta energy a low-rank SVD captures,
versus the same curve on the raw expert matrices (control).

Usage:
  .venv/bin/python tools/scripts/run_exp03_delta.py --model models/qwen3-30b-a3b-bf16 --tag qwen3_30b
  .venv/bin/python tools/scripts/run_exp03_delta.py --model models/qwen3-30b-a3b-bf16 --tag qwen3_30b --pilot
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rapidai_tools.delta_spectrum import energy_at_rank

ROOT = Path("/Volumes/x9/rapidAI")
PROJS = ("gate_proj", "up_proj", "down_proj")
RANK_FRACS = (0.01, 0.05, 0.10, 0.25, 0.50)
GATE_RANK_FRAC = 0.25
GATE_ENERGY = 0.70


def load_expert_stack(model_dir: Path, weight_map: dict, layer: int, proj: str) -> np.ndarray:
    """Return (E, out, in) float32 stack of one layer's expert matrices.

    Uses mx.load per shard (native bf16 support); shards are loaded lazily and
    released after use so peak memory stays ~one shard (~4 GB).
    """
    keys = {}
    e = 0
    while True:
        key = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
        shard = weight_map.get(key)
        if shard is None:
            break
        keys.setdefault(shard, []).append((e, key))
        e += 1
    if not keys:
        raise KeyError(f"no experts found for layer {layer} {proj}")
    experts = [None] * e
    for shard, entries in keys.items():
        tensors = mx.load(str(model_dir / shard))
        for idx, key in entries:
            experts[idx] = np.array(tensors[key].astype(mx.float32), copy=False)
        del tensors
        mx.clear_cache()
    return np.stack(experts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    model_dir = ROOT / args.model
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    n_layers = 1 + max(
        int(k.split(".")[2]) for k in weight_map if k.startswith("model.layers.")
    )
    layers = [0] if args.pilot else sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})
    projs = PROJS[:1] if args.pilot else PROJS

    results = defaultdict(dict)
    for layer in layers:
        for proj in projs:
            stack = load_expert_stack(model_dir, weight_map, layer, proj)
            base = stack.mean(axis=0)
            deltas = stack - base
            full_rank = min(stack.shape[1], stack.shape[2])
            curve_delta, curve_raw = {}, {}
            for frac in RANK_FRACS:
                r = max(1, int(full_rank * frac))
                curve_delta[frac] = energy_at_rank(deltas, r)
                curve_raw[frac] = energy_at_rank(stack, r)
            results[f"L{layer}"][proj] = {
                "shape": list(stack.shape),
                "full_rank": full_rank,
                "delta_energy_by_rank_frac": curve_delta,
                "raw_energy_by_rank_frac": curve_raw,
            }
            print(f"L{layer} {proj}: delta@25%={curve_delta[0.25]:.3f} raw@25%={curve_raw[0.25]:.3f}")

    mean_delta_25 = float(np.mean(
        [r[p]["delta_energy_by_rank_frac"][GATE_RANK_FRAC] for r in results.values() for p in r]
    ))
    out = {
        "experiment": "0.3 expert delta spectrum",
        "model": args.model,
        "pilot": args.pilot,
        "layers": layers,
        "gate_metric": f"mean delta energy at rank {GATE_RANK_FRAC:.0%}",
        "gate_threshold": GATE_ENERGY,
        "mean_delta_energy_at_25pct_rank": mean_delta_25,
        "gate": "live" if mean_delta_25 >= GATE_ENERGY else "kill",
        "per_layer": dict(results),
    }
    suffix = "_pilot" if args.pilot else ""
    (ROOT / f"docs/experiments/data/exp03_delta_{args.tag}{suffix}.json").write_text(
        json.dumps(out, indent=2)
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fracs = list(RANK_FRACS)
    for lname, projres in results.items():
        for proj, r in projres.items():
            ax.plot(fracs, [r["delta_energy_by_rank_frac"][f] for f in fracs], "o-",
                    alpha=0.7, label=f"{lname}/{proj} delta")
            ax.plot(fracs, [r["raw_energy_by_rank_frac"][f] for f in fracs], "x--",
                    alpha=0.4, label=f"{lname}/{proj} raw")
    ax.axhline(GATE_ENERGY, color="red", ls="--")
    ax.set_xlabel("rank fraction")
    ax.set_ylabel("Frobenius energy captured")
    ax.set_title(f"Expert delta SVD spectrum — {args.tag}")
    if len(results) <= 2:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / f"docs/experiments/data/exp03_spectrum_{args.tag}{suffix}.png", dpi=140)
    print(json.dumps({k: v for k, v in out.items() if k != "per_layer"}, indent=2))


if __name__ == "__main__":
    main()
