"""Experiment 0.1 — lookahead routing decay.

Capture per-token expert routing on generated text, then measure how the
expert-set overlap between token t and t+lag decays for lag = 1..8, versus
the uniform-random baseline k/n.

Usage:
  .venv/bin/python tools/scripts/run_exp01_routing.py --model models/olmoe-1b-7b-4bit --tag olmoe --pilot
  .venv/bin/python tools/scripts/run_exp01_routing.py --model models/qwen3-30b-a3b-4bit --tag qwen3_30b
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlx_lm import load, generate

from rapidai_tools.trace_capture import (
    BOUNDARY,
    install_taps,
    split_on_boundaries,
    to_trace_array,
)
from rapidai_tools.routing_stats import overlap_at_lag, random_baseline

ROOT = Path("/Volumes/x9/rapidAI")
MAX_LAG = 8

PROMPTS = [
    "Write a detailed essay about the history of the Ottoman Empire.",
    "Explain, step by step, how a B-tree database index works, with code examples.",
    "Write a short story about a lighthouse keeper on the Aegean coast.",
    "Derive the gradient of the softmax cross-entropy loss, showing every step.",
    "Translate the opening paragraph of Don Quixote into modern English and discuss its themes.",
] * 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    prompts = PROMPTS[:2] if args.pilot else PROMPTS
    max_tokens = 256 if args.pilot else args.max_tokens

    model, tokenizer = load(str(ROOT / args.model))
    store: list = []
    n_taps = install_taps(model, store)
    print(f"taps installed: {n_taps}")
    assert n_taps > 0, "no MoE gates found — model layout unexpected"

    for i, p in enumerate(prompts):
        store.append((BOUNDARY, None))
        messages = [{"role": "user", "content": p}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        mx.clear_cache()  # release KV/cache buffers between generations
        print(f"  generation {i + 1}/{len(prompts)} done ({len(store)} records)", flush=True)

    traces = [to_trace_array(seg) for seg in split_on_boundaries(store)]
    k = traces[0].shape[2]
    n = model.args.num_experts
    base = random_baseline(k, n)

    curve, ratio = {}, {}
    for lag in range(1, MAX_LAG + 1):
        vals = [overlap_at_lag(tr, lag) for tr in traces if tr.shape[0] > lag]
        curve[lag] = float(np.mean(vals))
        ratio[lag] = curve[lag] / base

    total_tokens = int(sum(tr.shape[0] for tr in traces))
    result = {
        "experiment": "0.1 lookahead routing decay",
        "model": args.model,
        "pilot": args.pilot,
        "tokens_captured": total_tokens,
        "k": int(k),
        "n_experts": int(n),
        "random_baseline": base,
        "overlap_by_lag": curve,
        "ratio_by_lag": ratio,
        "gate_threshold": 1.5,
        "gate": "live" if ratio[2] >= 1.5 else "kill",
    }

    suffix = "_pilot" if args.pilot else ""
    out_json = ROOT / f"docs/experiments/data/exp01_routing_{args.tag}{suffix}.json"
    out_json.write_text(json.dumps(result, indent=2))
    traces_dir = ROOT / "data/traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(traces_dir / f"routing_{args.tag}{suffix}.npz", *traces)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(ratio.keys()), list(ratio.values()), "o-", label=f"{args.tag} (k={k}, n={n})")
    ax.axhline(1.5, color="red", ls="--", label="live gate (1.5x)")
    ax.axhline(1.0, color="gray", ls=":", label="random baseline")
    ax.set_xlabel("lookahead lag (tokens)")
    ax.set_ylabel("overlap / random baseline")
    ax.set_title(f"Expert-routing overlap decay — {args.tag}, {total_tokens} tokens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / f"docs/experiments/data/exp01_decay_{args.tag}{suffix}.png", dpi=140)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
